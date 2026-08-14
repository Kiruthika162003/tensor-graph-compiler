from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import mlp_graph
from tgc.ir.graph import Graph, Node
from tgc.passes.fusion import fused_bytes, traffic_ratio, unfused_bytes

# What a graph costs, roughly.
#
# A cost model that claims to predict nanoseconds is wrong in a way that is hard to notice: it
# is calibrated on one machine, it is off by a factor on the next one, and nobody finds out
# because nothing checks it. A cost model that claims only to rank alternatives is wrong in a
# way that shows up the moment the autotuner disagrees with a measurement, which is where the
# disagreement belongs.
#
# So this is a roofline and nothing more. Work over the arithmetic peak, bytes over the
# bandwidth, and whichever is larger is the answer. It gets the shape of the answer right,
# which is what an optimiser needs to choose between two versions of the same computation, and
# it will not tell you how long anything takes.


@dataclass
class Machine:
    """The two numbers a roofline needs."""

    flops_per_second: float = 20e12
    bytes_per_second: float = 1.5e12
    name: str = "reference"

    def __post_init__(self) -> None:
        if self.flops_per_second <= 0 or self.bytes_per_second <= 0:
            raise ConfigError("both peaks have to be positive")

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at which a kernel stops being memory bound.

        Below this, more arithmetic is free. Above it, less traffic is free. Every fusion
        decision in the compiler is a bet about which side of this number a kernel sits on.
        """
        return self.flops_per_second / self.bytes_per_second

    def as_dict(self) -> dict[str, float | str]:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "flops_per_second": self.flops_per_second,
            "bytes_per_second": self.bytes_per_second,
            "ridge_point": round(self.ridge_point, 2),
        }


CPU = Machine(flops_per_second=500e9, bytes_per_second=50e9, name="cpu")
GPU = Machine(flops_per_second=20e12, bytes_per_second=1.5e12, name="gpu")
BANDWIDTH_STARVED = Machine(
    flops_per_second=100e12, bytes_per_second=1e12, name="bandwidth starved"
)

MACHINES = {machine.name: machine for machine in (CPU, GPU, BANDWIDTH_STARVED)}


def get_machine(name: str) -> Machine:
    """Look up a machine by name."""
    if name not in MACHINES:
        raise ConfigError(f"unknown machine {name!r}, expected one of {sorted(MACHINES)}")
    return MACHINES[name]


def node_elements(node: Node) -> int:
    """How many output elements a node produces, when that is knowable."""
    if not node.output.shape.is_static:
        raise ConfigError(f"{node.name} has a symbolic shape and cannot be costed")
    return node.output.shape.elements


def node_flops(node: Node) -> float:
    """Arithmetic one node performs.

    A matrix product is the exception that matters: its work is not proportional to its
    output, it is proportional to the output times the contracted dimension, and a cost model
    that misses that ranks a matmul alongside an addition.
    """
    if node.op.is_leaf:
        return 0.0
    cost = ops.cost_of(node.op)
    if node.op is ops.MATMUL:
        contracted = node.attrs.get("contracted")
        if contracted is None:
            raise ConfigError(f"{node.name} needs its contracted dimension to be costed")
        return cost.flops_per_element * node_elements(node) * contracted
    return cost.flops_per_element * node_elements(node)


def annotate_matmuls(graph: Graph) -> Graph:
    """Record the contracted dimension on every matrix product.

    Derived rather than stored at construction because it is a property of the operands and
    the builder should not have to know that the cost model exists. Recomputing it here keeps
    one definition of what a matmul costs.
    """
    annotated = []
    for node in graph.nodes:
        if node.op is not ops.MATMUL:
            annotated.append(node)
            continue
        left = graph.value(node.inputs[0])
        inner = left.shape.sizes[-1]
        if not inner.is_static:
            annotated.append(node)
            continue
        attrs = dict(node.attrs)
        attrs["contracted"] = inner.value
        annotated.append(Node(op=node.op, inputs=node.inputs, output=node.output, attrs=attrs))
    return graph.with_nodes(annotated)


def graph_flops(graph: Graph) -> float:
    """Total arithmetic a graph performs."""
    prepared = annotate_matmuls(graph)
    return sum(node_flops(node) for node in prepared.nodes)


@dataclass
class RooflineEstimate:
    """A prediction of which resource a graph runs out of first."""

    flops: float
    bytes_moved: float
    machine: Machine

    @property
    def compute_seconds(self) -> float:
        """Time if arithmetic were the only limit."""
        return self.flops / self.machine.flops_per_second

    @property
    def memory_seconds(self) -> float:
        """Time if bandwidth were the only limit."""
        return self.bytes_moved / self.machine.bytes_per_second

    @property
    def seconds(self) -> float:
        """The larger of the two, which is the roofline."""
        return max(self.compute_seconds, self.memory_seconds)

    @property
    def intensity(self) -> float:
        """Work done per byte moved."""
        if self.bytes_moved == 0:
            return 0.0
        return self.flops / self.bytes_moved

    @property
    def is_memory_bound(self) -> bool:
        """Whether the kernel sits below the ridge point."""
        return self.intensity < self.machine.ridge_point

    @property
    def utilisation(self) -> float:
        """Share of the machine's arithmetic peak the graph reaches."""
        if self.seconds == 0:
            return 0.0
        return self.flops / (self.seconds * self.machine.flops_per_second)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "flops": self.flops,
            "bytes": self.bytes_moved,
            "intensity": round(self.intensity, 4),
            "seconds": self.seconds,
            "memory_bound": self.is_memory_bound,
            "utilisation": round(self.utilisation, 4),
        }


def estimate(graph: Graph, machine: Machine = GPU, *, fused: bool = True) -> RooflineEstimate:
    """Roofline cost of a graph, fused or not."""
    moved = fused_bytes(graph) if fused else unfused_bytes(graph)
    return RooflineEstimate(flops=graph_flops(graph), bytes_moved=float(moved), machine=machine)


def fusion_speedup(graph: Graph, machine: Machine = GPU) -> float:
    """How much faster the roofline says a fused graph runs.

    Bounded by the traffic ratio and usually below it, because fusing a chain that was
    already compute bound saves bytes nobody was waiting on. That is the honest version of
    the fusion claim: the traffic falls by the full factor and the time does not.
    """
    unfused = estimate(graph, machine, fused=False).seconds
    fused = estimate(graph, machine, fused=True).seconds
    if fused == 0:
        raise ConfigError("a graph that takes no time cannot be compared")
    return unfused / fused


def compare_machines(graph: Graph) -> list[dict]:
    """The same graph against machines with different balances.

    Whether fusion is worth anything depends on the ridge point, and a compiler tuned on one
    machine makes decisions that are merely harmless on another.
    """
    rows = []
    for machine in MACHINES.values():
        result = estimate(graph, machine)
        row = result.as_dict()
        row["machine"] = machine.name
        row["fusion_speedup"] = round(fusion_speedup(graph, machine), 4)
        rows.append(row)
    return rows


def rank_by_cost(graphs: Sequence[tuple[str, Graph]], machine: Machine = GPU) -> list[str]:
    """Order candidate graphs cheapest first.

    The only thing this model is asked to do. An autotuner uses the ranking to decide what to
    measure, and the measurement decides what to keep.
    """
    if not graphs:
        raise ConfigError("there is nothing to rank")
    return [
        name for name, _ in sorted(graphs, key=lambda item: estimate(item[1], machine).seconds)
    ]


def kendall_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    """How closely two rankings agree, from minus one to one.

    Used to say how much the cost model's ordering resembles a measured one. A number near
    one means the model can be trusted to prune; a number near zero means it is choosing at
    random and the autotuner is doing all the work.
    """
    if sorted(first) != sorted(second):
        raise ConfigError("the two rankings have to cover the same candidates")
    if len(first) < 2:
        raise ConfigError("a ranking of one has no order to compare")

    positions = {name: index for index, name in enumerate(second)}
    concordant = 0
    discordant = 0
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            if positions[first[i]] < positions[first[j]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total


def traffic_against_speedup(graph: Graph, machine: Machine = GPU) -> dict:
    """What fusion saves in bytes against what it saves in time.

    The two are the same number only while the kernel is memory bound. Past the ridge point
    the traffic still falls by the full factor and the time stops moving, because the bytes
    fusion removes were not the ones anybody was waiting on.
    """
    result = estimate(graph, machine)
    return {
        "intensity": round(result.intensity, 2),
        "memory_bound": result.is_memory_bound,
        "traffic_ratio": round(traffic_ratio(graph), 4),
        "speedup": round(fusion_speedup(graph, machine), 4),
    }


def size_sweep(
    sizes: Sequence[tuple[int, int]] = ((8, 64), (128, 512), (512, 1024), (2048, 2048)),
    machine: Machine = GPU,
) -> list[dict]:
    """A feed forward block at several sizes, across the ridge point.

    The small one is memory bound and fusion buys everything the traffic ratio promises. The
    large one is compute bound and fusion buys nothing at all, while still reporting a
    perfectly real 24 percent reduction in bytes moved. A compiler that reports the byte
    number as a speedup is not lying about the bytes.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for batch, hidden in sizes:
        graph = mlp_graph(batch=batch, hidden=hidden)
        row = traffic_against_speedup(graph, machine)
        row["batch"] = batch
        row["hidden"] = hidden
        rows.append(row)
    return rows
