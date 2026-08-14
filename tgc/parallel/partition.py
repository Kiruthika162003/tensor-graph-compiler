from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.cost import annotate_matmuls, node_flops
from tgc.errors import ConfigError, ScheduleError
from tgc.ir.graph import Graph
from tgc.schedule.order import depth_first_order

# Splitting one graph across several devices.
#
# An assignment of nodes to devices decides two things that pull against each other. Every
# edge whose two ends land on different devices becomes a transfer, and every device's share
# of the arithmetic decides how long it works while the others wait. Minimising the transfers
# alone puts everything on one device; balancing the work alone cuts every edge.
#
# The measurement that matters is neither number on its own. It is the time the slowest device
# spends, plus the bytes crossing the links, weighted by how much slower a link is than the
# arithmetic. On a chain the answer is a contiguous split and there is nothing to discuss. On a
# graph with width there is a real choice, and a round robin assignment, which is what a naive
# scheduler produces, cuts almost every edge in the graph.


@dataclass
class Assignment:
    """Which device each node runs on."""

    devices: int
    placement: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.devices < 1:
            raise ConfigError(f"there has to be at least one device, got {self.devices}")
        wrong = {
            name: index
            for name, index in self.placement.items()
            if not 0 <= index < self.devices
        }
        if wrong:
            raise ConfigError(f"these nodes are placed on devices that do not exist: {wrong}")

    def device_of(self, name: str) -> int:
        """Where a value is produced."""
        if name not in self.placement:
            raise ScheduleError(f"{name} has not been placed")
        return self.placement[name]

    def nodes_on(self, device: int) -> list[str]:
        """Everything assigned to one device."""
        return sorted(name for name, index in self.placement.items() if index == device)

    @property
    def used_devices(self) -> int:
        """How many devices have anything on them."""
        return len(set(self.placement.values()))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "devices": self.devices,
            "used": self.used_devices,
            "placed": len(self.placement),
        }


@dataclass
class PartitionReport:
    """What one assignment costs."""

    strategy: str
    cut_edges: int = 0
    transferred_bytes: int = 0
    work_per_device: list[float] = field(default_factory=list)

    @property
    def total_work(self) -> float:
        """Arithmetic across the whole graph."""
        return sum(self.work_per_device)

    @property
    def slowest_device(self) -> float:
        """Work the busiest device does, which is what everybody waits for."""
        return max(self.work_per_device, default=0.0)

    @property
    def balance(self) -> float:
        """How evenly the work is spread across the devices that got any, from zero to one.

        The mean over the maximum, counting only devices with work on them. Counting the empty
        ones instead reports a single device placement as badly balanced, which is the wrong
        word for it: that placement is perfectly balanced over the one device it uses and its
        problem is that it uses one. The makespan already says that.
        """
        busy = [work for work in self.work_per_device if work > 0]
        if not busy or self.slowest_device == 0:
            return 1.0
        return (sum(busy) / len(busy)) / self.slowest_device

    def makespan(
        self, flops_per_second: float = 20e12, bytes_per_second: float = 50e9
    ) -> float:
        """Time the whole graph takes: the slowest device plus the transfers.

        The link is deliberately three orders of magnitude slower than the arithmetic, which
        is roughly the ratio between an interconnect and a compute unit and is why cut edges
        dominate any partition that was chosen for balance alone.
        """
        if flops_per_second <= 0 or bytes_per_second <= 0:
            raise ConfigError("both rates have to be positive")
        compute = self.slowest_device / flops_per_second
        transfer = self.transferred_bytes / bytes_per_second
        return compute + transfer

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "cut_edges": self.cut_edges,
            "transferred_bytes": self.transferred_bytes,
            "balance": round(self.balance, 4),
            "makespan": round(self.makespan(), 9),
        }


def evaluate(graph: Graph, assignment: Assignment, strategy: str = "") -> PartitionReport:
    """Cut edges, transferred bytes and work per device for one assignment."""
    prepared = annotate_matmuls(graph)
    work = [0.0] * assignment.devices
    cut = 0
    moved = 0

    for node in prepared.nodes:
        device = assignment.device_of(node.name)
        work[device] += node_flops(node)
        for operand in node.inputs:
            if assignment.device_of(operand) != device:
                cut += 1
                moved += graph.value(operand).bytes

    return PartitionReport(
        strategy=strategy,
        cut_edges=cut,
        transferred_bytes=moved,
        work_per_device=work,
    )


def place_everything(graph: Graph, devices: int = 4) -> Assignment:
    """Put the whole graph on one device.

    The baseline nobody proposes and every partition has to beat. It cuts nothing and uses one
    device, so a partition that is slower than this is a partition that should not exist.
    """
    placement = dict.fromkeys(graph.value_names, 0)
    return Assignment(devices=devices, placement=placement)


def place_contiguous(graph: Graph, devices: int = 4) -> Assignment:
    """Split the schedule into consecutive stages, one per device.

    Pipeline parallelism. On a chain it cuts exactly one edge per boundary, which is the least
    any partition using every device can cut, and the balance is whatever the schedule happened
    to give.
    """
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    order = depth_first_order(graph)
    per_device = max(1, -(-len(order) // devices))

    placement = {}
    for index, node in enumerate(order):
        placement[node.name] = min(devices - 1, index // per_device)
    for value in graph.inputs:
        placement[value.name] = 0
    return Assignment(devices=devices, placement=placement)


def place_round_robin(graph: Graph, devices: int = 4) -> Assignment:
    """Hand each node to the next device in turn.

    The worst thing anybody can do, and it does not even buy what it was meant to. It balances
    the node count exactly and the arithmetic not at all, since an exponential costs eight
    times an addition, so on a chain of alternating operations it lands at 0.56 balance while
    cutting 94 percent of the edges. Contiguous cuts 19 percent of them and comes out at 1.0.
    """
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    order = depth_first_order(graph)
    placement = {node.name: index % devices for index, node in enumerate(order)}
    for value in graph.inputs:
        placement[value.name] = 0
    return Assignment(devices=devices, placement=placement)


def place_by_balance(graph: Graph, devices: int = 4) -> Assignment:
    """Fill each device to an equal share of the arithmetic before moving on.

    Contiguous like the pipeline split and balanced by work rather than by node count, which
    matters as soon as the graph holds a matrix product next to an addition. Still cuts one
    edge per boundary on a chain, because the stages stay consecutive.
    """
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    prepared = annotate_matmuls(graph)
    order = depth_first_order(prepared)
    costs = {node.name: node_flops(node) for node in prepared.nodes}
    target = sum(costs.values()) / devices

    placement = {}
    device = 0
    accumulated = 0.0
    for node in order:
        if accumulated > target and device < devices - 1:
            device += 1
            accumulated = 0.0
        placement[node.name] = device
        accumulated += costs[node.name]
    for value in graph.inputs:
        placement[value.name] = 0
    return Assignment(devices=devices, placement=placement)


STRATEGIES = {
    "one device": place_everything,
    "contiguous": place_contiguous,
    "balanced": place_by_balance,
    "round robin": place_round_robin,
}


def get_strategy(name: str):
    """Look up a placement strategy by name."""
    if name not in STRATEGIES:
        raise ConfigError(f"unknown strategy {name!r}, expected one of {sorted(STRATEGIES)}")
    return STRATEGIES[name]


def compare_strategies(graph: Graph, devices: int = 4) -> list[dict]:
    """Every placement on the same graph, with cuts, balance and total time.

    Round robin has the best balance and the worst time by a wide margin, which is the whole
    lesson: balancing the arithmetic while ignoring the links optimises the cheaper of the two
    resources.
    """
    rows = []
    for name, strategy in STRATEGIES.items():
        rows.append(evaluate(graph, strategy(graph, devices), name).as_dict())
    return rows


def best_strategy(graph: Graph, devices: int = 4) -> str:
    """Whichever placement finishes soonest."""
    rows = compare_strategies(graph, devices)
    return min(rows, key=lambda row: (row["makespan"], row["strategy"]))["strategy"]


def device_sweep(graph: Graph, counts: Sequence[int] = (1, 2, 4, 8)) -> list[dict]:
    """The contiguous split across a range of device counts.

    More devices means more boundaries and more transfers, and the work per device falls
    linearly while the transfers rise. Past some count the links cost more than the arithmetic
    saved, and where that sits depends entirely on how fast the link is.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for devices in counts:
        report = evaluate(graph, place_contiguous(graph, devices), "contiguous")
        row = report.as_dict()
        row["devices"] = devices
        rows.append(row)
    return rows


def link_speed_sweep(
    graph: Graph, devices: int = 4, speeds: Sequence[float] = (5e9, 50e9, 500e9, 5e12)
) -> list[dict]:
    """Which placement wins at each interconnect speed.

    A fast enough link makes round robin competitive and a slow one makes a single device the
    right answer, and neither of those is a statement about the graph. A partitioner without
    the link speed in it is choosing on incomplete information.
    """
    if not speeds:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for speed in speeds:
        timings = {}
        for name, strategy in STRATEGIES.items():
            report = evaluate(graph, strategy(graph, devices), name)
            timings[name] = report.makespan(bytes_per_second=speed)
        winner = min(timings, key=lambda name: (timings[name], name))
        rows.append(
            {
                "bytes_per_second": speed,
                "winner": winner,
                "seconds": round(timings[winner], 9),
            }
        )
    return rows


def cut_edges_of(graph: Graph, assignment: Assignment) -> list[tuple[str, str]]:
    """Every edge that crosses a device boundary."""
    crossing = []
    for node in graph.nodes:
        device = assignment.device_of(node.name)
        for operand in node.inputs:
            if assignment.device_of(operand) != device:
                crossing.append((operand, node.name))
    return crossing


def total_edges(graph: Graph) -> int:
    """Every edge in the graph, for comparing against the cut."""
    return sum(len(node.inputs) for node in graph.nodes)


def cut_fraction(graph: Graph, assignment: Assignment) -> float:
    """Share of the graph's edges that become transfers."""
    edges = total_edges(graph)
    if edges == 0:
        return 0.0
    return len(cut_edges_of(graph, assignment)) / edges
