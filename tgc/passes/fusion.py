from __future__ import annotations

from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.graph import Graph, Node

# Running a chain of elementwise operations as one loop.
#
# An elementwise op reads element i of each input and writes element i of its output. A chain
# of them therefore does not need the intermediates to exist: element i can be carried
# through the whole chain in a register and written once. A chain of length n writes n
# tensors and reads n of them; fused it writes one and reads one.
#
# The condition is not that the ops are elementwise. It is that every intermediate is read
# exactly once, by the next op in the chain, and is not itself a graph output. A value read
# twice can still be fused into both readers, and doing so computes it twice. That is
# sometimes the right trade and it is never free, so this pass will not make it silently.
#
# Reductions end a group. A sum reads every element of its input to produce one output, so
# there is no element i to carry, and fusing an elementwise op into the reduction's consumer
# would need the reduction to have finished first. The elementwise chain feeding a reduction
# can be fused into it, which is a different transformation and is where most of the win on a
# softmax comes from.


@dataclass
class FusionGroup:
    """A run of nodes that can execute as one loop."""

    members: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    output: str = ""

    def __post_init__(self) -> None:
        if self.members and not self.output:
            self.output = self.members[-1]

    @property
    def size(self) -> int:
        """Nodes in the group."""
        return len(self.members)

    @property
    def is_trivial(self) -> bool:
        """Whether the group is a single node, which fusing does nothing for."""
        return self.size <= 1

    @property
    def intermediates(self) -> list[str]:
        """Values that no longer need to exist."""
        return self.members[:-1]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "size": self.size,
            "members": list(self.members),
            "inputs": list(self.inputs),
            "output": self.output,
        }


def can_start_group(node: Node) -> bool:
    """Whether a node can be the first in a fused loop."""
    return node.op.is_elementwise


def can_extend_group(
    node: Node, previous: str, use_counts: dict[str, int], outputs: set[str]
) -> bool:
    """Whether a node can join the group ending at a value.

    Three conditions, and the second is the one that gets skipped. The node has to be
    elementwise, it has to read the previous value exactly once in the whole graph, and the
    previous value must not be something the caller asked for. Fusing over a value that is
    read twice duplicates the work; fusing over a graph output deletes a result.
    """
    if not node.op.is_elementwise:
        return False
    if previous not in node.inputs:
        return False
    if use_counts.get(previous, 0) != 1:
        return False
    return previous not in outputs


def find_groups(graph: Graph) -> list[FusionGroup]:
    """Every maximal run of elementwise nodes that can share a loop."""
    use_counts = graph.use_counts()
    outputs = set(graph.outputs)
    producers = {node.name: node for node in graph.nodes}

    grouped: set[str] = set()
    groups: list[FusionGroup] = []

    for node in graph.nodes:
        if node.name in grouped or not can_start_group(node):
            continue
        members = [node.name]
        grouped.add(node.name)
        current = node.name

        while True:
            readers = [
                candidate
                for candidate in graph.nodes
                if candidate.name not in grouped
                and can_extend_group(candidate, current, use_counts, outputs)
            ]
            if len(readers) != 1:
                break
            following = readers[0]
            members.append(following.name)
            grouped.add(following.name)
            current = following.name

        inputs = _group_inputs(members, producers)
        groups.append(FusionGroup(members=members, inputs=inputs))
    return groups


def _group_inputs(members: list[str], producers: dict[str, Node]) -> list[str]:
    """Values a group reads from outside itself."""
    inside = set(members)
    seen: list[str] = []
    for name in members:
        for operand in producers[name].inputs:
            if operand not in inside and operand not in seen:
                seen.append(operand)
    return seen


@dataclass
class FusionReport:
    """What fusion would do to a graph."""

    groups: list[FusionGroup] = field(default_factory=list)

    @property
    def fused_groups(self) -> list[FusionGroup]:
        """Groups holding more than one node."""
        return [group for group in self.groups if not group.is_trivial]

    @property
    def nodes_fused(self) -> int:
        """Nodes that end up inside a multi node loop."""
        return sum(group.size for group in self.fused_groups)

    @property
    def buffers_removed(self) -> int:
        """Intermediates that no longer have to be written and read back."""
        return sum(group.size - 1 for group in self.fused_groups)

    @property
    def largest_group(self) -> int:
        """The longest chain found."""
        return max((group.size for group in self.groups), default=0)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": len(self.groups),
            "fused_groups": len(self.fused_groups),
            "nodes_fused": self.nodes_fused,
            "buffers_removed": self.buffers_removed,
            "largest_group": self.largest_group,
        }


def report_fusion(graph: Graph) -> FusionReport:
    """Which chains a fusing backend would form."""
    return FusionReport(groups=find_groups(graph))


def fused_bytes(graph: Graph) -> int:
    """Traffic a fused execution moves.

    Every group reads its outside inputs once and writes its final output once. The
    intermediates never reach memory, which is the entire benefit and is invisible in a node
    count.
    """
    groups = find_groups(graph)
    in_a_group = {name for group in groups for name in group.members}
    total = 0

    for group in groups:
        for name in group.inputs:
            total += graph.value(name).bytes
        total += graph.value(group.output).bytes
    for node in graph.nodes:
        if node.name in in_a_group:
            continue
        for name in node.inputs:
            total += graph.value(name).bytes
        total += node.output.bytes
    return total


def unfused_bytes(graph: Graph) -> int:
    """Traffic an execution that materialises every node moves."""
    total = 0
    for node in graph.nodes:
        for name in node.inputs:
            total += graph.value(name).bytes
        total += node.output.bytes
    return total


def traffic_ratio(graph: Graph) -> float:
    """How much less memory traffic fusion moves."""
    fused = fused_bytes(graph)
    if fused == 0:
        raise PassError("a graph that moves nothing cannot be compared")
    return unfused_bytes(graph) / fused


def peak_intermediates(graph: Graph) -> int:
    """Intermediate tensors an unfused execution has to allocate."""
    outputs = set(graph.outputs)
    return sum(1 for node in graph.nodes if node.name not in outputs and not node.op.is_leaf)


def fused_intermediates(graph: Graph) -> int:
    """Intermediate tensors a fused execution has to allocate."""
    return peak_intermediates(graph) - report_fusion(graph).buffers_removed


@dataclass
class FusedNode:
    """A group rewritten as a single node, for a backend to lower.

    The op list is kept rather than collapsed into an opaque kernel name, because everything
    downstream still needs to know what is inside: the cost model adds up the per element
    costs, the numerics checker wants to know whether a transcendental is in there, and a
    debugger wants to print it.
    """

    name: str
    op_names: list[str]
    inputs: list[str]
    output: str

    def __post_init__(self) -> None:
        if not self.op_names:
            raise ConfigError("a fused node runs at least one operation")

    @property
    def length(self) -> int:
        """Operations in the loop body."""
        return len(self.op_names)

    def flops_per_element(self) -> float:
        """Work one element costs, summed over the chain."""
        return sum(ops.cost_of(ops.get_op(name)).flops_per_element for name in self.op_names)

    def has_transcendental(self) -> bool:
        """Whether the loop body contains something expensive."""
        return any(ops.cost_of(ops.get_op(name)).transcendental for name in self.op_names)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "length": self.length,
            "ops": list(self.op_names),
            "flops_per_element": self.flops_per_element(),
            "transcendental": self.has_transcendental(),
        }


def to_fused_nodes(graph: Graph) -> list[FusedNode]:
    """Rewrite every non trivial group as one node."""
    producers = {node.name: node for node in graph.nodes}
    fused = []
    for group in find_groups(graph):
        if group.is_trivial:
            continue
        fused.append(
            FusedNode(
                name=group.output,
                op_names=[producers[name].op.name for name in group.members],
                inputs=list(group.inputs),
                output=group.output,
            )
        )
    return fused


def arithmetic_intensity(graph: Graph, *, fused: bool = True) -> float:
    """Work done per byte moved.

    The number that says whether a kernel is limited by the memory system or by the
    arithmetic units. An elementwise chain is memory bound at every length until it is fused,
    at which point the same arithmetic is spread over a fraction of the traffic, and that is
    the whole reason the pass exists.
    """
    total_flops = 0.0
    for node in graph.nodes:
        if node.op.is_leaf:
            continue
        cost = ops.cost_of(node.op)
        try:
            elements = node.output.shape.elements
        except Exception:
            continue
        total_flops += cost.flops_per_element * elements
    moved = fused_bytes(graph) if fused else unfused_bytes(graph)
    if moved == 0:
        raise PassError("a graph that moves nothing has no intensity")
    return total_flops / moved
