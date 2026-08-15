from __future__ import annotations

from dataclasses import dataclass, field

from tgc.errors import ConfigError, GraphError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node

# Which values are really the same memory.
#
# A reshape does not move anything. Nor does a transpose that only relabels axes, nor a
# broadcast that repeats a row. Whether a compiler materialises them or leaves them as views
# is its own choice, and the moment it chooses views it acquires an aliasing problem: two
# distinct names in the IR refer to one buffer, so writing through one changes the other.
#
# Every pass in this compiler that reuses storage has to ask this question. Buffer donation
# hands a node the buffer it read; if that buffer is a view of something still live, the write
# lands somewhere nobody expected.
#
# Checking which of the donation pass's conditions actually stops that produced a result worth
# recording. The shape check catches none of it, because a transpose of a square matrix has the
# shape it started with. What refuses every unsafe pair is the elementwise condition, since a
# view is not an elementwise operation. The pass is correct, and correct for a reason it does
# not state, which is one edit away from not being correct at all.

VIEW_OPS = (ops.RESHAPE, ops.TRANSPOSE, ops.BROADCAST_TO)


def is_view(node: Node) -> bool:
    """Whether a node produces a new name for existing memory rather than new memory."""
    return node.op in VIEW_OPS


def is_materialising(node: Node) -> bool:
    """Whether a node writes storage of its own."""
    return not is_view(node) and not node.op.is_leaf


@dataclass
class AliasSet:
    """Values that may refer to the same buffer."""

    root: str
    members: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.root:
            raise ConfigError("an alias set needs a root")
        self.members.add(self.root)

    @property
    def size(self) -> int:
        """How many names refer to this buffer."""
        return len(self.members)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"root": self.root, "size": self.size, "members": sorted(self.members)}


def alias_roots(graph: Graph) -> dict[str, str]:
    """Every value mapped to the buffer it ultimately refers to.

    A chain of views collapses to whatever materialised it. That is the only sound way to ask
    the question: a reshape of a transpose of a sum is three names and one buffer, and a pass
    that only looks one level up sees two of them as distinct.
    """
    roots: dict[str, str] = {value.name: value.name for value in graph.inputs}
    for node in graph.nodes:
        if is_view(node):
            source = node.inputs[0]
            if source not in roots:
                raise GraphError(f"{node.name} views {source}, which is not defined yet")
            roots[node.name] = roots[source]
        else:
            roots[node.name] = node.name
    return roots


def alias_sets(graph: Graph) -> list[AliasSet]:
    """Every buffer with the names that refer to it."""
    roots = alias_roots(graph)
    grouped: dict[str, AliasSet] = {}
    for name, root in roots.items():
        if root not in grouped:
            grouped[root] = AliasSet(root=root)
        grouped[root].members.add(name)
    return [grouped[key] for key in sorted(grouped)]


def may_alias(graph: Graph, first: str, second: str) -> bool:
    """Whether two names might refer to the same buffer."""
    roots = alias_roots(graph)
    if first not in roots or second not in roots:
        raise GraphError(f"cannot compare {first} and {second}, one of them is not defined")
    return roots[first] == roots[second]


def aliased_names(graph: Graph) -> list[str]:
    """Values that share a buffer with something else."""
    return sorted(
        name
        for alias_set in alias_sets(graph)
        if alias_set.size > 1
        for name in alias_set.members
    )


def materialised_buffers(graph: Graph) -> int:
    """How many distinct buffers a view based execution actually needs."""
    return len(alias_sets(graph))


def named_values(graph: Graph) -> int:
    """How many names the graph holds, which is what a materialising execution allocates."""
    return len(graph.value_names)


@dataclass
class AliasReport:
    """How much of a graph is views rather than storage."""

    names: int = 0
    buffers: int = 0
    view_nodes: int = 0

    @property
    def saved_buffers(self) -> int:
        """Allocations a view based execution avoids."""
        return self.names - self.buffers

    @property
    def view_fraction(self) -> float:
        """Share of the names that are views of something else."""
        if self.names == 0:
            return 0.0
        return self.saved_buffers / self.names

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "names": self.names,
            "buffers": self.buffers,
            "saved_buffers": self.saved_buffers,
            "view_fraction": round(self.view_fraction, 4),
            "view_nodes": self.view_nodes,
        }


def analyse(graph: Graph) -> AliasReport:
    """Names, buffers and views for one graph."""
    return AliasReport(
        names=named_values(graph),
        buffers=materialised_buffers(graph),
        view_nodes=sum(1 for node in graph.nodes if is_view(node)),
    )


def view_chain_graph(length: int = 4, width: int = 16) -> Graph:
    """A value put through several views in a row.

    Four names and one buffer. A pass that resolves one level at a time sees the first and the
    last as unrelated, which is the mistake alias_roots exists to prevent.
    """
    if length < 1 or width < 2:
        raise ConfigError("the chain needs a positive length and a width above one")
    builder = Builder()
    x = builder.input([width, width], name="x")
    current = builder.relu(x)
    for _ in range(length):
        current = builder.transpose(current, [1, 0])
    return builder.finish(current)


def materialising_graph(length: int = 4, width: int = 16) -> Graph:
    """The same shape with real operations instead of views.

    The control. Every name here is its own buffer, so the difference between this graph and
    the view chain is exactly what aliasing analysis is measuring.
    """
    if length < 1 or width < 2:
        raise ConfigError("the chain needs a positive length and a width above one")
    builder = Builder()
    current = builder.input([width, width], name="x")
    for _ in range(length + 1):
        current = builder.relu(current)
    return builder.finish(current)


def compare_graphs() -> list[dict]:
    """A chain of views against a chain of operations.

    The view chain holds six names in two buffers and the materialising one holds six names in
    six. That ratio is the whole argument for tracking views rather than materialising them,
    and the aliasing it introduces is the whole cost.
    """
    rows = []
    for label, graph in (
        ("view chain", view_chain_graph()),
        ("materialising chain", materialising_graph()),
    ):
        row = analyse(graph).as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def unsafe_donations(graph: Graph) -> list[tuple[str, str]]:
    """Pairs a donation pass would have to refuse because one is a view of the other.

    The interaction that makes this analysis worth having. Donating a buffer to a node whose
    output aliases it means the write lands in memory something else is still reading, and no
    shape or type check catches it, because the shapes and types are exactly what make a view
    a view.
    """
    roots = alias_roots(graph)
    unsafe = []
    for node in graph.nodes:
        for operand in node.inputs:
            if operand == node.name:
                continue
            if roots.get(operand) == roots.get(node.name):
                unsafe.append((operand, node.name))
    return unsafe


def which_check_refuses(graph: Graph) -> list[dict]:
    """Which of the donation pass's conditions actually stops each unsafe pair.

    Written to find out rather than to confirm, and the answer was not the one expected. The
    shape check catches none of them: a transpose of a square matrix has exactly the shape it
    started with, and so does a reshape that only regroups. What refuses them is the
    elementwise condition, because a view is not an elementwise operation.

    That makes the donation pass correct today and correct for a reason it does not state. The
    hazard is one step away: allow donation to a view, which is a tempting optimisation since a
    view performs no arithmetic, and the write lands in memory something else is still reading
    while every shape and type check passes.
    """
    rows = []
    for donor, receiver in unsafe_donations(graph):
        node = graph.producer_of(receiver)
        if node is None:
            continue
        rows.append(
            {
                "donor": donor,
                "receiver": receiver,
                "shapes_differ": graph.value(donor).shape != node.output.shape,
                "receiver_is_elementwise": node.op.is_elementwise,
            }
        )
    return rows


def refused_by_shape_alone(graph: Graph) -> int:
    """Unsafe pairs the shape check would catch on its own."""
    return sum(1 for row in which_check_refuses(graph) if row["shapes_differ"])


def refused_by_elementwise_alone(graph: Graph) -> int:
    """Unsafe pairs the elementwise check would catch on its own."""
    return sum(1 for row in which_check_refuses(graph) if not row["receiver_is_elementwise"])


def transpose_pair_aliases(width: int = 8) -> dict:
    """A transpose of a transpose, which is the same buffer under a third name."""
    if width < 2:
        raise ConfigError("the fixture needs a width above one")
    builder = Builder()
    x = builder.input([width, width * 2], name="x")
    once = builder.transpose(x, [1, 0])
    twice = builder.transpose(once, [1, 0])
    graph = builder.finish(builder.relu(twice))
    return {
        "names": named_values(graph),
        "buffers": materialised_buffers(graph),
        "x_aliases_twice": may_alias(graph, "x", twice),
        "x_aliases_the_output": may_alias(graph, "x", graph.outputs[0]),
    }
