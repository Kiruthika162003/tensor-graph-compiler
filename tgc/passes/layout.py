from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.ir.shape import contiguous_strides

# Which order the dimensions sit in memory, and how many transposes that costs.
#
# Two separate things get called layout and they behave differently. A transpose node in the
# graph is a real data movement that a pass can sometimes cancel. A layout preference is a
# statement about which order an operation would rather read, and satisfying it costs a
# transpose somewhere else.
#
# The rule that makes the first one work is that two transposes compose, and compose to
# nothing when their permutations are inverse. That is exact: it moves no data at all, so
# unlike almost everything else in this compiler it changes nothing about the arithmetic.
#
# The second one is a global decision dressed up as a local one. Every operation has a
# preference, the preferences conflict, and picking one layout for the whole graph is cheap to
# implement and leaves transposes wherever the preferences disagreed.

ROW_MAJOR = "row major"
COLUMN_MAJOR = "column major"
LAYOUTS = (ROW_MAJOR, COLUMN_MAJOR)


@dataclass(frozen=True)
class Layout:
    """The order a tensor's dimensions sit in memory."""

    name: str = ROW_MAJOR

    def __post_init__(self) -> None:
        if self.name not in LAYOUTS:
            raise ConfigError(f"unknown layout {self.name!r}, expected one of {list(LAYOUTS)}")

    @property
    def is_row_major(self) -> bool:
        """Whether the last dimension is the fastest varying."""
        return self.name == ROW_MAJOR

    def permutation_for(self, rank: int) -> tuple[int, ...]:
        """The permutation that turns this layout into the packed one."""
        if rank < 0:
            raise ConfigError("a rank cannot be negative")
        if self.is_row_major:
            return tuple(range(rank))
        return tuple(reversed(range(rank)))

    def __str__(self) -> str:
        return self.name


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    """The permutation that undoes another."""
    if sorted(permutation) != list(range(len(permutation))):
        raise ConfigError(f"{list(permutation)} is not a permutation")
    result = [0] * len(permutation)
    for position, axis in enumerate(permutation):
        result[axis] = position
    return tuple(result)


def compose_permutations(outer: Sequence[int], inner: Sequence[int]) -> tuple[int, ...]:
    """The single permutation equivalent to applying one and then the other."""
    if len(outer) != len(inner):
        raise ConfigError("two permutations of different ranks do not compose")
    return tuple(inner[axis] for axis in outer)


def is_identity(permutation: Sequence[int]) -> bool:
    """Whether a permutation moves nothing."""
    return tuple(permutation) == tuple(range(len(permutation)))


@dataclass
class TransposeReport:
    """What the transpose pass found."""

    cancelled: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    remaining: int = 0

    @property
    def removed(self) -> int:
        """Transposes that no longer move data."""
        return len(self.cancelled) + len(self.merged)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "cancelled": len(self.cancelled),
            "merged": len(self.merged),
            "removed": self.removed,
            "remaining": self.remaining,
        }


def count_transposes(graph: Graph) -> int:
    """How many transposes a graph performs."""
    return sum(1 for node in graph.nodes if node.op is ops.TRANSPOSE)


def transpose_bytes(graph: Graph) -> int:
    """Data the transposes move.

    A transpose is not free even when it is only a relabelling, because a consumer that
    expects packed memory has to see packed memory, and the compiler cannot assume otherwise
    without carrying a layout on every value.
    """
    return sum(node.output.bytes * 2 for node in graph.nodes if node.op is ops.TRANSPOSE)


def cancel_transposes(graph: Graph) -> Graph:
    """Remove pairs of transposes that undo each other, and merge the rest.

    Exact in a way nothing else in this file is. Composing two permutations moves no data and
    changes no arithmetic, so the result is bit identical rather than merely close.
    """
    producers = {node.name: node for node in graph.nodes}
    replacement: dict[str, str] = {}
    kept: list[Node] = []

    for node in graph.nodes:
        rewritten = node.replace_inputs(replacement)
        producers[node.name] = rewritten
        if rewritten.op is not ops.TRANSPOSE:
            kept.append(rewritten)
            continue

        inner = producers.get(rewritten.inputs[0])
        if inner is None or inner.op is not ops.TRANSPOSE:
            kept.append(rewritten)
            continue

        combined = compose_permutations(
            rewritten.attrs["permutation"], inner.attrs["permutation"]
        )
        if is_identity(combined):
            replacement[node.name] = replacement.get(inner.inputs[0], inner.inputs[0])
            continue
        merged = Node(
            op=ops.TRANSPOSE,
            inputs=inner.inputs,
            output=rewritten.output,
            attrs={"permutation": combined},
        )
        producers[node.name] = merged
        kept.append(merged)

    outputs = [replacement.get(name, name) for name in graph.outputs]
    return Graph(nodes=kept, inputs=list(graph.inputs), outputs=outputs)


def report_transposes(graph: Graph) -> TransposeReport:
    """What cancelling would remove, without removing it.

    The cancelled list is the ones that disappeared entirely and the merged list is the ones
    that survived as a single composed permutation, which are different outcomes: the first
    removes a data movement and the second replaces two with one.
    """
    after = cancel_transposes(graph)
    survivors = {node.name for node in after.nodes if node.op is ops.TRANSPOSE}
    before = [node.name for node in graph.nodes if node.op is ops.TRANSPOSE]
    return TransposeReport(
        cancelled=[name for name in before if name not in survivors],
        merged=sorted(survivors),
        remaining=len(survivors),
    )


PREFERENCES = {
    "matmul": COLUMN_MAJOR,
    "sum": ROW_MAJOR,
    "mean": ROW_MAJOR,
    "max": ROW_MAJOR,
}


def preferred_layout(node: Node) -> str:
    """Which order an operation would rather read.

    Elementwise operations have no preference, which is what makes the whole question
    tractable: a long chain of them can be given whichever layout its neighbours wanted, and
    only the reductions and the matrix products actually argue.
    """
    if node.op.is_elementwise or node.op.is_leaf:
        return ""
    return PREFERENCES.get(node.op.name, ROW_MAJOR)


def layout_conflicts(graph: Graph, layout: str) -> list[str]:
    """Nodes that would rather have had the other layout."""
    if layout not in LAYOUTS:
        raise ConfigError(f"unknown layout {layout!r}")
    conflicting = []
    for node in graph.nodes:
        preference = preferred_layout(node)
        if preference and preference != layout:
            conflicting.append(node.name)
    return conflicting


def global_layout_cost(graph: Graph, layout: str) -> int:
    """Bytes moved by the transposes one global layout leaves behind."""
    return sum(graph.value(name).bytes * 2 for name in layout_conflicts(graph, layout))


def best_global_layout(graph: Graph) -> str:
    """Whichever single layout leaves the least data movement."""
    return min(LAYOUTS, key=lambda layout: (global_layout_cost(graph, layout), layout))


def boundary_cost(graph: Graph) -> int:
    """Bytes moved by transposing between neighbours that disagree.

    The honest cost of giving every node its preference. Counting satisfied preferences
    instead reports zero for the per node policy on every graph, which is true and useless:
    the cost of that policy is not in the nodes, it is at the edges where two preferences
    meet. Walked over edges for that reason.
    """
    total = 0
    producers = {node.name: node for node in graph.nodes}
    for node in graph.nodes:
        consumer = preferred_layout(node)
        if not consumer:
            continue
        for name in node.inputs:
            source = producers.get(name)
            if source is None:
                continue
            producer = preferred_layout(source)
            if producer and producer != consumer:
                total += graph.value(name).bytes * 2
    return total


def compare_layout_policies(graph: Graph) -> list[dict]:
    """One global layout against giving every node its preference.

    The global policy pays for every node that wanted the other order. The per node policy
    pays at every boundary where two neighbours disagree, which on a graph whose elementwise
    chains separate the reductions from the matrix products is far fewer places.
    """
    rows = []
    for layout in LAYOUTS:
        rows.append(
            {
                "policy": f"global {layout}",
                "conflicts": len(layout_conflicts(graph, layout)),
                "bytes_moved": global_layout_cost(graph, layout),
            }
        )
    rows.append(
        {
            "policy": "per node",
            "conflicts": 0,
            "bytes_moved": boundary_cost(graph),
        }
    )
    return rows


def transposed_pair_graph(width: int = 32) -> Graph:
    """A transpose immediately undone by another.

    What a frontend emits when a user writes two transposes, and what a shape manipulating
    library emits without anybody writing anything.
    """
    builder = Builder()
    x = builder.input([width, width * 2], name="x")
    once = builder.transpose(x, [1, 0])
    twice = builder.transpose(once, [1, 0])
    return builder.finish(builder.relu(twice))


def transposed_chain_graph(width: int = 8) -> Graph:
    """Three transposes that compose into one."""
    builder = Builder()
    x = builder.input([width, width * 2, width * 3], name="x")
    first = builder.transpose(x, [1, 2, 0])
    second = builder.transpose(first, [1, 2, 0])
    third = builder.transpose(second, [1, 2, 0])
    return builder.finish(builder.relu(third))


def strides_for(graph: Graph, name: str, layout: Layout) -> tuple[int, ...]:
    """The strides a value would have under a layout.

    Computed rather than stored, so that a pass claiming a value is contiguous can be checked
    against the packed strides instead of trusting a flag that may have stopped being true
    three passes ago.
    """
    value = graph.value(name)
    packed = contiguous_strides(value.shape)
    if layout.is_row_major:
        return packed
    permutation = layout.permutation_for(value.shape.rank)
    return tuple(packed[axis] for axis in permutation)


def is_contiguous_under(graph: Graph, name: str, layout: Layout) -> bool:
    """Whether a value is densely packed in a given layout."""
    value = graph.value(name)
    if value.shape.rank < 2:
        return True
    return strides_for(graph, name, layout) == contiguous_strides(value.shape)


def check_permutation_algebra(rank: int = 4) -> dict:
    """Verify that composing a permutation with its inverse gives the identity.

    A three line proof and a one line check, and the check is here because the cancellation
    pass rests entirely on it. A compiler that gets this wrong produces transposed tensors
    that look right in every shape assertion and hold the wrong numbers.
    """
    if rank < 1:
        raise PassError("a permutation of rank zero has nothing to check")
    permutation = tuple(reversed(range(rank)))
    inverse = inverse_permutation(permutation)
    return {
        "permutation": permutation,
        "inverse": inverse,
        "composes_to_identity": is_identity(compose_permutations(permutation, inverse)),
        "self_inverse": permutation == inverse,
    }
