from __future__ import annotations

from dataclasses import dataclass, field

from tgc.analysis.liveness import compute_intervals, peak_bytes
from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.passes.dce import eliminate_dead_code
from tgc.schedule.order import depth_first_order

# Removing broadcasts that never needed to be materialised.
#
# A broadcast_to node writes a wide tensor whose every row is a copy of a narrow one. If the
# only readers of that wide tensor are elementwise operations, none of them needed the copy:
# broadcasting is already what an elementwise operation does to mismatched shapes, so each
# reader can take the narrow tensor directly and produce the same answer.
#
# The saving is the wide tensor, which is the widest thing in the graph by construction. On a
# graph that broadcasts a column across a matrix it is the difference between holding one
# column and holding the whole matrix, and the pass costs nothing because the broadcast was
# never doing any arithmetic.
#
# It stops at anything that is not elementwise. A matrix product reading a broadcast operand
# genuinely needs the shape, and a reduction over the broadcast axis reads every copy.


@dataclass
class SinkReport:
    """Which broadcasts can be removed and which cannot."""

    sunk: list[str] = field(default_factory=list)
    kept: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Broadcasts removed."""
        return len(self.sunk)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "sunk": self.count,
            "kept": len(self.kept),
            "reasons": sorted(set(self.kept.values())),
        }


def is_broadcast(node: Node) -> bool:
    """Whether a node widens a tensor without computing anything."""
    return node.op is ops.BROADCAST_TO


def can_sink(graph: Graph, name: str) -> tuple[bool, str]:
    """Whether a broadcast can be dropped and its readers pointed at the source."""
    node = graph.producer_of(name)
    if node is None or not is_broadcast(node):
        return False, "not a broadcast"
    if name in graph.outputs:
        return False, "the broadcast is a graph output, so the caller wants the wide tensor"

    readers = graph.consumers_of(name)
    if not readers:
        return False, "nothing reads it"
    for reader in readers:
        if not reader.op.is_elementwise:
            return False, f"{reader.op.name} needs the widened shape"
    return True, ""


def sink_broadcasts(graph: Graph) -> Graph:
    """Point every sinkable broadcast's readers at its source instead.

    Exact. An elementwise operation broadcasts mismatched operands anyway, so reading the
    narrow tensor performs the same arithmetic on the same values in the same order, and the
    output shape is unchanged because the other operand still carries the width.
    """
    replacement: dict[str, str] = {}
    for node in graph.nodes:
        allowed, _ = can_sink(graph, node.name)
        if allowed:
            replacement[node.name] = node.inputs[0]

    rewritten = [node.replace_inputs(replacement) for node in graph.nodes]
    outputs = [replacement.get(name, name) for name in graph.outputs]
    return eliminate_dead_code(
        Graph(nodes=rewritten, inputs=list(graph.inputs), outputs=outputs)
    )


def report_sinking(graph: Graph) -> SinkReport:
    """What sinking would remove, and why it declined the rest."""
    report = SinkReport()
    for node in graph.nodes:
        if not is_broadcast(node):
            continue
        allowed, reason = can_sink(graph, node.name)
        if allowed:
            report.sunk.append(node.name)
        else:
            report.kept[node.name] = reason
    return report


def broadcast_bytes(graph: Graph) -> int:
    """Storage the materialised broadcasts occupy."""
    return sum(node.output.bytes for node in graph.nodes if is_broadcast(node))


def elementwise_broadcast_graph(width: int = 64, readers: int = 3) -> Graph:
    """A column broadcast across a matrix and read only by elementwise operations.

    Which is what a bias add or a normalisation scale looks like once a frontend has made the
    shapes match. The wide tensor is the largest value in the graph and no operation ever
    needed it.
    """
    if width < 2 or readers < 1:
        raise ConfigError("the fixture needs a width above one and at least one reader")
    builder = Builder()
    x = builder.input([width, width], name="x")
    column = builder.sum(x, axes=[1], keepdims=True)
    wide = builder.broadcast_to(column, [width, width])

    current = x
    for _ in range(readers):
        current = builder.add(current, wide)
    return builder.finish(current)


def contracting_broadcast_graph(width: int = 64) -> Graph:
    """The same broadcast read by a matrix product, which genuinely needs the shape.

    The control. A matrix product contracts over the broadcast axis and reads every copy, so
    the wide tensor has to exist and the pass has to leave it alone.
    """
    if width < 2:
        raise ConfigError("the fixture needs a width above one")
    builder = Builder()
    x = builder.input([width, width], name="x")
    column = builder.sum(x, axes=[1], keepdims=True)
    wide = builder.broadcast_to(column, [width, width])
    return builder.finish(builder.matmul(x, wide))


def reduced_broadcast_graph(width: int = 64) -> Graph:
    """A broadcast read by a reduction over the axis it widened.

    Also has to be kept, for a different reason: the reduction reads every copy, so removing
    the broadcast would sum one column instead of a matrix full of them.
    """
    if width < 2:
        raise ConfigError("the fixture needs a width above one")
    builder = Builder()
    x = builder.input([width, width], name="x")
    column = builder.sum(x, axes=[1], keepdims=True)
    wide = builder.broadcast_to(column, [width, width])
    return builder.finish(builder.sum(wide, axes=[0]))


def measure_sinking(width: int = 64, readers: int = 3) -> dict:
    """Peak memory and node count with the broadcasts sunk and without."""
    graph = elementwise_broadcast_graph(width, readers)
    sunk = sink_broadcasts(graph)
    return {
        "sunk": report_sinking(graph).count,
        "nodes_before": len(graph.nodes),
        "nodes_after": len(sunk.nodes),
        "broadcast_bytes_before": broadcast_bytes(graph),
        "broadcast_bytes_after": broadcast_bytes(sunk),
        "peak_before": peak_bytes(compute_intervals(graph, depth_first_order(graph))),
        "peak_after": peak_bytes(compute_intervals(sunk, depth_first_order(sunk))),
    }


def compare_fixtures(width: int = 64) -> list[dict]:
    """The sinkable graph against the two that are not.

    Three graphs with the same broadcast in them. One loses it entirely and the other two keep
    it, and the reasons are different: a matrix product needs the shape and a reduction reads
    every copy.
    """
    rows = []
    for label, graph in (
        ("elementwise readers", elementwise_broadcast_graph(width)),
        ("matmul reader", contracting_broadcast_graph(width)),
        ("reduction reader", reduced_broadcast_graph(width)),
    ):
        report = report_sinking(graph)
        rows.append(
            {
                "graph": label,
                "sunk": report.count,
                "kept": len(report.kept),
                "reason": next(iter(report.kept.values()), ""),
            }
        )
    return rows


def check_sinking(graph: Graph, name: str) -> None:
    """Raise with the reason if a broadcast cannot be sunk."""
    allowed, reason = can_sink(graph, name)
    if not allowed:
        raise PassError(f"cannot sink {name}: {reason}")


def saving_fraction(width: int = 64, readers: int = 3) -> float:
    """Share of the peak memory that sinking gives back."""
    result = measure_sinking(width, readers)
    if result["peak_before"] == 0:
        return 0.0
    return 1.0 - result["peak_after"] / result["peak_before"]
