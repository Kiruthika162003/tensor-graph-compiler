from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tgc.analysis.liveness import compute_intervals, peak_bytes
from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.passes.dce import eliminate_dead_code

# Trading arithmetic for memory by computing something twice.
#
# A value that is produced early and read late occupies storage for the whole gap. If it is
# cheap to recompute, recomputing it at the point of use costs the arithmetic once more and
# frees the storage for everything in between. That is the entire idea, and it is the only
# lever in the compiler that makes a graph slower on purpose.
#
# The interesting structure is a chain. Storing every intermediate costs the length of the
# chain in memory and nothing extra in arithmetic. Storing none of them and recomputing from
# the input costs one buffer and quadratic arithmetic. Storing every kth costs both terms in
# the middle, and the k that balances them is the square root of the length, which is a
# genuinely nice result and is easy to check rather than assert.


@dataclass
class RematPlan:
    """Which values to keep and which to recompute."""

    kept: list[str]
    recomputed: list[str]
    extra_flops: float = 0.0

    @property
    def stored(self) -> int:
        """Values held in memory."""
        return len(self.kept)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kept": self.stored,
            "recomputed": len(self.recomputed),
            "extra_flops": self.extra_flops,
        }


@dataclass
class CheckpointCost:
    """What a checkpointing interval costs on a chain of a given length.

    Written as a closed form rather than by building the graph, because the point is the shape
    of the tradeoff and building a chain of ten thousand nodes to see it is a slow way to
    learn arithmetic.
    """

    length: int
    interval: int
    element_bytes: int = 4
    elements: int = 1024

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ConfigError(f"a chain has at least one step, got {self.length}")
        if not 1 <= self.interval <= self.length:
            raise ConfigError(
                f"the interval must be between one and the chain length, got {self.interval}"
            )

    @property
    def checkpoints(self) -> int:
        """Values kept in memory."""
        return math.ceil(self.length / self.interval)

    @property
    def peak_values(self) -> int:
        """Values alive at the worst moment.

        The checkpoints, plus the segment being recomputed. That second term is what stops
        an arbitrarily long interval from being free.
        """
        return self.checkpoints + self.interval

    @property
    def memory_bytes(self) -> int:
        """Storage the plan needs."""
        return self.peak_values * self.elements * self.element_bytes

    @property
    def extra_steps(self) -> int:
        """Recomputation the plan performs, in chain steps.

        Every segment is recomputed once during the backward walk, which is one extra pass
        over everything that is not a checkpoint.
        """
        return self.length - self.checkpoints

    @property
    def overhead(self) -> float:
        """Share of extra arithmetic the plan pays."""
        return self.extra_steps / self.length

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "interval": self.interval,
            "checkpoints": self.checkpoints,
            "peak_values": self.peak_values,
            "memory_bytes": self.memory_bytes,
            "extra_steps": self.extra_steps,
            "overhead": round(self.overhead, 4),
        }


def best_interval(length: int) -> int:
    """The checkpoint interval that minimises peak storage.

    The square root of the chain length, and the derivation is one line: keeping every kth
    value holds n over k checkpoints plus k for the segment being redone, and that sum is
    smallest when the two terms are equal.
    """
    if length < 1:
        raise ConfigError(f"a chain has at least one step, got {length}")
    candidates = range(1, length + 1)
    return min(
        candidates, key=lambda k: (CheckpointCost(length=length, interval=k).peak_values, k)
    )


def sweep_intervals(length: int = 64) -> list[dict]:
    """Every checkpoint interval on a chain, with its memory and its overhead."""
    if length < 1:
        raise ConfigError(f"a chain has at least one step, got {length}")
    return [
        CheckpointCost(length=length, interval=interval).as_dict()
        for interval in range(1, length + 1)
    ]


def square_root_agreement(lengths: Sequence[int] = (16, 64, 256, 1024, 4096)) -> list[dict]:
    """Whether the integer square root is as good as the exhaustive optimum.

    Checked rather than asserted, and the first version of the check was the wrong one. The
    optimal interval is not always the square root: for a length of a thousand the sweep
    picks twenty eight while the square root is thirty two, and both hold sixty four values,
    because the minimum is flat across a wide plateau and the tie is broken toward the
    smaller interval. Comparing the intervals therefore reports a disagreement that costs
    nothing. Comparing the peaks is the claim worth making, and it holds exactly.
    """
    if not lengths:
        raise ConfigError("there is nothing to check")
    rows = []
    for length in lengths:
        chosen = best_interval(length)
        root = max(1, min(length, round(math.sqrt(length))))
        rows.append(
            {
                "length": length,
                "best_interval": chosen,
                "square_root": root,
                "best_peak": CheckpointCost(length=length, interval=chosen).peak_values,
                "root_peak": CheckpointCost(length=length, interval=root).peak_values,
            }
        )
    return rows


def rematerialise(graph: Graph, names: Sequence[str]) -> Graph:
    """Recompute the named values at each point they are read.

    Every reader gets its own copy of the producing node, so the original value is no longer
    live across the gap between them. Only worth doing for a cheap node with distant readers,
    which is exactly what select_candidates looks for.
    """
    wanted = list(names)
    unknown = [name for name in wanted if name not in {node.name for node in graph.nodes}]
    if unknown:
        raise PassError(f"cannot rematerialise {unknown}, which nothing produces")

    producers = {node.name: node for node in graph.nodes}
    for name in wanted:
        if producers[name].op.is_leaf:
            raise PassError(f"{name} is a leaf and has nothing to recompute")

    rebuilt: list[Node] = []
    copies: dict[tuple[str, str], str] = {}
    counter = 0

    for node in graph.nodes:
        if node.name in wanted and node.name not in graph.outputs:
            rebuilt.append(node)
            continue
        replacement: dict[str, str] = {}
        for operand in node.inputs:
            if operand not in wanted:
                continue
            source = producers[operand]
            counter += 1
            copy_name = f"{operand}_remat{counter}"
            copies[(node.name, operand)] = copy_name
            rebuilt.append(
                Node(
                    op=source.op,
                    inputs=source.inputs,
                    output=type(source.output)(
                        name=copy_name,
                        shape=source.output.shape,
                        dtype=source.output.dtype,
                    ),
                    attrs=dict(source.attrs),
                )
            )
            replacement[operand] = copy_name
        rebuilt.append(node.replace_inputs(replacement))

    return eliminate_dead_code(graph.with_nodes(rebuilt))


def select_candidates(graph: Graph, *, min_gap: int = 2) -> list[str]:
    """Values worth recomputing: cheap to produce and read a long way from where they were.

    The gap is the whole criterion. Recomputing a value read immediately saves nothing and
    costs the arithmetic, and a pass that does not check the distance makes every graph
    slower for no reason.
    """
    if min_gap < 1:
        raise ConfigError(f"the gap must be positive, got {min_gap}")
    positions = {node.name: index for index, node in enumerate(graph.nodes)}
    outputs = set(graph.outputs)

    chosen = []
    for node in graph.nodes:
        if node.op.is_leaf or node.name in outputs:
            continue
        readers = [positions[reader.name] for reader in graph.consumers_of(node.name)]
        if not readers:
            continue
        if max(readers) - positions[node.name] >= min_gap:
            chosen.append(node.name)
    return chosen


def long_range_graph(depth: int = 6, width: int = 64) -> Graph:
    """A cheap value produced early and read at the very end.

    The shape everybody reaches for first, and it saves nothing. Recomputing the activation
    at the point of use keeps its own input alive instead, and the input is the same size, so
    one live tensor is swapped for another of equal size and the peak does not move.
    """
    if depth < 2:
        raise ConfigError("the chain needs somewhere to be long across")
    builder = Builder()
    x = builder.input([width, width], name="x")
    early = builder.relu(x)
    current = early
    for _ in range(depth):
        current = builder.tanh(current)
    return builder.finish(builder.add(current, early))


def expanding_graph(depth: int = 6, width: int = 64, held: int = 4) -> Graph:
    """Several large values, each rebuildable from a small one, all read at the end.

    The shape rematerialisation actually pays for, and it took two attempts to build. One
    long lived tensor saves nothing: the final addition needs it and the chain result at the
    same instant, so the peak sits at that addition either way. Several of them is different.
    Held together they are all alive across the chain; recomputed at the point of use only
    one exists at a time, and each is rebuilt from a column a width smaller.
    """
    if depth < 2:
        raise ConfigError("the chain needs somewhere to be long across")
    if held < 1:
        raise ConfigError(f"there has to be something to hold, got {held}")
    builder = Builder()
    x = builder.input([width, width], name="x")

    wides = []
    for index in range(held):
        column = builder.sum(x, axes=[1], keepdims=True)
        shifted = builder.add(column, builder.constant(float(index + 1)))
        wides.append(builder.broadcast_to(shifted, [width, width]))

    current = builder.relu(x)
    for _ in range(depth):
        current = builder.tanh(current)
    for wide in wides:
        current = builder.add(current, wide)
    return builder.finish(current)


def measure_rematerialisation(
    depth: int = 6, width: int = 64, *, expanding: bool = True
) -> dict:
    """Peak memory and node count with the long lived value recomputed, and without.

    Run on both fixtures. On the expanding one the saving is real, and on the plain chain it
    is exactly zero, which is the result that says what the pass is actually for: it helps
    when the thing recomputed is larger than what it is recomputed from.
    """
    graph = (
        expanding_graph(depth=depth, width=width)
        if expanding
        else long_range_graph(depth=depth, width=width)
    )
    candidates = select_candidates(graph, min_gap=depth)
    if not candidates:
        raise PassError("the fixture produced nothing worth recomputing")

    rewritten = rematerialise(graph, candidates)
    before = peak_bytes(compute_intervals(graph))
    after = peak_bytes(compute_intervals(rewritten))
    return {
        "candidates": candidates,
        "peak_before": before,
        "peak_after": after,
        "saved": before - after,
        "saved_fraction": round((before - after) / before, 4) if before else 0.0,
        "nodes_before": len(graph.nodes),
        "nodes_after": len(rewritten.nodes),
    }
