from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder
from tgc.ir.dtype import FLOAT16, FLOAT32, DType
from tgc.ir.graph import Graph
from tgc.verify.reference import random_feeds, run, to_torch

# Splitting one reduction into several partial ones.
#
# A sum over ten thousand elements is a serial dependence ten thousand long. Split into
# sixteen partial sums and a final sum of those, it is sixteen independent chains and one
# short one, which is the only way a reduction uses more than one execution unit.
#
# The arithmetic is not the same arithmetic. Addition is not associative in floating point, so
# regrouping the terms changes the answer, and this is the one transformation in the compiler
# that is worth doing anyway. The split version is usually more accurate rather than less: a
# serial sum lets the running total grow until each new term falls below its last bit, and
# sixteen shorter sums each stay small enough to keep adding.
#
# So the honest framing is not that splitting is a rounding detail to be tolerated. It is that
# the serial version was never the reference, and the tests below measure both against a
# double precision answer rather than against each other.


@dataclass
class SplitPlan:
    """How one reduction is broken up."""

    length: int
    parts: int

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ConfigError(f"a reduction has at least one term, got {self.length}")
        if not 1 <= self.parts <= self.length:
            raise ConfigError(
                f"the part count must be between one and the length, got {self.parts}"
            )

    @property
    def per_part(self) -> int:
        """Terms in each partial reduction."""
        return math.ceil(self.length / self.parts)

    @property
    def serial_depth(self) -> int:
        """Longest chain of dependent additions.

        The partial sums run at the same time, so the depth is one partial plus the final
        combination. That is the number parallelism actually reduces, and it bottoms out at
        the square root of the length for the same reason checkpointing does.
        """
        return self.per_part + self.parts - 1

    @property
    def speedup(self) -> float:
        """How much shorter the dependence chain gets."""
        return self.length / self.serial_depth

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "length": self.length,
            "parts": self.parts,
            "per_part": self.per_part,
            "serial_depth": self.serial_depth,
            "speedup": round(self.speedup, 3),
        }


def best_part_count(length: int) -> int:
    """The number of parts that gives the shortest dependence chain."""
    if length < 1:
        raise ConfigError(f"a reduction has at least one term, got {length}")

    def depth(parts: int) -> tuple[int, int]:
        return (SplitPlan(length, parts).serial_depth, parts)

    return min(range(1, length + 1), key=depth)


def sweep_parts(length: int = 4096) -> list[dict]:
    """Dependence depth across a range of part counts."""
    if length < 1:
        raise ConfigError(f"a reduction has at least one term, got {length}")
    parts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    return [SplitPlan(length, count).as_dict() for count in parts if count <= length]


def serial_sum(values: torch.Tensor, dtype: DType = FLOAT32) -> float:
    """Add every term one at a time, in order."""
    total = torch.zeros((), dtype=to_torch(dtype))
    for value in values.to(to_torch(dtype)):
        total = total + value
    return float(total)


def split_sum(values: torch.Tensor, parts: int = 16, dtype: DType = FLOAT32) -> float:
    """Add in partial chunks and then add the partials.

    The regrouping, done explicitly rather than handed to a library, because the whole point
    is which additions happen in which order and a library is free to choose differently.
    """
    if parts < 1:
        raise ConfigError(f"the part count must be positive, got {parts}")
    length = values.numel()
    if parts > length:
        raise PassError(f"cannot split {length} terms into {parts} parts")

    per_part = math.ceil(length / parts)
    partials = []
    for start in range(0, length, per_part):
        partials.append(serial_sum(values[start : start + per_part], dtype))
    return serial_sum(torch.tensor(partials, dtype=to_torch(dtype)), dtype)


def exact_sum(values: torch.Tensor) -> float:
    """The answer in double precision, which is the reference both versions are judged by."""
    return float(values.to(torch.float64).sum())


def accuracy_comparison(
    length: int = 100_000, parts: int = 16, dtype: DType = FLOAT32, seed: int = 0
) -> dict:
    """Serial and split against a double precision answer.

    Both compared to the truth rather than to each other, which is what makes the result
    meaningful. The serial sum is not the reference; it is one of the two candidates.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(length, generator=generator).to(to_torch(dtype))

    exact = exact_sum(values)
    serial = serial_sum(values, dtype)
    split = split_sum(values, parts, dtype)
    return {
        "length": length,
        "parts": parts,
        "dtype": dtype.name,
        "exact": exact,
        "serial_error": abs(serial - exact) / abs(exact),
        "split_error": abs(split - exact) / abs(exact),
        "split_is_closer": abs(split - exact) < abs(serial - exact),
    }


def compare_widths(length: int = 100_000, parts: int = 16) -> list[dict]:
    """The same comparison in half and single precision.

    The narrower the accumulator, the sooner a serial sum stalls and the more splitting helps.
    In half precision the serial version stops moving entirely and the split one is still
    counting.
    """
    rows = []
    for dtype in (FLOAT16, FLOAT32):
        row = accuracy_comparison(length=length, parts=parts, dtype=dtype)
        rows.append(row)
    return rows


def compare_part_counts(
    length: int = 100_000, counts: Sequence[int] = (1, 2, 4, 16, 64, 256)
) -> list[dict]:
    """Accuracy across a range of part counts.

    More parts is more accurate up to a point and then stops helping, because the final
    combination becomes its own serial sum. The best split is the one that balances the two,
    which is the same square root that appears in checkpointing and for the same reason.
    """
    if not counts:
        raise ConfigError("there is nothing to compare")
    generator = torch.Generator().manual_seed(0)
    values = torch.rand(length, generator=generator).to(torch.float32)
    exact = exact_sum(values)

    rows = []
    for parts in counts:
        if parts > length:
            continue
        result = split_sum(values, parts)
        rows.append(
            {
                "parts": parts,
                "error": abs(result - exact) / abs(exact),
                "serial_depth": SplitPlan(length, parts).serial_depth,
            }
        )
    return rows


@dataclass
class ReductionReport:
    """Which reductions in a graph are worth splitting."""

    candidates: list[str] = field(default_factory=list)
    lengths: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Reductions found."""
        return len(self.candidates)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"candidates": self.count, "lengths": dict(self.lengths)}


def find_reductions(graph: Graph, *, minimum: int = 64) -> ReductionReport:
    """Reductions long enough that splitting them is worth the extra node.

    A sum over four elements has a dependence chain of four and splitting it into two adds a
    node to save one step. The threshold exists so the pass does not make every graph longer
    for nothing.
    """
    if minimum < 1:
        raise ConfigError(f"the threshold must be positive, got {minimum}")
    report = ReductionReport()
    for node in graph.nodes:
        if node.op.name not in ("sum", "mean"):
            continue
        source = graph.value(node.inputs[0])
        if not source.shape.is_static:
            continue
        axes = node.attrs.get("axes", ())
        length = 1
        for axis in axes:
            resolved = axis + source.shape.rank if axis < 0 else axis
            size = source.shape.sizes[resolved]
            length *= size.value or 1
        if length >= minimum:
            report.candidates.append(node.name)
            report.lengths[node.name] = length
    return report


def long_reduction_graph(length: int = 4096) -> Graph:
    """A single sum over a long axis, which is what splitting is for."""
    if length < 2:
        raise ConfigError(f"the axis has to hold something, got {length}")
    builder = Builder()
    x = builder.input([4, length], name="x")
    return builder.finish(builder.sum(x, axes=[1]))


def short_reduction_graph(length: int = 8) -> Graph:
    """A sum too short to be worth splitting."""
    if length < 2:
        raise ConfigError(f"the axis has to hold something, got {length}")
    builder = Builder()
    x = builder.input([4, length], name="x")
    return builder.finish(builder.sum(x, axes=[1]))


def measure_graph_candidates() -> list[dict]:
    """Which of two graphs holds a reduction worth splitting."""
    rows = []
    for label, graph in (
        ("long", long_reduction_graph()),
        ("short", short_reduction_graph()),
    ):
        report = find_reductions(graph)
        rows.append({"graph": label, "candidates": report.count})
    return rows


def random_feeds_for(graph: Graph, seed: int = 0) -> dict[str, torch.Tensor]:
    """Inputs for a reduction fixture."""
    return random_feeds(graph, seed=seed, positive=True)


def graph_sum_matches_torch(length: int = 4096) -> bool:
    """Whether the interpreter's reduction agrees with the library's on a long axis."""
    graph = long_reduction_graph(length)
    feeds = random_feeds_for(graph)
    return bool(torch.allclose(run(graph, feeds)[0], feeds["x"].sum(dim=1), atol=1e-3))
