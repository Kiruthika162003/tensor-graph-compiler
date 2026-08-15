from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, ScheduleError

# Whether two iterations of a loop touch the same element, and which one goes first.
#
# Every loop transformation in codegen/loops.py is legal only if the dependences allow it, and
# that file deliberately does not model them. This one does, for the case that covers most
# tensor code: array subscripts that are affine in the loop variables, which is what an index
# like a[2 * i + 3] is.
#
# The test that decides it is older than most of the rest of compilers and is one line. Two
# iterations touch the same element when 2 * i + 3 equals 2 * j + 3 for some i and j in range,
# which rearranges to a linear equation, and a linear equation with integer coefficients has an
# integer solution exactly when the greatest common divisor of the coefficients divides the
# constant. That is the whole of the greatest common divisor test.
#
# It says no or maybe, never yes. When it says the divisor does not divide, there is provably
# no dependence and the transformation is safe. When it says it does, there may be a
# dependence and there may not, and a compiler that treats maybe as yes is correct and a
# compiler that treats it as no is not.


@dataclass(frozen=True)
class Subscript:
    """An array index that is affine in one loop variable: stride times index plus offset."""

    stride: int
    offset: int = 0

    def at(self, index: int) -> int:
        """The element this subscript reaches on a given iteration."""
        return self.stride * index + self.offset

    def __str__(self) -> str:
        if self.stride == 1 and self.offset == 0:
            return "i"
        if self.offset == 0:
            return f"{self.stride}i"
        sign = "+" if self.offset > 0 else "-"
        return f"{self.stride}i {sign} {abs(self.offset)}"


NO_DEPENDENCE = "none"
MAYBE_DEPENDENCE = "maybe"


@dataclass
class DependenceResult:
    """What the test concluded about a pair of subscripts."""

    verdict: str
    reason: str = ""
    witness: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.verdict not in (NO_DEPENDENCE, MAYBE_DEPENDENCE):
            raise ConfigError(f"unknown verdict {self.verdict!r}")

    @property
    def is_independent(self) -> bool:
        """Whether the two accesses provably never collide."""
        return self.verdict == NO_DEPENDENCE

    @property
    def blocks_reordering(self) -> bool:
        """Whether a transformation has to assume a dependence.

        Maybe blocks it. A compiler that treats maybe as no is not conservative, it is wrong,
        and the wrongness surfaces as an answer that depends on the optimisation level.
        """
        return not self.is_independent

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "verdict": self.verdict,
            "independent": self.is_independent,
            "reason": self.reason,
            "witness": self.witness,
        }


def gcd_test(write: Subscript, read: Subscript) -> DependenceResult:
    """The greatest common divisor test on two affine subscripts.

    A collision needs write.stride * i + write.offset to equal read.stride * j + read.offset,
    which is a linear equation in i and j. It has an integer solution exactly when the greatest
    common divisor of the two strides divides the difference of the offsets, so when it does
    not there is provably no dependence.
    """
    if write.stride == 0 and read.stride == 0:
        if write.offset == read.offset:
            return DependenceResult(
                verdict=MAYBE_DEPENDENCE, reason="both accesses are the same fixed element"
            )
        return DependenceResult(
            verdict=NO_DEPENDENCE, reason="two different fixed elements never collide"
        )

    divisor = math.gcd(abs(write.stride), abs(read.stride))
    difference = read.offset - write.offset
    if divisor == 0 or difference % divisor != 0:
        return DependenceResult(
            verdict=NO_DEPENDENCE,
            reason=f"gcd {divisor} does not divide the offset difference {difference}",
        )
    return DependenceResult(
        verdict=MAYBE_DEPENDENCE,
        reason=f"gcd {divisor} divides the offset difference {difference}",
    )


def brute_force(write: Subscript, read: Subscript, extent: int) -> DependenceResult:
    """Check every pair of iterations directly, which is only possible for small extents.

    The exact answer, and the thing the gcd test is checked against. It exists because a test
    that says maybe is only useful if the cases where it says no are genuinely no, and the
    only way to know that is to enumerate.
    """
    if extent < 1:
        raise ConfigError(f"the extent must be positive, got {extent}")
    for i in range(extent):
        for j in range(extent):
            if write.at(i) == read.at(j):
                return DependenceResult(
                    verdict=MAYBE_DEPENDENCE,
                    reason="found a colliding pair",
                    witness=(i, j),
                )
    return DependenceResult(verdict=NO_DEPENDENCE, reason="no pair collides")


def gcd_never_says_no_when_there_is_one(
    strides: Sequence[int] = (1, 2, 3, 4, 6),
    offsets: Sequence[int] = range(-4, 5),
    extent: int = 24,
) -> dict:
    """Sweep the test against the enumeration and count where they differ.

    The only direction that matters is the unsound one. A case where the gcd test says no and
    the enumeration finds a collision would make every transformation resting on it wrong, and
    the sweep exists to find one rather than to trust that none is there.
    """
    if extent < 1:
        raise ConfigError(f"the extent must be positive, got {extent}")
    checked = 0
    unsound = 0
    conservative = 0

    for write_stride in strides:
        for read_stride in strides:
            for write_offset in offsets:
                for read_offset in offsets:
                    write = Subscript(stride=write_stride, offset=write_offset)
                    read = Subscript(stride=read_stride, offset=read_offset)
                    predicted = gcd_test(write, read)
                    actual = brute_force(write, read, extent)
                    checked += 1
                    if predicted.is_independent and not actual.is_independent:
                        unsound += 1
                    if not predicted.is_independent and actual.is_independent:
                        conservative += 1
    return {
        "checked": checked,
        "unsound": unsound,
        "conservative": conservative,
        "conservative_rate": round(conservative / checked, 4) if checked else 0.0,
    }


@dataclass
class Distance:
    """How far apart in iterations two accesses to the same element are."""

    value: int | None = None

    @property
    def is_known(self) -> bool:
        """Whether the distance is a single number rather than a range."""
        return self.value is not None

    @property
    def is_loop_carried(self) -> bool:
        """Whether the dependence crosses an iteration boundary.

        A distance of zero means each iteration only depends on itself, so the loop is
        parallel. Anything else means iteration k reads what iteration k minus d wrote, and
        running them at the same time gets one of the two answers at random.
        """
        return self.is_known and self.value != 0

    @property
    def permits_parallelism(self) -> bool:
        """Whether the iterations can run at the same time."""
        return self.is_known and self.value == 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "distance": self.value,
            "loop_carried": self.is_loop_carried,
            "parallel": self.permits_parallelism,
        }


def distance_between(write: Subscript, read: Subscript, extent: int) -> Distance:
    """The iteration distance when both accesses have the same stride.

    Only defined for matched strides, which is the case that covers a loop reading what it
    wrote a fixed number of steps ago. Different strides give a distance that varies with the
    iteration, and calling that a distance is where dependence analysis usually goes wrong.
    """
    if extent < 1:
        raise ConfigError(f"the extent must be positive, got {extent}")
    if write.stride != read.stride:
        raise ScheduleError(
            "the distance is only a single number when the strides match, and these are "
            f"{write.stride} and {read.stride}"
        )
    if write.stride == 0:
        return Distance(value=0 if write.offset == read.offset else None)
    difference = write.offset - read.offset
    if difference % write.stride != 0:
        return Distance(value=None)
    return Distance(value=difference // write.stride)


def loop_is_parallel(write: Subscript, read: Subscript, extent: int) -> bool:
    """Whether every iteration is independent of every other."""
    result = gcd_test(write, read)
    if result.is_independent:
        return True
    try:
        return distance_between(write, read, extent).permits_parallelism
    except ScheduleError:
        return False


def classify(write: Subscript, read: Subscript, extent: int = 32) -> dict:
    """Everything the analysis can say about one pair of subscripts."""
    result = gcd_test(write, read)
    row = result.as_dict()
    row["write"] = str(write)
    row["read"] = str(read)
    row["parallel"] = loop_is_parallel(write, read, extent)
    return row


def worked_examples(extent: int = 32) -> list[dict]:
    """The cases worth having in front of you.

    Reading and writing the same element is parallel. Reading one step behind is not. Even
    and odd indices provably never meet, which is the answer the gcd test exists to give and
    the only one it gives with certainty.
    """
    pairs = [
        (Subscript(stride=1), Subscript(stride=1)),
        (Subscript(stride=1), Subscript(stride=1, offset=-1)),
        (Subscript(stride=2), Subscript(stride=2, offset=1)),
        (Subscript(stride=2), Subscript(stride=4)),
        (Subscript(stride=3), Subscript(stride=2)),
    ]
    return [classify(write, read, extent) for write, read in pairs]


@dataclass
class LoopSummary:
    """What can be said about a whole loop from its accesses."""

    accesses: list[tuple[Subscript, Subscript]] = field(default_factory=list)
    extent: int = 32

    def __post_init__(self) -> None:
        if self.extent < 1:
            raise ConfigError(f"the extent must be positive, got {self.extent}")

    @property
    def is_parallel(self) -> bool:
        """Whether every pair permits parallelism."""
        return all(loop_is_parallel(write, read, self.extent) for write, read in self.accesses)

    @property
    def blocking_pairs(self) -> list[str]:
        """The pairs that stop the loop being parallel."""
        return [
            f"{write} against {read}"
            for write, read in self.accesses
            if not loop_is_parallel(write, read, self.extent)
        ]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "accesses": len(self.accesses),
            "parallel": self.is_parallel,
            "blocking": self.blocking_pairs,
        }


def imprecision_falls_with_extent(extents: Sequence[int] = (2, 3, 4, 8, 24)) -> list[dict]:
    """How often the test says maybe where the enumeration says no, across loop lengths.

    Forty percent of pairs at an extent of two, and none at all by twenty four. The test
    ignores the loop bounds entirely, so it is conservative exactly when the colliding
    iterations exist as integers but lie outside the range the loop actually runs. On a short
    loop that is most pairs; on a long one it is almost none.

    The unsound count is zero at every extent, and that is the property everything else rests
    on. A single case where the test said no and the enumeration found a collision would make
    every transformation resting on it wrong.
    """
    if not extents:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for extent in extents:
        row = gcd_never_says_no_when_there_is_one(extent=extent)
        row["extent"] = extent
        rows.append(row)
    return rows
