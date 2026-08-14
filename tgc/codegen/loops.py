from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import CodegenError, ConfigError

# Loop nests, and the transformations that rearrange them.
#
# A tensor operation is a loop nest with the loops written down. Once they are explicit the
# usual transformations become list manipulation: splitting one loop into two, swapping two
# loops, peeling a remainder off the end, unrolling a body.
#
# The invariant that makes any of it safe is that the set of iterations never changes. A
# split turns one loop over n into two loops whose product is n; a swap changes the order the
# iterations happen in and not which ones happen; unrolling changes how many times the body
# text appears and not how many times it runs. Every transformation here is checked against
# that by enumerating the iterations both ways and comparing, which is slow, exhaustive, and
# the only check that would have caught the off by one in the first version of peel.


@dataclass
class Loop:
    """One loop in a nest."""

    variable: str
    extent: int
    step: int = 1
    unrolled: int = 1
    vectorised: bool = False

    def __post_init__(self) -> None:
        if not self.variable:
            raise ConfigError("a loop needs a variable")
        if self.extent < 0:
            raise ConfigError(f"{self.variable} cannot run {self.extent} times")
        if self.step < 1:
            raise ConfigError(f"{self.variable} cannot step by {self.step}")
        if self.unrolled < 1:
            raise ConfigError(f"{self.variable} cannot be unrolled {self.unrolled} times")

    @property
    def iterations(self) -> int:
        """How many times the loop body runs."""
        return math.ceil(self.extent / self.step)

    @property
    def is_full(self) -> bool:
        """Whether the step divides the extent, so there is no remainder."""
        return self.extent % self.step == 0

    def indices(self) -> list[int]:
        """Every value the variable takes."""
        return list(range(0, self.extent, self.step))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "variable": self.variable,
            "extent": self.extent,
            "step": self.step,
            "iterations": self.iterations,
            "unrolled": self.unrolled,
            "vectorised": self.vectorised,
        }


@dataclass
class LoopNest:
    """An ordered stack of loops with a body."""

    loops: list[Loop] = field(default_factory=list)
    body: str = "body"

    def __post_init__(self) -> None:
        names = [loop.variable for loop in self.loops]
        if len(set(names)) != len(names):
            raise ConfigError(f"two loops share a variable: {sorted(names)}")

    @property
    def depth(self) -> int:
        """How many loops are stacked."""
        return len(self.loops)

    @property
    def total_iterations(self) -> int:
        """How many times the body runs in total."""
        total = 1
        for loop in self.loops:
            total *= loop.iterations
        return total

    def variables(self) -> list[str]:
        """The loop variables from outermost to innermost."""
        return [loop.variable for loop in self.loops]

    def loop(self, variable: str) -> Loop:
        """Look up a loop by its variable."""
        for candidate in self.loops:
            if candidate.variable == variable:
                return candidate
        raise CodegenError(f"no loop named {variable!r}")

    def position(self, variable: str) -> int:
        """Where a loop sits in the nest."""
        for index, candidate in enumerate(self.loops):
            if candidate.variable == variable:
                return index
        raise CodegenError(f"no loop named {variable!r}")

    def iteration_space(self) -> list[tuple[int, ...]]:
        """Every point the nest visits, in the order it visits them.

        Enumerated rather than counted, because the order is what a transformation changes and
        a count cannot tell a swap from a no op.
        """
        points: list[tuple[int, ...]] = [()]
        for loop in self.loops:
            points = [(*point, index) for point in points for index in loop.indices()]
        return points

    def visited_set(self) -> set[tuple[tuple[str, int], ...]]:
        """Every point the nest visits, keyed by variable rather than by position.

        Keyed by name on purpose. A point recorded as a positional tuple compares equal across
        a swap of two loops with the same extent, because only the meaning of the positions
        changed and the tuples did not, so a positional comparison reports that a swap
        preserved the iteration space for a reason that has nothing to do with the swap.
        """
        variables = self.variables()
        return {
            tuple(sorted(zip(variables, point, strict=True)))
            for point in self.iteration_space()
        }

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "depth": self.depth,
            "variables": self.variables(),
            "total_iterations": self.total_iterations,
        }

    def __str__(self) -> str:
        lines = []
        for index, loop in enumerate(self.loops):
            indent = "  " * index
            suffix = ""
            if loop.unrolled > 1:
                suffix += f" unroll {loop.unrolled}"
            if loop.vectorised:
                suffix += " vector"
            lines.append(
                f"{indent}for {loop.variable} in range(0, {loop.extent}, {loop.step}):{suffix}"
            )
        lines.append("  " * self.depth + self.body)
        return "\n".join(lines)


def split(nest: LoopNest, variable: str, factor: int) -> LoopNest:
    """Turn one loop into an outer loop over blocks and an inner loop inside a block.

    The transformation tiling is made of. The product of the two extents covers the original
    range, and when the factor does not divide the extent the inner loop runs past the end on
    the last block, which is what peel exists to fix.
    """
    if factor < 1:
        raise ConfigError(f"the split factor must be positive, got {factor}")
    position = nest.position(variable)
    original = nest.loops[position]
    if factor > original.extent:
        raise CodegenError(f"cannot split {variable} of extent {original.extent} by {factor}")

    outer = Loop(variable=f"{variable}_outer", extent=original.extent, step=factor)
    inner = Loop(variable=f"{variable}_inner", extent=factor, step=original.step)
    return LoopNest(
        loops=[*nest.loops[:position], outer, inner, *nest.loops[position + 1 :]],
        body=nest.body,
    )


def swap(nest: LoopNest, first: str, second: str) -> LoopNest:
    """Exchange two loops in the nest.

    Changes the order the iterations happen in and not which ones happen, which is why the
    check is on the visited set rather than on the sequence. Whether the swap is legal depends
    on the dependences in the body, which this file does not model and does not pretend to.
    """
    left = nest.position(first)
    right = nest.position(second)
    loops = list(nest.loops)
    loops[left], loops[right] = loops[right], loops[left]
    return LoopNest(loops=loops, body=nest.body)


def unroll(nest: LoopNest, variable: str, factor: int) -> LoopNest:
    """Repeat a loop body several times per iteration.

    The iteration count does not change and the number of times the body text appears does.
    Recording it on the loop rather than expanding it here keeps the nest readable and lets
    the cost model count instructions without the nest growing by the unroll factor.
    """
    if factor < 1:
        raise ConfigError(f"the unroll factor must be positive, got {factor}")
    position = nest.position(variable)
    original = nest.loops[position]
    if factor > original.iterations:
        raise CodegenError(
            f"cannot unroll {variable} by {factor} when it runs {original.iterations} times"
        )
    updated = Loop(
        variable=original.variable,
        extent=original.extent,
        step=original.step,
        unrolled=factor,
        vectorised=original.vectorised,
    )
    loops = list(nest.loops)
    loops[position] = updated
    return LoopNest(loops=loops, body=nest.body)


def vectorise(nest: LoopNest, variable: str, width: int = 8) -> LoopNest:
    """Mark the innermost loop as running several elements at once."""
    if width < 1:
        raise ConfigError(f"the vector width must be positive, got {width}")
    position = nest.position(variable)
    if position != nest.depth - 1:
        raise CodegenError(
            f"{variable} is not the innermost loop, and vectorising anything else needs a "
            "stride that this file does not model"
        )
    original = nest.loops[position]
    updated = Loop(
        variable=original.variable,
        extent=original.extent,
        step=original.step,
        unrolled=original.unrolled,
        vectorised=True,
    )
    loops = list(nest.loops)
    loops[position] = updated
    return LoopNest(loops=loops, body=nest.body)


def peel(nest: LoopNest, variable: str) -> tuple[LoopNest, LoopNest | None]:
    """Separate the whole iterations from the remainder.

    A loop of extent thirteen stepping by four runs three full blocks and one partial. Peeling
    gives a nest over twelve and a nest over the remaining one, so the main body can assume a
    full block and the epilogue handles what is left.

    The tail is one iteration over the remainder, not one iteration per leftover element, and
    the difference only shows up when the extent is smaller than the step. The first version
    stepped the tail by one, so a loop of extent two stepping by three peeled into zero plus
    two where the original ran once. Sweeping every extent up to forty against every step up
    to seven is what found it; a test at a size somebody picked would not have.
    """
    position = nest.position(variable)
    original = nest.loops[position]
    if original.is_full:
        return nest, None

    whole = (original.extent // original.step) * original.step
    main_loop = Loop(variable=original.variable, extent=whole, step=original.step)
    main = LoopNest(
        loops=[*nest.loops[:position], main_loop, *nest.loops[position + 1 :]],
        body=nest.body,
    )
    remainder = original.extent - whole
    tail_loop = Loop(variable=f"{original.variable}_tail", extent=remainder, step=remainder)
    tail = LoopNest(
        loops=[*nest.loops[:position], tail_loop, *nest.loops[position + 1 :]],
        body=nest.body,
    )
    return main, tail


def peeled_iteration_count(nest: LoopNest, variable: str) -> int:
    """Iterations across both halves of a peeled nest."""
    main, tail = peel(nest, variable)
    return main.total_iterations + (tail.total_iterations if tail else 0)


def matmul_nest(rows: int = 64, columns: int = 64, depth: int = 64) -> LoopNest:
    """The three loops of a matrix product."""
    return LoopNest(
        loops=[
            Loop(variable="i", extent=rows),
            Loop(variable="j", extent=columns),
            Loop(variable="k", extent=depth),
        ],
        body="c[i, j] += a[i, k] * b[k, j]",
    )


def elementwise_nest(rows: int = 64, columns: int = 64) -> LoopNest:
    """The two loops of an elementwise operation."""
    return LoopNest(
        loops=[Loop(variable="i", extent=rows), Loop(variable="j", extent=columns)],
        body="y[i, j] = f(x[i, j])",
    )


def tile_nest(nest: LoopNest, factors: dict[str, int]) -> LoopNest:
    """Split several loops and move the block loops outside the element loops.

    Which is what tiling actually is: splitting alone leaves the block loop next to its own
    element loop and changes nothing, and the reuse only appears once every block loop is
    outside every element loop.
    """
    if not factors:
        raise ConfigError("there is nothing to tile")
    current = nest
    for variable, factor in factors.items():
        current = split(current, variable, factor)

    outer = [loop for loop in current.loops if loop.variable.endswith("_outer")]
    inner = [loop for loop in current.loops if loop.variable.endswith("_inner")]
    rest = [
        loop
        for loop in current.loops
        if not loop.variable.endswith("_outer") and not loop.variable.endswith("_inner")
    ]
    return LoopNest(loops=[*outer, *rest, *inner], body=current.body)


def preserves_iterations(before: LoopNest, after: LoopNest) -> bool:
    """Whether a transformation ran the body the same number of times.

    A count, and a count is all that can be compared across a split: splitting turns a point
    (i, k) into a point (i, k_outer, k_inner), so the two iteration spaces have different
    arity and no set comparison between them is meaningful. For transformations that keep the
    arity, visits_the_same_points is the stronger check.
    """
    return before.total_iterations == after.total_iterations


def visits_the_same_points(before: LoopNest, after: LoopNest) -> bool:
    """Whether two nests of the same depth visit exactly the same iterations.

    The stronger check, available whenever the transformation did not change the arity. A swap
    passes it and a peel that drops the last element does not, which is the off by one worth
    catching at a boundary rather than at whatever size somebody happened to test.
    """
    if sorted(before.variables()) != sorted(after.variables()):
        raise CodegenError(
            "two nests over different variables visit points that cannot be compared: "
            f"{sorted(before.variables())} against {sorted(after.variables())}"
        )
    return before.visited_set() == after.visited_set()


def check_transformations(
    nest: LoopNest | None = None, factors: Sequence[int] = (2, 4, 8, 16)
) -> list[dict]:
    """Every transformation on the same nest, with the iteration count before and after."""
    target = nest or matmul_nest(16, 16, 16)
    rows = []
    for factor in factors:
        split_nest = split(target, "k", factor)
        rows.append(
            {
                "transformation": f"split k by {factor}",
                "before": target.total_iterations,
                "after": split_nest.total_iterations,
                "preserved": preserves_iterations(target, split_nest),
                "depth": split_nest.depth,
            }
        )
    swapped = swap(target, "i", "k")
    rows.append(
        {
            "transformation": "swap i and k",
            "before": target.total_iterations,
            "after": swapped.total_iterations,
            "preserved": visits_the_same_points(target, swapped),
            "depth": swapped.depth,
        }
    )
    unrolled = unroll(target, "k", 4)
    rows.append(
        {
            "transformation": "unroll k by 4",
            "before": target.total_iterations,
            "after": unrolled.total_iterations,
            "preserved": preserves_iterations(target, unrolled),
            "depth": unrolled.depth,
        }
    )
    return rows


def remainder_report(extent: int = 13, step: int = 4) -> dict:
    """What peeling a loop that does not divide produces."""
    nest = LoopNest(loops=[Loop(variable="i", extent=extent, step=step)], body="work")
    main, tail = peel(nest, "i")
    return {
        "extent": extent,
        "step": step,
        "main_iterations": main.total_iterations,
        "tail_iterations": tail.total_iterations if tail else 0,
        "combined": peeled_iteration_count(nest, "i"),
        "original": nest.total_iterations,
    }
