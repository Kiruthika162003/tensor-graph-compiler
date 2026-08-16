from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError

# Choosing between several implementations of the same operation.
#
# A matrix product has more than one right implementation. A library kernel is the best thing
# available for a large square product and the worst thing available for a product of two rows,
# where its setup costs more than the arithmetic. A specialised path for thin products beats it
# there and loses everywhere else. Which one to call is a decision the compiler has to make from
# the shapes, and this file is about how well that can be done.
#
# The honest way to ask is to compare a cheap model against a better one rather than against
# nothing. The cheap model here counts arithmetic and bytes; the detailed one adds the two terms
# the cheap one leaves out, a fixed cost per launch and the waste from a product whose
# dimensions do not fill the kernel tile. Both are models and neither is the machine, but the
# difference between them is the kind of thing that separates a model that ranks correctly from
# one that does not.
#
# What the comparison says is worse than expected and more useful for it. The cheap model never
# picks anything but the library. Its mean regret and its worst regret are the same numbers, to
# four decimal places, as choosing the library on every shape without looking. A model that
# counts arithmetic and bytes is not a weak dispatcher, it is not a dispatcher: the two terms it
# leaves out are the only terms that separate these kernels anywhere the decision is close.
#
# The two candidates disagree on more than half the population and the disagreements cost a
# factor of eight on average, because they are concentrated exactly where the cheap model is
# blind. So the usual defence of a rough cost model, that it does not have to be accurate as
# long as it ranks, is the right idea and is not automatically true: it has to be accurate in
# the terms that do the ranking.


@dataclass(frozen=True)
class ProductShape:
    """The three dimensions of a matrix product."""

    rows: int
    inner: int
    columns: int

    def __post_init__(self) -> None:
        if min(self.rows, self.inner, self.columns) < 1:
            raise ConfigError(
                f"a product cannot be {self.rows} by {self.inner} by {self.columns}"
            )

    @property
    def arithmetic(self) -> float:
        """Multiply adds the product performs."""
        return 2.0 * self.rows * self.inner * self.columns

    @property
    def elements(self) -> int:
        """Numbers read and written, in the best case."""
        return self.rows * self.inner + self.inner * self.columns + self.rows * self.columns

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rows": self.rows, "inner": self.inner, "columns": self.columns}


@dataclass
class Machine:
    """What the hardware does per second, and what it costs to start."""

    name: str
    flops_per_second: float
    bytes_per_second: float
    launch_seconds: float
    tile: int = 64

    def __post_init__(self) -> None:
        if min(self.flops_per_second, self.bytes_per_second) <= 0:
            raise ConfigError("a machine has to do some work per second")
        if self.launch_seconds < 0:
            raise ConfigError("a launch cannot take negative time")
        if self.tile < 1:
            raise ConfigError(f"a tile of {self.tile} is not a tile")


ACCELERATOR = Machine(
    name="accelerator",
    flops_per_second=1.2e13,
    bytes_per_second=9.0e11,
    launch_seconds=5e-6,
    tile=64,
)

PROCESSOR = Machine(
    name="processor",
    flops_per_second=2.0e11,
    bytes_per_second=5.0e10,
    launch_seconds=2e-7,
    tile=16,
)


@dataclass
class Kernel:
    """One implementation, what it can run and what it costs.

    The cost is a multiplier on the arithmetic and one on the traffic rather than a formula per
    kernel. Two numbers is enough to express the only difference that matters here, which is
    that a specialised kernel does less setup and reaches a smaller share of peak.
    """

    name: str
    arithmetic_factor: float
    traffic_factor: float
    launch_factor: float
    run: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    minimum_rows: int = 1
    maximum_rows: int = 1 << 30

    def applies_to(self, shape: ProductShape) -> bool:
        """Whether this kernel can run a given product at all."""
        return self.minimum_rows <= shape.rows <= self.maximum_rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kernel": self.name,
            "arithmetic_factor": self.arithmetic_factor,
            "traffic_factor": self.traffic_factor,
        }


def _library(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The ordinary product."""
    return left @ right


def _transposed(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The product computed the other way round and turned back.

    Real rather than decorative. A library that is faster on column major operands is reached by
    transposing both sides, and whether that is worth the two transposes is exactly the kind of
    question a dispatcher is for.
    """
    return (right.transpose(0, 1) @ left.transpose(0, 1)).transpose(0, 1)


def _row_at_a_time(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """One row of the output at a time.

    What a specialised kernel for a thin product does. It never builds a tile, so it wastes
    nothing on a product with two rows and reaches a small share of peak on anything larger.
    """
    return torch.stack([row @ right for row in left])


KERNELS = (
    Kernel(
        name="library",
        arithmetic_factor=1.0,
        traffic_factor=1.0,
        launch_factor=1.0,
        run=_library,
    ),
    Kernel(
        name="transposed",
        arithmetic_factor=1.0,
        traffic_factor=1.6,
        launch_factor=1.0,
        run=_transposed,
    ),
    Kernel(
        name="thin",
        arithmetic_factor=6.0,
        traffic_factor=1.0,
        launch_factor=0.1,
        run=_row_at_a_time,
        maximum_rows=8,
    ),
)


def applicable(shape: ProductShape) -> list[Kernel]:
    """Every kernel that can run a given product."""
    return [kernel for kernel in KERNELS if kernel.applies_to(shape)]


def cheap_cost(kernel: Kernel, shape: ProductShape, machine: Machine = ACCELERATOR) -> float:
    """Seconds, counting only arithmetic and traffic.

    The model most compilers use. It has no term for starting a kernel and no term for a product
    that does not fill a tile, which is exactly why it gets small shapes wrong.
    """
    compute = shape.arithmetic * kernel.arithmetic_factor / machine.flops_per_second
    traffic = shape.elements * 4 * kernel.traffic_factor / machine.bytes_per_second
    return max(compute, traffic)


def detailed_cost(kernel: Kernel, shape: ProductShape, machine: Machine = ACCELERATOR) -> float:
    """Seconds, with the launch and the tile waste included.

    The launch is a fixed cost the arithmetic cannot amortise on a small product. The tile waste
    is the arithmetic a kernel does on the padding when a dimension is not a multiple of its
    tile, which for a product of two rows on a tile of sixty four is thirty two times the useful
    work.
    """
    padded = ProductShape(
        rows=_round_up(shape.rows, machine.tile) if kernel.name != "thin" else shape.rows,
        inner=shape.inner,
        columns=_round_up(shape.columns, machine.tile)
        if kernel.name != "thin"
        else shape.columns,
    )
    compute = padded.arithmetic * kernel.arithmetic_factor / machine.flops_per_second
    traffic = shape.elements * 4 * kernel.traffic_factor / machine.bytes_per_second
    return machine.launch_seconds * kernel.launch_factor + max(compute, traffic)


def _round_up(size: int, multiple: int) -> int:
    """The next multiple of a tile at or above a size."""
    return ((size + multiple - 1) // multiple) * multiple


def select(
    shape: ProductShape, machine: Machine = ACCELERATOR, *, detailed: bool = False
) -> str:
    """The kernel a model would choose."""
    candidates = applicable(shape)
    if not candidates:
        raise ConfigError(f"no kernel can run {shape.as_dict()}")
    cost = detailed_cost if detailed else cheap_cost
    return min(candidates, key=lambda kernel: cost(kernel, shape, machine)).name


def kernel_named(name: str) -> Kernel:
    """One kernel by name."""
    for kernel in KERNELS:
        if kernel.name == name:
            return kernel
    raise ConfigError(f"unknown kernel {name!r}")


def every_kernel_computes_the_same_product(
    shape: ProductShape | None = None, *, seed: int = 0, tolerance: float = 1e-4
) -> list[dict]:
    """Each implementation against the ordinary product, on the same numbers.

    None of them is bit identical to the others and all of them are within a rounding unit,
    which is the expected answer: they perform the same multiplications in different orders and
    float addition is not associative. A dispatcher that swapped between them silently would
    therefore change the answer between runs, and that is worth knowing before it is deployed
    rather than after.
    """
    target = shape if shape is not None else ProductShape(rows=4, inner=32, columns=16)
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(target.rows, target.inner, generator=generator)
    right = torch.randn(target.inner, target.columns, generator=generator)
    exact = left @ right

    rows = []
    for kernel in applicable(target):
        result = kernel.run(left, right)
        gap = float((result - exact).abs().max())
        scale = float(exact.abs().max())
        rows.append(
            {
                "kernel": kernel.name,
                "identical": bool(torch.equal(result, exact)),
                "relative_gap": gap / scale if scale else gap,
                "agrees": (gap / scale if scale else gap) <= tolerance,
            }
        )
    return rows


def shape_population(count: int = 200) -> list[ProductShape]:
    """A spread of shapes covering the small and the large.

    Deliberately weighted toward the small end. Every dispatcher does well on shapes where one
    kernel wins by an order of magnitude, and the only place the decision is interesting is
    where the candidates are close, which for these kernels is at few rows.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    shapes = []
    row_choices = (1, 2, 4, 8, 16, 64, 256)
    size_choices = (16, 64, 256, 1024)
    for index in range(count):
        rows = row_choices[index % len(row_choices)]
        inner = size_choices[(index // len(row_choices)) % len(size_choices)]
        columns = size_choices[
            (index // (len(row_choices) * len(size_choices))) % len(size_choices)
        ]
        shapes.append(ProductShape(rows=rows, inner=inner, columns=columns))
    return shapes


def regret(shape: ProductShape, chosen: str, machine: Machine = ACCELERATOR) -> float:
    """How much slower the chosen kernel is than the best available, under the detailed model.

    A ratio rather than a difference, because the shapes span four orders of magnitude and a
    difference in seconds would be a report about the largest one.
    """
    candidates = applicable(shape)
    best = min(detailed_cost(kernel, shape, machine) for kernel in candidates)
    picked = detailed_cost(kernel_named(chosen), shape, machine)
    return picked / best if best else 1.0


def compare_strategies(machine: Machine = ACCELERATOR, count: int = 200) -> list[dict]:
    """Three ways of choosing, over the same population of shapes.

    Always the library, the cheap model, and the detailed model. The last one is the floor by
    construction, since the regret is measured against it, and it is here to give the other two
    a scale rather than as a candidate.
    """
    shapes = shape_population(count)
    strategies = {
        "always the library": lambda _shape: "library",
        "cheap model": lambda shape: select(shape, machine),
        "detailed model": lambda shape: select(shape, machine, detailed=True),
    }
    rows = []
    for name, choose in strategies.items():
        penalties = [regret(shape, choose(shape), machine) for shape in shapes]
        rows.append(
            {
                "strategy": name,
                "mean_regret": round(sum(penalties) / len(penalties), 4),
                "worst_regret": round(max(penalties), 4),
                "perfect_choices": sum(1 for value in penalties if value <= 1.0 + 1e-9),
                "shapes": len(shapes),
            }
        )
    return rows


def the_cheap_model_is_the_default_in_disguise(machine: Machine = ACCELERATOR) -> dict:
    """Whether counting arithmetic and bytes chooses anything at all.

    It does not. Every number the cheap model produces is the number choosing the library
    without looking produces, because the library wins on arithmetic and traffic at every shape
    and only loses once the launch and the tile waste are counted. The model is doing work and
    arriving where it started.
    """
    rows = {row["strategy"]: row for row in compare_strategies(machine)}
    fixed = rows["always the library"]
    model = rows["cheap model"]
    return {
        "fixed_worst_regret": fixed["worst_regret"],
        "model_worst_regret": model["worst_regret"],
        "fixed_mean_regret": fixed["mean_regret"],
        "model_mean_regret": model["mean_regret"],
        "identical": fixed["mean_regret"] == model["mean_regret"]
        and fixed["worst_regret"] == model["worst_regret"],
    }


def what_the_missing_terms_are_worth(machine: Machine = ACCELERATOR) -> dict:
    """How much regret the launch and the tile terms account for.

    All of it. The detailed model is the floor by construction, so the gap between it and the
    cheap one is exactly what those two terms buy, and here it is a factor of five on average
    and ten at the worst shape.
    """
    rows = {row["strategy"]: row for row in compare_strategies(machine)}
    return {
        "without_them": rows["cheap model"]["mean_regret"],
        "with_them": rows["detailed model"]["mean_regret"],
        "worst_without_them": rows["cheap model"]["worst_regret"],
    }


def where_the_models_disagree(machine: Machine = ACCELERATOR, count: int = 200) -> dict:
    """How often the cheap model picks a different kernel, and what that costs.

    More than half of them, and expensively. The disagreements are concentrated exactly where
    the cheap model is blind rather than where the candidates are close, so the mean cost of one
    is a factor of eight rather than a few percent. The comfortable version of this result, that
    a rough model errs only where it does not matter, is not what happens here.
    """
    shapes = shape_population(count)
    disagreements = []
    for shape in shapes:
        cheap = select(shape, machine)
        detailed = select(shape, machine, detailed=True)
        if cheap != detailed:
            disagreements.append((shape, regret(shape, cheap, machine)))
    return {
        "shapes": len(shapes),
        "disagreements": len(disagreements),
        "share": round(len(disagreements) / len(shapes), 4),
        "mean_regret_where_they_disagree": round(
            sum(value for _, value in disagreements) / len(disagreements), 4
        )
        if disagreements
        else 1.0,
        "worst_regret_where_they_disagree": round(
            max((value for _, value in disagreements), default=1.0), 4
        ),
    }


def the_cheap_model_fails_on_small_shapes(machine: Machine = ACCELERATOR) -> list[dict]:
    """Which shapes the cheap model gets wrong, listed rather than counted.

    All of them are thin, at eight rows and below, which is where the thin kernel is allowed to
    run at all. The cheap model has no launch term and no tile term, both of which only matter
    when the arithmetic is too small to hide them, so its errors sit exactly on its omissions.
    """
    rows = []
    for rows_count in (1, 2, 4, 8, 16, 64, 256):
        shape = ProductShape(rows=rows_count, inner=256, columns=256)
        cheap = select(shape, machine)
        detailed = select(shape, machine, detailed=True)
        rows.append(
            {
                "rows": rows_count,
                "cheap_picks": cheap,
                "detailed_picks": detailed,
                "agree": cheap == detailed,
            }
        )
    return rows


def machine_changes_the_answer(count: int = 200) -> dict:
    """Whether the right kernel depends on the machine, which it does.

    A processor has a launch cost four hundred times smaller than an accelerator's and a tile
    four times smaller, so the shapes where a specialised kernel wins are different ones. A
    dispatcher with the choice baked in is right on one machine.
    """
    shapes = shape_population(count)
    differ = sum(
        1
        for shape in shapes
        if select(shape, ACCELERATOR, detailed=True) != select(shape, PROCESSOR, detailed=True)
    )
    return {
        "shapes": len(shapes),
        "different_choices": differ,
        "share": round(differ / len(shapes), 4),
    }


@dataclass
class Timing:
    """A measured time and how much it moved between runs."""

    kernel: str
    median: float
    spread: float
    samples: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kernel": self.kernel,
            "median": self.median,
            "spread": self.spread,
            "samples": self.samples,
        }


def time_kernel(
    kernel: Kernel, shape: ProductShape, *, repeats: int = 9, seed: int = 0
) -> Timing:
    """A median of repeated runs, with the spread reported alongside.

    The median rather than the mean, and the spread rather than nothing, for the same reason
    runtime/profile.py gives: a single timing on a shared machine measures the machine's mood.
    Nothing in the test suite asserts on these numbers, because a number that depends on what
    else is running is not something to assert on.
    """
    if repeats < 1:
        raise ConfigError(f"a measurement needs a run, got {repeats}")
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(shape.rows, shape.inner, generator=generator)
    right = torch.randn(shape.inner, shape.columns, generator=generator)
    kernel.run(left, right)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        kernel.run(left, right)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return Timing(
        kernel=kernel.name,
        median=samples[len(samples) // 2],
        spread=samples[-1] - samples[0],
        samples=len(samples),
    )


def measure_all(shape: ProductShape | None = None, *, repeats: int = 9) -> list[dict]:
    """Every applicable kernel, timed on one shape."""
    target = shape if shape is not None else ProductShape(rows=4, inner=256, columns=256)
    return [
        time_kernel(kernel, target, repeats=repeats).as_dict() for kernel in applicable(target)
    ]


def the_spread_is_wider_than_the_gap(shape: ProductShape | None = None) -> dict:
    """Whether a measurement can tell the kernels apart at all.

    Often it cannot. The spread between the fastest and slowest run of one kernel is frequently
    larger than the difference between two kernels' medians, which means a dispatcher choosing
    by one measurement is choosing by noise. That is the argument for having a model at all
    rather than measuring at runtime.
    """
    rows = measure_all(shape)
    medians = sorted(row["median"] for row in rows)
    gap = medians[1] - medians[0] if len(medians) > 1 else 0.0
    widest = max(row["spread"] for row in rows)
    return {
        "smallest_gap_between_kernels": gap,
        "widest_spread_within_one": widest,
        "measurement_is_conclusive": gap > widest,
    }


@dataclass
class DispatchReport:
    """A summary of how a strategy did."""

    strategy: str
    mean_regret: float = 1.0
    worst_regret: float = 1.0
    rows: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "mean_regret": round(self.mean_regret, 4),
            "worst_regret": round(self.worst_regret, 4),
        }


def report_for(strategy: str, machine: Machine = ACCELERATOR) -> DispatchReport:
    """One strategy's numbers, packaged."""
    rows = {row["strategy"]: row for row in compare_strategies(machine)}
    if strategy not in rows:
        raise ConfigError(f"unknown strategy {strategy!r}, expected one of {sorted(rows)}")
    return DispatchReport(
        strategy=strategy,
        mean_regret=rows[strategy]["mean_regret"],
        worst_regret=rows[strategy]["worst_regret"],
    )


def kernels_that_never_win(machine: Machine = ACCELERATOR, count: int = 200) -> list[str]:
    """Implementations the detailed model never picks.

    Worth reporting rather than deleting on sight. A kernel that never wins on this population
    of shapes may still win on a machine with different ratios, and the honest thing is to say
    it did not win here rather than to conclude it is useless.
    """
    chosen = {select(shape, machine, detailed=True) for shape in shape_population(count)}
    return sorted(kernel.name for kernel in KERNELS if kernel.name not in chosen)


def coverage(shapes: Sequence[ProductShape] | None = None) -> dict:
    """Whether every shape has at least one kernel that can run it."""
    population = list(shapes) if shapes is not None else shape_population()
    uncovered = [shape.as_dict() for shape in population if not applicable(shape)]
    return {"shapes": len(population), "uncovered": uncovered}
