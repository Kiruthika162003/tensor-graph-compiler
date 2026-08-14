from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tgc.analysis.cost import kendall_agreement
from tgc.errors import ConfigError, ScheduleError
from tgc.schedule.tiling import (
    MatmulShape,
    Tile,
    effective_traffic,
    square_tile,
)

# Searching for a tile size, and being clear about what the search is for.
#
# The cost model in analysis/cost.py ranks candidates and does not predict times. This file is
# where that distinction earns its keep: the model prunes a large space down to a handful, and
# the measurement decides between them, because the model is missing everything that is not
# traffic and arithmetic.
#
# The missing things are modelled here as a deterministic perturbation rather than as noise,
# because they are not noise. A tile whose side does not divide the problem leaves a ragged
# remainder that runs at a fraction of the speed of a full tile, and a tile that is not a
# multiple of the vector width wastes lanes on every iteration. Both are perfectly
# predictable and neither is in a roofline.


@dataclass
class Candidate:
    """One tiling under consideration."""

    side: int
    model_cost: float = 0.0
    measured_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.side < 1:
            raise ConfigError(f"a tile side must be positive, got {self.side}")

    @property
    def error(self) -> float:
        """How far the model was from the measurement, relatively."""
        if self.measured_cost == 0:
            return 0.0
        return abs(self.model_cost - self.measured_cost) / self.measured_cost

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "side": self.side,
            "model_cost": round(self.model_cost, 2),
            "measured_cost": round(self.measured_cost, 2),
            "error": round(self.error, 4),
        }


def model_cost(shape: MatmulShape, tile: Tile, cache_bytes: int) -> float:
    """What the roofline style model says a tiling costs."""
    return float(effective_traffic(shape, tile, cache_bytes))


def measured_cost(
    shape: MatmulShape, tile: Tile, cache_bytes: int, vector_width: int = 8
) -> float:
    """What the tiling actually costs, including the effects the model omits.

    Two penalties, both deterministic. A tile side that does not divide the problem leaves a
    remainder iteration doing partial work, and a side that is not a multiple of the vector
    width wastes lanes on every single iteration rather than only at the edges. The second one
    is the larger effect and is entirely invisible to a traffic model.
    """
    if vector_width < 1:
        raise ConfigError("a vector is at least one element wide")
    base = model_cost(shape, tile, cache_bytes)

    ragged = 1.0
    for size, side in (
        (shape.rows, tile.rows),
        (shape.columns, tile.columns),
        (shape.depth, tile.depth),
    ):
        remainder = size % side
        if remainder:
            ragged *= 1.0 + (side - remainder) / size

    lanes = tile.columns % vector_width
    wasted = 1.0 if lanes == 0 else 1.0 + (vector_width - lanes) / vector_width
    return base * ragged * wasted


@dataclass
class TuningResult:
    """What one search found and what it cost to find it."""

    strategy: str
    chosen: int
    measurements: int
    best_available: int
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def found_the_best(self) -> bool:
        """Whether the search landed on the true optimum."""
        return self.chosen == self.best_available

    def regret(self, costs: dict[int, float]) -> float:
        """How much worse the chosen tiling is than the best one."""
        if self.chosen not in costs or self.best_available not in costs:
            raise ScheduleError("the cost table does not cover the chosen tilings")
        best = costs[self.best_available]
        if best == 0:
            return 0.0
        return costs[self.chosen] / best - 1.0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "chosen": self.chosen,
            "measurements": self.measurements,
            "found_the_best": self.found_the_best,
        }


# Two candidate sets, and they tell opposite stories.
#
# The coarse one spans two orders of magnitude of traffic, so the term the model captures
# dwarfs the terms it misses and the model alone picks the winner. The narrow one sits either
# side of the cache limit, where every candidate has almost the same traffic and the vector
# width decides, which is invisible to the model. A tuner spends its time in the second
# regime, because the first one was settled by whoever chose the powers of two.
SIDES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
NEAR_LIMIT_SIDES = tuple(range(120, 148))


def cost_table(
    shape: MatmulShape, cache_bytes: int, sides: Sequence[int] = SIDES
) -> dict[int, float]:
    """The measured cost of every candidate, which a real tuner cannot afford."""
    if not sides:
        raise ConfigError("there is nothing to tune over")
    return {side: measured_cost(shape, square_tile(side), cache_bytes) for side in sides}


def exhaustive(
    shape: MatmulShape, cache_bytes: int, sides: Sequence[int] = SIDES
) -> TuningResult:
    """Measure every candidate. Correct, and the thing every other strategy is judged by."""
    costs = cost_table(shape, cache_bytes, sides)
    best = min(costs, key=lambda side: (costs[side], side))
    return TuningResult(
        strategy="exhaustive",
        chosen=best,
        measurements=len(costs),
        best_available=best,
        candidates=[
            Candidate(
                side=side,
                model_cost=model_cost(shape, square_tile(side), cache_bytes),
                measured_cost=costs[side],
            )
            for side in sides
        ],
    )


def model_only(
    shape: MatmulShape, cache_bytes: int, sides: Sequence[int] = SIDES
) -> TuningResult:
    """Trust the model and measure nothing.

    Free, and right whenever the candidates differ by more than the model's blind spots. Over
    powers of two that is always, because traffic spans two orders of magnitude and the
    penalties the model misses are bounded by a factor of two. Over a narrow band near the
    cache limit it is wrong, because there the traffic is flat and the vector width is the
    only thing left deciding.
    """
    costs = cost_table(shape, cache_bytes, sides)
    modelled = {side: model_cost(shape, square_tile(side), cache_bytes) for side in sides}
    chosen = min(modelled, key=lambda side: (modelled[side], side))
    return TuningResult(
        strategy="model only",
        chosen=chosen,
        measurements=0,
        best_available=min(costs, key=lambda side: (costs[side], side)),
    )


def model_then_measure(
    shape: MatmulShape,
    cache_bytes: int,
    sides: Sequence[int] = SIDES,
    *,
    shortlist: int = 4,
) -> TuningResult:
    """Rank with the model, measure the top few, keep the best.

    What the model is for. It does not have to be accurate, only good enough that the true
    optimum survives the shortlist. That holds in both regimes here while measuring a fraction
    of the space, which is the argument for keeping an inaccurate model rather than replacing
    it with a better one.
    """
    if shortlist < 1:
        raise ConfigError(f"the shortlist must hold something, got {shortlist}")
    costs = cost_table(shape, cache_bytes, sides)
    modelled = sorted(
        sides, key=lambda side: (model_cost(shape, square_tile(side), cache_bytes), side)
    )
    top = modelled[:shortlist]
    chosen = min(top, key=lambda side: (costs[side], side))
    return TuningResult(
        strategy=f"model then measure {shortlist}",
        chosen=chosen,
        measurements=len(top),
        best_available=min(costs, key=lambda side: (costs[side], side)),
    )


def random_search(
    shape: MatmulShape,
    cache_bytes: int,
    sides: Sequence[int] = SIDES,
    *,
    budget: int = 4,
    seed: int = 0,
) -> TuningResult:
    """Measure a random handful. The control the model has to beat."""
    if budget < 1:
        raise ConfigError(f"the budget must be positive, got {budget}")
    costs = cost_table(shape, cache_bytes, sides)
    generator = random.Random(seed)
    sampled = generator.sample(list(sides), min(budget, len(sides)))
    chosen = min(sampled, key=lambda side: (costs[side], side))
    return TuningResult(
        strategy=f"random {budget}",
        chosen=chosen,
        measurements=len(sampled),
        best_available=min(costs, key=lambda side: (costs[side], side)),
    )


STRATEGIES: dict[str, Callable[..., TuningResult]] = {
    "exhaustive": exhaustive,
    "model only": model_only,
    "model then measure": model_then_measure,
    "random": random_search,
}


def compare_strategies(
    shape: MatmulShape | None = None, cache_bytes: int = 256 * 1024
) -> list[dict]:
    """Every search strategy on the same problem, with its cost and its regret."""
    target = shape or MatmulShape()
    costs = cost_table(target, cache_bytes)
    rows = []
    for strategy in STRATEGIES.values():
        result = strategy(target, cache_bytes)
        row = result.as_dict()
        row["regret"] = round(result.regret(costs), 4)
        rows.append(row)
    return rows


def model_ranking_agreement(
    shape: MatmulShape | None = None, cache_bytes: int = 256 * 1024
) -> float:
    """How closely the model's ordering resembles the measured one.

    The number that says whether the model is worth keeping. Near one and it can prune the
    space safely; near zero and the shortlist it produces is a random sample with extra steps.
    """
    target = shape or MatmulShape()
    costs = cost_table(target, cache_bytes)
    modelled = sorted(
        costs, key=lambda side: (model_cost(target, square_tile(side), cache_bytes), side)
    )
    measured = sorted(costs, key=lambda side: (costs[side], side))
    return kendall_agreement([str(side) for side in modelled], [str(side) for side in measured])


def shortlist_sweep(
    shape: MatmulShape | None = None, cache_bytes: int = 256 * 1024
) -> list[dict]:
    """How large a shortlist has to be before it contains the true optimum."""
    target = shape or MatmulShape()
    costs = cost_table(target, cache_bytes)
    rows = []
    for size in range(1, len(SIDES) + 1):
        result = model_then_measure(target, cache_bytes, shortlist=size)
        rows.append(
            {
                "shortlist": size,
                "chosen": result.chosen,
                "found_the_best": result.found_the_best,
                "regret": round(result.regret(costs), 4),
            }
        )
    return rows


def compare_regimes(
    shape: MatmulShape | None = None, cache_bytes: int = 256 * 1024
) -> list[dict]:
    """The same strategies over a coarse candidate set and over a narrow one.

    Over powers of two the model alone is exactly right and measuring buys nothing. Over the
    band around the cache limit the model picks a tile 22 percent worse than the best one,
    because every candidate there moves almost the same bytes and the model cannot see the
    lanes being wasted. Both rows are the same code on the same machine.
    """
    target = shape or MatmulShape(500, 500, 500)
    rows = []
    for label, sides in (("coarse", SIDES), ("near the limit", NEAR_LIMIT_SIDES)):
        costs = cost_table(target, cache_bytes, sides)
        for strategy in (model_only, model_then_measure, exhaustive):
            result = strategy(target, cache_bytes, sides)
            rows.append(
                {
                    "candidates": label,
                    "strategy": result.strategy,
                    "chosen": result.chosen,
                    "measurements": result.measurements,
                    "regret": round(result.regret(costs), 4),
                }
            )
    return rows


def model_blind_spot(shape: MatmulShape | None = None, cache_bytes: int = 256 * 1024) -> dict:
    """How much the model loses in the regime where its blind spots decide."""
    target = shape or MatmulShape(500, 500, 500)
    costs = cost_table(target, cache_bytes, NEAR_LIMIT_SIDES)
    modelled = model_only(target, cache_bytes, NEAR_LIMIT_SIDES)
    return {
        "model_pick": modelled.chosen,
        "measured_best": modelled.best_available,
        "regret": round(modelled.regret(costs), 4),
        "traffic_spread": round(max(costs.values()) / min(costs.values()), 3),
    }
