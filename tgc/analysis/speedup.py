from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError

# What a pass is worth once the rest of the program is allowed to have an opinion.
#
# Every pass in tgc/passes reports what it did to the thing it touched: fusion removes so many
# intermediates, tiling raises the hit rate, layout removes a transpose. None of those numbers
# is the number anybody wants, which is what happened to the step time. A pass that makes its
# category ten times faster is worth almost nothing if the category is five percent of the step,
# and the arithmetic that says so is old and keeps being rediscovered.
#
# So this takes a measured breakdown of where a step goes, applies passes to it, and reports
# what came out. Three results:
#
# Ranking passes by local speedup gives a different order than ranking them by what they save,
# and the disagreement is not subtle: the pass with the largest local number is not the pass
# with the largest saving. Second, individual savings do not add. Two passes worth six percent
# each are not worth twelve percent together, because the second one is optimising a step the
# first one already shortened, and adding them up overstates the pair by about a point. Third,
# the credit depends on the order they are applied in even though the final time does not, which
# is worth knowing before anybody puts a percentage next to a pass name in a changelog.
#
# The last part is a budget. Compile time is finite, and choosing which passes to run under one
# is a knapsack. Greedy by saving per second is what a pass manager does, and against every
# subset it is right at all five budgets tried, for a reason worth stating rather than glossing:
# the only pass expensive enough to be worth skipping cheap ones for is not good enough to be
# worth it. Raise that one pass until it is and the rule loses immediately, which is the honest
# version of the result.

MILLISECONDS = 1.0


@dataclass(frozen=True)
class Profile:
    """Where a step spends its time, by category, in milliseconds."""

    times: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.times:
            raise ConfigError("a profile with no categories measures nothing")
        names = [name for name, _ in self.times]
        if len(set(names)) != len(names):
            raise ConfigError(f"a category appears twice in {names}")
        for name, value in self.times:
            if value < 0:
                raise ConfigError(f"category {name} cannot take {value} milliseconds")
        if self.total <= 0:
            raise ConfigError("a profile has to spend some time somewhere")

    @property
    def total(self) -> float:
        """Step time, in milliseconds."""
        return sum(value for _, value in self.times)

    @property
    def categories(self) -> tuple[str, ...]:
        """The category names, in the order they were measured."""
        return tuple(name for name, _ in self.times)

    def time_in(self, category: str) -> float:
        """What one category costs."""
        for name, value in self.times:
            if name == category:
                return value
        raise ConfigError(f"unknown category {category!r}, expected one of {self.categories}")

    def share_of(self, category: str) -> float:
        """What fraction of the step one category is."""
        return self.time_in(category) / self.total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"total": round(self.total, 3), **dict(self.times)}


@dataclass(frozen=True)
class Optimisation:
    """A pass, what it speeds up locally, and what it costs to run."""

    name: str
    effects: tuple[tuple[str, float], ...]
    compile_cost: float

    def __post_init__(self) -> None:
        if not self.effects:
            raise ConfigError(f"pass {self.name} does not claim to do anything")
        for category, factor in self.effects:
            if factor <= 0:
                raise ConfigError(f"pass {self.name} claims a factor of {factor} on {category}")
        if self.compile_cost < 0:
            raise ConfigError(f"pass {self.name} cannot cost {self.compile_cost} seconds")

    @property
    def largest_claim(self) -> float:
        """The biggest local speedup it reports. What ends up in the pass description."""
        return max(factor for _, factor in self.effects)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "pass": self.name,
            "claim": round(self.largest_claim, 3),
            "categories": [category for category, _ in self.effects],
            "compile_cost": self.compile_cost,
        }


def apply_one(profile: Profile, optimisation: Optimisation) -> Profile:
    """The profile after one pass, with each touched category divided by its factor."""
    effects = dict(optimisation.effects)
    for category in effects:
        profile.time_in(category)
    return Profile(
        times=tuple(
            (name, value / effects[name]) if name in effects else (name, value)
            for name, value in profile.times
        )
    )


def apply_all(profile: Profile, optimisations: Sequence[Optimisation]) -> Profile:
    """The profile after a sequence of passes."""
    current = profile
    for optimisation in optimisations:
        current = apply_one(current, optimisation)
    return current


def end_to_end(profile: Profile, optimisations: Sequence[Optimisation]) -> float:
    """How much faster the step got, all in."""
    return profile.total / apply_all(profile, optimisations).total


def saving(profile: Profile, optimisations: Sequence[Optimisation]) -> float:
    """How much of the step went away, as a fraction."""
    return 1.0 - apply_all(profile, optimisations).total / profile.total


def ceiling_for(profile: Profile, category: str) -> float:
    """The best any pass on one category can ever do to the step.

    One over the part of the step it does not touch. A category that is a fifth of the time
    cannot be worth more than a quarter however completely it is removed, and no measurement or
    cleverness moves that number.
    """
    share = profile.share_of(category)
    if share >= 1.0:
        raise ConfigError(f"category {category} is the whole step, so there is no ceiling")
    return 1.0 / (1.0 - share)


def transformer_step() -> Profile:
    """A measured breakdown of one training step, normalised to a hundred milliseconds.

    The shape every model of this size has: the products dominate, the elementwise work is the
    next largest piece, and there is a tail of reductions, copies and per kernel overhead that
    nobody looks at until the first two have been dealt with.
    """
    return Profile(
        times=(
            ("matmul", 55.0),
            ("elementwise", 22.0),
            ("reduction", 11.0),
            ("copy", 7.0),
            ("overhead", 5.0),
        )
    )


def memory_bound_step() -> Profile:
    """A step with no large products in it, where the tail is the whole story."""
    return Profile(
        times=(
            ("matmul", 8.0),
            ("elementwise", 46.0),
            ("reduction", 21.0),
            ("copy", 17.0),
            ("overhead", 8.0),
        )
    )


FUSION = Optimisation(
    name="fusion",
    effects=(("elementwise", 2.5), ("overhead", 2.0)),
    compile_cost=3.0,
)
TILING = Optimisation(name="tiling", effects=(("matmul", 1.25),), compile_cost=8.0)
LAYOUT = Optimisation(name="layout", effects=(("matmul", 1.1), ("copy", 3.0)), compile_cost=2.0)
VECTORISE = Optimisation(name="vectorise", effects=(("reduction", 1.8),), compile_cost=1.0)
FOLDING = Optimisation(name="folding", effects=(("elementwise", 1.05),), compile_cost=0.5)
PASSES = (FUSION, TILING, LAYOUT, VECTORISE, FOLDING)


def rank_by_claim() -> list[dict]:
    """The passes sorted by the number each one reports about itself."""
    return [
        optimisation.as_dict()
        for optimisation in sorted(PASSES, key=lambda item: -item.largest_claim)
    ]


def rank_by_saving(profile: Profile | None = None) -> list[dict]:
    """The passes sorted by what they take off the step."""
    target = profile if profile is not None else transformer_step()
    rows = [
        {
            "pass": optimisation.name,
            "claim": round(optimisation.largest_claim, 3),
            "saving": round(saving(target, [optimisation]), 5),
            "compile_cost": optimisation.compile_cost,
        }
        for optimisation in PASSES
    ]
    return sorted(rows, key=lambda row: -row["saving"])


def the_two_rankings_disagree(profile: Profile | None = None) -> dict:
    """Whether the pass with the biggest claim is the pass worth the most.

    It is not. Layout reports the largest factor in the set, a three on the copies, and comes
    third by saving because the copies are seven percent of the step. Fusion reports a smaller
    factor and takes the most off, because it is working on a fifth of the step and on the
    overhead at the same time. Nothing about the two rankings is close.
    """
    target = profile if profile is not None else transformer_step()
    by_claim = [row["pass"] for row in rank_by_claim()]
    by_saving = [row["pass"] for row in rank_by_saving(target)]
    return {
        "by_claim": by_claim,
        "by_saving": by_saving,
        "same_order": by_claim == by_saving,
        "top_claim": by_claim[0],
        "top_saving": by_saving[0],
    }


def the_ceiling_is_the_share(profile: Profile | None = None) -> list[dict]:
    """What each category is worth if a pass removed it entirely.

    The products are worth two and a quarter and the overhead is worth five percent, and those
    two numbers are fixed before anybody writes a pass. A category that is five percent of a
    step is not a place to spend a month.
    """
    target = profile if profile is not None else transformer_step()
    return [
        {
            "category": category,
            "share": round(target.share_of(category), 4),
            "ceiling": round(ceiling_for(target, category), 4),
        }
        for category in target.categories
    ]


def a_perfect_pass_on_a_small_category(category: str = "overhead") -> dict:
    """What removing one whole category is worth against a good pass on a large one.

    Less. Deleting the launch overhead completely is worth five percent, and fusion, which does
    not delete anything, is worth more than three times that, because a fifth of the step is
    worth more than a twentieth of it however thoroughly the twentieth is dealt with.
    """
    profile = transformer_step()
    perfect = Optimisation(
        name=f"perfect {category}", effects=((category, 1e9),), compile_cost=0.0
    )
    return {
        "category": category,
        "perfect_saving": round(saving(profile, [perfect]), 5),
        "fusion_saving": round(saving(profile, [FUSION]), 5),
        "fusion_wins": saving(profile, [FUSION]) > saving(profile, [perfect]),
        "ceiling": round(ceiling_for(profile, category), 4),
    }


def savings_do_not_add(profile: Profile | None = None) -> dict:
    """What happens when the individual savings are summed instead of composed.

    They come out too high. The second pass is shortening a step the first one already
    shortened, so its share of what is left is smaller than its share of what there was, and
    adding the two numbers double counts the overlap.
    """
    target = profile if profile is not None else transformer_step()
    chosen = [FUSION, TILING, LAYOUT, VECTORISE]
    separate = sum(saving(target, [optimisation]) for optimisation in chosen)
    together = saving(target, chosen)
    return {
        "sum_of_parts": round(separate, 5),
        "measured": round(together, 5),
        "overstated_by": round(separate - together, 5),
        "sum_is_higher": separate > together,
    }


def speedups_multiply_on_disjoint_categories() -> dict:
    """And what the right composition rule is when two passes do not overlap.

    The step times multiply, not the savings. Tiling and vectorising touch different categories
    and the combined step is exactly the product of the two individually, to the last bit the
    arithmetic carries.
    """
    profile = transformer_step()
    both = apply_all(profile, [TILING, VECTORISE]).total
    chained = apply_one(apply_one(profile, TILING), VECTORISE).total
    return {
        "together": round(both, 6),
        "one_then_the_other": round(chained, 6),
        "identical": abs(both - chained) < 1e-12,
        "speedup": round(end_to_end(profile, [TILING, VECTORISE]), 5),
    }


def the_order_does_not_change_the_result() -> dict:
    """Whether applying the same passes in a different order lands anywhere else.

    It does not. Each pass divides its categories, and division commutes, so every ordering of
    the same set produces the same step to the last bit. That is a property of this model and
    not of a real compiler, where passes enable and disable each other.
    """
    profile = transformer_step()
    forward = apply_all(profile, [FUSION, TILING, LAYOUT, VECTORISE]).total
    reverse = apply_all(profile, [VECTORISE, LAYOUT, TILING, FUSION]).total
    return {
        "forward": round(forward, 9),
        "reverse": round(reverse, 9),
        "identical": abs(forward - reverse) < 1e-9,
    }


def but_the_credit_does(profile: Profile | None = None) -> dict:
    """Whether the value attributed to one pass depends on when it ran.

    It does, and in the direction that flatters whoever measures last. Fusion is worth fifteen
    percent of the raw step and twenty one percent of the step the other three passes have
    already been through, because those three shortened the categories fusion does not touch and
    left it a larger share of a smaller total. Both numbers are correct, which is why a
    percentage next to a pass name means nothing without saying what it was measured against.
    """
    target = profile if profile is not None else transformer_step()
    others = [TILING, LAYOUT, VECTORISE]
    first = saving(target, [FUSION])
    after = saving(apply_all(target, others), [FUSION])
    return {
        "applied_first": round(first, 5),
        "applied_last": round(after, 5),
        "same": abs(first - after) < 1e-9,
        "ratio": round(first / after, 4) if after else 0.0,
    }


def affordable(optimisations: Sequence[Optimisation], budget: float) -> bool:
    """Whether a set of passes fits the compile time budget."""
    if budget < 0:
        raise ConfigError(f"a budget of {budget} seconds is not a budget")
    return sum(optimisation.compile_cost for optimisation in optimisations) <= budget


def best_under_budget(
    profile: Profile, budget: float, available: Sequence[Optimisation] = PASSES
) -> tuple[Optimisation, ...]:
    """Every affordable subset, and the one that takes the most off the step."""
    if not available:
        raise ConfigError("there are no passes to choose from")
    best: tuple[Optimisation, ...] = ()
    best_saving = 0.0
    for size in range(len(available) + 1):
        for chosen in itertools.combinations(available, size):
            if not affordable(chosen, budget):
                continue
            value = saving(profile, chosen)
            if value > best_saving:
                best, best_saving = chosen, value
    return best


def greedy_under_budget(
    profile: Profile, budget: float, available: Sequence[Optimisation] = PASSES
) -> tuple[Optimisation, ...]:
    """Take the pass with the best saving per second of compile time, then the next.

    The rule a pass manager with a budget actually implements, because it does not need to
    know the other passes exist. It re-measures the saving against the current profile each
    time, which is the part that is easy to get wrong and the part that makes it work.
    """
    if not available:
        raise ConfigError("there are no passes to choose from")
    if budget < 0:
        raise ConfigError(f"a budget of {budget} seconds is not a budget")
    chosen: list[Optimisation] = []
    current = profile
    spent = 0.0
    remaining = list(available)
    while remaining:
        scored = [
            (
                saving(current, [optimisation]) / max(optimisation.compile_cost, 1e-9),
                optimisation,
            )
            for optimisation in remaining
            if spent + optimisation.compile_cost <= budget
        ]
        if not scored:
            break
        _, pick = max(scored, key=lambda item: item[0])
        chosen.append(pick)
        current = apply_one(current, pick)
        spent += pick.compile_cost
        remaining.remove(pick)
    return tuple(chosen)


def budget_sweep(
    budgets: Sequence[float] = (1.0, 3.0, 6.0, 10.0, 15.0),
    profile: Profile | None = None,
) -> list[dict]:
    """What the two selections buy at each budget."""
    if not budgets:
        raise ConfigError("there is nothing to sweep")
    target = profile if profile is not None else transformer_step()
    rows = []
    for budget in budgets:
        best = best_under_budget(target, budget)
        rule = greedy_under_budget(target, budget)
        rows.append(
            {
                "budget": budget,
                "best": round(saving(target, best), 5),
                "greedy": round(saving(target, rule), 5),
                "best_passes": [optimisation.name for optimisation in best],
                "greedy_passes": [optimisation.name for optimisation in rule],
                "matches": abs(saving(target, best) - saving(target, rule)) < 1e-12,
            }
        )
    return rows


def the_greedy_selection_matches_the_search() -> dict:
    """How often taking the best saving per second lands on the best set.

    Every budget in the sweep, and it is worth being clear about why rather than calling the
    rule sound. Tiling is the only pass expensive enough to be worth skipping cheap ones for,
    and it saves eleven percent where the cheap ones together save thirty. There is no budget
    at which giving up three passes to afford it pays.
    """
    rows = budget_sweep()
    return {
        "budgets": len(rows),
        "matching": sum(1 for row in rows if row["matches"]),
        "losing": [row["budget"] for row in rows if not row["matches"]],
        "worst_gap": round(max(row["best"] - row["greedy"] for row in rows), 5),
    }


STRONG_TILING = Optimisation(name="strong tiling", effects=(("matmul", 2.5),), compile_cost=8.0)
STRONG_PASSES = (FUSION, STRONG_TILING, LAYOUT, VECTORISE, FOLDING)


def the_rule_loses_when_the_expensive_pass_is_good_enough(budget: float = 8.0) -> dict:
    """The condition that makes the cheap rule wrong, made true.

    Give tiling a factor of two and a half instead of a quarter and it takes a third off the
    step on its own, more than the four cheap passes together. The rule still sorts by saving
    per second, still buys the cheap ones first, and still ends up with too little budget left
    for the one pass that was worth the whole budget. That is the shape of every knapsack
    counterexample and it needs the expensive item to actually be the best item.
    """
    profile = transformer_step()
    best = best_under_budget(profile, budget, STRONG_PASSES)
    rule = greedy_under_budget(profile, budget, STRONG_PASSES)
    return {
        "budget": budget,
        "best": round(saving(profile, best), 5),
        "greedy": round(saving(profile, rule), 5),
        "best_passes": [optimisation.name for optimisation in best],
        "greedy_passes": [optimisation.name for optimisation in rule],
        "gap": round(saving(profile, best) - saving(profile, rule), 5),
    }


def the_rate_and_the_saving_point_different_ways() -> dict:
    """Why that happens, in the numbers the rule actually sorts on.

    Tiling has the best saving in the set and three passes have a better saving per second. The
    rule buys all three, which costs six of the eight seconds, and then cannot afford the eight
    second pass it was always going to want. Nothing in the rule is wrong. It is answering the
    question of what to buy next, and the budget is asking what to buy at all.
    """
    profile = transformer_step()
    rates = {
        optimisation.name: saving(profile, [optimisation]) / optimisation.compile_cost
        for optimisation in STRONG_PASSES
    }
    savings = {
        optimisation.name: saving(profile, [optimisation]) for optimisation in STRONG_PASSES
    }
    better = [name for name, rate in rates.items() if rate > rates["strong tiling"]]
    return {
        "best_saving": max(savings, key=lambda name: savings[name]),
        "better_rate_than_tiling": sorted(better),
        "their_total_cost": sum(
            optimisation.compile_cost
            for optimisation in STRONG_PASSES
            if optimisation.name in better
        ),
        "tiling_rate": round(rates["strong tiling"], 6),
        "tiling_saving": round(savings["strong tiling"], 5),
    }


def compare_profiles() -> list[dict]:
    """The same passes against two different steps.

    Fusion wins on both, which is the boring half of the answer. The interesting half is what
    happens underneath it: tiling is a close second on the step dominated by products and worth
    one and a half percent on the memory bound one, a factor of seven between the same pass on
    two programs. A pass ordering tuned on one of these is not a pass ordering for the other.
    """
    rows = []
    for label, profile in (
        ("transformer", transformer_step()),
        ("memory bound", memory_bound_step()),
    ):
        ranked = rank_by_saving(profile)
        rows.append(
            {
                "profile": label,
                "best_pass": ranked[0]["pass"],
                "best_saving": ranked[0]["saving"],
                "fusion": round(saving(profile, [FUSION]), 5),
                "tiling": round(saving(profile, [TILING]), 5),
                "all_of_them": round(saving(profile, PASSES), 5),
            }
        )
    return rows


def the_margin_depends_on_the_program() -> dict:
    """How far the same passes move between the two steps.

    Fusion doubles and tiling falls to a seventh. The winner is the same and everything about
    what to do after the winner is different, which is the part a fixed pass ordering gets
    wrong.
    """
    rows = {row["profile"]: row for row in compare_profiles()}
    return {
        "same_winner": rows["transformer"]["best_pass"] == rows["memory bound"]["best_pass"],
        "fusion_ratio": round(
            rows["memory bound"]["fusion"] / rows["transformer"]["fusion"], 3
        ),
        "tiling_ratio": round(
            rows["memory bound"]["tiling"] / rows["transformer"]["tiling"], 3
        ),
        "gap_widened": rows["memory bound"]["fusion"] / rows["memory bound"]["tiling"]
        > rows["transformer"]["fusion"] / rows["transformer"]["tiling"],
    }


def an_unknown_category_is_refused() -> bool:
    """Whether a pass naming a category the profile does not have is caught.

    A pass that claims a factor on something the step does not do would otherwise apply
    cleanly, change nothing, and be reported as having no effect rather than as a mistake.
    """
    try:
        apply_one(
            transformer_step(),
            Optimisation(name="ghost", effects=(("nowhere", 2.0),), compile_cost=0.0),
        )
    except ConfigError:
        return True
    return False


def a_pass_that_claims_nothing_is_refused() -> bool:
    """Whether a pass with no effects at all is refused at construction."""
    try:
        Optimisation(name="empty", effects=(), compile_cost=1.0)
    except ConfigError:
        return True
    return False


def an_empty_profile_is_refused() -> bool:
    """Whether a profile with no categories is refused."""
    try:
        Profile(times=())
    except ConfigError:
        return True
    return False
