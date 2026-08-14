from __future__ import annotations

import pytest

from tgc.analysis.liveness import Interval, compute_intervals, peak_bytes, total_bytes
from tgc.errors import AllocationError, ConfigError
from tgc.ir.builder import branching_graph, elementwise_chain, mlp_graph, softmax_graph
from tgc.memory.planner import (
    STRATEGIES,
    Allocation,
    Plan,
    compare_strategies,
    floor_miss_rate,
    fragmentation,
    get_strategy,
    packing_hazard,
    plan_first_fit,
    plan_is_valid,
    plan_largest_first,
    plan_longest_lived_first,
    plan_without_reuse,
    random_intervals,
    savings_against_no_reuse,
    validate_plan,
)

GRAPHS = (elementwise_chain(8), softmax_graph(), mlp_graph(), branching_graph(4, 3))


class TestAllocation:
    def test_the_end_is_past_the_last_byte(self):
        assert Allocation(name="a", offset=8, size=4).end == 12

    def test_two_placements_that_share_a_byte_overlap(self):
        assert Allocation(name="a", offset=0, size=8).overlaps(
            Allocation(name="b", offset=4, size=8)
        )

    def test_two_that_touch_do_not(self):
        assert not Allocation(name="a", offset=0, size=8).overlaps(
            Allocation(name="b", offset=8, size=8)
        )

    def test_a_negative_offset_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot sit at"):
            Allocation(name="a", offset=-1, size=8)

    def test_a_negative_size_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot occupy"):
            Allocation(name="a", offset=0, size=-1)

    def test_it_serialises(self):
        assert Allocation(name="a", offset=8, size=4).as_dict()["offset"] == 8


class TestValidation:
    def test_every_strategy_produces_a_valid_plan(self):
        # A plan that overlaps two live tensors produces corrupted numbers rather than a
        # crash, which is the worst failure mode a compiler has.
        for graph in GRAPHS:
            intervals = compute_intervals(graph)
            for strategy in STRATEGIES.values():
                validate_plan(intervals, strategy(intervals))

    def test_a_plan_missing_a_value_is_caught(self):
        intervals = compute_intervals(elementwise_chain(4))
        with pytest.raises(AllocationError, match="no placement"):
            validate_plan(intervals, Plan(allocations=[]))

    def test_a_plan_that_overlaps_two_live_values_is_caught(self):
        intervals = [
            Interval(name="a", start=0, end=2, size=8),
            Interval(name="b", start=1, end=3, size=8),
        ]
        broken = Plan(
            allocations=[
                Allocation(name="a", offset=0, size=8),
                Allocation(name="b", offset=0, size=8),
            ]
        )
        with pytest.raises(AllocationError, match="alive together"):
            validate_plan(intervals, broken)

    def test_sharing_bytes_between_values_that_never_coexist_is_fine(self):
        intervals = [
            Interval(name="a", start=0, end=1, size=8),
            Interval(name="b", start=2, end=3, size=8),
        ]
        shared = Plan(
            allocations=[
                Allocation(name="a", offset=0, size=8),
                Allocation(name="b", offset=0, size=8),
            ]
        )
        assert plan_is_valid(intervals, shared)


class TestStrategies:
    def test_never_reusing_needs_the_sum_of_everything(self):
        intervals = compute_intervals(elementwise_chain(8))
        assert plan_without_reuse(intervals).arena_bytes == total_bytes(intervals)

    def test_reuse_gives_most_of_a_chain_back(self):
        intervals = compute_intervals(elementwise_chain(8))
        assert savings_against_no_reuse(intervals, plan_largest_first(intervals)) > 0.7

    def test_no_plan_can_beat_the_floor(self):
        for graph in GRAPHS:
            intervals = compute_intervals(graph)
            floor = peak_bytes(intervals)
            for strategy in STRATEGIES.values():
                assert strategy(intervals).arena_bytes >= floor

    def test_placing_the_biggest_first_reaches_the_floor_on_the_hazard(self):
        intervals = packing_hazard()
        assert plan_largest_first(intervals).arena_bytes == peak_bytes(intervals)

    def test_placing_in_production_order_does_not(self):
        # A small tensor placed early sits in the middle of the arena and forces every later
        # large one above it.
        intervals = packing_hazard()
        assert plan_first_fit(intervals).arena_bytes > peak_bytes(intervals)

    def test_the_gap_on_the_hazard_is_a_third_of_the_arena(self):
        intervals = packing_hazard()
        first = plan_first_fit(intervals).arena_bytes
        largest = plan_largest_first(intervals).arena_bytes
        assert first / largest == pytest.approx(1.346, abs=0.01)

    def test_sorting_by_lifetime_is_plausible_and_worse(self):
        # The arena is measured in bytes and not in conflicts, so a long lived small tensor
        # at offset zero blocks nothing worth blocking.
        assert floor_miss_rate("longest lived first") > floor_miss_rate("largest first")

    def test_sorting_by_size_misses_the_floor_rarely(self):
        assert floor_miss_rate("largest first") < 0.1

    def test_production_order_misses_it_far_more_often(self):
        assert floor_miss_rate("first fit") > 4 * floor_miss_rate("largest first")

    def test_never_reusing_almost_never_reaches_it(self):
        assert floor_miss_rate("no reuse") > 0.9

    def test_a_zero_trial_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            floor_miss_rate("largest first", trials=0)

    def test_it_looks_up_a_strategy_by_name(self):
        assert get_strategy("largest first") is plan_largest_first

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            get_strategy("simulated annealing")


class TestComparison:
    def test_it_reports_the_floor_alongside_the_strategies(self):
        rows = compare_strategies(packing_hazard())
        assert rows[-1]["strategy"] == "floor"

    def test_the_floor_row_has_no_overhead(self):
        assert compare_strategies(packing_hazard())[-1]["overhead"] == 0.0

    def test_it_says_which_reach_the_floor(self):
        rows = {row["strategy"]: row for row in compare_strategies(packing_hazard())}
        assert rows["largest first"]["reaches_floor"]
        assert not rows["first fit"]["reaches_floor"]

    def test_a_chain_is_easy_for_everybody(self):
        rows = compare_strategies(compute_intervals(elementwise_chain(8)))
        reusing = [row for row in rows if row["strategy"] != "no reuse"]
        assert all(row["reaches_floor"] for row in reusing)

    def test_fragmentation_is_how_far_above_the_floor_it_sits(self):
        intervals = packing_hazard()
        assert fragmentation(intervals, plan_largest_first(intervals)) == 0.0
        assert fragmentation(intervals, plan_first_fit(intervals)) > 0.2

    def test_an_empty_plan_has_no_fragmentation(self):
        assert fragmentation([], Plan()) == 0.0

    def test_saving_nothing_against_nothing_is_zero(self):
        assert savings_against_no_reuse([], Plan()) == 0.0

    def test_it_serialises(self):
        intervals = compute_intervals(elementwise_chain(4))
        assert plan_largest_first(intervals).as_dict()["strategy"] == "largest first"


class TestRandomIntervals:
    def test_the_same_seed_gives_the_same_intervals(self):
        assert random_intervals(seed=3) == random_intervals(seed=3)

    def test_a_different_seed_does_not(self):
        assert random_intervals(seed=1) != random_intervals(seed=2)

    def test_the_count_is_what_was_asked_for(self):
        assert len(random_intervals(count=12)) == 12

    def test_an_empty_set_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one interval"):
            random_intervals(count=0)

    def test_every_random_set_is_placed_validly_by_every_strategy(self):
        for seed in range(20):
            intervals = random_intervals(seed=seed)
            for strategy in STRATEGIES.values():
                validate_plan(intervals, strategy(intervals))

    def test_the_hazard_is_reproducible(self):
        assert packing_hazard() == packing_hazard()

    def test_the_hazard_holds_four_large_tensors_and_four_small(self):
        sizes = [i.size for i in packing_hazard()]
        assert sum(1 for size in sizes if size == 16384) == 4
        assert sum(1 for size in sizes if size < 16384) == 4


class TestLongestLived:
    def test_it_still_produces_a_valid_plan(self):
        intervals = compute_intervals(branching_graph(4, 3))
        validate_plan(intervals, plan_longest_lived_first(intervals))

    def test_it_is_worse_than_sorting_by_size_on_a_wide_graph(self):
        intervals = compute_intervals(branching_graph(4, 3))
        assert (
            plan_longest_lived_first(intervals).arena_bytes
            > plan_largest_first(intervals).arena_bytes
        )
