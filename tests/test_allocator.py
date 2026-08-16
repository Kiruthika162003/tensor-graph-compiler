from __future__ import annotations

import pytest

from tgc.errors import AllocationError, ConfigError
from tgc.ir.builder import branching_graph, mlp_graph, softmax_graph
from tgc.runtime.allocator import (
    POLICIES,
    Allocator,
    AllocatorStats,
    Block,
    a_graph_trace_reuses_more_than_a_random_one,
    a_zero_sized_request_is_served,
    caching_against_returning,
    compare_graphs,
    compare_policies,
    freeing_twice_is_refused,
    graph_traces,
    random_trace,
    returning_gives_up_every_reuse,
    round_up,
    rounding_buys_reuse,
    rounding_costs_waste,
    rounding_predicts_its_own_waste,
    run_trace,
    size_classes_are_the_compromise,
    trace_from,
)


class TestRounding:
    def test_exact_fitting_rounds_nothing(self):
        assert round_up(1000, "exact") == 1000

    def test_powers_of_two_round_up(self):
        assert round_up(1000, "power of two") == 1024

    def test_size_classes_land_in_between(self):
        assert round_up(1000, "size classes") == 1024
        assert round_up(700, "size classes") == 768

    def test_a_power_of_two_is_left_alone(self):
        assert round_up(1024, "power of two") == 1024

    def test_nothing_rounds_to_nothing(self):
        assert round_up(0, "power of two") == 0

    def test_a_negative_size_is_refused(self):
        with pytest.raises(ConfigError, match="cannot allocate"):
            round_up(-1, "exact")

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            round_up(64, "guesswork")

    def test_the_average_waste_is_a_quarter(self):
        # A size drawn uniformly lands on average a quarter of the way below the power above it.
        assert rounding_predicts_its_own_waste()["measured_waste"] == 0.25

    def test_a_zero_sample_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            rounding_predicts_its_own_waste(samples=0)


class TestAllocator:
    def test_a_freed_block_comes_back(self):
        allocator = Allocator()
        allocator.free(allocator.allocate(1024))
        allocator.allocate(1024)
        assert allocator.stats.reused == 1

    def test_a_different_capacity_does_not(self):
        allocator = Allocator(policy="exact")
        allocator.free(allocator.allocate(1000))
        allocator.allocate(1001)
        assert allocator.stats.reused == 0

    def test_but_a_rounded_one_might(self):
        allocator = Allocator(policy="power of two")
        allocator.free(allocator.allocate(1000))
        allocator.allocate(1001)
        assert allocator.stats.reused == 1

    def test_freeing_twice_is_refused(self):
        # Two entries in a free list for one block hands the same memory to two callers.
        assert freeing_twice_is_refused()

    def test_freeing_something_never_allocated_is_refused(self):
        with pytest.raises(AllocationError, match="not allocated"):
            Allocator().free(7)

    def test_a_zero_sized_request_is_served(self):
        assert a_zero_sized_request_is_served()["footprint"] == 0

    def test_the_footprint_covers_what_is_held_and_cached(self):
        allocator = Allocator()
        first = allocator.allocate(1024)
        allocator.allocate(1024)
        allocator.free(first)
        assert allocator.footprint == 2048

    def test_a_returning_allocator_gives_it_back(self):
        allocator = Allocator(give_back=True)
        allocator.free(allocator.allocate(1024))
        assert allocator.footprint == 0

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            Allocator(policy="guesswork")

    def test_an_empty_run_has_no_reuse(self):
        assert AllocatorStats().reuse_rate == 0.0

    def test_and_no_waste(self):
        assert AllocatorStats().internal_waste == 0.0

    def test_it_serialises(self):
        assert Allocator().as_dict()["policy"] == "power of two"

    def test_a_block_knows_its_waste(self):
        assert Block(size=1000, capacity=1024).waste == 24

    def test_a_block_smaller_than_its_size_is_refused(self):
        with pytest.raises(ConfigError, match="cannot have capacity"):
            Block(size=1024, capacity=512)

    def test_it_serialises_too(self):
        assert Block(size=1000, capacity=1024).as_dict()["waste"] == 24


class TestTraces:
    def test_a_graph_becomes_an_allocate_and_free_sequence(self):
        trace = trace_from(softmax_graph())
        assert any(kind == "allocate" for kind, _ in trace)
        assert any(kind == "free" for kind, _ in trace)

    def test_one_allocation_per_node(self):
        graph = mlp_graph()
        trace = trace_from(graph)
        assert sum(1 for kind, _ in trace if kind == "allocate") == len(graph.nodes)

    def test_four_graphs_are_traced(self):
        assert len(graph_traces()) == 4

    def test_a_random_trace_allocates_and_frees(self):
        trace = random_trace(100)
        assert 0 < sum(1 for kind, _ in trace if kind == "free") < 100

    def test_an_empty_trace_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to replay"):
            run_trace([], Allocator())

    def test_an_unknown_event_is_refused(self):
        with pytest.raises(ConfigError, match="unknown event"):
            run_trace([("borrow", 64)], Allocator())

    def test_a_zero_length_random_trace_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            random_trace(count=0)


class TestPolicies:
    def test_rounding_buys_everything_on_arbitrary_sizes(self):
        # Two arbitrary sizes are never equal, so exact fitting reuses nothing.
        result = rounding_buys_reuse()
        assert result["exact_reuse"] == 0.0
        assert result["power_of_two_reuse"] > 0.8

    def test_and_costs_a_quarter_of_what_it_hands_out(self):
        assert 0.2 < rounding_costs_waste()["power_of_two_waste"] < 0.3

    def test_size_classes_keep_most_of_the_reuse(self):
        assert size_classes_are_the_compromise()["reuse_against_the_coarse_one"] > 0.9

    def test_for_three_fifths_of_the_waste(self):
        assert size_classes_are_the_compromise()["waste_against_the_coarse_one"] < 0.7

    def test_exact_fitting_wastes_nothing(self):
        assert rounding_costs_waste()["exact_waste"] == 0.0

    def test_but_on_a_graph_every_policy_is_identical(self):
        # Every tensor in a graph already has a power of two size, so there is nothing to round.
        rows = compare_policies()
        assert len({row["reuse_rate"] for row in rows}) == 1
        assert len({row["internal_waste"] for row in rows}) == 1

    def test_which_is_where_the_reuse_actually_comes_from(self):
        result = a_graph_trace_reuses_more_than_a_random_one()
        assert result["graph_exact_reuse"] > 0.5
        assert result["random_exact_reuse"] == 0.0

    def test_every_policy_is_compared(self):
        assert len(compare_policies()) == len(POLICIES)


class TestCaching:
    def test_caching_costs_nothing_on_a_graph_trace(self):
        # A graph frees a value and immediately asks for one the same size.
        assert caching_against_returning()["extra_memory"] == 1.0

    def test_and_buys_the_whole_reuse_rate(self):
        assert caching_against_returning()["caching_reuse"] > 0.7

    def test_a_returning_allocator_reuses_nothing(self):
        assert returning_gives_up_every_reuse()["returning_reuses_nothing"]

    def test_the_branching_graph_is_the_one_with_anything_to_say(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["branching"]["reuse_rate"] > rows["mlp"]["reuse_rate"]

    def test_every_graph_is_reported(self):
        assert len(compare_graphs()) == 4

    def test_the_high_water_is_at_least_the_live_peak(self):
        rows = compare_graphs()
        assert all(row["high_water"] >= row["live_peak"] for row in rows)

    def test_a_graph_with_several_values_alive_holds_more(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["branching"]["live_peak"] > rows["softmax"]["live_peak"]

    def test_a_trace_can_be_replayed_through_any_policy(self):
        trace = trace_from(branching_graph())
        for policy in POLICIES:
            allocator = Allocator(policy=policy)
            run_trace(trace, allocator)
            assert allocator.stats.requests == len(branching_graph().nodes)
