from __future__ import annotations

import pytest

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import elementwise_chain, mlp_graph, softmax_graph
from tgc.runtime.profile import (
    NodeProfile,
    Timing,
    arena_allocation_cost,
    compare_execution,
    deterministic_output,
    hot_node_share,
    model_against_measurement,
    profile_nodes,
    ranking_agreement,
    time_call,
    time_compiled,
    time_compiled_reusing_arena,
    time_interpreter,
    warmup_matters,
)

SIZES = [
    ("small", mlp_graph(batch=8, hidden=32)),
    ("medium", mlp_graph(batch=64, hidden=128)),
    ("large", mlp_graph(batch=256, hidden=256)),
]


class TestTiming:
    def test_the_median_is_the_middle_sample(self):
        assert Timing(label="t", samples=[1.0, 2.0, 9.0]).median == 2.0

    def test_the_fastest_is_the_smallest(self):
        assert Timing(label="t", samples=[1.0, 2.0, 9.0]).fastest == 1.0

    def test_the_median_ignores_one_bad_sample(self):
        # A mean over ten runs where one was interrupted reports the interruption.
        clean = Timing(label="t", samples=[1.0, 1.0, 1.0])
        disturbed = Timing(label="t", samples=[1.0, 1.0, 1.0, 1.0, 50.0])
        assert disturbed.median == clean.median

    def test_the_spread_says_whether_a_comparison_is_meaningful(self):
        assert Timing(label="t", samples=[1.0, 1.0, 1.0]).spread == 0.0
        assert Timing(label="t", samples=[1.0, 3.0]).spread > 0.5

    def test_a_single_sample_has_no_spread(self):
        assert Timing(label="t", samples=[1.0]).spread == 0.0

    def test_an_empty_timing_has_no_median(self):
        with pytest.raises(ScheduleError, match="no samples"):
            _ = Timing(label="t").median

    def test_nor_a_fastest(self):
        with pytest.raises(ScheduleError, match="no samples"):
            _ = Timing(label="t").fastest

    def test_a_nameless_timing_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a label"):
            Timing(label="")

    def test_a_negative_duration_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            Timing(label="t", samples=[-1.0])

    def test_it_serialises(self):
        assert Timing(label="t", samples=[1.0, 2.0]).as_dict()["samples"] == 2


class TestTimingCalls:
    def test_it_takes_the_requested_number_of_samples(self):
        assert time_call(lambda: None, repeats=5, warmups=1).count == 5

    def test_the_warm_ups_are_not_among_them(self):
        calls = []
        time_call(lambda: calls.append(1), repeats=3, warmups=2)
        assert len(calls) == 5

    def test_zero_repeats_are_rejected(self):
        with pytest.raises(ConfigError, match="at least one repeat"):
            time_call(lambda: None, repeats=0)

    def test_a_negative_warm_up_count_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            time_call(lambda: None, warmups=-1)

    def test_every_sample_is_a_real_duration(self):
        timing = time_call(lambda: None, repeats=4)
        assert all(sample >= 0 for sample in timing.samples)


class TestExecution:
    def test_both_paths_can_be_timed(self):
        graph = mlp_graph(batch=32, hidden=64)
        assert time_interpreter(graph, repeats=3).count == 3
        assert time_compiled(graph, repeats=3).count == 3

    def test_the_arena_can_be_reused_across_calls(self):
        graph = mlp_graph(batch=32, hidden=64)
        assert time_compiled_reusing_arena(graph, repeats=3).count == 3

    def test_the_comparison_reports_its_own_spread(self):
        # So the ratio can be read honestly rather than quoted on its own.
        result = compare_execution(mlp_graph(batch=32, hidden=64), repeats=5)
        assert result["worst_spread"] > 0

    def test_the_allocation_cost_is_reported_either_way(self):
        # It is a small number on small graphs and the noise is larger, so the test checks
        # that both medians exist rather than pretending the difference is resolvable.
        result = arena_allocation_cost(mlp_graph(batch=32, hidden=64), repeats=5)
        assert result["fresh_arena_median"] > 0
        assert result["reused_arena_median"] > 0

    def test_repeated_runs_produce_the_same_answer(self):
        # Timing a thing many times is only meaningful if the thing is the same each time,
        # and the arena is reused across calls.
        for graph in (softmax_graph(), mlp_graph(batch=32, hidden=64), elementwise_chain(8)):
            assert deterministic_output(graph)

    def test_the_first_call_is_reported_separately(self):
        result = warmup_matters(mlp_graph(batch=32, hidden=64), repeats=5)
        assert result["first_call"] > 0
        assert result["steady_median"] > 0


class TestModelAgainstMeasurement:
    def test_the_model_ranks_three_sizes_correctly(self):
        # The ranking is the only thing the model claims, and comparing its predicted seconds
        # against a measurement would be comparing a model of a GPU to a run on a CPU.
        assert ranking_agreement(SIZES, repeats=3) == 1.0

    def test_every_graph_gets_a_row(self):
        rows = model_against_measurement(SIZES, repeats=3)
        assert len(rows) == len(SIZES)

    def test_each_row_carries_its_spread(self):
        rows = model_against_measurement(SIZES, repeats=3)
        assert all("spread" in row for row in rows)

    def test_the_predicted_times_are_ordered_by_size(self):
        rows = model_against_measurement(SIZES, repeats=3)
        predicted = [row["predicted_seconds"] for row in rows]
        assert predicted == sorted(predicted)

    def test_comparing_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            model_against_measurement([])

    def test_a_ranking_of_one_has_no_order(self):
        with pytest.raises(ConfigError, match="no order to compare"):
            ranking_agreement(SIZES[:1])


class TestNodeProfile:
    def test_every_node_gets_a_time(self):
        graph = mlp_graph(batch=32, hidden=64)
        profile = profile_nodes(graph, repeats=3)
        assert set(profile.per_node) == {node.name for node in graph.nodes}

    def test_the_hottest_nodes_come_first(self):
        profile = profile_nodes(mlp_graph(batch=64, hidden=128), repeats=3)
        hottest = profile.hottest(2)
        assert hottest[0][1] >= hottest[1][1]

    def test_one_node_accounts_for_a_large_share(self):
        # Which is the argument for measuring before optimising.
        assert hot_node_share(mlp_graph(batch=64, hidden=128), repeats=3) > 0.2

    def test_the_shares_add_up_to_one(self):
        profile = profile_nodes(mlp_graph(batch=32, hidden=64), repeats=3)
        total = sum(profile.share_of(name) for name in profile.per_node)
        assert total == pytest.approx(1.0)

    def test_an_unprofiled_node_is_rejected(self):
        profile = profile_nodes(softmax_graph(), repeats=3)
        with pytest.raises(ScheduleError, match="was not profiled"):
            profile.share_of("nothing")

    def test_a_zero_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            NodeProfile(per_node={"a": 1.0}).hottest(0)

    def test_an_empty_profile_totals_nothing(self):
        assert NodeProfile().total == 0.0

    def test_it_serialises(self):
        assert profile_nodes(softmax_graph(), repeats=3).as_dict()["nodes"] == 5
