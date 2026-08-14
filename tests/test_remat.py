from __future__ import annotations

import math

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.graph import validate
from tgc.passes.remat import (
    CheckpointCost,
    RematPlan,
    best_interval,
    expanding_graph,
    long_range_graph,
    measure_rematerialisation,
    rematerialise,
    select_candidates,
    square_root_agreement,
    sweep_intervals,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


class TestCheckpointCost:
    def test_keeping_everything_costs_no_extra_arithmetic(self):
        assert CheckpointCost(length=64, interval=1).extra_steps == 0

    def test_and_holds_the_whole_chain(self):
        assert CheckpointCost(length=64, interval=1).checkpoints == 64

    def test_keeping_one_holds_one_checkpoint(self):
        assert CheckpointCost(length=64, interval=64).checkpoints == 1

    def test_but_still_needs_the_segment_being_redone(self):
        # Which is what stops an arbitrarily long interval from being free.
        assert CheckpointCost(length=64, interval=64).peak_values == 65

    def test_the_middle_holds_far_less_than_either_end(self):
        ends = CheckpointCost(length=64, interval=1).peak_values
        middle = CheckpointCost(length=64, interval=8).peak_values
        assert middle < ends / 4

    def test_the_overhead_is_the_share_recomputed(self):
        assert CheckpointCost(length=64, interval=8).overhead == pytest.approx(56 / 64)

    def test_an_interval_past_the_length_is_rejected(self):
        with pytest.raises(ConfigError, match="between one and the chain length"):
            CheckpointCost(length=8, interval=9)

    def test_a_zero_interval_is_rejected(self):
        with pytest.raises(ConfigError, match="between one and the chain length"):
            CheckpointCost(length=8, interval=0)

    def test_an_empty_chain_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one step"):
            CheckpointCost(length=0, interval=1)

    def test_it_serialises(self):
        assert CheckpointCost(length=64, interval=8).as_dict()["checkpoints"] == 8


class TestSquareRoot:
    def test_the_best_interval_on_a_square_is_its_root(self):
        assert best_interval(64) == 8
        assert best_interval(4096) == 64

    def test_the_sweep_agrees_with_the_search(self):
        rows = sweep_intervals(64)
        assert min(rows, key=lambda row: row["peak_values"])["interval"] == best_interval(64)

    def test_the_integer_root_is_as_good_as_the_optimum_everywhere(self):
        # The claim worth making. Comparing intervals reports a disagreement that costs
        # nothing, because the minimum is flat across a wide plateau.
        for row in square_root_agreement([16, 20, 50, 100, 500, 1000, 4096]):
            assert row["root_peak"] == row["best_peak"]

    def test_the_intervals_themselves_sometimes_differ(self):
        rows = {row["length"]: row for row in square_root_agreement([50, 1000])}
        assert rows[1000]["best_interval"] != rows[1000]["square_root"]

    def test_the_peak_grows_like_the_root(self):
        small = CheckpointCost(length=100, interval=best_interval(100)).peak_values
        large = CheckpointCost(length=10000, interval=best_interval(10000)).peak_values
        assert large / small == pytest.approx(math.sqrt(100), rel=0.1)

    def test_an_empty_length_list_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to check"):
            square_root_agreement([])

    def test_a_zero_length_chain_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one step"):
            best_interval(0)


class TestSelection:
    def test_a_value_read_far_from_where_it_was_made_is_a_candidate(self):
        graph = long_range_graph(depth=6)
        assert select_candidates(graph, min_gap=6) == ["v0"]

    def test_a_value_read_immediately_is_not(self):
        # Recomputing it saves nothing and costs the arithmetic.
        graph = long_range_graph(depth=6)
        assert "v1" not in select_candidates(graph, min_gap=6)

    def test_an_output_is_never_a_candidate(self):
        graph = long_range_graph(depth=6)
        assert graph.outputs[0] not in select_candidates(graph, min_gap=1)

    def test_a_zero_gap_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            select_candidates(long_range_graph(), min_gap=0)

    def test_a_smaller_gap_finds_more(self):
        graph = long_range_graph(depth=6)
        assert len(select_candidates(graph, min_gap=1)) > len(
            select_candidates(graph, min_gap=6)
        )


class TestRewriting:
    def test_the_answer_does_not_change(self):
        graph = expanding_graph(depth=4, width=16)
        candidates = select_candidates(graph, min_gap=4)
        feeds = random_feeds(graph)
        rewritten = rematerialise(graph, candidates)
        assert outputs_agree(run(graph, feeds), run(rewritten, feeds))

    def test_the_result_still_validates(self):
        graph = expanding_graph(depth=4, width=16)
        validate(rematerialise(graph, select_candidates(graph, min_gap=4)))

    def test_rematerialising_something_absent_is_rejected(self):
        with pytest.raises(PassError, match="which nothing produces"):
            rematerialise(long_range_graph(), ["nothing"])

    def test_rematerialising_a_leaf_is_rejected(self):
        graph = long_range_graph()
        constant = next((node.name for node in graph.nodes if node.op.is_leaf), None)
        if constant is not None:
            with pytest.raises(PassError, match="has nothing to recompute"):
                rematerialise(graph, [constant])

    def test_rematerialising_nothing_leaves_the_graph_alone(self):
        graph = long_range_graph()
        assert len(rematerialise(graph, []).nodes) == len(graph.nodes)


class TestMeasurement:
    def test_several_long_lived_tensors_are_worth_recomputing(self):
        result = measure_rematerialisation(expanding=True)
        assert result["saved"] > 0
        assert result["saved_fraction"] > 0.4

    def test_a_single_long_lived_tensor_is_not(self):
        # The final addition needs it and the chain result at the same instant, so the peak
        # sits at that addition either way and recomputing swaps one live tensor for another
        # of equal size.
        assert measure_rematerialisation(expanding=False)["saved"] == 0

    def test_the_fixture_finds_something_to_recompute_either_way(self):
        assert measure_rematerialisation(expanding=True)["candidates"]
        assert measure_rematerialisation(expanding=False)["candidates"]

    def test_a_chain_with_no_room_is_rejected(self):
        with pytest.raises(ConfigError, match="somewhere to be long across"):
            expanding_graph(depth=1)

    def test_a_fixture_holding_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="something to hold"):
            expanding_graph(held=0)


class TestPlan:
    def test_it_counts_what_it_keeps(self):
        assert RematPlan(kept=["a", "b"], recomputed=["c"]).stored == 2

    def test_it_serialises(self):
        assert RematPlan(kept=["a"], recomputed=["b", "c"]).as_dict()["recomputed"] == 2
