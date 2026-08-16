from __future__ import annotations

import math

import pytest
import torch

from tgc.analysis.attention import (
    AttentionShape,
    a_small_block_costs_almost_no_accuracy,
    attention_graph,
    block_size_changes_the_overhead,
    block_size_sweep,
    extra_arithmetic,
    memory_for,
    naive_attention,
    random_head,
    sequence_sweep,
    streaming_attention,
    streaming_matches_the_reference,
    the_block_size_trade,
    the_graph_agrees_with_the_reference,
    the_graph_has_a_square_intermediate,
    the_running_maximum_is_not_optional,
    the_saving_grows_with_the_sequence,
    unstable_streaming_attention,
    where_the_exponential_overflows,
)
from tgc.errors import ConfigError
from tgc.ir import op as ops


class TestShape:
    def test_the_scores_are_square_in_the_sequence(self):
        assert AttentionShape(sequence=64, width=8).score_elements == 4096

    def test_and_the_inputs_are_not(self):
        assert AttentionShape(sequence=64, width=8).input_elements == 3 * 64 * 8

    def test_the_arithmetic_covers_both_products(self):
        assert AttentionShape(sequence=64, width=8).arithmetic == 4 * 64 * 64 * 8

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            AttentionShape(sequence=0, width=8)

    def test_it_serialises(self):
        assert AttentionShape(sequence=64, width=8).as_dict()["scores"] == 4096


class TestCorrectness:
    def test_the_streaming_version_matches_the_reference(self):
        assert streaming_matches_the_reference()["relative_gap"] < 1e-5

    def test_but_not_bit_for_bit(self):
        # It adds the contributions in blocks and rescales between them, which is a different
        # order of the same additions.
        assert not streaming_matches_the_reference()["identical"]

    def test_a_block_as_large_as_the_sequence_is_the_naive_version(self):
        shape = AttentionShape(sequence=64, width=16)
        queries, keys, values = random_head(shape)
        streamed = streaming_attention(queries, keys, values, block=64)
        assert float((streamed - naive_attention(queries, keys, values)).abs().max()) < 1e-6

    def test_a_block_of_one_still_agrees(self):
        rows = {row["block"]: row for row in block_size_sweep()}
        assert rows[1]["largest_gap"] < 1e-5

    def test_and_costs_only_half_again_in_error(self):
        # Two hundred and fifty six times as many rescalings for half again the error.
        assert a_small_block_costs_almost_no_accuracy()["ratio"] < 2.0

    def test_a_block_of_nothing_is_refused(self):
        shape = AttentionShape(sequence=16, width=8)
        queries, keys, values = random_head(shape)
        with pytest.raises(ConfigError, match="has to hold something"):
            streaming_attention(queries, keys, values, block=0)

    def test_an_empty_block_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            block_size_sweep(blocks=())

    def test_the_softmax_weights_still_sum_to_one(self):
        shape = AttentionShape(sequence=32, width=8)
        queries, keys, _ = random_head(shape)
        ones = torch.ones(shape.sequence, 1)
        result = streaming_attention(queries, keys, ones)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


class TestStability:
    def test_without_the_running_maximum_the_answer_is_nan(self):
        # The exponential of anything above about eighty nine is an infinity in float32.
        assert not the_running_maximum_is_not_optional()["unstable_is_finite"]

    def test_with_it_the_answer_is_finite(self):
        assert the_running_maximum_is_not_optional()["stable_is_finite"]

    def test_and_still_matches_the_reference(self):
        assert the_running_maximum_is_not_optional()["stable_gap"] < 1e-5

    def test_the_scores_really_do_pass_the_overflow_point(self):
        assert the_running_maximum_is_not_optional()["largest_score"] > 89.0

    def test_the_boundary_is_two_doublings_from_an_ordinary_scale(self):
        rows = {row["scale"]: row for row in where_the_exponential_overflows()}
        assert rows[1.0]["finite"]
        assert not rows[8.0]["finite"]

    def test_a_zero_block_is_refused_there_too(self):
        shape = AttentionShape(sequence=16, width=8)
        queries, keys, values = random_head(shape)
        with pytest.raises(ConfigError, match="has to hold something"):
            unstable_streaming_attention(queries, keys, values, block=0)

    def test_an_empty_scale_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            where_the_exponential_overflows(scales=())


class TestMemory:
    def test_a_long_sequence_saves_a_great_deal(self):
        assert memory_for(AttentionShape(sequence=8192, width=64))["ratio"] > 20.0

    def test_a_short_one_saves_nothing(self):
        # The streaming version holds the whole matrix as its block plus three running values.
        assert memory_for(AttentionShape(sequence=128, width=64))["ratio"] < 1.0

    def test_the_saving_grows_with_the_sequence(self):
        assert the_saving_grows_with_the_sequence()["grew"]

    def test_by_more_than_an_order_of_magnitude_over_the_sweep(self):
        result = the_saving_grows_with_the_sequence()
        assert result["at_the_longest"] / result["at_the_shortest"] > 10.0

    def test_a_block_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            memory_for(AttentionShape(sequence=64, width=8), block=0)

    def test_an_empty_sequence_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            sequence_sweep(lengths=())


class TestOverhead:
    def test_the_rescaling_is_a_fifth_of_a_percent_at_a_sensible_block(self):
        shape = AttentionShape(sequence=2048, width=64)
        assert extra_arithmetic(shape, 128)["share"] < 0.01

    def test_and_a_quarter_of_the_work_at_a_block_of_one(self):
        shape = AttentionShape(sequence=2048, width=64)
        assert extra_arithmetic(shape, 1)["share"] > 0.2

    def test_it_falls_like_one_over_the_block(self):
        rows = {row["block"]: row for row in block_size_changes_the_overhead()}
        assert rows[16]["share"] / rows[128]["share"] == pytest.approx(8.0, rel=0.05)

    def test_memory_and_overhead_move_in_opposite_directions(self):
        result = the_block_size_trade()
        assert result["memory_grows"]
        assert result["overhead_falls"]

    def test_a_block_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            extra_arithmetic(AttentionShape(sequence=64, width=8), block=0)

    def test_an_empty_overhead_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            block_size_changes_the_overhead(blocks=())


class TestGraph:
    def test_the_graph_computes_what_the_reference_does(self):
        # The fixture is written twice, and one that computed something else would make every
        # measurement taken on it meaningless.
        assert the_graph_agrees_with_the_reference()["relative_gap"] == 0.0

    def test_the_middle_of_the_graph_is_larger_than_its_ends(self):
        assert the_graph_has_a_square_intermediate()["ratio"] > 1.0

    def test_the_graph_holds_two_products(self):
        graph = attention_graph()
        assert sum(1 for node in graph.nodes if node.op is ops.MATMUL) == 2

    def test_and_the_softmax_that_makes_it_attention(self):
        graph = attention_graph()
        names = {node.op.name for node in graph.nodes}
        assert {"max", "exp", "sum", "div"} <= names

    def test_the_scale_is_one_over_the_root_of_the_width(self):
        graph = attention_graph(sequence=32, width=16)
        literals = [
            float(node.attrs["value"]) for node in graph.nodes if node.op is ops.CONSTANT
        ]
        assert literals == [pytest.approx(1.0 / math.sqrt(16))]

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            attention_graph(sequence=0)
