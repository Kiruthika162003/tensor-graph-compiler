from __future__ import annotations

import pytest
import torch

from tgc.analysis.embedding import (
    TableShape,
    a_token_outside_the_table_is_refused,
    assigning_instead_of_adding_is_wrong,
    batch_sweep,
    dense_gradient,
    duplicates_are_common_in_real_text,
    lookup,
    memory_comparison,
    most_of_the_table_is_never_read,
    optimiser_state_for,
    random_batch,
    rows_touched_per_step,
    scatter_without_adding,
    sparse_gradient,
    sparse_updates_change_the_answer,
    state_memory_comparison,
    the_forward_pass_does_no_arithmetic,
    the_gradient_matches_autograd,
    the_saving_is_the_vocabulary_over_the_batch,
    the_sparse_form_is_the_same_gradient,
    the_table_is_most_of_a_small_model,
    vocabulary_sweep,
)
from tgc.errors import ConfigError


class TestShape:
    def test_the_table_is_the_vocabulary_by_the_width(self):
        assert TableShape(1000, 16, 200).table_elements == 16000

    def test_the_batch_reads_one_row_per_token(self):
        assert TableShape(1000, 16, 200).read_elements == 3200

    def test_the_density_is_the_batch_over_the_vocabulary(self):
        assert TableShape(1000, 16, 200).density == 0.2

    def test_and_never_exceeds_one(self):
        assert TableShape(10, 16, 200).density == 1.0

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            TableShape(0, 16, 200)

    def test_it_serialises(self):
        assert TableShape(1000, 16, 200).as_dict()["density"] == 0.2


class TestGradient:
    def test_the_dense_gradient_matches_autograd(self):
        assert the_gradient_matches_autograd()["identical"]

    def test_the_sparse_form_expands_back_to_it(self):
        # A lost contribution is a row that trains slower than it should for reasons nothing
        # will report.
        assert the_sparse_form_is_the_same_gradient()["identical"]

    def test_and_holds_only_the_rows_that_were_read(self):
        result = the_sparse_form_is_the_same_gradient()
        assert 0 < result["touched_rows"] < 1000

    def test_assigning_instead_of_adding_is_wrong(self):
        result = assigning_instead_of_adding_is_wrong()
        assert result["largest_gap"] > 1.0

    def test_but_has_the_right_shape(self):
        # Which is the shape of a bug that survives a shape check.
        assert assigning_instead_of_adding_is_wrong()["same_shape"]

    def test_the_commonest_token_appears_hundreds_of_times(self):
        assert assigning_instead_of_adding_is_wrong()["commonest_token_appears"] > 100

    def test_a_uniform_batch_would_hide_it(self):
        # A test written against a uniform batch would pass with the wrong scatter in it.
        rows = duplicates_are_common_in_real_text()
        assert rows["uniform"]["commonest"] < rows["skewed"]["commonest"] / 10

    def test_the_scatter_and_the_merge_agree_on_the_rows(self):
        shape = TableShape(100, 4, 200)
        _, indices, cotangent = random_batch(shape, skewed=True)
        first, _ = sparse_gradient(indices, cotangent)
        second, _ = scatter_without_adding(indices, cotangent)
        assert torch.equal(first, second)

    def test_a_rank_one_table_is_refused(self):
        with pytest.raises(ConfigError, match="a table is a matrix"):
            lookup(torch.randn(8), torch.tensor([0]))

    def test_a_token_outside_the_table_is_refused(self):
        assert a_token_outside_the_table_is_refused()

    def test_the_dense_gradient_is_the_size_of_the_table(self):
        shape = TableShape(50, 4, 20)
        table, indices, cotangent = random_batch(shape)
        assert dense_gradient(table, indices, cotangent).shape == table.shape


class TestMemory:
    def test_a_large_vocabulary_makes_the_dense_form_enormous(self):
        assert memory_comparison()["ratio"] == 25.0

    def test_the_ratio_is_the_vocabulary_over_the_batch(self):
        assert the_saving_is_the_vocabulary_over_the_batch()["all_match"]

    def test_a_small_vocabulary_makes_it_the_smaller_of_the_two(self):
        rows = {row["vocabulary"]: row for row in vocabulary_sweep()}
        assert rows[1000]["ratio"] < 1.0

    def test_a_large_batch_narrows_the_gap(self):
        rows = {row["tokens"]: row for row in batch_sweep()}
        assert rows[8192]["ratio"] < rows[128]["ratio"]

    def test_but_never_closes_it(self):
        # Most of a large vocabulary goes untouched in any single step.
        rows = {row["tokens"]: row for row in batch_sweep()}
        assert rows[8192]["ratio"] > 1.0

    def test_an_empty_vocabulary_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            vocabulary_sweep(sizes=())

    def test_an_empty_batch_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            batch_sweep(sizes=())


class TestWork:
    def test_a_lookup_does_no_arithmetic(self):
        # There is nothing a compiler can fuse into it.
        assert the_forward_pass_does_no_arithmetic()["multiplies"] == 0

    def test_and_reads_what_it_writes(self):
        result = the_forward_pass_does_no_arithmetic()
        assert result["elements_read"] == result["elements_written"]

    def test_a_step_touches_a_small_share_of_the_table(self):
        result = rows_touched_per_step(steps=10)
        assert result["share_per_step"] < 0.1

    def test_and_most_of_it_is_never_touched_at_all(self):
        assert most_of_the_table_is_never_read(steps=10)["share_never_touched"] > 0.5

    def test_a_zero_step_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            rows_touched_per_step(steps=0)


class TestOptimiser:
    def test_a_sparse_update_holds_less_state(self):
        assert state_memory_comparison()["ratio"] == 25.0

    def test_the_dense_state_covers_the_whole_table(self):
        shape = TableShape(1000, 16, 200)
        assert optimiser_state_for(shape, sparse=False) == 2 * 1000 * 16 * 4

    def test_and_the_sparse_one_covers_the_batch(self):
        shape = TableShape(1000, 16, 200)
        assert optimiser_state_for(shape, sparse=True) == 2 * 200 * 16 * 4

    def test_skipping_the_decay_changes_the_moment(self):
        # The sparse rule is a different optimiser and it is the one everybody runs.
        assert sparse_updates_change_the_answer()["ratio"] != 1.0

    def test_by_more_than_a_rounding_difference(self):
        result = sparse_updates_change_the_answer()
        assert abs(result["ratio"] - 1.0) > 0.05

    def test_a_zero_step_count_is_refused(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            sparse_updates_change_the_answer(steps=0)

    def test_the_table_is_most_of_a_small_model(self):
        assert the_table_is_most_of_a_small_model()["share"] > 0.3

    def test_a_zero_width_model_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            the_table_is_most_of_a_small_model(width=0)
