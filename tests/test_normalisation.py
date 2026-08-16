from __future__ import annotations

import pytest
import torch

from tgc.analysis.normalisation import (
    NormShape,
    a_rank_three_input_is_refused,
    an_unknown_normalisation_is_refused,
    arithmetic_per_element,
    batch_norm,
    batch_norm_depends_on_the_batch,
    compare_all,
    compare_cost,
    fold_into_the_weight,
    matches_the_library,
    nothing_is_best_on_every_axis,
    offset_sweep,
    one_reduction_instead_of_two,
    random_rows,
    reductions_for,
    rms_norm,
    the_fold_computes_the_same_thing,
    the_fold_removes_an_operation,
    the_gap_grows_with_the_offset,
    the_root_mean_square_version_stops_normalising,
    the_row_norms_do_not,
    the_two_row_norms_agree_on_centred_data,
    why_it_only_works_at_inference,
)
from tgc.errors import ConfigError


class TestCorrectness:
    def test_the_layer_version_matches_the_library(self):
        assert matches_the_library()["layer_gap"] < 1e-5

    def test_and_so_does_the_batch_version(self):
        assert matches_the_library()["batch_gap"] < 1e-5

    def test_the_root_mean_square_one_matches_on_centred_data(self):
        # It has no library equivalent, so it is checked where it is the same function.
        assert matches_the_library()["rms_against_layer_on_centred"] < 1e-6

    def test_to_the_rounding_unit(self):
        assert the_two_row_norms_agree_on_centred_data()["relative_gap"] < 1e-6

    def test_a_rank_three_input_is_refused(self):
        assert a_rank_three_input_is_refused()

    def test_a_zero_epsilon_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            rms_norm(torch.randn(4, 8), torch.ones(8), epsilon=0.0)

    def test_a_zero_dimension_shape_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            NormShape(0, 8)

    def test_it_serialises(self):
        assert NormShape(8, 32).as_dict()["elements"] == 256


class TestOffsets:
    def test_the_two_row_norms_differ_even_at_no_offset(self):
        # A finite row is not centred just because it was drawn from something that is.
        rows = {row["offset"]: row for row in offset_sweep()}
        assert rows[0.0]["largest_gap"] > 0.1

    def test_and_the_gap_grows_with_the_offset(self):
        assert the_gap_grows_with_the_offset()["grew"]

    def test_by_more_than_an_order_of_magnitude(self):
        result = the_gap_grows_with_the_offset()
        assert result["at_ten"] > 10 * result["at_zero"]

    def test_the_layer_version_keeps_a_unit_spread(self):
        result = the_root_mean_square_version_stops_normalising()
        assert abs(result["layer_spread_at_ten"] - result["layer_spread_at_zero"]) < 0.01

    def test_the_other_one_stops_normalising(self):
        # At an offset of ten it is dividing by a number with almost nothing to do with the
        # variation it was meant to normalise.
        result = the_root_mean_square_version_stops_normalising()
        assert result["rms_spread_at_ten"] < 0.2

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            offset_sweep(offsets=())


class TestCost:
    def test_the_root_mean_square_version_does_one_reduction(self):
        assert reductions_for("rms") == 1

    def test_where_the_others_do_two(self):
        assert reductions_for("layer") == 2
        assert reductions_for("batch") == 2

    def test_and_three_eighths_less_work(self):
        assert one_reduction_instead_of_two()["operations_saved"] == 0.375

    def test_an_unknown_normalisation_is_refused(self):
        assert an_unknown_normalisation_is_refused()

    def test_and_when_asked_for_its_reductions(self):
        with pytest.raises(ConfigError, match="unknown normalisation"):
            reductions_for("magic")

    def test_three_versions_are_costed(self):
        assert len(compare_cost()) == 3

    def test_the_cheapest_is_the_one_with_one_reduction(self):
        rows = compare_cost()
        cheapest = min(rows, key=lambda row: row["operations"])
        assert cheapest["normalisation"] == "rms"


class TestBatchDependence:
    def test_a_batch_normalisation_depends_on_the_other_rows(self):
        # Which means it cannot be evaluated for one example or split without communicating.
        assert batch_norm_depends_on_the_batch()["changed"]

    def test_and_by_a_lot(self):
        assert batch_norm_depends_on_the_batch()["largest_change"] > 1.0

    def test_the_row_wise_versions_do_not(self):
        result = the_row_norms_do_not()
        assert not result["layer_changed"]
        assert not result["rms_changed"]

    def test_the_dependence_is_recorded_in_the_comparison(self):
        rows = {row["normalisation"]: row for row in compare_all()}
        assert rows["batch"]["depends_on_the_batch"]
        assert not rows["layer"]["depends_on_the_batch"]


class TestFolding:
    def test_the_fold_computes_the_same_thing(self):
        assert the_fold_computes_the_same_thing()["relative_gap"] < 1e-5

    def test_though_not_bit_for_bit(self):
        # It scales the weight once rather than the output once per row.
        assert not the_fold_computes_the_same_thing()["identical"]

    def test_it_removes_a_whole_operation(self):
        result = the_fold_removes_an_operation()
        assert len(result["after"]) == len(result["before"]) - 1

    def test_only_the_batch_version_folds(self):
        assert nothing_is_best_on_every_axis()["foldable"] == ["batch"]

    def test_and_it_is_not_the_cheapest_one(self):
        assert not nothing_is_best_on_every_axis()["one_wins_everything"]

    def test_the_fold_needs_stored_statistics(self):
        # During training they come from the batch and change every step.
        assert why_it_only_works_at_inference()["statistics_come_from_the_batch"]

    def test_a_zero_epsilon_fold_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            fold_into_the_weight(
                torch.randn(4, 4),
                torch.zeros(4),
                torch.ones(4),
                torch.zeros(4),
                (torch.zeros(1, 4), torch.ones(4)),
                epsilon=0.0,
            )

    def test_the_folded_weight_has_the_original_shape(self):
        weight = torch.randn(8, 4)
        folded, _ = fold_into_the_weight(
            weight,
            torch.zeros(4),
            torch.ones(4),
            torch.zeros(4),
            (torch.zeros(1, 4), torch.ones(4)),
        )
        assert folded.shape == weight.shape


class TestComparison:
    def test_nothing_wins_on_every_axis(self):
        result = nothing_is_best_on_every_axis()
        assert result["cheapest"] not in result["foldable"]

    def test_two_of_the_three_are_batch_independent(self):
        assert len(nothing_is_best_on_every_axis()["batch_independent"]) == 2

    def test_every_version_appears_in_the_table(self):
        assert len(compare_all()) == 3

    def test_the_stored_statistics_path_is_an_affine_map(self):
        values, gain, shift = random_rows(NormShape(4, 8))
        running = (torch.zeros(1, 8), torch.ones(8))
        doubled = batch_norm(values * 2, gain, shift, running=running)
        single = batch_norm(values, gain, shift, running=running)
        assert torch.allclose(doubled, 2 * single, atol=1e-5)

    def test_the_arithmetic_table_covers_every_version(self):
        assert all(arithmetic_per_element(kind) > 0 for kind in ("layer", "rms", "batch"))
