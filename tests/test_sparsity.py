from __future__ import annotations

import pytest
import torch

from tgc.analysis.sparsity import (
    SparsityReport,
    a_block_of_one_is_unstructured,
    accuracy_against_speed,
    block_size_sweep,
    break_even_density,
    compare_patterns,
    density_sweep,
    indices_undo_the_saving,
    measure,
    output_error,
    overhead_sweep,
    prune_blocks,
    prune_n_of_m,
    prune_unstructured,
    speedup_from,
    storage_bytes,
    storage_sweep,
    structure_costs_little_accuracy,
    the_gap_narrows_as_the_weight_thins,
    the_ranking_survives_the_contraction,
    weight_matrix,
)
from tgc.errors import ConfigError


class TestPruning:
    def test_unstructured_pruning_keeps_the_requested_count(self):
        values = weight_matrix(16, 16)
        assert int((prune_unstructured(values, 0.5) != 0).sum()) == 128

    def test_and_keeps_the_largest_ones(self):
        values = torch.tensor([[1.0, -5.0], [0.1, 3.0]])
        pruned = prune_unstructured(values, 0.5)
        assert set(pruned.flatten().tolist()) == {0.0, -5.0, 3.0}

    def test_block_pruning_keeps_whole_tiles(self):
        values = weight_matrix(16, 16)
        pruned = prune_blocks(values, 0.5, 4)
        tiles = pruned.reshape(4, 4, 4, 4)
        counts = {int((tiles[i, :, j, :] != 0).sum()) for i in range(4) for j in range(4)}
        assert counts == {0, 16}

    def test_n_of_m_keeps_two_from_every_four(self):
        values = weight_matrix(16, 16)
        grouped = prune_n_of_m(values, 2, 4).flatten().reshape(-1, 4)
        assert set((grouped != 0).sum(dim=1).tolist()) == {2}

    def test_a_density_of_one_keeps_everything(self):
        values = weight_matrix(16, 16)
        assert torch.equal(prune_unstructured(values, 1.0), values)

    def test_a_density_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ConfigError, match="has to be in"):
            prune_unstructured(weight_matrix(8, 8), 1.5)

    def test_block_pruning_needs_a_matrix(self):
        with pytest.raises(ConfigError, match="needs a matrix"):
            prune_blocks(torch.randn(16), 0.5)

    def test_a_shape_that_does_not_divide_is_refused(self):
        with pytest.raises(ConfigError, match="does not divide"):
            prune_blocks(weight_matrix(10, 10), 0.5, 4)

    def test_keeping_more_than_a_group_holds_is_refused(self):
        with pytest.raises(ConfigError, match="cannot keep"):
            prune_n_of_m(weight_matrix(16, 16), keep=5, group=4)

    def test_a_count_that_does_not_group_is_refused(self):
        with pytest.raises(ConfigError, match="do not divide into groups"):
            prune_n_of_m(torch.randn(10), 2, 4)

    def test_an_empty_tensor_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to prune"):
            prune_unstructured(torch.tensor([]), 0.5)


class TestReport:
    def test_density_and_sparsity_add_to_one(self):
        report = SparsityReport(kept=25, total=100)
        assert report.density + report.sparsity == 1.0

    def test_an_empty_tensor_is_refused(self):
        with pytest.raises(ConfigError, match="has to have elements"):
            SparsityReport(kept=0, total=0)

    def test_keeping_more_than_there_is_is_refused(self):
        with pytest.raises(ConfigError, match="cannot keep"):
            SparsityReport(kept=10, total=5)

    def test_it_serialises(self):
        assert SparsityReport(kept=50, total=100).as_dict()["density"] == 0.5

    def test_measuring_reports_what_survived(self):
        values = weight_matrix(16, 16)
        assert measure(values, prune_unstructured(values, 0.25)).density == 0.25


class TestPatterns:
    def test_unstructured_is_the_floor(self):
        rows = {row["pattern"]: row for row in compare_patterns()}
        assert rows["unstructured"]["error"] < rows["2 of 4"]["error"]
        assert rows["unstructured"]["error"] < rows["4 by 4 blocks"]["error"]

    def test_the_hardware_pattern_costs_about_a_third_more(self):
        result = structure_costs_little_accuracy()
        assert 1.3 < result["structure_penalty"] < 1.5

    def test_and_blocks_cost_more_than_double(self):
        assert structure_costs_little_accuracy()["block_penalty"] > 2.0

    def test_every_pattern_keeps_the_same_count(self):
        counts = {row["kept"] for row in compare_patterns()}
        assert len(counts) == 1

    def test_a_block_of_one_is_unstructured_pruning(self):
        # The block method scores tiles by the sum of their squares and the unstructured method
        # scores elements by magnitude, and those are the same order only at a tile of one.
        result = a_block_of_one_is_unstructured()
        assert result["unstructured"] == result["blocks_of_one"]

    def test_larger_tiles_cost_more(self):
        errors = [row["error"] for row in block_size_sweep()]
        assert errors == sorted(errors)

    def test_an_empty_block_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            block_size_sweep(sizes=())


class TestDensity:
    def test_error_climbs_as_the_weight_thins(self):
        rows = density_sweep()
        assert rows[-1]["unstructured"] > rows[0]["unstructured"]

    def test_the_penalty_for_structure_narrows_rather_than_widening(self):
        # Being clever about which elements to drop matters most when few are being dropped.
        assert the_gap_narrows_as_the_weight_thins()["narrowed"]

    def test_by_a_factor_of_eight(self):
        result = the_gap_narrows_as_the_weight_thins()
        assert result["ratio_at_ninety_percent"] / result["ratio_at_ten_percent"] > 5.0

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            density_sweep(densities=())


class TestStorage:
    def test_a_dense_format_stores_four_bytes_an_element(self):
        assert storage_bytes(1000, 1.0, pattern="dense") == 4000

    def test_an_unstructured_one_stores_eight_for_every_survivor(self):
        assert storage_bytes(1000, 0.5, pattern="unstructured") == 500 * 8

    def test_so_at_half_density_it_is_exactly_as_large_as_dense(self):
        # A position is the same size as the value it points at.
        assert indices_undo_the_saving()["density"] == 0.5

    def test_and_larger_above_that(self):
        rows = {row["density"]: row for row in storage_sweep()}
        assert rows[0.9]["unstructured_saves"] < 1.0

    def test_the_hardware_pattern_saves_from_the_start(self):
        rows = {row["density"]: row for row in storage_sweep()}
        assert rows[0.9]["n_of_m_saves"] > 1.0

    def test_but_loses_at_very_low_density(self):
        # Its two bits per group are paid whether or not anything survives in the group.
        rows = {row["density"]: row for row in storage_sweep()}
        assert rows[0.05]["n_of_m_saves"] < rows[0.05]["unstructured_saves"]

    def test_an_unknown_pattern_is_refused(self):
        with pytest.raises(ConfigError, match="unknown pattern"):
            storage_bytes(1000, 0.5, pattern="magic")

    def test_an_empty_tensor_is_refused(self):
        with pytest.raises(ConfigError, match="has to have elements"):
            storage_bytes(0, 0.5)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            storage_sweep(densities=())


class TestOutput:
    def test_the_error_carries_through_a_contraction_unchanged(self):
        # The same result the rounding error gave, and for the same reason.
        rows = output_error()
        for values in rows.values():
            assert abs(values["output_error"] - values["weight_error"]) < 0.02

    def test_so_the_ranking_of_the_patterns_survives(self):
        assert the_ranking_survives_the_contraction()["same_order"]

    def test_and_unstructured_leads_it(self):
        assert the_ranking_survives_the_contraction()["by_output"][0] == "unstructured"


class TestSpeed:
    def test_half_density_with_no_overhead_doubles(self):
        assert speedup_from(0.5) == 2.0

    def test_overhead_eats_the_gain(self):
        rows = {row["overhead"]: row for row in overhead_sweep()}
        assert rows[0.5]["speedup"] == 1.0

    def test_the_break_even_is_one_minus_the_overhead(self):
        assert break_even_density(0.2) == 0.8

    def test_a_negative_overhead_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            speedup_from(0.5, overhead=-0.1)

    def test_an_overhead_of_one_is_refused(self):
        with pytest.raises(ConfigError, match="has to be in"):
            break_even_density(1.0)

    def test_a_barely_pruned_weight_is_slower_than_dense(self):
        # The overhead is larger than the tenth of the work it skipped.
        rows = accuracy_against_speed()
        assert rows[0]["speedup"] < 1.0

    def test_and_a_heavily_pruned_one_is_much_faster(self):
        rows = accuracy_against_speed()
        assert rows[-1]["speedup"] > 3.0

    def test_the_error_climbs_the_whole_way_too(self):
        errors = [row["error"] for row in accuracy_against_speed()]
        assert errors == sorted(errors)

    def test_an_empty_trade_table_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            accuracy_against_speed(densities=())

    def test_an_empty_overhead_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            overhead_sweep(overheads=())
