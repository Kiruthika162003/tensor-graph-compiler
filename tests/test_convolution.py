from __future__ import annotations

import pytest
import torch

from tgc.analysis.convolution import (
    ConvShape,
    a_kernel_that_does_not_fit_is_refused,
    a_one_by_one_convolution_is_already_a_product,
    arithmetic_intensity,
    as_matrix_product,
    compare_routes,
    direct_tile,
    expand,
    half_padding_keeps_the_size,
    kernel_sweep,
    larger_tiles_save_more_and_hurt_more,
    matches_the_library,
    materialising_costs,
    materialising_halves_the_intensity,
    multiplies_saved,
    nothing_dominates,
    output_size_rules,
    random_inputs,
    the_expansion_is_larger_than_the_input,
    the_saving_grows_but_the_transform_does_too,
    traffic_for,
    winograd_error_against_a_plain_sum,
    winograd_matches_the_direct_form,
    winograd_tile,
)
from tgc.errors import ConfigError


class TestShape:
    def test_the_output_extent_follows_the_formula(self):
        assert ConvShape(1, 1, 1, 32, 3).output_size == 30

    def test_half_padding_keeps_the_size(self):
        assert half_padding_keeps_the_size()["all_preserved"]

    def test_a_stride_of_two_halves_it(self):
        assert ConvShape(1, 1, 1, 32, 3, stride=2, padding=1).output_size == 16

    def test_a_kernel_larger_than_its_input_is_refused(self):
        assert a_kernel_that_does_not_fit_is_refused()

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            ConvShape(0, 1, 1, 32, 3)

    def test_a_stride_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="does not move"):
            ConvShape(1, 1, 1, 32, 3, stride=0)

    def test_negative_padding_is_refused(self):
        with pytest.raises(ConfigError, match="is not padding"):
            ConvShape(1, 1, 1, 32, 3, padding=-1)

    def test_it_serialises(self):
        assert ConvShape(1, 1, 1, 32, 3).as_dict()["output_size"] == 30

    def test_four_configurations_are_tabulated(self):
        assert len(output_size_rules()) == 4


class TestExpansion:
    def test_the_expansion_and_product_match_the_library(self):
        # A wrongly indexed expansion still gives a matrix of the right shape.
        assert matches_the_library()["relative_gap"] < 1e-6

    def test_and_the_right_shape(self):
        assert matches_the_library()["shape_matches"]

    def test_the_expansion_has_one_row_per_output_position(self):
        shape = ConvShape(2, 3, 4, 8, 3, padding=1)
        source, _ = random_inputs(shape)
        assert expand(source, shape).shape[0] == shape.positions

    def test_and_one_column_per_receptive_field_element(self):
        shape = ConvShape(2, 3, 4, 8, 3, padding=1)
        source, _ = random_inputs(shape)
        assert expand(source, shape).shape[1] == shape.window

    def test_it_is_the_kernel_area_larger_than_the_input(self):
        result = the_expansion_is_larger_than_the_input()
        assert result["factor"] == result["kernel_area"]

    def test_across_every_kernel_size(self):
        assert all(row["factor"] == row["kernel_area"] for row in kernel_sweep())

    def test_a_one_by_one_convolution_expands_to_itself(self):
        assert a_one_by_one_convolution_is_already_a_product()["expansion_is_free"]

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(ConfigError, match="rank four"):
            expand(torch.randn(3, 8, 8), ConvShape(1, 3, 4, 8, 3))

    def test_an_empty_kernel_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            kernel_sweep(kernels=())

    def test_the_product_gives_the_channel_layout_back(self):
        shape = ConvShape(2, 3, 4, 8, 3, padding=1)
        source, weights = random_inputs(shape)
        result = as_matrix_product(source, weights, shape)
        assert list(result.shape) == [2, 4, 8, 8]


class TestTraffic:
    def test_writing_the_expansion_down_costs_traffic(self):
        assert materialising_costs()["ratio"] > 1.0

    def test_by_more_than_a_factor_of_six(self):
        assert materialising_costs()["ratio"] > 6.0

    def test_and_the_arithmetic_does_not_change(self):
        shape = ConvShape(1, 16, 32, 32, 3, padding=1)
        assert traffic_for(shape, materialised=True) > traffic_for(shape, materialised=False)

    def test_so_the_intensity_falls_by_the_same_factor(self):
        result = materialising_halves_the_intensity()
        assert result["ratio"] == materialising_costs()["ratio"]

    def test_the_inline_route_is_the_more_intense_one(self):
        shape = ConvShape(1, 16, 32, 32, 3, padding=1)
        assert arithmetic_intensity(shape, materialised=False) > arithmetic_intensity(
            shape, materialised=True
        )


class TestWinograd:
    def test_the_transform_computes_the_same_two_outputs(self):
        assert winograd_matches_the_direct_form()["relative_gap"] < 1e-6

    def test_but_not_bit_for_bit(self):
        assert winograd_matches_the_direct_form()["largest_gap"] > 0.0

    def test_it_is_half_again_as_inaccurate_as_the_direct_form(self):
        # Less than the method's reputation suggests, because the smallest form's transforms
        # are made of ones and halves, which are exact.
        assert 1.2 < winograd_error_against_a_plain_sum()["ratio"] < 3.0

    def test_a_tile_of_the_wrong_size_is_refused(self):
        with pytest.raises(ConfigError, match="four inputs and three taps"):
            winograd_tile(torch.randn(5), torch.randn(3))

    def test_the_direct_tile_computes_a_sliding_dot_product(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        taps = torch.tensor([1.0, 0.0, 0.0])
        assert direct_tile(values, taps).tolist() == [1.0, 2.0]

    def test_the_smallest_form_saves_two_and_a_quarter(self):
        assert multiplies_saved()["saving"] == 2.25

    def test_larger_tiles_save_more(self):
        savings = [row["saving"] for row in larger_tiles_save_more_and_hurt_more()]
        assert savings == sorted(savings)

    def test_and_need_larger_transforms(self):
        result = the_saving_grows_but_the_transform_does_too()
        assert result["transform_size_at_eight"] > result["transform_size_at_two"]

    def test_a_zero_tile_is_refused(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            multiplies_saved(kernel=3, output_tile=0)

    def test_an_empty_tile_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            larger_tiles_save_more_and_hurt_more(tiles=())

    def test_a_zero_sample_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            winograd_matches_the_direct_form(samples=0)

    def test_and_for_the_error_comparison_too(self):
        with pytest.raises(ConfigError, match="must be positive"):
            winograd_error_against_a_plain_sum(samples=0)


class TestRoutes:
    def test_three_routes_are_compared(self):
        assert len(compare_routes()) == 3

    def test_winograd_does_the_least_arithmetic(self):
        assert nothing_dominates()["fewest_multiplies"] == "winograd"

    def test_and_is_the_only_inexact_one(self):
        assert nothing_dominates()["exact_routes"] == ["direct", "expansion"]

    def test_the_expansion_moves_the_most(self):
        rows = {row["route"]: row for row in compare_routes()}
        assert rows["expansion"]["bytes"] > rows["direct"]["bytes"]

    def test_and_does_no_less_arithmetic(self):
        rows = {row["route"]: row for row in compare_routes()}
        assert rows["expansion"]["multiplies"] == rows["direct"]["multiplies"]

    def test_nothing_wins_on_every_axis(self):
        result = nothing_dominates()
        assert result["fewest_multiplies"] != result["fewest_bytes"]
