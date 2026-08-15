from __future__ import annotations

import pytest
import torch

from tgc.analysis.quantize import (
    BIT_WIDTHS,
    ErrorReport,
    Scheme,
    asymmetric_scheme,
    best_quantile,
    bit_width_sweep,
    bits_needed_for,
    clipped_scheme,
    clipping_helps_the_mean_and_hurts_the_worst,
    contraction_length_sweep,
    dequantise,
    distribution_changes_the_answer,
    each_bit_roughly_doubles_the_error,
    error_by_distribution,
    even_matrix,
    measure_error,
    memory_saved,
    peaked_values,
    per_row_is_worth_it,
    per_row_round_trip,
    per_row_schemes,
    quantile_sweep,
    quantise,
    relative_error_survives_a_contraction,
    relative_rms,
    round_trip,
    saving_against_error,
    symmetric_scheme,
    the_mean_error_is_not_monotonic,
    the_worst_case_does_grow,
    uneven_matrix,
    unevenness_decides_the_gain,
    uniform_values,
    weight_error_against_output_error,
)
from tgc.errors import ConfigError


class TestScheme:
    def test_eight_bits_gives_two_hundred_and_fifty_six_levels(self):
        assert Scheme(bits=8, scale=1.0).levels == 256

    def test_a_symmetric_scheme_loses_one_level_to_the_sign(self):
        # Which is why removing a bit does not exactly double the step.
        scheme = Scheme(bits=8, scale=1.0)
        assert scheme.high == 127
        assert scheme.low == -128

    def test_an_asymmetric_one_keeps_them_all(self):
        assert Scheme(bits=8, scale=1.0, symmetric=False).high == 255

    def test_the_representable_range_follows_the_scale(self):
        low, high = Scheme(bits=4, scale=0.5).representable_range
        assert (low, high) == (-4.0, 3.5)

    def test_one_bit_is_refused(self):
        with pytest.raises(ConfigError, match="at least two bits"):
            Scheme(bits=1, scale=1.0)

    def test_a_zero_scale_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            Scheme(bits=8, scale=0.0)

    def test_it_serialises(self):
        assert Scheme(bits=4, scale=1.0).as_dict()["levels"] == 16


class TestFitting:
    def test_a_symmetric_scheme_is_sized_by_the_largest_magnitude(self):
        values = torch.tensor([-2.0, 1.0])
        assert symmetric_scheme(values, 8).scale == pytest.approx(2.0 / 127)

    def test_an_asymmetric_one_is_sized_by_the_span(self):
        values = torch.tensor([1.0, 5.0])
        assert asymmetric_scheme(values, 8).zero_point == 1.0

    def test_a_tensor_of_zeros_gets_a_usable_scheme(self):
        assert symmetric_scheme(torch.zeros(8), 8).scale == 1.0

    def test_a_constant_tensor_gets_one_too(self):
        assert asymmetric_scheme(torch.full((8,), 3.0), 8).zero_point == 3.0

    def test_an_empty_tensor_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to quantise"):
            symmetric_scheme(torch.tensor([]))


class TestRoundTrip:
    def test_the_integers_stay_inside_the_scheme(self):
        values = torch.randn(64)
        scheme = symmetric_scheme(values, 4)
        integers = quantise(values, scheme)
        assert int(integers.min()) >= scheme.low
        assert int(integers.max()) <= scheme.high

    def test_dequantising_undoes_the_scale(self):
        scheme = Scheme(bits=8, scale=0.5)
        assert float(dequantise(torch.tensor([4.0]), scheme)) == 2.0

    def test_a_round_trip_stays_within_half_a_step(self):
        values = torch.randn(256)
        scheme = symmetric_scheme(values, 8)
        assert float((round_trip(values, scheme) - values).abs().max()) <= scheme.scale

    def test_more_bits_means_less_error(self):
        values = torch.randn(256)
        wide = measure_error(values, symmetric_scheme(values, 8))
        narrow = measure_error(values, symmetric_scheme(values, 4))
        assert wide.largest < narrow.largest

    def test_nothing_clips_when_the_scheme_covers_the_range(self):
        values = torch.randn(256)
        assert measure_error(values, symmetric_scheme(values, 8)).clipped == 0

    def test_an_empty_report_reads_as_zero(self):
        assert ErrorReport().as_dict()["largest"] == 0.0

    def test_the_root_mean_square_measure_is_against_the_data(self):
        values = torch.full((16,), 4.0)
        assert relative_rms(torch.full((16,), 2.0), values) == 0.5


class TestBitWidths:
    def test_each_bit_removed_roughly_doubles_the_error(self):
        result = each_bit_roughly_doubles_the_error(even_matrix().flatten())
        assert result["smallest_ratio"] > 2.0

    def test_but_never_exactly(self):
        # A symmetric scheme spends one level on the sign, so the step ratio between four bits
        # and three is seven over three rather than two.
        result = each_bit_roughly_doubles_the_error(even_matrix().flatten())
        assert result["smallest_ratio"] != 2.0

    def test_every_width_is_swept(self):
        assert len(bit_width_sweep(torch.randn(256))) == len(BIT_WIDTHS)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            bit_width_sweep(torch.randn(16), widths=())


class TestPerRow:
    def test_by_worst_case_a_scale_per_row_buys_nothing(self):
        # The worst error lives in the row that set the shared scale, so giving that row its
        # own scale hands it back the same number.
        assert per_row_is_worth_it(uneven_matrix())["worst_improvement"] == 1.0

    def test_by_mean_error_it_buys_a_great_deal(self):
        assert per_row_is_worth_it(uneven_matrix())["mean_improvement"] > 4.0

    def test_and_much_less_on_an_even_matrix(self):
        rows = {row["matrix"]: row for row in unevenness_decides_the_gain()}
        assert rows["even"]["mean_improvement"] < rows["uneven"]["mean_improvement"]

    def test_one_scheme_per_row(self):
        assert len(per_row_schemes(even_matrix(8, 8))) == 8

    def test_the_shape_survives(self):
        assert list(per_row_round_trip(even_matrix(8, 8)).shape) == [8, 8]

    def test_a_vector_has_no_rows_to_scale(self):
        with pytest.raises(ConfigError, match="need a matrix"):
            per_row_schemes(torch.randn(16))

    def test_an_even_factor_is_refused(self):
        with pytest.raises(ConfigError, match="make one row larger"):
            uneven_matrix(factor=1.0)


class TestClipping:
    def test_clipping_lowers_the_mean_error(self):
        result = clipping_helps_the_mean_and_hurts_the_worst(uneven_matrix().flatten())
        assert result["mean_improved"]

    def test_and_raises_the_worst(self):
        result = clipping_helps_the_mean_and_hurts_the_worst(uneven_matrix().flatten())
        assert result["worst_got_worse"]

    def test_the_mean_error_is_not_monotonic_in_the_clipping_point(self):
        # Which rules out picking one by walking downhill from either end.
        assert not the_mean_error_is_not_monotonic(uneven_matrix().flatten())["monotonic"]

    def test_the_best_clipping_point_is_not_the_full_range(self):
        assert best_quantile(uneven_matrix().flatten()) < 1.0

    def test_a_tighter_scheme_clips_more(self):
        values = uneven_matrix().flatten()
        tight = measure_error(values, clipped_scheme(values, 4, 0.99)).clipped
        full = measure_error(values, clipped_scheme(values, 4, 1.0)).clipped
        assert tight > full

    def test_a_quantile_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ConfigError, match="has to be in"):
            clipped_scheme(torch.randn(64), 8, 1.5)

    def test_an_empty_quantile_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            quantile_sweep(torch.randn(64), quantiles=())


class TestContraction:
    def test_the_root_mean_square_error_survives_a_contraction(self):
        # The signal and the noise are sums of the same number of independent terms, so both
        # grow at the same rate and the ratio between them does not move.
        assert relative_error_survives_a_contraction()["all_near_one"]

    def test_a_contraction_of_one_is_not_a_sum_and_changes_nothing(self):
        rows = contraction_length_sweep()
        assert rows[0]["rms_ratio"] == 1.0
        assert rows[0]["worst_ratio"] == 1.0

    def test_the_worst_case_does_grow_with_the_contraction(self):
        result = the_worst_case_does_grow()
        assert result["grew"]
        assert result["longest"] > 3.0

    def test_the_two_measures_disagree_at_four_thousand_terms(self):
        row = contraction_length_sweep()[-1]
        assert row["worst_ratio"] > 2 * row["rms_ratio"]

    def test_a_zero_shape_is_refused(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            weight_error_against_output_error(inner=0)

    def test_an_empty_length_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            contraction_length_sweep(lengths=())


class TestDistribution:
    def test_a_peaked_tensor_needs_more_bits_than_a_flat_one(self):
        # Same extremes, different shape. A symmetric scheme spends its levels evenly across a
        # range where the data is not.
        rows = {row["shape"]: row for row in distribution_changes_the_answer()}
        assert rows["peaked"]["bits_needed"] > rows["uniform"]["bits_needed"]

    def test_even_though_their_ranges_match(self):
        rows = {row["shape"]: row for row in distribution_changes_the_answer()}
        assert abs(rows["peaked"]["largest"] - rows["uniform"]["largest"]) < 0.01

    def test_the_peaked_one_is_worse_at_every_width(self):
        assert all(row["peaked"] > row["uniform"] for row in error_by_distribution())

    def test_a_flat_tensor_needs_five_bits_for_four_percent(self):
        assert bits_needed_for(uniform_values(), 0.04) == 5

    def test_a_tolerance_no_width_reaches_returns_the_ceiling(self):
        assert bits_needed_for(peaked_values(), 1e-12) == 17

    def test_a_negative_tolerance_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            bits_needed_for(torch.randn(64), -1.0)

    def test_an_empty_distribution_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            error_by_distribution(widths=())


class TestSaving:
    def test_four_bits_is_an_eighth_of_a_float(self):
        assert memory_saved(1024, 4)["ratio"] == 8.0

    def test_eight_bits_is_a_quarter(self):
        assert memory_saved(1024, 8)["ratio"] == 4.0

    def test_an_empty_tensor_saves_nothing(self):
        assert memory_saved(0, 8)["quantised_bytes"] == 0

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ConfigError, match="cannot have"):
            memory_saved(-1, 8)

    def test_one_bit_is_refused(self):
        with pytest.raises(ConfigError, match="at least two bits"):
            memory_saved(1024, 1)

    def test_the_saving_and_the_error_move_together(self):
        rows = saving_against_error()
        assert rows[0]["ratio"] < rows[-1]["ratio"]
        assert rows[0]["relative_error"] < rows[-1]["relative_error"]

    def test_an_empty_trade_table_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            saving_against_error(widths=())
