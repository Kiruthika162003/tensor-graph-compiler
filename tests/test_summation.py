from __future__ import annotations

import numpy as np
import pytest

from tgc.analysis.summation import (
    ORDERS,
    a_double_input_is_refused,
    a_split_changes_the_answer,
    a_split_wider_than_the_input_is_refused,
    alternating,
    an_empty_reduction_is_refused,
    an_unknown_order_is_refused,
    compare_inputs,
    compare_orders,
    compensated,
    compensation_is_flat,
    double_precision_removes_the_problem,
    error_grows_with_length,
    error_of,
    exact,
    flops_per_element,
    mixed_magnitudes,
    pairwise,
    partition_sweep,
    partitioned,
    relative_error,
    reordering_usually_improves_it,
    reversing_the_input_changes_the_total,
    sequential,
    sorted_ascending,
    sorting_helps_and_costs,
    the_best_split_is_the_square_root,
    the_sequential_order_stops_adding,
    the_tree_beats_the_loop,
    uniform,
    uniform_data_cares_too,
)
from tgc.errors import ConfigError


class TestKernels:
    def test_every_order_gets_the_same_answer_on_a_short_input(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert {kernel(values) for kernel in ORDERS.values()} == {10.0}

    def test_the_reference_is_the_double_precision_sum(self):
        # The float32 values widened, summed exactly, and rounded once. Not the double sum of
        # the decimal literals, which is a different number.
        values = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        assert abs(exact(values) - 0.6000000163912773) < 1e-15

    def test_one_partition_is_the_sequential_order(self):
        values = uniform(count=256)
        assert partitioned(values, 1) == sequential(values)

    def test_a_block_larger_than_the_input_makes_the_tree_sequential(self):
        values = uniform(count=64)
        assert pairwise(values, block=1024) == sequential(values)

    def test_the_compensated_loop_is_at_least_as_good(self):
        values = mixed_magnitudes(count=1024)
        reference = exact(values)
        assert relative_error(compensated(values), reference) <= relative_error(
            sequential(values), reference
        )

    def test_sorting_a_positive_input_is_a_permutation(self):
        values = uniform(count=128)
        assert abs(sorted_ascending(values) - exact(values)) < 1e-2

    def test_an_empty_reduction_is_refused(self):
        assert an_empty_reduction_is_refused()

    def test_a_double_input_is_refused(self):
        assert a_double_input_is_refused()

    def test_a_matrix_is_refused(self):
        with pytest.raises(ConfigError, match="runs over a vector"):
            sequential(np.zeros((2, 2), dtype=np.float32))

    def test_a_zero_block_is_refused(self):
        with pytest.raises(ConfigError, match="not a block"):
            pairwise(uniform(count=16), block=0)

    def test_a_zero_partition_count_is_refused(self):
        with pytest.raises(ConfigError, match="not a split"):
            partitioned(uniform(count=16), 0)

    def test_a_split_wider_than_the_input_is_refused(self):
        assert a_split_wider_than_the_input_is_refused()

    def test_a_relative_error_against_zero_is_refused(self):
        with pytest.raises(ConfigError, match="not a number"):
            relative_error(1.0, 0.0)

    def test_an_unknown_order_is_refused(self):
        assert an_unknown_order_is_refused()

    def test_and_when_asked_for_its_cost(self):
        with pytest.raises(ConfigError, match="unknown order"):
            flops_per_element("magic")

    def test_a_one_value_reduction_is_refused_as_a_fixture(self):
        with pytest.raises(ConfigError, match="not a reduction"):
            mixed_magnitudes(count=1)

    def test_a_zero_leader_is_refused(self):
        with pytest.raises(ConfigError, match="not a leader"):
            mixed_magnitudes(leader=0.0)


class TestAbsorption:
    def test_the_loop_absorbs_everything_after_the_leader(self):
        result = the_sequential_order_stops_adding()
        assert result["absorbed"] == result["values"] - 1

    def test_so_the_answer_is_the_leader(self):
        assert sequential(mixed_magnitudes(count=4096)) == 1e9

    def test_and_the_error_is_the_whole_tail(self):
        assert the_sequential_order_stops_adding()["error"] > 4e-6

    def test_the_tree_does_not_lose_the_tail(self):
        values = mixed_magnitudes(count=4096)
        assert pairwise(values) > 1e9


class TestLength:
    def test_the_loop_degrades_in_proportion_to_the_length(self):
        rows = {row["count"]: row for row in error_grows_with_length()}
        assert rows[4096]["sequential"] > 3.5 * rows[1024]["sequential"]

    def test_and_the_tree_does_not_degrade_at_all(self):
        rows = {row["count"]: row for row in error_grows_with_length()}
        assert rows[65536]["pairwise"] < rows[1024]["pairwise"]

    def test_the_tree_beats_the_loop_by_three_orders(self):
        assert the_tree_beats_the_loop()["ratio"] > 1000

    def test_for_the_same_number_of_additions(self):
        assert the_tree_beats_the_loop()["same_flops"]

    def test_the_compensated_loop_stays_flat(self):
        assert not compensation_is_flat()["grew"]

    def test_within_a_few_parts_in_a_hundred_million(self):
        assert compensation_is_flat()["spread"] < 1e-7

    def test_an_empty_length_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            error_grows_with_length(counts=())

    def test_compensation_costs_four_operations_per_element(self):
        assert flops_per_element("compensated") == 4

    def test_where_the_tree_costs_one(self):
        assert flops_per_element("pairwise") == 1


class TestSorting:
    def test_the_sort_beats_the_loop(self):
        assert sorting_helps_and_costs()["beats_the_loop"]

    def test_and_lands_exactly_where_the_tree_lands(self):
        assert sorting_helps_and_costs()["matches_the_tree"]

    def test_but_it_is_the_worst_order_on_signed_data(self):
        rows = {row["input"]: row for row in compare_inputs()}
        row = rows["alternating"]
        assert row["sorted"] == max(row[name] for name in ORDERS)

    def test_by_three_orders_of_magnitude(self):
        rows = {row["input"]: row for row in compare_inputs()}
        row = rows["alternating"]
        assert row["sorted"] > 1000 * row["pairwise"]

    def test_the_tree_is_never_the_worst_on_any_input(self):
        for row in compare_inputs():
            assert row["pairwise"] < max(row[name] for name in ORDERS)

    def test_three_inputs_are_compared(self):
        assert len(compare_inputs()) == 3


class TestPartitions:
    def test_a_split_changes_the_answer(self):
        assert a_split_changes_the_answer()["all_different"]

    def test_by_thousands_on_a_total_of_a_billion(self):
        assert a_split_changes_the_answer()["spread"] > 1000

    def test_a_comparison_needs_two_counts(self):
        with pytest.raises(ConfigError, match="at least two"):
            a_split_changes_the_answer(counts=(4,))

    def test_the_error_bottoms_out_at_the_square_root(self):
        rows = partition_sweep()
        best = min(rows, key=lambda row: row["error"])
        assert best["partitions"] == 64

    def test_which_is_where_the_partition_length_matches_the_count(self):
        rows = {row["partitions"]: row for row in partition_sweep()}
        assert rows[64]["length"] == 64

    def test_and_the_two_ends_of_the_sweep_agree(self):
        # One partition is the sequential order, and one value per partition is the sequential
        # order again in the second stage.
        assert the_best_split_is_the_square_root()["the_ends_agree"]

    def test_the_bottom_is_eighty_times_better_than_either_end(self):
        assert the_best_split_is_the_square_root()["ratio"] > 80

    def test_so_more_partitions_is_not_monotonically_better(self):
        rows = {row["partitions"]: row for row in partition_sweep()}
        assert rows[256]["error"] > rows[64]["error"]

    def test_an_empty_partition_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            partition_sweep(counts=())


class TestAssociativity:
    def test_reversing_the_input_changes_the_total(self):
        assert not reversing_the_input_changes_the_total()["identical"]

    def test_by_the_whole_tail(self):
        assert reversing_the_input_changes_the_total()["gap"] > 4000

    def test_and_the_reversed_order_is_the_better_one(self):
        assert reordering_usually_improves_it()["backward_is_better"]

    def test_by_two_orders_of_magnitude(self):
        assert reordering_usually_improves_it()["ratio"] > 100

    def test_which_makes_the_rewrite_an_improvement_and_not_an_identity(self):
        result = reversing_the_input_changes_the_total()
        assert result["forward"] != result["backward"]


class TestPrecision:
    def test_double_precision_is_nine_orders_better(self):
        assert double_precision_removes_the_problem()["ratio"] > 1e8

    def test_and_beats_every_narrow_order(self):
        assert double_precision_removes_the_problem()["beats_every_narrow_order"]

    def test_at_twice_the_bytes(self):
        assert double_precision_removes_the_problem()["bytes_per_element"] == 8

    def test_uniform_data_is_not_exempt(self):
        assert uniform_data_cares_too()["ratio"] > 50

    def test_even_with_no_leader_to_swallow_anything(self):
        values = uniform(count=16384)
        assert float(values.max()) < 2.0

    def test_alternating_data_is_generated_with_both_signs(self):
        values = alternating(count=1024)
        assert float(values.min()) < 0 < float(values.max())

    def test_four_orders_are_reported(self):
        assert len(compare_orders()) == len(ORDERS)

    def test_the_loop_is_the_worst_of_them(self):
        rows = compare_orders()
        assert max(rows, key=lambda row: row["error"])["order"] == "sequential"

    def test_every_error_is_a_fraction(self):
        assert all(0 <= error_of(name, uniform(count=512)) < 1 for name in ORDERS)
