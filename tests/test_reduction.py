from __future__ import annotations

import math

import pytest
import torch

from tgc.errors import ConfigError, PassError
from tgc.ir.dtype import FLOAT16, FLOAT32
from tgc.passes.reduction import (
    ReductionReport,
    SplitPlan,
    accuracy_comparison,
    best_part_count,
    compare_part_counts,
    compare_widths,
    exact_sum,
    find_reductions,
    graph_sum_matches_torch,
    long_reduction_graph,
    measure_graph_candidates,
    serial_sum,
    short_reduction_graph,
    split_sum,
    sweep_parts,
)


class TestPlan:
    def test_one_part_is_the_serial_version(self):
        assert SplitPlan(length=4096, parts=1).serial_depth == 4096

    def test_splitting_shortens_the_dependence_chain(self):
        assert SplitPlan(length=4096, parts=16).serial_depth < 4096

    def test_the_depth_counts_the_final_combination(self):
        # The partials run at the same time; the combination does not.
        plan = SplitPlan(length=4096, parts=16)
        assert plan.serial_depth == plan.per_part + plan.parts - 1

    def test_the_best_split_is_the_square_root(self):
        # The same balance as checkpointing, and for the same reason: one term falls as the
        # other rises and the sum is smallest where they meet.
        assert best_part_count(4096) == 64
        assert best_part_count(4096) == int(math.sqrt(4096))

    def test_the_speedup_grows_with_the_parts_up_to_that_point(self):
        rows = sweep_parts(4096)
        speedups = [row["speedup"] for row in rows if row["parts"] <= 64]
        assert speedups == sorted(speedups)

    def test_a_part_count_above_the_length_is_rejected(self):
        with pytest.raises(ConfigError, match="between one and the length"):
            SplitPlan(length=8, parts=9)

    def test_an_empty_reduction_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one term"):
            SplitPlan(length=0, parts=1)

    def test_a_zero_length_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one term"):
            sweep_parts(0)

    def test_it_serialises(self):
        assert SplitPlan(length=4096, parts=16).as_dict()["per_part"] == 256


class TestAccuracy:
    def test_splitting_is_more_accurate_than_the_serial_sum(self):
        # Which is the opposite of how regrouping is usually described. A serial sum lets the
        # running total grow until each new term falls below its last bit.
        assert accuracy_comparison()["split_is_closer"]

    def test_both_are_judged_against_a_double_precision_answer(self):
        # The serial version is not the reference, it is one of the two candidates.
        result = accuracy_comparison()
        assert result["serial_error"] > 0
        assert result["split_error"] > 0

    def test_the_gain_is_large_in_single_precision(self):
        result = accuracy_comparison()
        assert result["serial_error"] > 5 * result["split_error"]

    def test_and_enormous_in_half_precision(self):
        # The serial version stops moving entirely and the split one is still counting.
        rows = {row["dtype"]: row for row in compare_widths(length=20_000)}
        assert rows["float16"]["serial_error"] > 0.5
        assert rows["float16"]["split_error"] < 0.01

    def test_single_precision_is_far_better_than_half_either_way(self):
        rows = {row["dtype"]: row for row in compare_widths(length=20_000)}
        assert rows["float32"]["serial_error"] < rows["float16"]["serial_error"]

    def test_more_parts_helps_until_it_stops(self):
        rows = compare_part_counts(length=20_000)
        errors = [row["error"] for row in rows]
        assert errors[0] > errors[-1]
        assert errors[-1] == errors[-2]

    def test_the_depth_keeps_falling_after_the_accuracy_flattens(self):
        rows = compare_part_counts(length=20_000)
        depths = [row["serial_depth"] for row in rows]
        assert depths == sorted(depths, reverse=True)

    def test_an_empty_comparison_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            compare_part_counts(counts=())


class TestSummation:
    def test_a_serial_sum_of_ones_counts_them(self):
        assert serial_sum(torch.ones(64)) == 64.0

    def test_a_split_sum_agrees_on_an_easy_case(self):
        values = torch.ones(64)
        assert split_sum(values, parts=8) == serial_sum(values)

    def test_the_exact_sum_is_the_double_precision_one(self):
        values = torch.rand(1000)
        assert exact_sum(values) == float(values.double().sum())

    def test_splitting_into_one_part_is_the_serial_sum(self):
        generator = torch.Generator().manual_seed(0)
        values = torch.rand(500, generator=generator)
        assert split_sum(values, parts=1) == serial_sum(values)

    def test_splitting_into_more_parts_than_terms_is_refused(self):
        with pytest.raises(PassError, match="cannot split"):
            split_sum(torch.ones(4), parts=8)

    def test_a_zero_part_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            split_sum(torch.ones(8), parts=0)

    def test_a_narrow_accumulator_stalls_on_a_serial_sum(self):
        assert serial_sum(torch.ones(20_000), FLOAT16) < 20_000

    def test_and_a_split_one_gets_much_further(self):
        assert split_sum(torch.ones(20_000), parts=64, dtype=FLOAT16) > serial_sum(
            torch.ones(20_000), FLOAT16
        )


class TestGraphs:
    def test_a_long_reduction_is_a_candidate(self):
        assert find_reductions(long_reduction_graph()).count == 1

    def test_a_short_one_is_not(self):
        # Splitting a sum over eight terms adds a node to save a step.
        assert find_reductions(short_reduction_graph()).count == 0

    def test_the_length_is_recorded(self):
        report = find_reductions(long_reduction_graph(4096))
        assert next(iter(report.lengths.values())) == 4096

    def test_the_threshold_decides(self):
        assert find_reductions(short_reduction_graph(8), minimum=4).count == 1

    def test_a_zero_threshold_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            find_reductions(long_reduction_graph(), minimum=0)

    def test_both_fixtures_are_compared(self):
        rows = {row["graph"]: row for row in measure_graph_candidates()}
        assert rows["long"]["candidates"] == 1
        assert rows["short"]["candidates"] == 0

    def test_a_degenerate_axis_is_rejected(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            long_reduction_graph(1)

    def test_the_interpreter_agrees_with_the_library(self):
        assert graph_sum_matches_torch()

    def test_an_empty_report_finds_nothing(self):
        assert ReductionReport().count == 0

    def test_it_serialises(self):
        assert find_reductions(long_reduction_graph()).as_dict()["candidates"] == 1

    def test_the_default_accumulator_is_the_wide_one(self):
        graph = long_reduction_graph()
        assert graph.value(graph.outputs[0]).dtype is FLOAT32
