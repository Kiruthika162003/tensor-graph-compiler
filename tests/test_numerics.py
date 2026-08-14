from __future__ import annotations

import pytest
import torch

from tgc.analysis.numerics import (
    Amplification,
    PrecisionPlan,
    accumulate_in,
    accumulator_comparison,
    benign_graph,
    cancellation_factor,
    cancelling_graph,
    compare_conditioning,
    compare_narrow_types,
    compare_precision_budgets,
    dangerous_nodes,
    elementwise_ops_are_benign,
    machine_epsilon,
    measured_amplification,
    plan_precision,
    predicted_amplifications,
    significant_digits,
    variance_two_ways,
    worst_amplification,
)
from tgc.errors import ConfigError
from tgc.ir.builder import Builder, elementwise_chain, layernorm_graph, softmax_graph
from tgc.ir.dtype import BFLOAT16, FLOAT16, FLOAT32, FLOAT64, INT32
from tgc.verify.reference import random_feeds


class TestEpsilon:
    def test_a_wider_type_has_a_smaller_gap(self):
        assert machine_epsilon(FLOAT32) < machine_epsilon(FLOAT16)

    def test_bfloat16_is_the_same_width_and_eight_times_coarser(self):
        # It spends its bits on range rather than on precision, which is the trade.
        assert BFLOAT16.bits == FLOAT16.bits
        assert machine_epsilon(BFLOAT16) == 8 * machine_epsilon(FLOAT16)

    def test_double_precision_is_the_finest(self):
        assert machine_epsilon(FLOAT64) < machine_epsilon(FLOAT32)

    def test_float32_carries_about_seven_digits(self):
        assert 7.0 < significant_digits(FLOAT32) < 7.5

    def test_float16_carries_about_three(self):
        assert 3.0 < significant_digits(FLOAT16) < 3.5

    def test_an_integer_type_has_no_epsilon(self):
        with pytest.raises(ConfigError, match="no epsilon known"):
            machine_epsilon(INT32)


class TestCancellation:
    def test_two_nearby_values_cancel_badly(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0 - 1e-6])
        assert cancellation_factor(left, right) > 1e5

    def test_two_distant_values_do_not(self):
        assert cancellation_factor(torch.tensor([5.0]), torch.tensor([1.0])) < 2.0

    def test_identical_values_are_treated_as_harmless(self):
        # The difference is exactly zero, so there is no relative error to amplify.
        assert cancellation_factor(torch.tensor([1.0]), torch.tensor([1.0])) == 1.0


class TestAmplification:
    def test_a_subtraction_of_nearby_values_is_dangerous(self):
        graph, feeds = cancelling_graph()
        assert dangerous_nodes(graph, feeds)

    def test_a_subtraction_of_distant_values_is_not(self):
        graph, feeds = benign_graph()
        assert dangerous_nodes(graph, feeds) == []

    def test_a_multiplication_adds_the_relative_errors(self):
        builder = Builder()
        x = builder.input([4], name="x")
        y = builder.input([4], name="y")
        graph = builder.finish(builder.mul(x, y))
        assert worst_amplification(graph) == 2.0

    def test_a_square_root_halves_the_error(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.sqrt(x))
        assert worst_amplification(graph) == 0.5

    def test_a_rectifier_carries_the_error_unchanged(self):
        assert elementwise_ops_are_benign(elementwise_chain(6))

    def test_an_addition_of_opposite_signs_is_a_subtraction(self):
        builder = Builder()
        x = builder.input([4], name="x")
        y = builder.input([4], name="y")
        graph = builder.finish(builder.add(x, y))
        feeds = {"x": torch.full((4,), 1.0), "y": torch.full((4,), -1.0 + 1e-6)}
        assert worst_amplification(graph, feeds) > 1e5

    def test_a_negative_factor_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot amplify"):
            Amplification(node="a", factor=-1.0)

    def test_every_node_gets_a_reason(self):
        graph, feeds = cancelling_graph()
        assert all(item.reason for item in predicted_amplifications(graph, feeds))

    def test_it_serialises(self):
        assert Amplification(node="a", factor=100.0).as_dict()["dangerous"]


class TestMeasurement:
    def test_the_measurement_agrees_with_the_prediction_on_a_cancellation(self):
        # A predicted condition number nobody checks is a comment with arithmetic in it.
        graph, feeds = cancelling_graph()
        predicted = worst_amplification(graph, feeds)
        measured = measured_amplification(graph, feeds)
        assert 0.5 < measured / predicted < 2.0

    def test_and_on_a_benign_graph(self):
        graph, feeds = benign_graph()
        predicted = worst_amplification(graph, feeds)
        measured = measured_amplification(graph, feeds)
        assert 0.5 < measured / predicted < 2.0

    def test_the_cancelling_graph_amplifies_by_millions(self):
        rows = {row["graph"]: row for row in compare_conditioning()}
        assert rows["cancelling"]["measured"] > 1e6

    def test_the_benign_one_amplifies_by_about_one(self):
        rows = {row["graph"]: row for row in compare_conditioning()}
        assert rows["benign"]["measured"] < 2.0

    def test_a_zero_perturbation_is_rejected(self):
        graph, feeds = benign_graph()
        with pytest.raises(ConfigError, match="has to be positive"):
            measured_amplification(graph, feeds, relative=0.0)

    def test_a_zero_gap_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            cancelling_graph(gap=0.0)


class TestAccumulation:
    def test_a_narrow_accumulator_stops_counting(self):
        # The running total grows until each new addend falls below its last bit.
        rows = {row["dtype"]: row for row in accumulator_comparison()}
        assert rows["float16"]["sum"] < 4096

    def test_bfloat16_stops_even_sooner(self):
        rows = {row["dtype"]: row for row in accumulator_comparison()}
        assert rows["bfloat16"]["sum"] < rows["float16"]["sum"]

    def test_float32_counts_all_the_way(self):
        rows = {row["dtype"]: row for row in accumulator_comparison()}
        assert rows["float32"]["relative_error"] == 0.0

    def test_the_half_precision_error_is_enormous(self):
        rows = {row["dtype"]: row for row in accumulator_comparison()}
        assert rows["float16"]["relative_error"] > 0.4

    def test_a_short_sum_is_fine_in_any_type(self):
        assert accumulate_in(torch.full((16,), 1.0), FLOAT16) == 16.0

    def test_a_zero_length_run_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            accumulator_comparison(count=0)


class TestVariance:
    def test_the_cancelling_formula_loses_everything(self):
        # The mean of squares minus the square of the mean, which keeps happening because it
        # is the formula people remember.
        result = variance_two_ways()
        assert result["naive_error"] > 1.0

    def test_it_can_even_return_a_negative_variance(self):
        assert variance_two_ways()["naive_is_negative"]

    def test_centring_first_is_exact_to_seven_digits(self):
        assert variance_two_ways()["stable_error"] < 1e-6

    def test_a_smaller_scale_hides_the_problem(self):
        assert variance_two_ways(scale=1.0)["naive_error"] < 0.1

    def test_a_single_value_has_no_variance(self):
        with pytest.raises(ConfigError, match="at least two values"):
            variance_two_ways(count=1)


class TestPrecisionPlanning:
    def test_a_generous_budget_narrows_most_of_a_layernorm(self):
        plan = plan_precision(layernorm_graph(), budget=1e-2)
        assert plan.share > 0.5

    def test_a_tight_budget_narrows_nothing(self):
        plan = plan_precision(layernorm_graph(), budget=1e-6)
        assert plan.count == 0

    def test_the_refusals_say_what_it_would_have_cost(self):
        plan = plan_precision(layernorm_graph(), budget=1e-6)
        assert any("would contribute" in reason for reason in plan.kept_wide.values())

    def test_bfloat16_narrows_less_than_float16_at_the_same_budget(self):
        # Same width, eight times the rounding, and that is the trade the choice makes.
        rows = {row["type"]: row for row in compare_narrow_types()}
        assert rows["bfloat16"]["narrowed"] < rows["float16"]["narrowed"]

    def test_tightening_the_budget_never_narrows_more(self):
        rows = compare_precision_budgets()
        counts = [row["narrowed"] for row in rows]
        assert counts == sorted(counts, reverse=True)

    def test_a_zero_budget_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            plan_precision(softmax_graph(), budget=0.0)

    def test_an_empty_budget_list_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            compare_precision_budgets(budgets=())

    def test_an_empty_plan_narrows_nothing(self):
        assert PrecisionPlan().share == 0.0

    def test_it_serialises(self):
        assert plan_precision(layernorm_graph(), budget=1e-2).as_dict()["narrowed"] > 0

    def test_a_dangerous_node_is_kept_wide(self):
        graph, feeds = cancelling_graph()
        plan = plan_precision(graph, feeds, budget=1e-3)
        assert plan.count == 0
        assert len(plan.kept_wide) == 1


class TestGraphs:
    def test_a_softmax_has_a_cancellation_by_construction(self):
        # Subtracting the maximum is the stable form, and the subtraction it introduces is
        # exactly a cancellation on the largest element of each row.
        graph = softmax_graph()
        feeds = random_feeds(graph, positive=True)
        assert worst_amplification(graph, feeds) > 1.0

    def test_a_chain_of_rectifiers_has_none(self):
        builder = Builder()
        current = builder.input([8, 8], name="x")
        for _ in range(6):
            current = builder.relu(builder.tanh(current))
        assert worst_amplification(builder.finish(current)) <= 1.0

    def test_a_chain_of_exponentials_is_badly_conditioned(self):
        # exp multiplies a relative error by the magnitude of its exponent, so stacking them
        # runs the amplification away long before the values themselves overflow.
        assert worst_amplification(elementwise_chain(6)) > 100.0

    def test_the_default_type_is_the_wide_one(self):
        assert softmax_graph().value("x").dtype is FLOAT32
