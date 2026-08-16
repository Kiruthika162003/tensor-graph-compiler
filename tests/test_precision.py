from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.dtype import BFLOAT16, FLOAT16, FLOAT32
from tgc.passes.precision import (
    POLICIES,
    PrecisionPlan,
    apply_plan,
    brain_float_does_not_overflow,
    bytes_of_intermediates,
    bytes_without_the_casts,
    cast_count,
    casts_inserted,
    compare_graphs,
    compare_policies,
    compile_with,
    every_fixture_survives_the_mixed_policy,
    half_precision_breaks_an_unshifted_softmax,
    measure_policy,
    memory_saved,
    mixing_costs_the_most_conversions,
    plan_for,
    products_only_is_the_useful_policy,
    reductions_stay_wide_whatever_the_policy,
    the_casts_eat_the_saving,
    the_range_and_the_mantissa_trade,
    what_summing_in_half_precision_would_do,
    where_float16_runs_out,
    where_the_accumulator_stalls,
)


class TestPlans:
    def test_the_wide_policy_leaves_everything_alone(self):
        plan = plan_for(mlp_graph(), "everything wide")
        assert plan.narrow_nodes == 0

    def test_the_narrow_policy_takes_everything(self):
        graph = mlp_graph()
        plan = plan_for(graph, "everything narrow")
        assert plan.narrow_nodes == len(graph.nodes)

    def test_the_mixed_policy_takes_only_the_products(self):
        graph = mlp_graph()
        plan = plan_for(graph, "products only")
        products = sum(1 for node in graph.nodes if node.op is ops.MATMUL)
        assert plan.narrow_nodes == products

    def test_the_brain_float_policy_uses_the_other_narrow_type(self):
        assert plan_for(mlp_graph(), "brain float").narrow is BFLOAT16

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            plan_for(mlp_graph(), "guesswork")

    def test_four_policies_are_available(self):
        assert len(POLICIES) == 4

    def test_a_plan_serialises(self):
        assert plan_for(mlp_graph(), "brain float").as_dict()["narrow"] == "bfloat16"

    def test_an_empty_plan_narrows_nothing(self):
        assert PrecisionPlan(policy="everything wide").narrow_nodes == 0


class TestRewriting:
    def test_the_wide_rewrite_inserts_no_casts(self):
        assert cast_count(compile_with(mlp_graph(), "everything wide")) == 0

    def test_a_narrow_rewrite_does(self):
        assert cast_count(compile_with(mlp_graph(), "everything narrow")) > 0

    def test_the_output_comes_back_wide_whatever_the_policy(self):
        for policy in POLICIES:
            graph = compile_with(mlp_graph(), policy)
            assert graph.value(graph.outputs[0]).dtype is FLOAT32

    def test_mixing_costs_the_most_conversions(self):
        # Every boundary between a narrow node and a wide one is a conversion, and narrowing one
        # op creates two boundaries around it.
        assert mixing_costs_the_most_conversions()["mixed_is_the_most"]

    def test_a_symbolic_shape_cannot_be_sized(self):
        builder = Builder()
        x = builder.input(["n", 8], name="x")
        graph = builder.finish(builder.relu(x))
        with pytest.raises(Exception, match="symbolic shape"):
            bytes_of_intermediates(graph)

    def test_every_policy_is_measured(self):
        assert len(compare_policies()) == len(POLICIES)

    def test_a_plan_can_be_applied_directly(self):
        graph = mlp_graph()
        rewritten = apply_plan(graph, plan_for(graph, "products only"))
        assert cast_count(rewritten) > 0

    def test_casts_are_reported_per_policy(self):
        assert len(casts_inserted()) == len(POLICIES)


class TestRange:
    def test_the_exponential_of_twelve_does_not_fit_in_half_precision(self):
        result = half_precision_breaks_an_unshifted_softmax()
        assert not result["narrow_is_finite"]

    def test_but_fits_comfortably_in_single(self):
        assert half_precision_breaks_an_unshifted_softmax()["wide_is_finite"]

    def test_and_the_shift_a_softmax_does_makes_it_fit(self):
        # Which is why the aggressive policy survives on the softmax fixture.
        assert half_precision_breaks_an_unshifted_softmax()["shifted_is_finite"]

    def test_the_boundary_is_between_eleven_and_twelve(self):
        rows = {row["input"]: row for row in where_float16_runs_out()}
        assert rows[11.0]["finite"]
        assert not rows[12.0]["finite"]

    def test_brain_float_never_overflows_there(self):
        assert all(row["brain_float_finite"] for row in brain_float_does_not_overflow())

    def test_while_float16_does(self):
        rows = brain_float_does_not_overflow()
        assert not all(row["float16_finite"] for row in rows)

    def test_but_brain_float_is_eight_times_less_accurate(self):
        assert the_range_and_the_mantissa_trade()["ratio"] > 6.0

    def test_an_empty_range_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            where_float16_runs_out(values=())

    def test_an_empty_brain_float_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            brain_float_does_not_overflow(values=())


class TestAccumulators:
    def test_a_sum_stays_wide_whatever_the_policy_asks(self):
        assert reductions_stay_wide_whatever_the_policy()["wide_outputs"] >= 1

    def test_a_maximum_does_not_need_to(self):
        # It has no accumulator and never grows past the largest thing it read.
        result = reductions_stay_wide_whatever_the_policy()
        assert result["wide_outputs"] < result["reductions"]

    def test_a_sequential_half_precision_total_stalls(self):
        assert what_summing_in_half_precision_would_do()["sequential_in_half_precision"] < 4096

    def test_at_two_thousand_and_forty_eight(self):
        assert what_summing_in_half_precision_would_do()["stalled_at"] == 2048

    def test_a_blocked_reduction_does_not(self):
        # Torch blocks its reduction, and the blocking is itself a fix for this.
        result = what_summing_in_half_precision_would_do()
        assert result["blocked_in_half_precision"] == result["expected"]

    def test_the_stall_is_where_the_mantissa_runs_out(self):
        result = where_the_accumulator_stalls()
        assert result["two_thousand_and_forty_eight_plus_one"] == 2048.0
        assert result["one_thousand_and_twenty_four_plus_one"] == 1025.0

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            what_summing_in_half_precision_would_do(count=0)


class TestMemory:
    def test_counting_the_casts_the_narrow_policies_cost_more(self):
        # Every cast materialises a copy of what it converted.
        assert memory_saved()["products only"]["with_casts"] < 1.0

    def test_folding_them_in_the_saving_appears(self):
        assert memory_saved()["products only"]["without_casts"] > 1.0

    def test_which_is_a_backend_decision_rather_than_a_policy_one(self):
        assert the_casts_eat_the_saving()["folding_is_what_saves"]

    def test_the_aggressive_policy_halves_the_memory_when_folded(self):
        assert memory_saved()["everything narrow"]["without_casts"] == 2.0

    def test_the_wide_policy_saves_nothing_either_way(self):
        rows = memory_saved()["everything wide"]
        assert rows["with_casts"] == 1.0
        assert rows["without_casts"] == 1.0

    def test_casts_are_excluded_from_the_folded_count(self):
        graph = compile_with(mlp_graph(), "everything narrow")
        assert bytes_without_the_casts(graph) < bytes_of_intermediates(graph)


class TestAccuracy:
    def test_narrowing_the_products_keeps_the_answer(self):
        assert products_only_is_the_useful_policy()["products_only_gap"] < 1e-3

    def test_and_stays_finite(self):
        assert products_only_is_the_useful_policy()["products_only_finite"]

    def test_brain_float_is_worse_at_the_same_shapes(self):
        result = products_only_is_the_useful_policy()
        assert result["brain_float_gap"] > result["products_only_gap"]

    def test_every_fixture_survives_the_mixed_policy(self):
        assert every_fixture_survives_the_mixed_policy()

    def test_three_fixtures_are_checked(self):
        assert len(compare_graphs()) == 3

    def test_a_softmax_has_no_product_to_narrow(self):
        result = measure_policy(softmax_graph(), "products only")
        assert result["relative_gap"] == 0.0

    def test_a_layernorm_has_none_either(self):
        assert measure_policy(layernorm_graph(), "products only")["relative_gap"] == 0.0

    def test_the_narrow_type_really_is_half_precision(self):
        graph = compile_with(mlp_graph(), "everything narrow")
        assert any(node.output.dtype is FLOAT16 for node in graph.nodes)

    def test_a_wide_graph_holds_no_narrow_values(self):
        graph = compile_with(mlp_graph(), "everything wide")
        assert all(node.output.dtype is FLOAT32 for node in graph.nodes)

    def test_the_measurement_reports_whether_it_stayed_finite(self):
        assert "finite" in measure_policy(mlp_graph(), "everything narrow")

    def test_a_narrow_tensor_really_loses_bits(self):
        values = torch.randn(1024)
        assert not torch.equal(values.to(torch.float16).to(torch.float32), values)
