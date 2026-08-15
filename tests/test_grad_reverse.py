from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, PassError
from tgc.grad.reverse import (
    COTANGENT,
    GradientResult,
    accumulation_matters,
    accumulation_report,
    agrees_with_torch,
    best_step,
    bitwise_agreement,
    check_every_fixture,
    compare_with_torch,
    curvature_decides_the_best_step,
    elementwise_share,
    finite_difference,
    gradient,
    growth_by_graph,
    ones_seed_hides_a_softmax,
    optimise,
    optimiser_gains,
    optimising_preserves_the_gradient,
    retype,
    split_feeds,
    step_size_sweep,
    torch_gradient,
)
from tgc.ir.builder import (
    Builder,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT32, FLOAT64
from tgc.verify.reference import random_feeds, run


def broadcast_graph():
    """A product against a column, so one operand is broadcast and one is not."""
    builder = Builder()
    x = builder.input([8, 32], name="x")
    column = builder.input([8, 1], name="column")
    return builder.finish(builder.mul(x, column))


class TestBuilding:
    def test_a_gradient_is_an_ordinary_graph(self):
        built = gradient(softmax_graph())
        assert built.graph.nodes

    def test_it_carries_the_forward_pass_with_it(self):
        built = gradient(softmax_graph())
        assert built.forward_nodes == len(softmax_graph().nodes)

    def test_the_cotangent_becomes_an_input(self):
        built = gradient(softmax_graph())
        assert COTANGENT in [value.name for value in built.graph.inputs]

    def test_seeding_with_ones_does_not(self):
        built = gradient(softmax_graph(), seed_is_input=False)
        assert COTANGENT not in [value.name for value in built.graph.inputs]

    def test_the_forward_output_can_be_kept(self):
        built = gradient(softmax_graph(), keep_forward=True)
        assert len(built.graph.outputs) == 2

    def test_and_is_left_out_by_default(self):
        assert len(gradient(softmax_graph()).graph.outputs) == 1

    def test_one_output_per_requested_input(self):
        built = gradient(mlp_graph(), ["x", "w_up"])
        assert len(built.graph.outputs) == 2

    def test_a_graph_with_two_outputs_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.relu(x), builder.exp(x))
        with pytest.raises(ConfigError, match="exactly one output"):
            gradient(graph)

    def test_an_unknown_target_is_refused(self):
        with pytest.raises(ConfigError, match="not inputs"):
            gradient(softmax_graph(), ["w"])

    def test_an_empty_target_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to differentiate"):
            gradient(softmax_graph(), [])

    def test_an_input_that_does_not_reach_the_output_is_refused(self):
        # Reported rather than answered with zeros. A gradient of zero and a gradient that was
        # never connected look the same to a caller and mean different things.
        builder = Builder()
        x = builder.input([4, 4], name="x")
        builder.input([4, 4], name="unused")
        graph = builder.finish(builder.relu(x))
        with pytest.raises(PassError, match="do not reach the output"):
            gradient(graph, ["unused"])

    def test_a_graph_already_holding_a_cotangent_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name=COTANGENT)
        graph = builder.finish(builder.relu(x))
        with pytest.raises(ConfigError, match="already has an input"):
            gradient(graph)

    def test_a_result_serialises(self):
        assert gradient(softmax_graph()).as_dict()["wrt"] == ["x"]

    def test_an_empty_result_has_no_growth(self):
        empty = GradientResult(graph=softmax_graph(), forward_nodes=0, backward_nodes=0)
        assert empty.growth == 0.0


class TestAgreementWithTorch:
    def test_a_chain_matches(self):
        assert agrees_with_torch(elementwise_chain(3))

    def test_a_diamond_matches(self):
        assert agrees_with_torch(diamond_graph())

    def test_a_softmax_matches(self):
        assert agrees_with_torch(softmax_graph(), tolerance=1e-5)

    def test_a_layernorm_matches(self):
        assert agrees_with_torch(layernorm_graph())

    def test_an_mlp_matches_on_every_parameter(self):
        assert agrees_with_torch(mlp_graph())

    def test_a_broadcast_operand_matches(self):
        # The gradient of the column is the sum across the axis it was broadcast along, and
        # leaving that sum out is the most common way a hand written backward pass is wrong.
        rows = {row["input"]: row for row in compare_with_torch(broadcast_graph())}
        assert rows["column"]["relative_gap"] == 0.0

    def test_the_broadcast_gradient_has_the_operand_shape(self):
        built = gradient(broadcast_graph(), ["column"])
        feeds = random_feeds(built.graph, positive=True)
        assert list(run(built.graph, feeds)[0].shape) == [8, 1]

    def test_an_overflowing_chain_is_reported_rather_than_differenced(self):
        # Infinity minus infinity is a nan, so the positions where both sides overflowed are
        # dropped. A chain of four exponentials reaches the top of float32 on ordinary inputs.
        assert compare_with_torch(elementwise_chain(4))[0]["overflowed"] > 0

    def test_and_still_agrees_where_it_is_finite(self):
        assert compare_with_torch(elementwise_chain(4))[0]["relative_gap"] == 0.0


class TestBitwiseAgreement:
    def test_an_mlp_agrees_to_the_last_bit(self):
        result = bitwise_agreement(mlp_graph())
        assert result["bit_identical"] == result["gradients"]

    def test_a_softmax_does_not(self):
        # Both build the same derivative out of different orders of the same operations and
        # float addition is not associative.
        assert bitwise_agreement(softmax_graph())["bit_identical"] == 0

    def test_but_agrees_to_a_millionth(self):
        assert bitwise_agreement(softmax_graph())["largest_relative_gap"] < 1e-5


class TestSeedChoice:
    def test_a_ones_seed_makes_the_softmax_check_vacuous(self):
        # The sum of a softmax row is one whatever the input was, so its derivative is zero
        # and a check built on that seed passes while measuring nothing.
        result = ones_seed_hides_a_softmax()
        assert result["with_a_ones_seed"] < 1e-6

    def test_and_a_real_cotangent_does_not(self):
        result = ones_seed_hides_a_softmax()
        assert result["with_a_real_cotangent"] > 1e-3

    def test_feeds_split_into_the_forward_ones_and_the_cotangent(self):
        graph = softmax_graph()
        feeds = random_feeds(gradient(graph).graph, positive=True)
        forward, cotangent = split_feeds(graph, feeds)
        assert set(forward) == {"x"}
        assert list(cotangent.shape) == [8, 32]

    def test_feeds_without_a_cotangent_are_refused(self):
        graph = softmax_graph()
        with pytest.raises(ConfigError, match="no 'cotangent'"):
            split_feeds(graph, random_feeds(graph, positive=True))

    def test_torch_defaults_to_ones_when_given_no_cotangent(self):
        graph = mlp_graph()
        feeds = random_feeds(graph, positive=True)
        plain = torch_gradient(graph, feeds, ["x"])[0]
        seeded = torch_gradient(graph, feeds, ["x"], torch.ones(8, 64))[0]
        assert torch.equal(plain, seeded)


class TestFiniteDifference:
    def test_the_analytic_gradient_survives_a_numerical_check(self):
        graph = softmax_graph()
        feeds = random_feeds(gradient(graph, ["x"]).graph, positive=True)
        assert finite_difference(graph, "x", feeds)["relative_gap"] < 1e-6

    def test_an_mlp_survives_it_too(self):
        graph = mlp_graph()
        feeds = random_feeds(gradient(graph, ["x"]).graph, positive=True)
        assert finite_difference(graph, "x", feeds)["relative_gap"] < 1e-6

    def test_the_error_falls_and_then_rises_with_the_step(self):
        # A large step measures the wrong thing because the function curves across it, and a
        # small one because the subtraction cancels almost every digit it had.
        gaps = [row["largest_gap"] for row in step_size_sweep(softmax_graph(), "x")]
        assert gaps.index(min(gaps)) not in (0, len(gaps) - 1)

    def test_the_smallest_step_is_not_the_best_one(self):
        rows = step_size_sweep(softmax_graph(), "x")
        smallest = min(row["step"] for row in rows)
        assert best_step(softmax_graph(), "x") > smallest

    def test_a_piecewise_linear_graph_has_no_interior_best_step(self):
        # An mlp is a product, a relu and another product, which is linear in its input along
        # every piece, so a central difference has no truncation error to trade against.
        rows = {row["graph"]: row for row in curvature_decides_the_best_step()}
        assert rows["mlp"]["at_an_end"]

    def test_and_a_softmax_does(self):
        rows = {row["graph"]: row for row in curvature_decides_the_best_step()}
        assert not rows["softmax"]["at_an_end"]

    def test_a_negative_step_is_refused(self):
        graph = softmax_graph()
        feeds = random_feeds(gradient(graph, ["x"]).graph, positive=True)
        with pytest.raises(ConfigError, match="has to be positive"):
            finite_difference(graph, "x", feeds, step=-1.0)

    def test_a_zero_sample_count_is_refused(self):
        graph = softmax_graph()
        feeds = random_feeds(gradient(graph, ["x"]).graph, positive=True)
        with pytest.raises(ConfigError, match="must be positive"):
            finite_difference(graph, "x", feeds, samples=0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            step_size_sweep(softmax_graph(), "x", steps=())


class TestRetyping:
    def test_a_graph_can_be_widened(self):
        assert retype(softmax_graph()).inputs[0].dtype == FLOAT64

    def test_the_original_is_left_alone(self):
        graph = softmax_graph()
        retype(graph)
        assert graph.inputs[0].dtype == FLOAT32

    def test_a_pinned_type_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.cast(x, FLOAT64))
        with pytest.raises(PassError, match="pins a type"):
            retype(graph)


class TestSize:
    def test_every_fixture_grows(self):
        assert all(row["growth"] > 1.0 for row in growth_by_graph())

    def test_an_elementwise_chain_is_the_cheap_end(self):
        rows = {row["graph"]: row for row in growth_by_graph()}
        assert rows["chain"]["growth"] < rows["softmax"]["growth"]

    def test_a_softmax_grows_the_most(self):
        # Two reductions, and a max differentiates into a broadcast, a subtraction, a step and
        # a multiply.
        rows = growth_by_graph()
        assert max(rows, key=lambda row: row["growth"])["graph"] == "softmax"

    def test_no_fixture_grows_past_five_times(self):
        assert all(row["growth"] < 5.0 for row in growth_by_graph())

    def test_a_normalisation_backward_pass_is_mostly_elementwise(self):
        assert elementwise_share(layernorm_graph())["share"] > 0.5

    def test_an_mlp_backward_pass_is_not(self):
        # A matmul differentiates into two matmuls and two transposes and none of those merge.
        assert elementwise_share(mlp_graph())["share"] < 0.25

    def test_a_graph_with_no_backward_nodes_reports_nothing(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        assert elementwise_share(builder.finish(builder.neg(x)))["share"] > 0


class TestOptimising:
    def test_the_default_pipeline_removes_part_of_a_gradient(self):
        assert optimiser_gains(softmax_graph())["removed"] > 0

    def test_the_ones_seed_makes_it_look_better_than_it_is(self):
        # Multiplications by a literal one, which the algebraic rules delete. A real cotangent
        # produces multiplications by an input, which they cannot.
        real = optimiser_gains(softmax_graph())["fraction_removed"]
        shortcut = optimiser_gains(softmax_graph(), seed_is_input=False)["fraction_removed"]
        assert shortcut > real

    def test_optimising_keeps_the_answer_on_every_fixture(self):
        for graph in (elementwise_chain(3), softmax_graph(), layernorm_graph(), mlp_graph()):
            assert optimising_preserves_the_gradient(graph)

    def test_the_optimised_graph_is_smaller(self):
        built = gradient(layernorm_graph())
        assert len(optimise(built).nodes) < len(built.graph.nodes)


class TestAccumulation:
    def test_a_value_read_twice_needs_its_gradients_summed(self):
        assert accumulation_report(diamond_graph()).accumulated == 1

    def test_a_softmax_reads_its_exponential_twice(self):
        assert accumulation_report(softmax_graph()).accumulated == 2

    def test_a_chain_needs_no_accumulation_at_all(self):
        assert accumulation_report(elementwise_chain(6)).accumulated == 0

    def test_the_largest_fan_in_is_recorded(self):
        assert accumulation_report(diamond_graph()).largest_fan_in == 2

    def test_every_fixture_is_reported(self):
        assert len(accumulation_matters()) == 4

    def test_it_serialises(self):
        assert accumulation_report(softmax_graph()).as_dict()["accumulated"] == 2


class TestEndToEnd:
    def test_every_fixture_passes_all_three_checks(self):
        for row in check_every_fixture():
            assert row["torch_gap"] < 1e-5
            assert row["finite_gap"] < 1e-5
            assert row["optimised_agrees"]
