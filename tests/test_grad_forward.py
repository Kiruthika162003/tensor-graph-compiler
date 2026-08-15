from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.grad.forward import (
    JvpResult,
    accumulation_is_reverse_mode_only,
    agrees_with_torch,
    broadcasting_needs_no_correction,
    compare_with_torch,
    dot_product_identity,
    fan_in_graph,
    identity_holds_everywhere,
    jvp,
    passes_for_a_full_jacobian,
    size_comparison,
    split_jvp_feeds,
    tangent_name,
    the_diamond_fixture_is_vacuous,
    which_mode_to_use,
)
from tgc.ir.builder import (
    Builder,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.verify.reference import random_feeds, run


class TestBuilding:
    def test_a_tangent_input_appears_for_each_target(self):
        built = jvp(mlp_graph(), ["x", "w_up"])
        names = [value.name for value in built.graph.inputs]
        assert tangent_name("x") in names
        assert tangent_name("w_up") in names

    def test_an_input_held_fixed_gets_no_tangent_input(self):
        # Expressed as a zero tangent rather than by leaving it out, because an operation still
        # has to be told what its other operand's tangent was.
        built = jvp(mlp_graph(), ["x"])
        names = [value.name for value in built.graph.inputs]
        assert tangent_name("w_up") not in names

    def test_the_result_is_a_single_tangent(self):
        assert len(jvp(softmax_graph()).graph.outputs) == 1

    def test_it_records_what_it_grew_from(self):
        built = jvp(softmax_graph())
        assert built.forward_nodes == len(softmax_graph().nodes)

    def test_a_graph_with_two_outputs_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.relu(x), builder.exp(x))
        with pytest.raises(ConfigError, match="exactly one output"):
            jvp(graph)

    def test_an_unknown_target_is_refused(self):
        with pytest.raises(ConfigError, match="not inputs"):
            jvp(softmax_graph(), ["w"])

    def test_an_empty_target_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to differentiate"):
            jvp(softmax_graph(), [])

    def test_a_graph_with_an_indicator_in_it_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.step(x))
        with pytest.raises(PassError, match="no forward rule"):
            jvp(graph)

    def test_a_result_serialises(self):
        assert jvp(softmax_graph()).as_dict()["wrt"] == ["x"]

    def test_an_empty_result_has_no_growth(self):
        empty = JvpResult(graph=softmax_graph(), forward_nodes=0, tangent_nodes=0)
        assert empty.growth == 0.0


class TestAgreementWithTorch:
    def test_a_chain_matches(self):
        assert agrees_with_torch(elementwise_chain(3))

    def test_a_shared_value_matches(self):
        assert agrees_with_torch(fan_in_graph())

    def test_a_softmax_matches(self):
        assert agrees_with_torch(softmax_graph())

    def test_a_layernorm_matches(self):
        assert agrees_with_torch(layernorm_graph())

    def test_an_mlp_matches(self):
        assert agrees_with_torch(mlp_graph())

    def test_a_direction_along_one_input_only_matches(self):
        assert agrees_with_torch(mlp_graph(), ["w_up"])

    def test_feeds_split_into_values_and_directions(self):
        graph = mlp_graph()
        feeds = random_feeds(jvp(graph, ["x"]).graph, positive=True)
        values, tangents = split_jvp_feeds(graph, feeds, ["x"])
        assert set(tangents) == {"x"}
        assert len(values) == len(graph.inputs)

    def test_feeds_without_a_direction_are_refused(self):
        graph = softmax_graph()
        with pytest.raises(ConfigError, match="no 'x_tangent'"):
            split_jvp_feeds(graph, random_feeds(graph, positive=True), ["x"])

    def test_an_overflowing_chain_is_reported(self):
        assert compare_with_torch(elementwise_chain(4))["overflowed"] > 0


class TestDotProductIdentity:
    def test_the_two_modes_agree_on_every_fixture(self):
        # A cotangent dotted with a forward derivative equals a reverse derivative dotted with
        # the direction, because both are the same bilinear form read from opposite sides.
        assert all(row["holds"] for row in identity_holds_everywhere())

    def test_it_is_checked_on_five_graphs(self):
        assert len(identity_holds_everywhere()) == 5

    def test_the_two_sides_are_reported_separately(self):
        result = dot_product_identity(softmax_graph())
        assert result["forward_side"] != 0.0
        assert result["reverse_side"] != 0.0

    def test_an_mlp_agrees_to_eight_digits(self):
        assert dot_product_identity(mlp_graph())["relative_gap"] < 1e-7

    def test_the_diamond_fixture_computes_nothing(self):
        # Its two branches are a relu and a negation of the same exponential, and an
        # exponential is positive, so the relu is the identity there and they cancel.
        result = the_diamond_fixture_is_vacuous()
        assert result["largest_output"] == 0.0
        assert result["largest_tangent"] == 0.0

    def test_which_is_why_a_different_fan_in_fixture_exists(self):
        graph = fan_in_graph()
        feeds = random_feeds(jvp(graph).graph, positive=True)
        assert float(run(jvp(graph).graph, feeds)[0].abs().max()) > 0.0

    def test_and_it_shares_a_value(self):
        assert max(fan_in_graph().use_counts().values()) == 2


class TestStructure:
    def test_reverse_mode_needs_a_sum_to_undo_a_broadcast(self):
        assert broadcasting_needs_no_correction()["reverse_mode_sums"] == 1

    def test_forward_mode_needs_none(self):
        # The tangent broadcast the same way the value did, because it is going the same way.
        assert broadcasting_needs_no_correction()["forward_mode_sums"] == 0

    def test_the_saving_from_dropping_accumulation_is_worth_nothing(self):
        # It is given straight back by the product rule, which costs two multiplies and an
        # addition going forwards against one multiply per operand going back.
        result = accumulation_is_reverse_mode_only()
        assert result["forward_mode_nodes"] == result["reverse_mode_nodes"]

    def test_the_fixture_really_does_share_a_value(self):
        assert accumulation_is_reverse_mode_only()["values_read_more_than_once"] == 1

    def test_reverse_mode_builds_the_bigger_graph_where_there_is_a_reduction(self):
        rows = {row["graph"]: row for row in size_comparison()}
        assert rows["layernorm"]["reverse_mode"] > rows["layernorm"]["forward_mode"]

    def test_and_the_same_size_where_everything_is_elementwise(self):
        rows = {row["graph"]: row for row in size_comparison()}
        assert rows["chain"]["reverse_mode"] == rows["chain"]["forward_mode"]

    def test_every_fixture_is_compared(self):
        assert len(size_comparison()) == 5


class TestChoosingAMode:
    def test_a_function_with_more_inputs_than_outputs_wants_reverse(self):
        assert passes_for_a_full_jacobian(mlp_graph())["reverse_is_cheaper"]

    def test_and_it_wins_by_the_number_of_parameters(self):
        rows = {row["graph"]: row for row in which_mode_to_use()}
        assert rows["mlp"]["ratio"] > 60

    def test_an_elementwise_function_has_no_preference(self):
        rows = {row["graph"]: row for row in which_mode_to_use()}
        assert rows["chain"]["ratio"] == 1.0

    def test_the_counts_are_in_elements_rather_than_tensors(self):
        result = passes_for_a_full_jacobian(softmax_graph())
        assert result["input_elements"] == 8 * 32

    def test_every_fixture_is_classified(self):
        assert len(which_mode_to_use()) == 4
