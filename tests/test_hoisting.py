from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import mlp_graph, softmax_graph
from tgc.passes.hoisting import (
    HoistReport,
    Split,
    a_graph_with_no_preprocessing_is_left_alone,
    an_unknown_parameter_is_refused,
    arithmetic_moved,
    break_even_calls,
    call_count_sweep,
    compare_graphs,
    declaring_nothing_hoists_nothing,
    hoistable,
    nothing_to_hoist,
    only_the_preprocessing_fixture_has_anything,
    parameter_only,
    preprocessed_weights,
    run_split,
    split,
    storage_cost,
    the_benefit_takes_a_hundred_calls,
    the_saving_is_traffic_rather_than_arithmetic,
    the_split_computes_the_same_thing,
    traffic_moved,
    what_moves,
)
from tgc.verify.reference import random_feeds, run


class TestFinding:
    def test_a_node_reading_only_weights_is_fixed(self):
        graph = preprocessed_weights()
        assert len(parameter_only(graph, ["w"])) == 3

    def test_a_node_reading_the_activation_is_not(self):
        graph = preprocessed_weights()
        fixed = set(parameter_only(graph, ["w"]))
        products = [node.name for node in graph.nodes if node.op is ops.MATMUL]
        assert not set(products) & fixed

    def test_the_boundary_is_what_the_body_still_reads(self):
        assert len(hoistable(preprocessed_weights(), ["w"])) == 1

    def test_declaring_nothing_finds_nothing(self):
        # A compiler that guessed which inputs were weights would be right most of the time and
        # would silently cache an activation the first time it was wrong.
        result = declaring_nothing_hoists_nothing()
        assert result["without_it"] == 0
        assert result["with_the_declaration"] > 0

    def test_an_unknown_parameter_is_refused(self):
        assert an_unknown_parameter_is_refused()

    def test_a_graph_with_no_preprocessing_offers_nothing(self):
        assert parameter_only(nothing_to_hoist(), ["w"]) == []

    def test_a_zero_dimension_fixture_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            preprocessed_weights(rows=0)


class TestSplitting:
    def test_the_split_computes_the_same_thing(self):
        # The split moves operations between two graphs and changes neither the operations nor
        # their order relative to each other.
        assert the_split_computes_the_same_thing()["identical"]

    def test_three_of_five_nodes_move(self):
        result = what_moves()
        assert result["moved"] == 3
        assert result["nodes"] == 5

    def test_the_body_keeps_the_rest(self):
        result = what_moves()
        assert result["body_nodes"] == result["nodes"] - result["moved"]

    def test_a_graph_with_nothing_to_move_is_returned_unchanged(self):
        assert a_graph_with_no_preprocessing_is_left_alone()["same_object"]

    def test_the_body_takes_the_hoisted_values_as_inputs(self):
        result = split(preprocessed_weights(), ["w"])
        names = [value.name for value in result.body.inputs]
        assert any(name.startswith("hoisted_") for name in names)

    def test_and_no_longer_takes_the_weight(self):
        result = split(preprocessed_weights(), ["w"])
        assert "w" not in [value.name for value in result.body.inputs]

    def test_the_prologue_takes_only_the_weight(self):
        result = split(preprocessed_weights(), ["w"])
        assert [value.name for value in result.prologue.inputs] == ["w"]

    def test_running_the_split_needs_both_halves(self):
        graph = preprocessed_weights()
        result = split(graph, ["w"])
        feeds = random_feeds(graph, positive=True)
        assert run_split(result, feeds, ["w"])[0].shape == run(graph, feeds)[0].shape

    def test_a_split_with_no_prologue_runs_the_body_alone(self):
        graph = nothing_to_hoist()
        result = split(graph, ["w"])
        feeds = random_feeds(graph, positive=True)
        assert run_split(result, feeds, ["w"])[0].equal(run(graph, feeds)[0])

    def test_an_empty_report_moves_nothing(self):
        assert HoistReport().moved == 0

    def test_a_split_serialises(self):
        assert split(preprocessed_weights(), ["w"]).as_dict()["moved"] == 3

    def test_a_split_with_no_prologue_says_so(self):
        assert not Split(prologue=None, body=nothing_to_hoist()).hoisted_anything


class TestWhatItSaves:
    def test_almost_no_arithmetic_moves(self):
        # The arithmetic in a layer is the product, and the product reads the activation.
        assert arithmetic_moved(preprocessed_weights(), ["w"])["share"] < 0.2

    def test_but_most_of_the_traffic_does(self):
        assert traffic_moved(preprocessed_weights(), ["w"])["share"] > 0.8

    def test_which_is_the_point_of_the_pass(self):
        result = the_saving_is_traffic_rather_than_arithmetic()
        assert result["traffic_share"] > 4 * result["arithmetic_share"]

    def test_the_hoisted_values_have_to_be_kept(self):
        # On this fixture that is a second copy of the weight.
        assert storage_cost()["ratio"] == 1.0

    def test_the_prologue_pays_for_itself_on_the_second_call(self):
        assert break_even_calls() == 2

    def test_a_graph_with_nothing_to_hoist_never_breaks_even(self):
        assert break_even_calls(nothing_to_hoist()) == 0

    def test_ten_calls_collect_about_half_of_it(self):
        assert the_benefit_takes_a_hundred_calls()["ten_calls_is_about_half"]

    def test_and_a_hundred_collect_most(self):
        assert the_benefit_takes_a_hundred_calls()["a_hundred_is_most_of_it"]

    def test_the_first_call_is_a_wash(self):
        rows = {row["calls"]: row for row in call_count_sweep()}
        assert rows[1]["ratio"] == 1.0

    def test_an_empty_call_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            call_count_sweep(counts=())


class TestFixtures:
    def test_none_of_the_standard_fixtures_offer_the_pass_anything(self):
        # They are written as a single layer applied to an activation, which is not what a
        # traced model looks like.
        assert only_the_preprocessing_fixture_has_anything()["with_work"] == ["preprocessed"]

    def test_five_graphs_are_compared(self):
        assert len(compare_graphs()) == 5

    def test_an_mlp_has_no_weight_preprocessing(self):
        assert split(mlp_graph(), ["w_up", "w_down", "b_up"]).report.moved == 0

    def test_a_softmax_has_no_parameters_at_all(self):
        assert split(softmax_graph(), []).report.moved == 0
