from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.graph import Node
from tgc.passes.outline import (
    Call,
    Occurrence,
    Program,
    a_shared_intermediate_is_refused,
    a_single_occurrence_is_left_alone,
    best_pattern_length,
    build_function,
    choose_occurrences,
    code_size,
    elementwise_layers,
    find_occurrences,
    fusion_lost_to_the_boundary,
    outline,
    outlining_is_worth_it_when_the_pattern_repeats,
    outlining_preserves_the_answer,
    parameters_of,
    pattern_length_sweep,
    run_program,
    size_by_layer_count,
    stacked_layers,
    the_cost_is_zero_where_there_was_already_a_boundary,
    the_function_is_the_same_graph_every_time,
    window_key,
)
from tgc.verify.reference import random_feeds, run


class TestMatching:
    def test_a_stack_of_layers_repeats_its_pattern(self):
        groups = find_occurrences(stacked_layers(4), 2)
        assert max(len(items) for items in groups.values()) == 4

    def test_the_parameters_are_what_a_window_reads_from_outside(self):
        graph = stacked_layers(2)
        assert parameters_of(graph.nodes[:2]) == ["x", "w"]

    def test_a_value_produced_inside_is_not_a_parameter(self):
        graph = stacked_layers(2)
        assert graph.nodes[0].name not in parameters_of(graph.nodes[:2])

    def test_two_windows_of_the_same_shape_share_a_key(self):
        graph = stacked_layers(4)
        assert window_key(graph, graph.nodes[:2]) == window_key(graph, graph.nodes[2:4])

    def test_windows_at_different_widths_do_not(self):
        narrow = stacked_layers(2, width=8)
        wide = stacked_layers(2, width=16)
        assert window_key(narrow, narrow.nodes[:2]) != window_key(wide, wide.nodes[:2])

    def test_a_zero_length_pattern_is_refused(self):
        with pytest.raises(ConfigError, match="needs some length"):
            find_occurrences(stacked_layers(2), 0)

    def test_a_window_whose_middle_escapes_is_refused(self):
        # A function returns one value, so a window that would have to return two is declined.
        result = a_shared_intermediate_is_refused()
        assert result["legal"] < result["windows"]


class TestOutlining:
    def test_four_layers_become_one_function_and_four_calls(self):
        program = outline(stacked_layers(4), 2)
        assert len(program.functions) == 1
        assert program.call_count == 4

    def test_and_the_answer_is_bit_identical(self):
        # A call performs the same operations on the same values in the same order.
        assert outlining_preserves_the_answer()["identical"]

    def test_on_an_elementwise_stack_too(self):
        assert outlining_preserves_the_answer(elementwise_layers(), 2)["identical"]

    def test_the_occurrences_really_are_one_function(self):
        # Checked by structural hash rather than by the matcher that grouped them.
        result = the_function_is_the_same_graph_every_time()
        assert result["distinct_functions"] == 1

    def test_a_pattern_that_appears_once_is_left_alone(self):
        # A function called once is the same nodes plus a boundary, for no saving.
        assert a_single_occurrence_is_left_alone()["calls"] == 0

    def test_a_graph_with_no_repeats_comes_back_whole(self):
        graph = softmax_graph()
        program = outline(graph, 3)
        assert program.call_count == 0
        assert len(program.steps) == len(graph.nodes)

    def test_the_function_takes_its_parameters_by_position(self):
        graph = stacked_layers(4)
        _, occurrences = choose_occurrences(graph, 2)
        function = build_function(graph, occurrences[1])
        assert [value.name for value in function.inputs] == ["arg0", "arg1"]

    def test_which_is_what_stops_a_name_collision(self):
        # An occurrence halfway down a graph takes a parameter called something the builder is
        # about to hand out.
        graph = stacked_layers(4)
        _, occurrences = choose_occurrences(graph, 2)
        assert all(build_function(graph, item).nodes for item in occurrences)


class TestCodeSize:
    def test_eight_layers_compile_from_the_code_of_one(self):
        result = code_size(stacked_layers(8))
        assert result["after"] == 2

    def test_the_saving_grows_with_the_layer_count(self):
        saved = [row["saved"] for row in size_by_layer_count()]
        assert saved == sorted(saved)

    def test_one_layer_saves_nothing(self):
        rows = {row["layers"]: row for row in outlining_is_worth_it_when_the_pattern_repeats()}
        assert not rows[1]["worth_it"]

    def test_two_layers_already_do(self):
        rows = {row["layers"]: row for row in outlining_is_worth_it_when_the_pattern_repeats()}
        assert rows[2]["worth_it"]

    def test_the_best_pattern_length_is_the_repeating_unit(self):
        assert best_pattern_length() == 2

    def test_a_run_of_one_saves_nothing(self):
        rows = {row["length"]: row for row in pattern_length_sweep()}
        assert rows[1]["saved"] == 0

    def test_and_a_run_longer_than_the_unit_matches_less_often(self):
        rows = {row["length"]: row for row in pattern_length_sweep()}
        assert rows[6]["calls"] < rows[2]["calls"]

    def test_an_empty_layer_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            size_by_layer_count(counts=())

    def test_an_empty_length_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            pattern_length_sweep(lengths=())

    def test_a_stack_of_no_layers_is_refused(self):
        with pytest.raises(ConfigError, match="at least one layer"):
            stacked_layers(0)

    def test_an_elementwise_stack_of_none_is_refused(self):
        with pytest.raises(ConfigError, match="at least one layer"):
            elementwise_layers(0)


class TestFusionCost:
    def test_an_unbroken_elementwise_graph_is_one_loop(self):
        assert fusion_lost_to_the_boundary()["loops_before"] == 1

    def test_and_becomes_one_loop_per_call(self):
        # Nothing merges across a call, so each call writes its result for the next to read.
        result = fusion_lost_to_the_boundary()
        assert result["loops_after"] == result["calls"]

    def test_which_is_eight_times_worse(self):
        result = fusion_lost_to_the_boundary()
        assert result["loops_after"] == 8 * result["loops_before"]

    def test_a_layer_with_a_product_in_it_loses_nothing(self):
        # The product was already breaking the fusion where the call boundary now sits.
        rows = the_cost_is_zero_where_there_was_already_a_boundary()
        assert rows["with_a_product"]["loops_after"] == rows["with_a_product"]["loops_before"]

    def test_and_the_elementwise_one_loses_everything(self):
        rows = the_cost_is_zero_where_there_was_already_a_boundary()
        assert rows["elementwise"]["loops_after"] > rows["elementwise"]["loops_before"]


class TestProgram:
    def test_a_program_counts_its_nodes_across_the_functions(self):
        program = outline(stacked_layers(8), 2)
        assert program.total_nodes == 2

    def test_an_empty_program_holds_nothing(self):
        assert Program().total_nodes == 0

    def test_it_serialises(self):
        assert outline(stacked_layers(4), 2).as_dict()["calls"] == 4

    def test_a_call_serialises(self):
        call = Call(function="body", arguments=("x", "w"), result="v1")
        assert call.as_dict()["function"] == "body"

    def test_an_occurrence_serialises(self):
        item = Occurrence(start=0, nodes=("a",), parameters=("x",), result="a")
        assert item.as_dict()["start"] == 0

    def test_running_without_an_input_is_refused(self):
        program = outline(stacked_layers(4), 2)
        with pytest.raises(PassError, match="no value supplied"):
            run_program(program, {})

    def test_a_call_with_the_wrong_argument_count_is_refused(self):
        graph = stacked_layers(4)
        program = outline(graph, 2)
        program.steps = [
            Call(function="body", arguments=("x",), result="v1")
            if isinstance(step, Call)
            else step
            for step in program.steps
        ]
        with pytest.raises(PassError, match="takes 2 arguments"):
            run_program(program, random_feeds(graph, positive=True))

    def test_a_program_with_no_calls_runs_as_the_graph_did(self):
        graph = softmax_graph()
        program = outline(graph, 3)
        feeds = random_feeds(graph, positive=True)
        assert run_program(program, feeds)[0].equal(run(graph, feeds)[0])

    def test_the_steps_are_nodes_and_calls(self):
        program = outline(stacked_layers(4), 2)
        assert all(isinstance(step, (Node, Call)) for step in program.steps)

    def test_a_graph_with_a_shared_intermediate_is_left_alone(self):
        builder = Builder()
        x = builder.input([16, 16], name="x")
        weight = builder.input([16, 16], name="w")
        product = builder.matmul(x, weight)
        graph = builder.finish(builder.add(builder.tanh(product), product))
        assert outline(graph, 2).call_count == 0
