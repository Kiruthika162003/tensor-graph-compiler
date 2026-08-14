from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.graph import Node, Value, validate
from tgc.ir.shape import shape
from tgc.passes.dce import (
    append_dead_node,
    dead_node_count,
    eliminate_dead_code,
    live_values,
    report_dead_code,
    unused_inputs,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


def graph_with_dead_branch():
    builder = Builder()
    x = builder.input([4, 4], name="x")
    dead = builder.exp(x)
    builder.relu(dead)
    return builder.finish(builder.neg(x))


def graph_with_print():
    builder = Builder()
    x = builder.input([4, 4], name="x")
    printed = builder.apply(ops.PRINT, x)
    return builder.finish(builder.neg(x)), printed


class TestLiveness:
    def test_an_output_is_live(self):
        graph = softmax_graph()
        assert graph.outputs[0] in live_values(graph)

    def test_everything_an_output_needs_is_live(self):
        graph = softmax_graph()
        assert live_values(graph) == graph.value_names

    def test_a_dead_branch_is_not(self):
        graph = graph_with_dead_branch()
        assert "v0" not in live_values(graph)
        assert "v1" not in live_values(graph)

    def test_a_side_effect_is_live_even_when_nothing_reads_it(self):
        # Deleting it turns a graph that printed a tensor into one that does not, and the
        # author of the print has no way to tell that a compiler pass is the reason.
        graph, printed = graph_with_print()
        assert printed in live_values(graph)

    def test_what_a_side_effect_reads_is_live_too(self):
        graph, _ = graph_with_print()
        assert "x" in live_values(graph)


class TestElimination:
    def test_a_dead_branch_is_removed(self):
        assert len(eliminate_dead_code(graph_with_dead_branch()).nodes) == 1

    def test_a_live_graph_is_left_alone(self):
        graph = softmax_graph()
        assert len(eliminate_dead_code(graph).nodes) == len(graph.nodes)

    def test_a_print_survives(self):
        graph, printed = graph_with_print()
        assert printed in {node.name for node in eliminate_dead_code(graph).nodes}

    def test_the_result_still_validates(self):
        validate(eliminate_dead_code(graph_with_dead_branch()))

    def test_the_answer_does_not_change(self):
        graph = graph_with_dead_branch()
        feeds = random_feeds(graph)
        assert outputs_agree(run(graph, feeds), run(eliminate_dead_code(graph), feeds))

    def test_a_chain_of_dead_nodes_goes_at_once(self):
        # Which is why it closes backwards to a fixed point rather than filtering once.
        builder = Builder()
        x = builder.input([4], name="x")
        current = x
        for _ in range(5):
            current = builder.exp(current)
        graph = builder.finish(builder.neg(x))
        assert len(eliminate_dead_code(graph).nodes) == 1

    def test_running_it_twice_changes_nothing_the_second_time(self):
        once = eliminate_dead_code(graph_with_dead_branch())
        assert len(eliminate_dead_code(once).nodes) == len(once.nodes)


class TestReporting:
    def test_it_names_what_it_would_remove(self):
        assert report_dead_code(graph_with_dead_branch()).removed == ["v0", "v1"]

    def test_it_names_what_it_kept_for_effects(self):
        graph, printed = graph_with_print()
        assert report_dead_code(graph).kept_for_effects == [printed]

    def test_a_live_graph_reports_nothing(self):
        assert report_dead_code(softmax_graph()).count == 0

    def test_the_count_matches_the_removal(self):
        graph = graph_with_dead_branch()
        before = len(graph.nodes)
        after = len(eliminate_dead_code(graph).nodes)
        assert dead_node_count(graph) == before - after

    def test_it_serialises(self):
        assert report_dead_code(graph_with_dead_branch()).as_dict()["removed"] == 2


class TestInputs:
    def test_an_unused_input_is_reported(self):
        # Reported rather than removed: dropping it changes the signature of the compiled
        # function, which breaks the caller rather than the graph.
        builder = Builder()
        x = builder.input([4], name="x")
        builder.input([4], name="unused")
        graph = builder.finish(builder.neg(x))
        assert unused_inputs(graph) == ["unused"]

    def test_a_used_input_is_not(self):
        assert unused_inputs(softmax_graph()) == []

    def test_and_it_is_not_removed(self):
        builder = Builder()
        x = builder.input([4], name="x")
        builder.input([4], name="unused")
        graph = builder.finish(builder.neg(x))
        assert len(eliminate_dead_code(graph).inputs) == 2


class TestFixture:
    def test_a_dead_node_can_be_appended(self):
        graph = softmax_graph()
        output = Value(name="dead", shape=shape(8, 32), dtype=graph.value("x").dtype)
        grown = append_dead_node(graph, Node(op=ops.NEG, inputs=("x",), output=output))
        assert len(grown.nodes) == len(graph.nodes) + 1

    def test_and_is_then_removed(self):
        graph = softmax_graph()
        output = Value(name="dead", shape=shape(8, 32), dtype=graph.value("x").dtype)
        grown = append_dead_node(graph, Node(op=ops.NEG, inputs=("x",), output=output))
        assert len(eliminate_dead_code(grown).nodes) == len(graph.nodes)

    def test_a_name_collision_is_rejected(self):
        graph = softmax_graph()
        output = Value(name="v0", shape=shape(8, 32), dtype=graph.value("x").dtype)
        with pytest.raises(ConfigError, match="already defined"):
            append_dead_node(graph, Node(op=ops.NEG, inputs=("x",), output=output))

    def test_the_answer_survives_the_round_trip(self):
        graph = softmax_graph()
        output = Value(name="dead", shape=shape(8, 32), dtype=graph.value("x").dtype)
        grown = append_dead_node(graph, Node(op=ops.NEG, inputs=("x",), output=output))
        feeds = random_feeds(graph)
        assert torch.equal(run(graph, feeds)[0], run(eliminate_dead_code(grown), feeds)[0])
