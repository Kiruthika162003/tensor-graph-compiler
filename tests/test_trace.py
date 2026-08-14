from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, GraphError
from tgc.frontend.trace import (
    Proxy,
    branching,
    compare_unrolling,
    eager,
    feed_forward,
    layernorm,
    make_loop,
    softmax,
    trace,
    trace_with_report,
    traced_matches_eager,
    unrolled_length,
)
from tgc.ir.graph import validate
from tgc.runtime.executor import compile_graph
from tgc.verify.reference import random_feeds, run


class TestTracing:
    def test_it_records_what_the_function_did(self):
        graph = trace(softmax, [[8, 32]])
        assert [node.op.name for node in graph.nodes] == ["max", "sub", "exp", "sum", "div"]

    def test_the_traced_graph_validates(self):
        for function, shapes in (
            (softmax, [[8, 32]]),
            (layernorm, [[8, 32]]),
            (feed_forward, [[8, 16], [16, 32], [32, 16]]),
        ):
            validate(trace(function, shapes))

    def test_the_inputs_are_named(self):
        graph = trace(softmax, [[8, 32]], names=["x"])
        assert graph.inputs[0].name == "x"

    def test_a_function_with_no_inputs_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one input"):
            trace(softmax, [])

    def test_a_name_per_input_is_required(self):
        with pytest.raises(ConfigError, match="one name per input"):
            trace(softmax, [[8, 32]], names=["a", "b"])

    def test_a_function_that_returns_nothing_is_rejected(self):
        with pytest.raises(GraphError, match="returned nothing"):
            trace(lambda _x: [], [[8, 8]])

    def test_a_function_that_returns_a_stranger_is_rejected(self):
        # The trace never saw it being computed, so there is nothing to emit for it.
        with pytest.raises(GraphError, match="never saw being computed"):
            trace(lambda _x: [torch.zeros(4)], [[8, 8]])

    def test_a_multi_output_function_is_traced(self):
        graph = trace(lambda x: [x.relu(), x.tanh()], [[8, 8]])
        assert len(graph.outputs) == 2


class TestEquivalence:
    def test_a_traced_softmax_matches_the_function_it_came_from(self):
        assert traced_matches_eager(softmax, [[8, 32]])

    def test_a_traced_layernorm_does_too(self):
        assert traced_matches_eager(layernorm, [[8, 32]])

    def test_and_a_feed_forward_block(self):
        assert traced_matches_eager(feed_forward, [[8, 16], [16, 32], [32, 16]])

    def test_the_proxy_is_spelled_like_a_tensor(self):
        # Which is the whole reason the comparison means anything. The first version of the
        # proxy took axis and keepdims, so the same source computed two different softmaxes
        # and the test passed for the wrong reason.
        assert hasattr(torch.zeros(2, 2), "amax")
        assert hasattr(Proxy(name="a", builder=None), "amax")

    def test_the_traced_graph_also_compiles_and_matches(self):
        graph = trace(softmax, [[8, 32]])
        feeds = random_feeds(graph, positive=True)
        assert torch.equal(compile_graph(graph)(feeds)[0], run(graph, feeds)[0])

    def test_a_constant_in_the_function_becomes_a_literal(self):
        graph = trace(layernorm, [[8, 32]])
        assert any(node.op.name == "constant" for node in graph.nodes)

    def test_arithmetic_with_a_python_number_works_either_way_round(self):
        assert traced_matches_eager(lambda x: 2.0 - (x * 3.0), [[4, 4]])

    def test_eager_runs_the_original_function(self):
        sample = torch.ones(4, 4)
        assert torch.equal(eager(lambda x: x + x, [sample]), sample * 2)


class TestLimits:
    def test_branching_on_a_tensor_is_refused(self):
        # Rather than recording the side it took and losing the other, which produces a graph
        # that validates, runs, and is wrong for half its inputs.
        with pytest.raises(GraphError, match="cannot branch"):
            trace(branching, [[8, 8]])

    def test_the_error_says_why(self):
        with pytest.raises(GraphError, match="loses the other"):
            trace(branching, [[8, 8]])

    def test_a_python_loop_is_unrolled(self):
        assert unrolled_length(make_loop(8), [[8, 8]]) == 8

    def test_the_node_count_follows_the_loop_bound(self):
        rows = compare_unrolling(make_loop, [[8, 8]])
        assert [row["nodes"] for row in rows] == [row["iterations"] for row in rows]

    def test_a_long_trace_is_noted(self):
        report = trace_with_report(make_loop(32), [[8, 8]])
        assert any("unrolled" in note for note in report.notes)

    def test_a_short_one_is_not(self):
        assert trace_with_report(softmax, [[8, 32]]).notes == []

    def test_a_loop_that_never_runs_is_rejected(self):
        with pytest.raises(ConfigError, match="has to run"):
            make_loop(0)

    def test_it_serialises(self):
        assert trace_with_report(softmax, [[8, 32]]).as_dict()["inputs"] == 1


class TestProxy:
    def test_a_matmul_needs_two_tensors(self):
        with pytest.raises(GraphError, match="needs two tensors"):
            trace(lambda x: x @ 2.0, [[4, 4]])

    def test_negation_is_recorded(self):
        graph = trace(lambda x: -x, [[4, 4]])
        assert graph.nodes[0].op.name == "neg"

    def test_division_by_a_number_is_recorded(self):
        graph = trace(lambda x: x / 2.0, [[4, 4]])
        assert {node.op.name for node in graph.nodes} == {"constant", "div"}

    def test_a_reversed_subtraction_keeps_its_order(self):
        graph = trace(lambda x: 1.0 - x, [[4, 4]])
        subtraction = next(node for node in graph.nodes if node.op.name == "sub")
        assert subtraction.inputs[1] == "in0"

    def test_it_prints_its_name(self):
        assert repr(Proxy(name="v3", builder=None)) == "Proxy(v3)"

    def test_a_traced_shape_follows_the_matmul(self):
        graph = trace(feed_forward, [[8, 16], [16, 32], [32, 16]])
        assert graph.value(graph.outputs[0]).shape.sizes[-1].value == 16
