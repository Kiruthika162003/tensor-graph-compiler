from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, GraphError
from tgc.frontend.importer import (
    COMPOSITE,
    DIRECT,
    UNSUPPORTED,
    ForeignGraph,
    ForeignNode,
    a_forward_reference_is_refused,
    an_output_that_was_never_produced_is_refused,
    an_unsupported_model_is_refused,
    classify,
    comments_and_blank_lines_are_ignored,
    composites_are_where_the_risk_is,
    coverage,
    coverage_of_the_sample,
    every_parse_error_names_the_problem,
    expansion,
    operation_groups,
    parse,
    parse_errors,
    sample_graph,
    the_axis_is_not_optional,
    the_lowering_grows_the_graph,
    the_sample_imports,
    to_graph,
    unsupported_operations,
)
from tgc.ir import op as ops
from tgc.verify.reference import run


class TestParsing:
    def test_a_model_parses_into_inputs_nodes_and_outputs(self):
        graph = sample_graph()
        assert (len(graph.inputs), len(graph.nodes), len(graph.outputs)) == (3, 3, 1)

    def test_shapes_are_read_from_the_declaration(self):
        assert sample_graph().inputs[0] == ("x", (8, 32))

    def test_attributes_are_read_as_numbers(self):
        node = next(node for node in sample_graph().nodes if node.op == "Softmax")
        assert node.attrs["axis"] == 1

    def test_a_string_attribute_stays_a_string(self):
        graph = parse('input x : 8x8\nv0 = Clip(x, mode="hard")\noutput v0')
        assert graph.nodes[0].attrs["mode"] == "hard"

    def test_comments_and_blank_lines_are_ignored(self):
        result = comments_and_blank_lines_are_ignored()
        assert result == {"inputs": 1, "nodes": 1}

    def test_an_empty_file_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to parse"):
            parse("   ")

    def test_every_parse_error_names_the_problem(self):
        result = every_parse_error_names_the_problem()
        assert result["matched"] == result["cases"]

    def test_five_kinds_of_bad_input_are_covered(self):
        assert len(parse_errors()) == 5

    def test_a_node_serialises(self):
        node = ForeignNode(name="v0", op="Relu", inputs=("x",))
        assert node.as_dict()["op"] == "Relu"

    def test_a_graph_serialises(self):
        assert sample_graph().as_dict()["nodes"] == 3

    def test_an_empty_graph_counts_nothing(self):
        assert ForeignGraph().op_counts == {}


class TestClassification:
    def test_a_rename_is_direct(self):
        assert classify("Relu") == "direct"

    def test_a_lowering_is_composite(self):
        assert classify("Softmax") == "composite"

    def test_anything_else_is_unsupported(self):
        assert classify("Conv") == "unsupported"

    def test_the_groups_are_disjoint(self):
        assert not set(DIRECT) & set(COMPOSITE)
        assert not set(COMPOSITE) & set(UNSUPPORTED)

    def test_the_sizes_are_reported(self):
        assert operation_groups()["direct"] == len(DIRECT)

    def test_the_sample_imports_entirely(self):
        assert coverage_of_the_sample()["importable_share"] == 1.0

    def test_an_empty_model_has_no_coverage(self):
        assert coverage(ForeignGraph(outputs=["x"]))["importable_share"] == 0.0

    def test_the_unsupported_ones_are_named(self):
        graph = parse("input x : 1x1\nv0 = Erf(x)\noutput v0")
        assert unsupported_operations(graph) == ["Erf"]


class TestLowering:
    def test_the_imported_graph_computes_what_the_model_meant(self):
        assert the_sample_imports()["agrees"]

    def test_a_lowering_that_ignored_the_axis_would_look_fine(self):
        # Same shape, same node count, and an output differing by almost the whole range.
        result = the_axis_is_not_optional()
        assert result["same_shape"]
        assert result["same_node_count"]

    def test_and_would_compute_a_different_function(self):
        assert the_axis_is_not_optional()["largest_gap"] > 0.5

    def test_three_foreign_nodes_become_eleven(self):
        result = the_lowering_grows_the_graph()
        assert result["foreign_nodes"] == 3
        assert result["lowered_nodes"] == 11

    def test_a_product_without_a_bias_lowers_to_one_node(self):
        graph = to_graph(parse("input x : 8x8\ninput w : 8x8\nv0 = Gemm(x, w)\noutput v0"))
        assert len(graph.nodes) == 1

    def test_a_clip_becomes_a_maximum_and_a_minimum(self):
        graph = to_graph(parse("input x : 4x4\nv0 = Clip(x, min=0, max=6)\noutput v0"))
        names = {node.op.name for node in graph.nodes}
        assert {"maximum", "minimum"} <= names

    def test_and_computes_the_clamp(self):
        graph = to_graph(parse("input x : 4x4\nv0 = Clip(x, min=0, max=1)\noutput v0"))
        values = torch.tensor([[-2.0, 0.5, 3.0, 1.0]] * 4)
        assert run(graph, {"x": values})[0].max().item() == 1.0

    def test_a_leaky_rectifier_keeps_a_slope_below_zero(self):
        graph = to_graph(parse("input x : 1x2\nv0 = LeakyRelu(x, alpha=0.5)\noutput v0"))
        values = torch.tensor([[-2.0, 2.0]])
        assert run(graph, {"x": values})[0].tolist() == [[-1.0, 2.0]]

    def test_a_reduce_mean_keeps_its_dimension(self):
        graph = to_graph(parse("input x : 4x8\nv0 = ReduceMean(x, axes=1)\noutput v0"))
        assert [size.value for size in graph.value(graph.outputs[0]).shape.sizes] == [4, 1]

    def test_composites_are_where_the_attributes_are(self):
        assert composites_are_where_the_risk_is()["all_composite"]


class TestRefusals:
    def test_an_unsupported_model_is_refused(self):
        # Dropping the operation and importing the rest gives a graph that computes most of a
        # model, which is worse than not importing it, because it runs.
        assert an_unsupported_model_is_refused()["refused"]

    def test_and_the_operation_is_named(self):
        assert an_unsupported_model_is_refused()["named"]

    def test_a_forward_reference_is_refused(self):
        assert a_forward_reference_is_refused()

    def test_an_output_nothing_produces_is_refused(self):
        assert an_output_that_was_never_produced_is_refused()

    def test_a_model_with_no_inputs_is_refused(self):
        with pytest.raises(ConfigError, match="at least one input"):
            to_graph(parse("v0 = Relu(v0)\noutput v0"))

    def test_a_product_with_one_operand_is_refused(self):
        with pytest.raises(GraphError, match="two or three operands"):
            to_graph(parse("input x : 8x8\nv0 = Gemm(x)\noutput v0"))

    def test_an_axis_outside_the_rank_is_refused(self):
        with pytest.raises(GraphError, match="outside a rank"):
            to_graph(parse("input x : 8x8\nv0 = Softmax(x, axis=5)\noutput v0"))

    def test_a_clip_that_keeps_nothing_is_refused(self):
        with pytest.raises(GraphError, match="keeps nothing"):
            to_graph(parse("input x : 4x4\nv0 = Clip(x, min=1, max=0)\noutput v0"))

    def test_a_negative_slope_is_refused(self):
        with pytest.raises(GraphError, match="not a leak"):
            to_graph(parse("input x : 4x4\nv0 = LeakyRelu(x, alpha=-1)\noutput v0"))

    def test_an_unknown_operation_reaches_the_lowering_as_unsupported(self):
        with pytest.raises(GraphError, match="no lowering"):
            to_graph(parse("input x : 4x4\nv0 = Wobble(x)\noutput v0"))


class TestExpansion:
    def test_the_softmax_is_the_expensive_lowering(self):
        graph = to_graph(parse("input x : 8x8\nv0 = Softmax(x, axis=1)\noutput v0"))
        assert len(graph.nodes) == 7

    def test_a_rename_costs_nothing(self):
        graph = to_graph(parse("input x : 8x8\nv0 = Relu(x)\noutput v0"))
        assert len(graph.nodes) == 1

    def test_the_ratio_is_reported(self):
        assert expansion(sample_graph())["ratio"] > 3.0

    def test_the_lowered_graph_holds_our_operations(self):
        graph = to_graph(sample_graph())
        assert all(node.op in ops.ALL_OPS for node in graph.nodes)
