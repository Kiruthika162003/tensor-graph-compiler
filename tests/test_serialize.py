from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.builder import (
    Builder,
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16, FLOAT32
from tgc.ir.graph import validate
from tgc.ir.serialize import (
    ParseError,
    dumps,
    format_shape,
    format_value,
    graphs_identical,
    loads,
    parse_shape,
    parse_value,
    round_trip_report,
    round_trips,
)
from tgc.ir.shape import shape
from tgc.verify.fuzz import generate_many
from tgc.verify.reference import outputs_agree, random_feeds, run

FIXTURES = {
    "softmax": softmax_graph(),
    "layernorm": layernorm_graph(),
    "mlp": mlp_graph(),
    "chain": elementwise_chain(6),
    "diamond": diamond_graph(),
    "branching": branching_graph(3, 2),
}


class TestPrinting:
    def test_a_graph_prints_with_a_header(self):
        assert dumps(softmax_graph()).startswith("graph(x: float32[8, 32]) {")

    def test_every_node_gets_a_line(self):
        text = dumps(softmax_graph())
        assert sum(1 for line in text.splitlines() if " = " in line) == 5

    def test_the_type_goes_on_the_line_that_defines_it(self):
        # A graph whose types were chosen by a pass has to round trip as itself and not as
        # whatever inference would have picked.
        assert "v0: float32[8, 1] = max(x)" in dumps(softmax_graph())

    def test_attributes_print_in_a_stable_order(self):
        text = dumps(softmax_graph())
        assert "{axes=[1], keepdims=true}" in text

    def test_the_outputs_are_named(self):
        graph = softmax_graph()
        assert f"return {graph.outputs[0]}" in dumps(graph)

    def test_a_symbolic_shape_prints_its_name(self):
        assert "batch" in dumps(mlp_graph(batch="batch"))


class TestValues:
    def test_a_shape_round_trips(self):
        assert parse_shape(format_shape(shape(2, 3, 4))) == shape(2, 3, 4)

    def test_a_symbolic_shape_round_trips(self):
        assert parse_shape(format_shape(shape("batch", 4))) == shape("batch", 4)

    def test_an_empty_shape_round_trips(self):
        assert parse_shape("").rank == 0

    def test_an_empty_dimension_is_rejected(self):
        with pytest.raises(ParseError, match="empty dimension"):
            parse_shape("2,,4")

    def test_a_boolean_round_trips(self):
        assert parse_value(format_value(True)) is True
        assert parse_value(format_value(False)) is False

    def test_a_tuple_round_trips(self):
        assert parse_value(format_value((1, 2, 3))) == (1, 2, 3)

    def test_an_empty_tuple_round_trips(self):
        assert parse_value(format_value(())) == ()

    def test_a_dtype_round_trips(self):
        assert parse_value(format_value(FLOAT16)) is FLOAT16

    def test_a_shape_attribute_round_trips(self):
        assert parse_value(format_value(shape(4, 8))) == shape(4, 8)

    def test_a_float_round_trips(self):
        assert parse_value(format_value(1.5)) == 1.5

    def test_an_integer_stays_an_integer(self):
        assert isinstance(parse_value("4"), int)


class TestRoundTrip:
    def test_every_fixture_survives(self):
        for name, graph in FIXTURES.items():
            assert round_trips(graph), name

    def test_a_broadcast_attribute_survives(self):
        builder = Builder()
        x = builder.input([1, 8], name="x")
        graph = builder.finish(builder.broadcast_to(x, [4, 8]))
        assert round_trips(graph)

    def test_a_cast_attribute_survives(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.cast(x, FLOAT16))
        assert round_trips(graph)

    def test_a_transpose_permutation_survives(self):
        builder = Builder()
        x = builder.input([2, 3, 4], name="x")
        graph = builder.finish(builder.transpose(x, [2, 0, 1]))
        assert round_trips(graph)

    def test_a_constant_survives(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.mul(x, builder.constant(2.5)))
        assert round_trips(graph)

    def test_generated_graphs_survive(self):
        # Fixtures were written by somebody with a format in mind, so the format has to
        # survive shapes nobody chose.
        assert round_trip_report(generate_many(40))["clean"]

    def test_a_parsed_graph_computes_the_same_thing(self):
        for graph in FIXTURES.values():
            feeds = random_feeds(graph, positive=True)
            assert outputs_agree(run(graph, feeds), run(loads(dumps(graph)), feeds))

    def test_a_parsed_graph_shares_nothing_with_its_source(self):
        graph = softmax_graph()
        parsed = loads(dumps(graph))
        assert parsed.nodes[0] is not graph.nodes[0]
        assert graphs_identical(graph, parsed)

    def test_checking_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to check"):
            round_trip_report([])


class TestParsing:
    def test_empty_text_is_rejected(self):
        with pytest.raises(ParseError, match="nothing to parse"):
            loads("")

    def test_text_that_is_not_a_graph_is_rejected(self):
        with pytest.raises(ParseError, match="not a graph header"):
            loads("hello\n")

    def test_a_broken_input_declaration_is_rejected(self):
        with pytest.raises(ParseError, match="input declaration"):
            loads("graph(x float32) {\n  return x\n}\n")

    def test_a_broken_assignment_is_rejected(self):
        text = "graph(x: float32[4]) {\n  this is not an assignment\n  return x\n}\n"
        with pytest.raises(ParseError, match="cannot read the assignment"):
            loads(text)

    def test_an_unknown_op_is_rejected(self):
        text = "graph(x: float32[4]) {\n  v0: float32[4] = convolve(x)\n  return v0\n}\n"
        with pytest.raises(ConfigError, match="unknown op"):
            loads(text)

    def test_a_parsed_graph_validates(self):

        validate(loads(dumps(softmax_graph())))

    def test_a_graph_that_reads_before_defining_is_rejected(self):
        text = "graph(x: float32[4]) {\n  v0: float32[4] = neg(missing)\n  return v0\n}\n"
        with pytest.raises(Exception, match="topological order"):
            loads(text)


class TestComparison:
    def test_a_graph_is_identical_to_itself(self):
        assert graphs_identical(softmax_graph(), softmax_graph())

    def test_different_node_counts_are_not_identical(self):
        graph = elementwise_chain(6)
        assert not graphs_identical(graph, graph.with_nodes(graph.nodes[:-1]))

    def test_different_input_types_are_not_identical(self):
        builder = Builder()
        first = builder.finish(builder.neg(builder.input([4], name="x")))
        other = Builder()
        second = other.finish(other.neg(other.input([4], dtype=FLOAT16, name="x")))
        assert not graphs_identical(first, second)

    def test_the_default_type_is_float32(self):
        assert softmax_graph().value("x").dtype is FLOAT32
