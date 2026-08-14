from __future__ import annotations

import pytest

from tgc.errors import ConfigError, GraphError, TypeInferenceError
from tgc.ir.builder import (
    Builder,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16, FLOAT32, INT32
from tgc.ir.graph import validate
from tgc.ir.shape import shape


class TestBuilder:
    def test_names_come_from_a_counter(self):
        # Names derived from op names collide the moment a graph has two additions, and a
        # collision in single assignment form silently reroutes a reader.
        builder = Builder()
        x = builder.input([4])
        first = builder.neg(x)
        second = builder.neg(x)
        assert first != second

    def test_an_input_can_be_named(self):
        builder = Builder()
        assert builder.input([4], name="x") == "x"

    def test_a_repeated_name_is_rejected(self):
        builder = Builder()
        builder.input([4], name="x")
        with pytest.raises(GraphError, match="already defined"):
            builder.input([4], name="x")

    def test_reading_an_undefined_value_is_rejected(self):
        with pytest.raises(GraphError, match="no value named"):
            Builder().neg("nothing")

    def test_returning_an_undefined_value_is_rejected(self):
        with pytest.raises(GraphError, match="not defined"):
            Builder().emit("nothing")

    def test_the_finished_graph_validates(self):
        builder = Builder()
        x = builder.input([4])
        validate(builder.finish(builder.relu(x)))

    def test_a_constant_is_a_scalar_by_default(self):
        builder = Builder()
        name = builder.constant(2.0)
        assert builder.finish(name).value(name).shape.rank == 0


class TestOperations:
    def test_a_sum_removes_its_axis(self):
        builder = Builder()
        x = builder.input([4, 8])
        total = builder.sum(x, axes=[1])
        assert builder.finish(total).value(total).shape == shape(4)

    def test_keeping_dimensions_leaves_a_one(self):
        builder = Builder()
        x = builder.input([4, 8])
        total = builder.sum(x, axes=[1], keepdims=True)
        assert builder.finish(total).value(total).shape == shape(4, 1)

    def test_a_matmul_meets_in_the_middle(self):
        builder = Builder()
        left = builder.input([4, 8])
        right = builder.input([8, 16])
        product = builder.matmul(left, right)
        assert builder.finish(product).value(product).shape == shape(4, 16)

    def test_a_mismatched_matmul_is_refused(self):
        builder = Builder()
        left = builder.input([4, 8])
        right = builder.input([16, 8])
        with pytest.raises(TypeInferenceError, match="do not meet"):
            builder.matmul(left, right)

    def test_a_cast_changes_the_type(self):
        builder = Builder()
        x = builder.input([4])
        narrowed = builder.cast(x, FLOAT16)
        assert builder.finish(narrowed).value(narrowed).dtype is FLOAT16

    def test_a_transpose_permutes(self):
        builder = Builder()
        x = builder.input([2, 3, 4])
        moved = builder.transpose(x, [2, 0, 1])
        assert builder.finish(moved).value(moved).shape == shape(4, 2, 3)

    def test_a_reshape_can_infer_one_dimension(self):
        builder = Builder()
        x = builder.input([4, 8])
        flat = builder.reshape(x, [-1])
        assert builder.finish(flat).value(flat).shape == shape(32)

    def test_a_broadcast_widens(self):
        builder = Builder()
        x = builder.input([1, 8])
        wide = builder.broadcast_to(x, [4, 8])
        assert builder.finish(wide).value(wide).shape == shape(4, 8)

    def test_an_integer_sum_stays_integral(self):
        builder = Builder()
        x = builder.input([4, 8], dtype=INT32)
        total = builder.sum(x, axes=[1])
        assert builder.finish(total).value(total).dtype is INT32

    def test_a_half_precision_sum_widens(self):
        builder = Builder()
        x = builder.input([4, 8], dtype=FLOAT16)
        total = builder.sum(x, axes=[1])
        assert builder.finish(total).value(total).dtype is FLOAT32


class TestFixtures:
    def test_the_chain_is_the_length_it_was_asked_for(self):
        assert len(elementwise_chain(6).nodes) == 6

    def test_every_node_in_it_is_elementwise(self):
        assert all(node.op.is_elementwise for node in elementwise_chain(6).nodes)

    def test_every_intermediate_in_it_is_read_once(self):
        # Which is the condition fusion needs, and the reason the chain is the fixture.
        graph = elementwise_chain(6)
        counts = graph.use_counts()
        assert all(counts[node.name] == 1 for node in graph.nodes)

    def test_a_chain_of_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one operation"):
            elementwise_chain(0)

    def test_the_diamond_shares_one_node_between_two_branches(self):
        assert len(diamond_graph().consumers_of("v0")) == 2

    def test_the_softmax_has_two_reductions_around_an_elementwise_chain(self):
        stats = softmax_graph().statistics()
        assert stats.counts["max"] == 1
        assert stats.counts["sum"] == 1

    def test_the_layernorm_reduces_twice(self):
        assert layernorm_graph().statistics().counts["mean"] == 2

    def test_the_mlp_has_two_matrix_products(self):
        assert mlp_graph().statistics().counts["matmul"] == 2

    def test_a_symbolic_batch_survives_the_mlp(self):
        graph = mlp_graph(batch="batch")
        assert not graph.value(graph.outputs[0]).shape.is_static

    def test_a_zero_width_mlp_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            mlp_graph(hidden=0)

    def test_every_fixture_validates(self):
        for graph in (
            elementwise_chain(4),
            diamond_graph(),
            mlp_graph(),
            softmax_graph(),
            layernorm_graph(),
        ):
            validate(graph)

    def test_every_fixture_is_float32(self):
        for graph in (softmax_graph(), layernorm_graph(), mlp_graph()):
            assert graph.value(graph.outputs[0]).dtype is FLOAT32
