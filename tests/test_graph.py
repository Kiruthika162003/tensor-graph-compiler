from __future__ import annotations

import pytest

from tgc.errors import ConfigError, GraphError, TypeInferenceError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, diamond_graph, elementwise_chain, softmax_graph
from tgc.ir.dtype import FLOAT16, FLOAT32, INT32
from tgc.ir.graph import (
    Graph,
    Node,
    Value,
    infer_output,
    is_valid,
    reachable_from_outputs,
    topological_order,
    validate,
)
from tgc.ir.shape import Shape, shape


def value(name: str, *sizes, dtype=FLOAT32) -> Value:
    return Value(name=name, shape=shape(*sizes), dtype=dtype)


class TestValue:
    def test_the_storage_follows_the_shape_and_type(self):
        assert value("a", 2, 3).bytes == 24

    def test_a_narrower_type_takes_less(self):
        assert value("a", 2, 3, dtype=FLOAT16).bytes == 12

    def test_a_nameless_value_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Value(name="", shape=Shape(), dtype=FLOAT32)

    def test_it_prints_its_type_and_shape(self):
        assert str(value("a", 2, 3)) == "a: float32[2, 3]"


class TestNode:
    def test_the_input_count_has_to_match_the_op(self):
        with pytest.raises(GraphError, match="takes 2 inputs"):
            Node(op=ops.ADD, inputs=("x",), output=value("y", 4))

    def test_a_node_is_named_after_the_value_it_produces(self):
        assert Node(op=ops.NEG, inputs=("x",), output=value("y", 4)).name == "y"

    def test_two_identical_nodes_share_a_signature(self):
        first = Node(op=ops.ADD, inputs=("a", "b"), output=value("c", 4))
        second = Node(op=ops.ADD, inputs=("a", "b"), output=value("d", 4))
        assert first.signature() == second.signature()

    def test_a_commutative_op_ignores_operand_order(self):
        first = Node(op=ops.ADD, inputs=("a", "b"), output=value("c", 4))
        second = Node(op=ops.ADD, inputs=("b", "a"), output=value("d", 4))
        assert first.signature() == second.signature()

    def test_a_non_commutative_one_does_not(self):
        first = Node(op=ops.SUB, inputs=("a", "b"), output=value("c", 4))
        second = Node(op=ops.SUB, inputs=("b", "a"), output=value("d", 4))
        assert first.signature() != second.signature()

    def test_attributes_are_part_of_the_signature(self):
        # A reduction over axis zero and one over axis one are the same op on the same input
        # and are not the same value.
        first = Node(op=ops.SUM, inputs=("a",), output=value("c", 4), attrs={"axes": (0,)})
        second = Node(op=ops.SUM, inputs=("a",), output=value("d", 4), attrs={"axes": (1,)})
        assert first.signature() != second.signature()

    def test_the_output_type_is_part_of_it_too(self):
        first = Node(op=ops.NEG, inputs=("a",), output=value("c", 4))
        second = Node(op=ops.NEG, inputs=("a",), output=value("d", 4, dtype=FLOAT16))
        assert first.signature() != second.signature()

    def test_inputs_can_be_renamed(self):
        node = Node(op=ops.ADD, inputs=("a", "b"), output=value("c", 4))
        assert node.replace_inputs({"a": "z"}).inputs == ("z", "b")

    def test_renaming_leaves_the_original_alone(self):
        node = Node(op=ops.ADD, inputs=("a", "b"), output=value("c", 4))
        node.replace_inputs({"a": "z"})
        assert node.inputs == ("a", "b")

    def test_it_prints_as_an_assignment(self):
        node = Node(op=ops.ADD, inputs=("a", "b"), output=value("c", 4))
        assert str(node) == "c = add(a, b)"


class TestGraph:
    def test_a_built_graph_validates(self):
        validate(softmax_graph())

    def test_it_finds_a_value_by_name(self):
        graph = softmax_graph()
        assert graph.value("x").name == "x"

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(GraphError, match="no value named"):
            softmax_graph().value("nothing")

    def test_a_graph_input_has_no_producer(self):
        assert softmax_graph().producer_of("x") is None

    def test_every_other_value_has_one(self):
        graph = softmax_graph()
        assert graph.producer_of(graph.outputs[0]) is not None

    def test_it_finds_the_consumers_of_a_value(self):
        # In the diamond, the shared node feeds two branches.
        graph = diamond_graph()
        assert len(graph.consumers_of("v0")) == 2

    def test_a_value_used_twice_counts_twice(self):
        # Counted per argument position, because a value used twice cannot have its storage
        # written over by the op reading it.
        builder = Builder()
        x = builder.input([4])
        graph = builder.finish(builder.mul(x, x))
        assert graph.use_counts()["v0"] == 2

    def test_an_output_counts_as_a_use(self):
        graph = elementwise_chain(2)
        assert graph.use_counts()[graph.outputs[0]] == 1

    def test_a_clone_can_be_rewritten_independently(self):
        graph = softmax_graph()
        clone = graph.clone()
        clone.nodes.pop()
        assert len(graph.nodes) == len(clone.nodes) + 1

    def test_the_statistics_tally_the_ops(self):
        stats = elementwise_chain(4).statistics()
        assert stats.total == 4

    def test_it_prints_as_a_readable_block(self):
        text = str(elementwise_chain(2))
        assert text.startswith("graph(x: float32[64, 64])")
        assert "return" in text


class TestValidation:
    def test_reading_before_defining_is_caught(self):
        graph = Graph(
            nodes=[Node(op=ops.NEG, inputs=("missing",), output=value("y", 4))],
            inputs=[value("x", 4)],
            outputs=["y"],
        )
        with pytest.raises(GraphError, match="topological order"):
            validate(graph)

    def test_assigning_twice_is_caught(self):
        graph = Graph(
            nodes=[
                Node(op=ops.NEG, inputs=("x",), output=value("y", 4)),
                Node(op=ops.NEG, inputs=("x",), output=value("y", 4)),
            ],
            inputs=[value("x", 4)],
            outputs=["y"],
        )
        with pytest.raises(GraphError, match="assigned twice"):
            validate(graph)

    def test_a_repeated_input_is_caught(self):
        graph = Graph(nodes=[], inputs=[value("x", 4), value("x", 4)], outputs=["x"])
        with pytest.raises(GraphError, match="declared twice"):
            validate(graph)

    def test_a_graph_with_no_outputs_is_caught(self):
        graph = Graph(nodes=[], inputs=[value("x", 4)], outputs=[])
        with pytest.raises(GraphError, match="computes nothing"):
            validate(graph)

    def test_an_undefined_output_is_caught(self):
        graph = Graph(nodes=[], inputs=[value("x", 4)], outputs=["nothing"])
        with pytest.raises(GraphError, match="not defined"):
            validate(graph)

    def test_the_predicate_agrees_with_the_check(self):
        assert is_valid(softmax_graph())
        assert not is_valid(Graph(nodes=[], inputs=[value("x", 4)], outputs=[]))


class TestInference:
    def test_an_elementwise_op_broadcasts_its_inputs(self):
        result = infer_output(ops.ADD, [value("a", 3, 1), value("b", 3, 4)], {}, "c")
        assert result.shape == shape(3, 4)

    def test_and_promotes_their_types(self):
        result = infer_output(ops.ADD, [value("a", 4, dtype=INT32), value("b", 4)], {}, "c")
        assert result.dtype is FLOAT32

    def test_a_division_of_integers_produces_a_float(self):
        left = value("a", 4, dtype=INT32)
        result = infer_output(ops.DIV, [left, left], {}, "c")
        assert result.dtype is FLOAT32

    def test_a_transcendental_of_integers_does_too(self):
        result = infer_output(ops.EXP, [value("a", 4, dtype=INT32)], {}, "c")
        assert result.dtype is FLOAT32

    def test_a_cast_needs_a_target(self):
        with pytest.raises(TypeInferenceError, match="needs a target dtype"):
            infer_output(ops.CAST, [value("a", 4)], {}, "c")

    def test_a_reduction_widens_a_narrow_accumulator(self):
        result = infer_output(ops.SUM, [value("a", 4, dtype=FLOAT16)], {"axes": (0,)}, "c")
        assert result.dtype is FLOAT32

    def test_a_maximum_keeps_the_input_type(self):
        # Nothing accumulates, so there is nothing to widen for.
        result = infer_output(ops.MAX, [value("a", 4, dtype=FLOAT16)], {"axes": (0,)}, "c")
        assert result.dtype is FLOAT16

    def test_a_reduction_needs_axes(self):
        with pytest.raises(TypeInferenceError, match="needs axes"):
            infer_output(ops.SUM, [value("a", 4)], {}, "c")

    def test_a_matrix_product_meets_in_the_middle(self):
        result = infer_output(ops.MATMUL, [value("a", 3, 4), value("b", 4, 5)], {}, "c")
        assert result.shape == shape(3, 5)

    def test_a_reshape_needs_sizes(self):
        with pytest.raises(TypeInferenceError, match="needs target sizes"):
            infer_output(ops.RESHAPE, [value("a", 4)], {}, "c")

    def test_a_transpose_needs_a_permutation(self):
        with pytest.raises(TypeInferenceError, match="needs a permutation"):
            infer_output(ops.TRANSPOSE, [value("a", 4)], {}, "c")

    def test_an_elementwise_op_with_no_inputs_is_rejected(self):
        with pytest.raises(TypeInferenceError, match="at least one input"):
            infer_output(ops.ADD, [], {}, "c")


class TestOrdering:
    def test_a_valid_graph_is_already_in_order(self):
        graph = softmax_graph()
        assert [node.name for node in topological_order(graph)] == [
            node.name for node in graph.nodes
        ]

    def test_a_shuffled_graph_comes_back_in_order(self):
        graph = elementwise_chain(4)
        shuffled = graph.with_nodes(list(reversed(graph.nodes)))
        ordered = [node.name for node in topological_order(shuffled)]
        assert ordered == [node.name for node in graph.nodes]

    def test_the_order_is_the_same_every_time(self):
        # Two runs on the same graph must agree, because the order decides the peak memory
        # and an order that varies between runs makes a regression impossible to bisect.
        graph = softmax_graph()
        assert topological_order(graph) == topological_order(graph)

    def test_a_cycle_is_reported(self):
        left = Node(op=ops.NEG, inputs=("b",), output=value("a", 4))
        right = Node(op=ops.NEG, inputs=("a",), output=value("b", 4))
        graph = Graph(nodes=[left, right], inputs=[value("x", 4)], outputs=["a"])
        with pytest.raises(GraphError, match="cycle"):
            topological_order(graph)


class TestReachability:
    def test_everything_an_output_needs_is_reachable(self):
        graph = softmax_graph()
        assert graph.outputs[0] in reachable_from_outputs(graph)
        assert "x" in reachable_from_outputs(graph)

    def test_a_value_nothing_reads_is_not(self):
        builder = Builder()
        x = builder.input([4], name="x")
        dead = builder.exp(x)
        graph = builder.finish(builder.neg(x))
        assert dead not in reachable_from_outputs(graph)
