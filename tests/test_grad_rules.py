from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.grad.rules import (
    NON_DIFFERENTIABLE,
    RULES,
    blocking_nodes,
    check_differentiable,
    coverage,
    differentiable_nodes,
    has_rule,
    rule_for,
    static_sizes,
    sum_to_shape,
)
from tgc.ir import op as ops
from tgc.ir.builder import Builder, mlp_graph, softmax_graph
from tgc.ir.shape import shape


class TestCoverage:
    def test_every_operation_either_has_a_rule_or_is_named(self):
        # An op added to the forward set without a rule turns into a runtime failure the first
        # time somebody differentiates a graph holding it, which is much later than it needs
        # to be.
        assert coverage()["missing"] == []

    def test_the_two_lists_account_for_the_whole_op_set(self):
        result = coverage()
        assert result["with_rules"] + result["deliberately_without"] == result["ops"]

    def test_a_leaf_has_no_rule(self):
        assert not has_rule(ops.INPUT)
        assert not has_rule(ops.CONSTANT)

    def test_the_indicator_has_no_rule(self):
        # It is flat everywhere it is defined and undefined at the one point that matters.
        assert "step" in NON_DIFFERENTIABLE
        assert not has_rule(ops.STEP)

    def test_a_cast_has_no_rule(self):
        # Its honest gradient is a cast back, and a cast down then up is not the identity.
        assert not has_rule(ops.CAST)

    def test_every_elementwise_op_but_the_two_named_ones_has_a_rule(self):
        missing = [
            op.name
            for op in ops.elementwise_ops()
            if not has_rule(op) and op.name not in NON_DIFFERENTIABLE
        ]
        assert missing == []

    def test_every_reduction_has_a_rule(self):
        assert all(has_rule(op) for op in ops.reduction_ops())

    def test_asking_for_a_rule_that_does_not_exist_is_refused(self):
        with pytest.raises(PassError, match="no gradient rule"):
            rule_for(ops.STEP)

    def test_the_table_is_keyed_by_op_name(self):
        assert all(name in ops.BY_NAME for name in RULES)


class TestGraphCoverage:
    def test_a_softmax_is_differentiable_end_to_end(self):
        check_differentiable(softmax_graph())

    def test_and_so_is_an_mlp(self):
        check_differentiable(mlp_graph())

    def test_the_differentiable_nodes_are_listed(self):
        assert len(differentiable_nodes(softmax_graph())) == 5

    def test_a_graph_with_an_indicator_in_it_is_refused(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.step(x))
        with pytest.raises(ConfigError, match="no gradient rule"):
            check_differentiable(graph)

    def test_and_the_offending_node_is_named(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.step(x))
        assert len(blocking_nodes(graph)) == 1

    def test_a_side_effect_passes_the_gradient_through(self):
        assert has_rule(ops.PRINT)


class TestSizes:
    def test_a_static_shape_reads_as_numbers(self):
        assert static_sizes(shape(8, 32)) == [8, 32]

    def test_a_symbolic_one_is_refused(self):
        # A named dimension does not say whether it was broadcast.
        with pytest.raises(PassError, match="symbolic dimension"):
            static_sizes(shape("batch", 32))

    def test_a_scalar_has_no_sizes(self):
        assert static_sizes(shape()) == []


class TestSumToShape:
    def test_a_matching_shape_is_left_alone(self):
        builder = Builder()
        x = builder.input([8, 32], name="x")
        assert sum_to_shape(builder, x, shape(8, 32), shape(8, 32)) == x

    def test_a_broadcast_axis_is_summed_back(self):
        builder = Builder()
        x = builder.input([8, 32], name="x")
        result = sum_to_shape(builder, x, shape(8, 32), shape(8, 1))
        assert static_sizes(builder.shape_of(result)) == [8, 1]

    def test_a_leading_axis_is_summed_away_entirely(self):
        builder = Builder()
        x = builder.input([4, 8, 32], name="x")
        result = sum_to_shape(builder, x, shape(4, 8, 32), shape(8, 32))
        assert static_sizes(builder.shape_of(result)) == [8, 32]

    def test_a_scalar_operand_collects_everything(self):
        builder = Builder()
        x = builder.input([8, 32], name="x")
        result = sum_to_shape(builder, x, shape(8, 32), shape())
        assert static_sizes(builder.shape_of(result)) == []

    def test_growing_a_cotangent_is_refused(self):
        builder = Builder()
        x = builder.input([8], name="x")
        with pytest.raises(PassError, match="cannot fit"):
            sum_to_shape(builder, x, shape(8), shape(4, 8))

    def test_a_mismatched_axis_is_refused(self):
        builder = Builder()
        x = builder.input([8, 32], name="x")
        with pytest.raises(PassError, match="does not reduce"):
            sum_to_shape(builder, x, shape(8, 32), shape(8, 16))
