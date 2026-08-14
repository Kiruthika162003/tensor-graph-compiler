from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.op import (
    ADD,
    CONSTANT,
    DIV,
    ELEMENTWISE,
    EXP,
    INPUT,
    MATMUL,
    MUL,
    PRINT,
    SUB,
    SUM,
    TANH,
    Op,
    OpCost,
    OpStats,
    cost_of,
    elementwise_ops,
    get_op,
    reduction_ops,
)


class TestOp:
    def test_an_elementwise_op_reads_element_i_and_writes_element_i(self):
        assert ADD.is_elementwise
        assert not SUM.is_elementwise
        assert not MATMUL.is_elementwise

    def test_a_leaf_produces_without_reading(self):
        assert INPUT.is_leaf
        assert CONSTANT.is_leaf
        assert INPUT.arity == 0

    def test_a_pure_op_can_be_deleted_when_nothing_reads_it(self):
        assert ADD.can_be_removed_if_unused

    def test_a_side_effecting_one_cannot(self):
        assert not PRINT.can_be_removed_if_unused

    def test_addition_and_multiplication_commute(self):
        assert ADD.commutative
        assert MUL.commutative

    def test_subtraction_and_division_do_not(self):
        assert not SUB.commutative
        assert not DIV.commutative

    def test_an_elementwise_op_may_write_over_its_input(self):
        assert ADD.can_write_over_input

    def test_a_reduction_may_not(self):
        assert not SUM.can_write_over_input

    def test_nor_may_one_that_reads_its_input_twice(self):
        # It would read a value it has already overwritten.
        twice = Op(name="square_sum", category=ELEMENTWISE, arity=1, reads_input_once=False)
        assert not twice.can_write_over_input

    def test_a_nameless_op_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Op(name="", category=ELEMENTWISE)

    def test_an_unknown_category_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown category"):
            Op(name="odd", category="magic")

    def test_a_negative_arity_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot take"):
            Op(name="odd", category=ELEMENTWISE, arity=-2)

    def test_a_commutative_op_takes_two_inputs(self):
        with pytest.raises(ConfigError, match="commutative but takes"):
            Op(name="odd", category=ELEMENTWISE, arity=1, commutative=True)

    def test_it_looks_up_by_name(self):
        assert get_op("add") is ADD

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown op"):
            get_op("convolve")

    def test_it_prints_as_its_name(self):
        assert str(ADD) == "add"


class TestCategories:
    def test_every_elementwise_op_says_so(self):
        assert all(op.is_elementwise for op in elementwise_ops())

    def test_the_arithmetic_ops_are_all_there(self):
        names = {op.name for op in elementwise_ops()}
        assert {"add", "sub", "mul", "div", "exp", "relu"} <= names

    def test_no_reduction_is_elementwise(self):
        assert not any(op.is_elementwise for op in reduction_ops())

    def test_the_reductions_are_all_there(self):
        assert {op.name for op in reduction_ops()} == {"sum", "mean", "max"}


class TestCost:
    def test_a_transcendental_costs_more_than_an_addition(self):
        assert cost_of(TANH).flops_per_element > cost_of(ADD).flops_per_element

    def test_and_says_that_it_is_one(self):
        assert cost_of(EXP).transcendental
        assert not cost_of(ADD).transcendental

    def test_a_division_costs_more_than_a_multiplication(self):
        assert cost_of(DIV).flops_per_element > cost_of(MUL).flops_per_element

    def test_an_unlisted_op_costs_something(self):
        assert cost_of(Op(name="unlisted", category=ELEMENTWISE)).flops_per_element == 1.0

    def test_negative_work_is_rejected(self):
        with pytest.raises(ConfigError, match="negative work"):
            OpCost(flops_per_element=-1.0)


class TestStats:
    def test_it_counts_occurrences(self):
        stats = OpStats()
        stats.add(ADD)
        stats.add(ADD)
        stats.add(SUM)
        assert stats.counts == {"add": 2, "sum": 1}

    def test_the_total_adds_up(self):
        stats = OpStats()
        for op in (ADD, MUL, SUM):
            stats.add(op)
        assert stats.total == 3

    def test_it_counts_by_category(self):
        stats = OpStats()
        for op in (ADD, MUL, SUM):
            stats.add(op)
        assert stats.count_in_category(ELEMENTWISE) == 2

    def test_an_unknown_category_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown category"):
            OpStats().count_in_category("magic")

    def test_an_empty_tally_counts_nothing(self):
        assert OpStats().total == 0

    def test_it_serialises_in_a_stable_order(self):
        stats = OpStats()
        for op in (SUM, ADD):
            stats.add(op)
        assert list(stats.as_dict()) == ["add", "sum"]
