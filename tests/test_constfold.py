from __future__ import annotations

import math

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.dtype import FLOAT16, FLOAT32, FLOAT64, INT32
from tgc.ir.graph import validate
from tgc.passes.constfold import (
    can_fold_into,
    careless_fold,
    chained_fold,
    constant_environment,
    constant_node,
    constant_value,
    evaluate_constant,
    fold_constants,
    folding_precision_gap,
    is_constant,
    is_foldable,
    report_folding,
    round_to,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


def foldable_graph():
    """A scalar expression that does not depend on any input."""
    builder = Builder()
    x = builder.input([4], name="x")
    two = builder.constant(2.0)
    three = builder.constant(3.0)
    scale = builder.mul(two, three)
    return builder.finish(builder.mul(x, scale))


class TestRounding:
    def test_a_double_survives_itself(self):
        assert round_to(0.1, FLOAT64) == 0.1

    def test_a_single_rounds(self):
        # Folding in float64 and storing in float32 gives a constant the runtime graph would
        # never have produced.
        assert round_to(0.1, FLOAT32) != 0.1

    def test_a_half_rounds_further(self):
        assert abs(round_to(0.1, FLOAT16) - 0.1) > abs(round_to(0.1, FLOAT32) - 0.1)

    def test_an_integer_type_truncates(self):
        assert round_to(2.7, INT32) == 2.0

    def test_an_overflow_becomes_infinite_rather_than_raising(self):
        assert math.isinf(round_to(1e300, FLOAT32))

    def test_the_sign_survives_an_overflow(self):
        assert round_to(-1e300, FLOAT32) == -math.inf

    def test_the_gap_is_reported(self):
        assert folding_precision_gap(0.1, FLOAT32) > 0
        assert folding_precision_gap(0.1, FLOAT64) == 0.0


class TestAdmissibility:
    def test_an_ordinary_value_folds(self):
        assert can_fold_into(2.5, FLOAT32)[0]

    def test_an_overflow_does_not(self):
        # exp of a hundred is inf, and inf is a perfectly good float32 value, so nothing
        # raises and the folded graph is quietly different.
        allowed, reason = can_fold_into(math.exp(100), FLOAT32)
        assert not allowed
        assert "overflow" in reason

    def test_a_value_that_fits_float32_and_not_float16_is_caught(self):
        assert can_fold_into(1e30, FLOAT32)[0]
        assert not can_fold_into(1e30, FLOAT16)[0]

    def test_a_nan_does_not_fold(self):
        allowed, reason = can_fold_into(float("nan"), FLOAT32)
        assert not allowed
        assert "nan" in reason

    def test_a_fraction_does_not_fold_into_an_integer_type(self):
        allowed, reason = can_fold_into(2.5, INT32)
        assert not allowed
        assert "fraction" in reason

    def test_an_integer_too_large_for_its_type_does_not_fold(self):
        assert not can_fold_into(2.0**40, INT32)[0]

    def test_an_integer_that_fits_does(self):
        assert can_fold_into(42.0, INT32)[0]


class TestFolding:
    def test_a_constant_expression_becomes_a_literal(self):
        folded = fold_constants(foldable_graph())
        assert is_constant(folded.node("v2"))

    def test_the_literal_holds_the_right_value(self):
        folded = fold_constants(foldable_graph())
        assert constant_value(folded.node("v2")) == 6.0

    def test_the_graph_is_the_same_length(self):
        # Folding replaces a node rather than removing it. Removal is dead code elimination's
        # job, and doing it here means the two passes disagree about node counts.
        graph = foldable_graph()
        assert len(fold_constants(graph).nodes) == len(graph.nodes)

    def test_a_graph_with_nothing_to_fold_is_left_alone(self):
        graph = softmax_graph()
        assert len(fold_constants(graph).nodes) == len(graph.nodes)

    def test_the_answer_does_not_change(self):
        graph = foldable_graph()
        feeds = random_feeds(graph)
        assert outputs_agree(run(graph, feeds), run(fold_constants(graph), feeds))

    def test_the_result_still_validates(self):
        validate(fold_constants(foldable_graph()))

    def test_a_chain_folds_all_the_way(self):
        builder = Builder()
        x = builder.input([4], name="x")
        current = builder.constant(2.0)
        for _ in range(4):
            current = builder.add(current, builder.constant(1.0))
        graph = builder.finish(builder.mul(x, current))
        folded = fold_constants(graph)
        assert constant_value(folded.node(graph.nodes[-2].name)) == 6.0

    def test_a_tensor_shaped_expression_is_left_alone(self):
        # Folding a whole tensor turns a graph into its own weights, which is a different
        # transformation, and running it by accident is how a small graph acquires a hundred
        # megabytes of literals.
        builder = Builder()
        x = builder.input([64, 64], name="x")
        two = builder.constant(2.0)
        wide = builder.broadcast_to(two, [64, 64])
        graph = builder.finish(builder.mul(x, wide))
        assert not is_constant(fold_constants(graph).node(wide))

    def test_an_overflowing_fold_is_declined(self):
        builder = Builder()
        x = builder.input([4], name="x")
        big = builder.constant(100.0)
        blown = builder.exp(big)
        graph = builder.finish(builder.mul(x, blown))
        assert not is_constant(fold_constants(graph).node(blown))

    def test_a_division_by_zero_is_declined(self):
        builder = Builder()
        x = builder.input([4], name="x")
        one = builder.constant(1.0)
        zero = builder.constant(0.0)
        blown = builder.div(one, zero)
        graph = builder.finish(builder.mul(x, blown))
        assert not is_constant(fold_constants(graph).node(blown))

    def test_a_logarithm_of_a_negative_is_declined(self):
        builder = Builder()
        x = builder.input([4], name="x")
        negative = builder.constant(-1.0)
        blown = builder.log(negative)
        graph = builder.finish(builder.mul(x, blown))
        assert not is_constant(fold_constants(graph).node(blown))

    def test_running_it_twice_changes_nothing_the_second_time(self):
        once = fold_constants(foldable_graph())
        assert len(fold_constants(once).nodes) == len(once.nodes)


class TestFoldability:
    def test_a_node_reading_an_input_is_not_foldable(self):
        graph = softmax_graph()
        assert not is_foldable(graph.node("v1"), {})

    def test_a_constant_is_not_folded_again(self):
        graph = foldable_graph()
        assert not is_foldable(graph.node("v0"), constant_environment(graph))

    def test_a_node_with_known_operands_is(self):
        graph = foldable_graph()
        assert is_foldable(graph.node("v2"), constant_environment(graph))

    def test_the_environment_holds_every_literal(self):
        assert constant_environment(foldable_graph()) == {"v0": 2.0, "v1": 3.0}

    def test_asking_a_non_constant_for_its_value_is_rejected(self):
        with pytest.raises(PassError, match="not a constant"):
            constant_value(softmax_graph().node("v1"))

    def test_an_unfoldable_op_is_rejected(self):
        graph = softmax_graph()
        with pytest.raises(PassError, match="cannot be folded"):
            evaluate_constant(graph.node("v0"), [1.0])


class TestReporting:
    def test_it_names_what_it_folded(self):
        assert report_folding(foldable_graph()).folded == {"v2": 6.0}

    def test_it_says_why_it_declined(self):
        builder = Builder()
        x = builder.input([4], name="x")
        blown = builder.exp(builder.constant(100.0))
        graph = builder.finish(builder.mul(x, blown))
        assert "overflow" in report_folding(graph).refused[blown]

    def test_a_graph_with_nothing_to_fold_reports_nothing(self):
        assert report_folding(softmax_graph()).count == 0

    def test_it_serialises(self):
        assert report_folding(foldable_graph()).as_dict()["folded"] == 1


class TestAccumulationOrder:
    def test_folding_step_by_step_is_not_folding_in_double(self):
        # The runtime graph rounds at every step. A folder that sums in double and rounds
        # once at the end has computed a number the graph would never have produced.
        values = [1e8] + [1.0] * 100
        assert chained_fold(values) != careless_fold(values)

    def test_the_careless_version_keeps_what_the_graph_would_lose(self):
        values = [1e8] + [1.0] * 100
        assert careless_fold(values) > chained_fold(values)

    def test_the_gap_is_large_enough_to_notice(self):
        values = [1e8] + [1.0] * 100
        assert abs(careless_fold(values) - chained_fold(values)) > 90

    def test_a_gentler_magnitude_hides_it(self):
        values = [1.0] * 100
        assert chained_fold(values) == careless_fold(values)

    def test_adding_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to add"):
            chained_fold([])


class TestFixture:
    def test_a_literal_can_be_built_by_hand(self):
        assert constant_value(constant_node("c", 2.5)) == 2.5

    def test_it_is_a_constant(self):
        assert is_constant(constant_node("c", 2.5))

    def test_it_takes_the_type_it_was_given(self):
        assert constant_node("c", 2.5, FLOAT16).output.dtype is FLOAT16
