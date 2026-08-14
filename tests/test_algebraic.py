from __future__ import annotations

import math

import pytest
import torch

from tgc.errors import ConfigError
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.graph import validate
from tgc.passes.algebraic import (
    ALL_RULES,
    EXACT_RULES,
    FAST_RULES,
    Context,
    Rule,
    build_context,
    get_rule,
    inexact_rules_that_fire,
    report_simplification,
    simplify,
    simplify_fast,
)
from tgc.passes.constfold import fold_constants
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.divergence import (
    Divergence,
    exact_rules_are_exact,
    measure_divergence,
    measure_rule,
    reciprocal_round_trip_failure_rate,
    rule_table,
    worst_relative_gap,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


def times_one():
    builder = Builder()
    x = builder.input([4], name="x")
    return builder.finish(builder.mul(x, builder.constant(1.0)))


def times_zero():
    builder = Builder()
    x = builder.input([4], name="x")
    return builder.finish(builder.mul(x, builder.constant(0.0)))


def negated_twice():
    builder = Builder()
    x = builder.input([4], name="x")
    return builder.finish(builder.neg(builder.neg(x)))


class TestRule:
    def test_a_rule_that_changes_the_answer_has_to_say_how(self):
        with pytest.raises(ConfigError, match="does not say how"):
            Rule(name="odd", match=lambda _node, _context: None, exact=False)

    def test_an_exact_rule_needs_no_note(self):
        assert Rule(name="fine", match=lambda _node, _context: None, exact=True).exact

    def test_a_nameless_rule_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Rule(name="", match=lambda _node, _context: None, exact=True)

    def test_something_uncallable_is_rejected(self):
        with pytest.raises(ConfigError, match="not callable"):
            Rule(name="odd", match=42, exact=True)

    def test_it_looks_up_by_name(self):
        assert get_rule("mul_by_one").exact

    def test_an_unknown_rule_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown rule"):
            get_rule("distribute")

    def test_every_fast_rule_admits_it_is_inexact(self):
        assert all(not rule.exact for rule in FAST_RULES)

    def test_every_exact_rule_claims_to_be_exact(self):
        assert all(rule.exact for rule in EXACT_RULES)

    def test_the_table_lists_them_all(self):
        assert len(rule_table()) == len(ALL_RULES)


class TestContext:
    def test_it_finds_a_literal(self):
        context = build_context(times_one())
        assert context.constant("v0") == 1.0

    def test_a_non_literal_has_no_value(self):
        assert build_context(softmax_graph()).constant("v1") is None

    def test_it_matches_a_particular_literal(self):
        context = build_context(times_one())
        assert context.is_constant("v0", 1.0)
        assert not context.is_constant("v0", 2.0)

    def test_it_finds_a_producer(self):
        assert build_context(softmax_graph()).producer("v1") is not None

    def test_an_input_has_no_producer(self):
        assert build_context(softmax_graph()).producer("x") is None

    def test_an_empty_context_knows_nothing(self):
        assert Context().constant("x") is None


class TestExactRules:
    def test_multiplying_by_one_disappears(self):
        assert len(simplify(times_one()).nodes) == 1

    def test_the_output_is_rewired(self):
        assert simplify(times_one()).outputs == ["x"]

    def test_negating_twice_rewires_to_the_original(self):
        # The inner negation is left behind, because removing what nothing reads is dead
        # code elimination's job. A pass that does both makes the two disagree about node
        # counts and makes neither measurable on its own.
        assert simplify(negated_twice()).outputs == ["x"]

    def test_and_dead_code_removes_what_is_left(self):
        assert len(eliminate_dead_code(simplify(negated_twice())).nodes) == 0

    def test_dividing_by_one_disappears(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.div(x, builder.constant(1.0)))
        assert simplify(graph).outputs == ["x"]

    def test_adding_zero_disappears(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.add(x, builder.constant(0.0)))
        assert simplify(graph).outputs == ["x"]

    def test_a_commuted_identity_still_matches(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.mul(builder.constant(1.0), x))
        assert simplify(graph).outputs == ["x"]

    def test_multiplying_by_zero_is_left_alone_by_default(self):
        # A compiler that silently swaps exact arithmetic for approximate arithmetic has
        # changed the program without telling anybody.
        assert len(simplify(times_zero()).nodes) == 2

    def test_a_graph_with_no_identities_is_left_alone(self):
        graph = softmax_graph()
        assert len(simplify(graph).nodes) == len(graph.nodes)

    def test_the_result_still_validates(self):
        for graph in (times_one(), negated_twice(), softmax_graph()):
            validate(simplify(graph))

    def test_the_answer_is_bit_identical(self):
        for graph in (times_one(), negated_twice()):
            feeds = random_feeds(graph)
            assert outputs_agree(run(graph, feeds), run(simplify(graph), feeds))

    def test_every_exact_rule_is_bit_identical_on_its_own_example(self):
        # A claim of exactness that nobody checks is a comment.
        assert all(row["identical"] for row in exact_rules_are_exact())


class TestFastRules:
    def test_multiplying_by_zero_disappears_when_asked(self):
        assert len(simplify_fast(times_zero()).nodes) == 1

    def test_and_the_output_becomes_the_zero(self):
        assert simplify_fast(times_zero()).outputs == ["v0"]

    def test_the_graph_reports_which_inexact_rules_it_is_exposed_to(self):
        assert inexact_rules_that_fire(times_zero()) == ["mul_by_zero"]

    def test_an_exact_graph_is_exposed_to_none(self):
        assert inexact_rules_that_fire(negated_twice()) == []

    def test_a_square_root_squared_disappears_when_asked(self):
        builder = Builder()
        x = builder.input([4], name="x")
        root = builder.sqrt(x)
        graph = builder.finish(builder.mul(root, root))
        assert simplify_fast(graph).outputs == ["x"]

    def test_the_result_still_validates(self):
        validate(simplify_fast(times_zero()))


class TestDivergence:
    def test_not_one_fast_rule_is_exact(self):
        assert not any(row["exact"] for row in measure_divergence())

    def test_multiplying_infinity_by_zero_changes_nan_into_zero(self):
        # Which reaches a masked position in every attention implementation ever written.
        result = measure_rule("mul_by_zero")
        assert math.isnan(result.baseline)
        assert result.rewritten == 0.0
        assert result.produced_nan

    def test_cancelling_an_addition_is_wrong_by_the_whole_value(self):
        result = measure_rule("add_then_subtract")
        assert result.baseline == 0.0
        assert result.rewritten == 1.0
        assert result.relative_gap == 1.0

    def test_a_square_root_round_trip_is_wrong_in_the_last_bit(self):
        assert measure_rule("sqrt_squared").relative_gap < 1e-6

    def test_the_magnitudes_are_not_one_category(self):
        # Treating them as one is how a fast math flag gets turned on for a whole compiler.
        gaps = {row["rule"]: row["relative_gap"] for row in measure_divergence()}
        assert gaps["add_then_subtract"] > 1e6 * gaps["sqrt_squared"]

    def test_the_worst_finite_gap_is_the_whole_answer(self):
        assert worst_relative_gap() == 1.0

    def test_a_rule_with_no_worked_example_is_rejected(self):
        with pytest.raises(ConfigError, match="no worked example"):
            measure_rule("mul_by_one")

    def test_a_reciprocal_round_trip_is_usually_exact(self):
        # Which is why testing the rule on a single convenient number says nothing.
        assert reciprocal_round_trip_failure_rate() < 0.2

    def test_and_fails_often_enough_to_matter(self):
        assert reciprocal_round_trip_failure_rate() > 0.1

    def test_a_zero_sample_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            reciprocal_round_trip_failure_rate(samples=0)

    def test_a_nan_disagreement_reports_an_infinite_gap(self):
        result = Divergence(rule="r", baseline=float("nan"), rewritten=0.0, produced_nan=True)
        assert math.isinf(result.relative_gap)
        assert not result.is_exact

    def test_two_zeros_agree(self):
        assert Divergence(rule="r", baseline=0.0, rewritten=0.0).relative_gap == 0.0

    def test_it_serialises(self):
        assert measure_rule("sqrt_squared").as_dict()["rule"] == "sqrt_squared"


class TestReporting:
    def test_it_names_which_rule_fired(self):
        assert report_simplification(times_one()).applied == {"v1": "mul_by_one"}

    def test_it_flags_the_inexact_ones(self):
        report = report_simplification(times_zero(), rules=ALL_RULES)
        assert report.changed_the_answer
        assert report.inexact_applied == ["mul_by_zero"]

    def test_the_exact_set_never_flags_anything(self):
        assert not report_simplification(times_zero()).changed_the_answer

    def test_it_serialises(self):
        assert report_simplification(times_one()).as_dict()["applied"] == 1


class TestInteraction:
    def test_folding_and_simplifying_compose(self):
        # Folding two times a half into one makes the multiplication an identity, which the
        # simplifier can then remove. Neither pass gets there alone.
        builder = Builder()
        x = builder.input([4], name="x")
        scale = builder.mul(builder.constant(2.0), builder.constant(0.5))
        graph = builder.finish(builder.mul(x, scale))
        assert simplify(fold_constants(graph)).outputs == ["x"]

    def test_and_the_answer_survives(self):
        builder = Builder()
        x = builder.input([4], name="x")
        scale = builder.mul(builder.constant(2.0), builder.constant(0.5))
        graph = builder.finish(builder.mul(x, scale))
        feeds = random_feeds(graph)
        simplified = simplify(fold_constants(graph))
        assert torch.equal(run(graph, feeds)[0], run(simplified, feeds)[0])
