from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.builder import Builder
from tgc.ir.rewriting import (
    ADD_ZERO,
    EXACT_RULES,
    INEXACT_RULES,
    LOOPING_RULES,
    SUB_SELF,
    WILDCARD,
    Pattern,
    RewriteReport,
    Rule,
    a_pattern_needs_an_operation,
    a_pattern_of_the_wrong_arity_never_matches,
    a_pattern_operand_has_to_be_one,
    a_repeated_name_binds_once,
    a_wildcard_matches_anything,
    a_zero_round_limit_is_refused,
    apply_once,
    either_rule_alone_settles,
    every_rule_fires,
    literal,
    logarithm_graph,
    looping_graph,
    match,
    messy_graph,
    rewrite,
    rules_are_shorter_than_the_pass,
    rules_that_undo_each_other,
    the_engine_reaches_a_fixed_point,
    the_inexact_rule_is_not_applied_by_default,
    the_rewrite_preserves_the_answer,
    what_the_inexact_rule_costs,
)


class TestPatterns:
    def test_a_pattern_names_its_bindings(self):
        assert Pattern("add", ("x", "y")).names == ("x", "y")

    def test_a_nested_pattern_collects_them_too(self):
        pattern = Pattern("neg", (Pattern("mul", ("x", "y")),))
        assert pattern.names == ("x", "y")

    def test_a_repeated_name_appears_once(self):
        assert Pattern("sub", ("x", "x")).names == ("x",)

    def test_a_pattern_prints_readably(self):
        assert str(Pattern("add", ("x", "y"))) == "add(x, y)"

    def test_an_empty_operation_is_refused(self):
        assert a_pattern_needs_an_operation()

    def test_an_operand_that_is_neither_is_refused(self):
        assert a_pattern_operand_has_to_be_one()

    def test_a_literal_matches_a_value(self):
        builder = Builder()
        name = builder.constant(1.0)
        graph = builder.finish(name)
        assert match(graph, name, literal(1.0)) == {}

    def test_and_refuses_a_different_one(self):
        builder = Builder()
        name = builder.constant(2.0)
        graph = builder.finish(name)
        assert match(graph, name, literal(1.0)) is None


class TestMatching:
    def test_a_repeated_name_requires_the_same_value(self):
        # A pattern language that let the two positions bind independently would match every
        # subtraction and rewrite them all to zero.
        result = a_repeated_name_binds_once()
        assert result["matches_the_same_value"]
        assert result["refuses_two_values"]

    def test_a_wildcard_accepts_anything(self):
        assert a_wildcard_matches_anything()["matched"]

    def test_and_binds_nothing(self):
        assert a_wildcard_matches_anything()["bound_nothing"]

    def test_the_wrong_arity_never_matches(self):
        result = a_pattern_of_the_wrong_arity_never_matches()
        assert result["right_arity"]
        assert result["wrong_arity"]

    def test_an_input_matches_nothing(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.neg(x))
        assert match(graph, "x", Pattern("neg", ("y",))) is None

    def test_the_wildcard_is_an_underscore(self):
        assert WILDCARD == "_"


class TestRewriting:
    def test_every_exact_rule_fires_on_the_fixture(self):
        result = every_rule_fires()
        assert result["fired"] == result["rules"]

    def test_and_each_one_fires_once(self):
        assert set(every_rule_fires()["counts"].values()) == {1}

    def test_the_rewrite_keeps_the_answer_exactly(self):
        # A rule that changed the answer and was labelled exact would be the worst kind of
        # mistake here, because the whole design rests on the label.
        assert the_rewrite_preserves_the_answer()["identical"]

    def test_and_makes_the_graph_smaller(self):
        result = the_rewrite_preserves_the_answer()
        assert result["nodes_after"] < result["nodes_before"]

    def test_it_settles_in_two_rounds(self):
        # The second round is the one that proves the first finished.
        result = the_engine_reaches_a_fixed_point()
        assert result["rounds"] == 2
        assert result["settled"]

    def test_a_single_pass_applies_each_rule_at_most_once_per_node(self):
        _, counts = apply_once(messy_graph(), EXACT_RULES)
        assert all(count == 1 for count in counts.values())

    def test_an_empty_rule_set_changes_nothing(self):
        graph = messy_graph()
        rewritten, counts = apply_once(graph, [])
        assert rewritten is graph
        assert counts == {}

    def test_a_zero_round_limit_is_refused(self):
        assert a_zero_round_limit_is_refused()

    def test_an_empty_report_applied_nothing(self):
        assert RewriteReport().total == 0

    def test_a_report_serialises(self):
        assert rewrite(messy_graph())[1].as_dict()["settled"]

    def test_a_rule_serialises(self):
        assert ADD_ZERO.as_dict()["exact"]

    def test_a_zero_sized_fixture_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            messy_graph(0)


class TestSafety:
    def test_an_inexact_rule_is_left_alone_by_default(self):
        # The safety has to live where the writer cannot forget it, which is the rule.
        assert the_inexact_rule_is_not_applied_by_default()["without_permission"] == 0

    def test_and_applied_when_asked_for(self):
        assert the_inexact_rule_is_not_applied_by_default()["with_permission"] == 1

    def test_the_inexact_rule_really_changes_the_answer(self):
        assert not what_the_inexact_rule_costs()["identical"]

    def test_though_only_in_the_last_few_digits(self):
        assert what_the_inexact_rule_costs()["relative_gap"] < 1e-6

    def test_every_exact_rule_is_labelled(self):
        assert all(rule.exact for rule in EXACT_RULES)

    def test_and_every_inexact_one_too(self):
        assert not any(rule.exact for rule in INEXACT_RULES)

    def test_the_rule_set_is_listed(self):
        assert len(rules_are_shorter_than_the_pass()["names"]) == 5


class TestConvergence:
    def test_two_rules_that_undo_each_other_never_settle(self):
        assert not rules_that_undo_each_other()["settled"]

    def test_and_the_engine_stops_at_the_limit(self):
        assert rules_that_undo_each_other()["rounds"] == 8

    def test_naming_both_rules(self):
        assert len(rules_that_undo_each_other()["rules_involved"]) == 2

    def test_the_graph_it_returns_is_still_correct(self):
        # A graph rewritten eight times is still a correct graph.
        assert rules_that_undo_each_other()["still_correct"]

    def test_either_rule_alone_settles(self):
        # A rule that is fine in isolation says nothing about the set it joins.
        result = either_rule_alone_settles()
        assert result["first_settles"]
        assert result["second_settles"]

    def test_but_the_pair_does_not(self):
        assert not either_rule_alone_settles()["together_settles"]

    def test_the_looping_pair_is_two_rules(self):
        assert len(LOOPING_RULES) == 2

    def test_a_rule_can_emit_several_nodes(self):
        graph = looping_graph()
        rewritten, _ = apply_once(graph, [SUB_SELF])
        assert len(rewritten.nodes) >= len(graph.nodes) - 1

    def test_the_logarithm_fixture_holds_the_pattern(self):
        graph = logarithm_graph()
        assert match(graph, graph.outputs[0], INEXACT_RULES[0].pattern) is not None

    def test_a_custom_rule_can_be_added(self):
        rule = Rule(
            name="negate twice",
            pattern=Pattern("neg", (Pattern("neg", ("x",)),)),
            replacement=lambda _builder, bindings: bindings["x"],
        )
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.neg(builder.neg(x)))
        rewritten, _ = rewrite(graph, [rule])
        assert len(rewritten.nodes) < len(graph.nodes)
