from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import layernorm_graph, mlp_graph, softmax_graph
from tgc.passes.cse import eliminate_common_subexpressions
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.layout import cancel_transposes
from tgc.verify.mutation import (
    MUTANTS,
    Mutant,
    MutationReport,
    caught_by_extreme_inputs,
    caught_by_outputs,
    caught_by_side_effects,
    cse_that_ignores_attributes,
    dce_that_forgets_side_effects,
    every_mutant_is_caught_by_something,
    fixture_rank_hides_a_mutant,
    fixture_set_hides_a_mutant,
    fixtures,
    generated_graphs_catch_almost_nothing,
    graph_with_a_side_effect,
    graph_with_a_zero,
    multiply_by_zero_is_zero,
    mutants_on_generated_graphs,
    mutation_report,
    overflowing_graph,
    random_inputs_are_not_enough,
    rank_three_graph,
    reductions_that_lose_an_axis,
    transposes_that_always_cancel,
    two_reductions_of_one_value,
    which_check_is_needed,
)


class TestTheRealPassesSurvive:
    def test_dead_code_elimination_is_not_caught(self):
        # The point of comparison. If a correct pass registered as a mutant, the checks would
        # be measuring something other than correctness.
        assert not any(caught_by_outputs(eliminate_dead_code, graph) for _, graph in fixtures())

    def test_nor_is_subexpression_elimination(self):
        graph = two_reductions_of_one_value()
        assert not caught_by_outputs(eliminate_common_subexpressions, graph)

    def test_nor_is_transpose_cancellation(self):
        assert not caught_by_outputs(cancel_transposes, rank_three_graph())

    def test_and_the_real_dce_keeps_the_print(self):
        assert not caught_by_side_effects(eliminate_dead_code, graph_with_a_side_effect())


class TestMutants:
    def test_every_mutant_is_caught_by_something(self):
        assert every_mutant_is_caught_by_something()["all_caught"]

    def test_six_of_them_are_tried(self):
        assert len(MUTANTS) == 6

    def test_each_one_says_what_it_got_wrong(self):
        assert all(mutant.mistake for mutant in MUTANTS)

    def test_a_mutant_serialises(self):
        assert MUTANTS[0].as_dict()["mutant"] == "dce keeps no side effects"

    def test_the_dropped_print_survives_a_value_comparison(self):
        # A print returns its input, so deleting it leaves every number exactly where it was.
        assert not caught_by_outputs(dce_that_forgets_side_effects, graph_with_a_side_effect())

    def test_and_is_caught_by_looking_at_the_side_effects(self):
        assert caught_by_side_effects(dce_that_forgets_side_effects, graph_with_a_side_effect())

    def test_the_zero_rule_is_wrong_only_on_infinities(self):
        graph = graph_with_a_zero()
        assert not caught_by_outputs(multiply_by_zero_is_zero, graph)
        assert caught_by_extreme_inputs(multiply_by_zero_is_zero, graph)

    def test_the_axis_shift_is_caught_immediately(self):
        assert caught_by_outputs(reductions_that_lose_an_axis, softmax_graph())

    def test_a_graph_with_nothing_to_break_reports_no_side_effect_change(self):
        assert not caught_by_side_effects(dce_that_forgets_side_effects, softmax_graph())


class TestFixturesWereTheProblem:
    def test_the_transpose_mutant_is_invisible_on_matrices(self):
        # At rank two the only permutation worth writing is its own inverse, so the mutant is
        # not a mistake there. Every fixture in the suite was a matrix.
        assert fixture_rank_hides_a_mutant()["the_fixtures_were_the_problem"]

    def test_and_visible_the_moment_a_graph_has_three_axes(self):
        assert caught_by_outputs(transposes_that_always_cancel, rank_three_graph())

    def test_the_subexpression_mutant_needs_a_pair_to_merge(self):
        assert fixture_set_hides_a_mutant()["the_fixtures_were_the_problem"]

    def test_and_a_graph_with_one_catches_it(self):
        assert caught_by_outputs(cse_that_ignores_attributes, two_reductions_of_one_value())

    def test_the_new_fixture_reduces_one_value_along_two_axes(self):
        graph = two_reductions_of_one_value()
        reductions = [node for node in graph.nodes if node.op is ops.SUM]
        assert len({node.attrs["axes"] for node in reductions}) == 2

    def test_the_rank_three_fixture_does_not_cancel_its_transposes(self):
        # Two rotations of three axes compose into a third rotation, not the identity.
        graph = rank_three_graph()
        assert sum(1 for node in graph.nodes if node.op is ops.TRANSPOSE) == 2

    def test_a_zero_size_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            rank_three_graph(0)

    def test_an_empty_reduction_fixture_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            two_reductions_of_one_value(0)


class TestWhatEachCheckBuys:
    def test_comparing_outputs_catches_four_of_six(self):
        assert random_inputs_are_not_enough()["caught_by_outputs"] == 4

    def test_and_leaves_two(self):
        survivors = random_inputs_are_not_enough()["survivors"]
        assert survivors == ["dce keeps no side effects", "multiply by zero is zero"]

    def test_the_zero_rule_needs_an_infinity_to_show_up(self):
        rows = {row["mutant"]: row["cheapest_check"] for row in which_check_is_needed()}
        assert rows["multiply by zero is zero"] == "comparing outputs on infinities and nans"

    def test_the_cheapest_check_is_named_for_each(self):
        rows = {row["mutant"]: row["cheapest_check"] for row in which_check_is_needed()}
        assert rows["dce keeps no side effects"] == "checking the side effects survived"

    def test_nothing_is_left_without_a_check(self):
        assert all(
            row["cheapest_check"] != "nothing here catches it"
            for row in which_check_is_needed()
        )

    def test_the_overflow_mutant_is_caught_by_the_ordinary_comparison(self):
        rows = {row["mutant"]: row["cheapest_check"] for row in which_check_is_needed()}
        assert rows["folding clamps an overflow"] == "comparing outputs"

    def test_the_overflowing_fixture_really_overflows(self):
        graph = overflowing_graph()
        literals = [node for node in graph.nodes if node.op is ops.CONSTANT]
        assert any(float(node.attrs["value"]) >= 1e30 for node in literals)


class TestFuzzingIsNotEnough:
    def test_four_mutants_never_show_up_on_a_generated_graph(self):
        assert len(generated_graphs_catch_almost_nothing()["never_caught"]) == 4

    def test_and_the_best_of_the_rest_shows_up_on_a_quarter(self):
        assert generated_graphs_catch_almost_nothing()["best_share"] <= 0.25

    def test_every_mutant_is_tried_on_the_generated_set(self):
        assert len(mutants_on_generated_graphs()) == len(MUTANTS)

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            mutants_on_generated_graphs(count=0)


class TestReport:
    def test_the_report_covers_every_mutant(self):
        assert mutation_report().total == len(MUTANTS)

    def test_an_empty_report_has_no_survivors(self):
        assert MutationReport().survivors == []

    def test_it_serialises(self):
        assert mutation_report().as_dict()["mutants"] == len(MUTANTS)

    def test_a_mutant_that_changes_nothing_is_not_caught(self):
        harmless = Mutant(name="identity", transform=lambda graph: graph, mistake="none")
        assert not caught_by_outputs(harmless.transform, mlp_graph())

    def test_a_mutant_that_empties_a_graph_is_caught(self):
        broken = Mutant(
            name="empty",
            transform=lambda graph: graph.with_nodes([]),
            mistake="removed everything",
        )
        assert caught_by_outputs(broken.transform, layernorm_graph())
