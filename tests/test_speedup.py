from __future__ import annotations

import pytest

from tgc.analysis.speedup import (
    FOLDING,
    FUSION,
    LAYOUT,
    PASSES,
    STRONG_PASSES,
    TILING,
    VECTORISE,
    Optimisation,
    Profile,
    a_pass_that_claims_nothing_is_refused,
    a_perfect_pass_on_a_small_category,
    affordable,
    an_empty_profile_is_refused,
    an_unknown_category_is_refused,
    apply_all,
    apply_one,
    best_under_budget,
    budget_sweep,
    but_the_credit_does,
    ceiling_for,
    compare_profiles,
    end_to_end,
    greedy_under_budget,
    memory_bound_step,
    rank_by_claim,
    rank_by_saving,
    saving,
    savings_do_not_add,
    speedups_multiply_on_disjoint_categories,
    the_ceiling_is_the_share,
    the_greedy_selection_matches_the_search,
    the_margin_depends_on_the_program,
    the_order_does_not_change_the_result,
    the_rate_and_the_saving_point_different_ways,
    the_rule_loses_when_the_expensive_pass_is_good_enough,
    the_two_rankings_disagree,
    transformer_step,
)
from tgc.errors import ConfigError


class TestProfiles:
    def test_the_step_adds_up_to_a_hundred_milliseconds(self):
        assert transformer_step().total == 100.0

    def test_the_products_are_more_than_half(self):
        assert transformer_step().share_of("matmul") > 0.5

    def test_the_memory_bound_step_is_the_other_way_round(self):
        assert memory_bound_step().share_of("elementwise") > memory_bound_step().share_of(
            "matmul"
        )

    def test_an_empty_profile_is_refused(self):
        assert an_empty_profile_is_refused()

    def test_a_repeated_category_is_refused(self):
        with pytest.raises(ConfigError, match="appears twice"):
            Profile(times=(("matmul", 1.0), ("matmul", 2.0)))

    def test_a_negative_time_is_refused(self):
        with pytest.raises(ConfigError, match="cannot take"):
            Profile(times=(("matmul", -1.0),))

    def test_a_step_that_takes_no_time_is_refused(self):
        with pytest.raises(ConfigError, match="some time somewhere"):
            Profile(times=(("matmul", 0.0), ("copy", 0.0)))

    def test_an_unknown_category_is_refused(self):
        with pytest.raises(ConfigError, match="unknown category"):
            transformer_step().time_in("nowhere")

    def test_it_serialises(self):
        assert transformer_step().as_dict()["matmul"] == 55.0


class TestPasses:
    def test_a_pass_that_claims_nothing_is_refused(self):
        assert a_pass_that_claims_nothing_is_refused()

    def test_a_zero_factor_is_refused(self):
        with pytest.raises(ConfigError, match="claims a factor"):
            Optimisation(name="bad", effects=(("matmul", 0.0),), compile_cost=1.0)

    def test_a_negative_compile_cost_is_refused(self):
        with pytest.raises(ConfigError, match="cannot cost"):
            Optimisation(name="bad", effects=(("matmul", 2.0),), compile_cost=-1.0)

    def test_a_pass_naming_a_missing_category_is_refused(self):
        assert an_unknown_category_is_refused()

    def test_applying_a_pass_only_touches_its_categories(self):
        after = apply_one(transformer_step(), VECTORISE)
        assert after.time_in("matmul") == 55.0

    def test_and_divides_the_one_it_names(self):
        after = apply_one(transformer_step(), VECTORISE)
        assert abs(after.time_in("reduction") - 11.0 / 1.8) < 1e-12

    def test_a_pass_touching_two_categories_divides_both(self):
        after = apply_one(transformer_step(), FUSION)
        assert after.time_in("elementwise") == 8.8
        assert after.time_in("overhead") == 2.5

    def test_the_largest_claim_is_the_one_reported(self):
        assert LAYOUT.largest_claim == 3.0

    def test_it_serialises(self):
        assert FUSION.as_dict()["compile_cost"] == 3.0


class TestRankings:
    def test_the_two_rankings_disagree(self):
        assert not the_two_rankings_disagree()["same_order"]

    def test_the_biggest_claim_is_not_the_biggest_saving(self):
        result = the_two_rankings_disagree()
        assert result["top_claim"] != result["top_saving"]

    def test_layout_claims_the_most(self):
        assert rank_by_claim()[0]["pass"] == "layout"

    def test_and_comes_third_by_saving(self):
        assert [row["pass"] for row in rank_by_saving()].index("layout") == 2

    def test_fusion_saves_the_most(self):
        assert rank_by_saving()[0]["pass"] == "fusion"

    def test_every_pass_appears_in_both_rankings(self):
        assert len(rank_by_claim()) == len(rank_by_saving()) == len(PASSES)


class TestCeilings:
    def test_the_ceiling_is_one_over_what_is_left(self):
        profile = transformer_step()
        assert abs(ceiling_for(profile, "matmul") - 1 / 0.45) < 1e-12

    def test_the_products_cap_at_two_and_a_quarter(self):
        rows = {row["category"]: row for row in the_ceiling_is_the_share()}
        assert 2.2 < rows["matmul"]["ceiling"] < 2.25

    def test_the_overhead_caps_at_five_percent(self):
        rows = {row["category"]: row for row in the_ceiling_is_the_share()}
        assert rows["overhead"]["ceiling"] < 1.06

    def test_a_category_that_is_the_whole_step_has_no_ceiling(self):
        with pytest.raises(ConfigError, match="no ceiling"):
            ceiling_for(Profile(times=(("matmul", 10.0),)), "matmul")

    def test_deleting_the_overhead_entirely_loses_to_fusion(self):
        assert a_perfect_pass_on_a_small_category()["fusion_wins"]

    def test_by_a_factor_of_three(self):
        result = a_perfect_pass_on_a_small_category()
        assert result["fusion_saving"] / result["perfect_saving"] > 3.0

    def test_every_category_has_a_ceiling(self):
        assert len(the_ceiling_is_the_share()) == len(transformer_step().categories)


class TestComposition:
    def test_the_savings_do_not_add(self):
        result = savings_do_not_add()
        assert result["measured"] != result["sum_of_parts"]

    def test_and_adding_them_overstates_the_result(self):
        assert savings_do_not_add()["sum_is_higher"]

    def test_by_about_a_point(self):
        assert 0.005 < savings_do_not_add()["overstated_by"] < 0.02

    def test_the_step_times_compose_exactly(self):
        assert speedups_multiply_on_disjoint_categories()["identical"]

    def test_the_order_does_not_change_the_result(self):
        assert the_order_does_not_change_the_result()["identical"]

    def test_but_the_credit_does(self):
        assert not but_the_credit_does()["same"]

    def test_and_the_later_measurement_is_the_flattering_one(self):
        result = but_the_credit_does()
        assert result["applied_last"] > result["applied_first"]

    def test_applying_nothing_changes_nothing(self):
        profile = transformer_step()
        assert apply_all(profile, []).total == profile.total

    def test_and_saves_nothing(self):
        assert saving(transformer_step(), []) == 0.0

    def test_the_whole_set_is_worth_forty_percent(self):
        assert 0.4 < saving(transformer_step(), PASSES) < 0.41

    def test_which_is_a_speedup_of_one_and_two_thirds(self):
        assert 1.65 < end_to_end(transformer_step(), PASSES) < 1.7


class TestBudget:
    def test_a_set_that_fits_is_affordable(self):
        assert affordable([VECTORISE, FOLDING], 2.0)

    def test_a_set_that_does_not_is_not(self):
        assert not affordable([TILING, FUSION], 10.0)

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ConfigError, match="not a budget"):
            affordable([FOLDING], -1.0)

    def test_an_empty_pass_set_is_refused(self):
        with pytest.raises(ConfigError, match="no passes to choose from"):
            best_under_budget(transformer_step(), 5.0, ())

    def test_and_so_is_an_empty_set_for_the_rule(self):
        with pytest.raises(ConfigError, match="no passes to choose from"):
            greedy_under_budget(transformer_step(), 5.0, ())

    def test_a_budget_of_nothing_buys_nothing(self):
        assert greedy_under_budget(transformer_step(), 0.0) == ()

    def test_the_search_never_exceeds_the_budget(self):
        for budget in (1.0, 4.0, 9.0):
            assert affordable(best_under_budget(transformer_step(), budget), budget)

    def test_and_neither_does_the_rule(self):
        for budget in (1.0, 4.0, 9.0):
            assert affordable(greedy_under_budget(transformer_step(), budget), budget)

    def test_the_rule_matches_the_search_at_every_budget(self):
        assert the_greedy_selection_matches_the_search()["losing"] == []

    def test_with_no_gap_at_all(self):
        assert the_greedy_selection_matches_the_search()["worst_gap"] == 0.0

    def test_the_sweep_buys_more_as_the_budget_grows(self):
        counts = [len(row["greedy_passes"]) for row in budget_sweep()]
        assert counts == sorted(counts)

    def test_an_empty_budget_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            budget_sweep(budgets=())


class TestKnapsack:
    def test_the_rule_loses_once_the_expensive_pass_is_good_enough(self):
        assert the_rule_loses_when_the_expensive_pass_is_good_enough()["gap"] > 0.02

    def test_by_buying_four_cheap_passes_instead_of_one(self):
        result = the_rule_loses_when_the_expensive_pass_is_good_enough()
        assert result["best_passes"] == ["strong tiling"]
        assert len(result["greedy_passes"]) == 4

    def test_the_expensive_pass_has_the_best_saving(self):
        assert the_rate_and_the_saving_point_different_ways()["best_saving"] == "strong tiling"

    def test_and_three_passes_have_a_better_rate(self):
        result = the_rate_and_the_saving_point_different_ways()
        assert len(result["better_rate_than_tiling"]) == 3

    def test_which_together_cost_most_of_the_budget(self):
        assert the_rate_and_the_saving_point_different_ways()["their_total_cost"] == 6.0

    def test_a_large_enough_budget_makes_the_question_moot(self):
        profile = transformer_step()
        best = best_under_budget(profile, 30.0, STRONG_PASSES)
        rule = greedy_under_budget(profile, 30.0, STRONG_PASSES)
        assert abs(saving(profile, best) - saving(profile, rule)) < 1e-12


class TestPrograms:
    def test_the_same_pass_wins_on_both_steps(self):
        assert the_margin_depends_on_the_program()["same_winner"]

    def test_but_fusion_is_worth_twice_as_much_on_the_memory_bound_one(self):
        assert the_margin_depends_on_the_program()["fusion_ratio"] > 2.0

    def test_and_tiling_a_seventh(self):
        assert the_margin_depends_on_the_program()["tiling_ratio"] < 0.15

    def test_so_the_gap_between_them_widens(self):
        assert the_margin_depends_on_the_program()["gap_widened"]

    def test_two_profiles_are_compared(self):
        assert len(compare_profiles()) == 2

    def test_the_whole_set_is_worth_more_on_the_memory_bound_step(self):
        rows = {row["profile"]: row for row in compare_profiles()}
        assert rows["memory bound"]["all_of_them"] > rows["transformer"]["all_of_them"]
