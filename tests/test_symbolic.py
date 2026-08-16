from __future__ import annotations

import pytest

from tgc.errors import ConfigError, TypeInferenceError
from tgc.ir.symbolic import (
    Expr,
    Facts,
    Guard,
    Quotient,
    a_fact_about_sixteen_does_not_settle_thirty_two,
    a_guard_that_fails,
    a_missing_value_is_refused,
    add,
    ceil_divide,
    coefficients_divide,
    constant,
    divides,
    divisibility_of_a_sum,
    equal,
    evaluate,
    evaluate_quotient,
    evaluation_agrees_with_the_symbols,
    facts_remove_guards,
    floor_divide,
    guards_for,
    multiply,
    needs_a_tail,
    normal_form_makes_equality_decidable,
    one_fact_answers_every_question,
    padding_always_divides,
    padding_instead_of_a_guard,
    rebuild,
    scale,
    symbol,
    the_identity_holds_when_it_divides,
    the_identity_that_is_false,
    the_rules_are_incomplete,
    tile_count,
    tiling_questions,
)


class TestExpressions:
    def test_a_constant_is_constant(self):
        assert constant(8).is_constant

    def test_a_symbol_is_not(self):
        assert not symbol("n").is_constant

    def test_a_sum_collects_like_terms(self):
        assert equal(add(symbol("n"), symbol("n")), scale(symbol("n"), 2))

    def test_a_product_expands(self):
        left = multiply(add(symbol("n"), constant(1)), add(symbol("n"), constant(1)))
        right = add(add(multiply(symbol("n"), symbol("n")), scale(symbol("n"), 2)), constant(1))
        assert equal(left, right)

    def test_two_spellings_of_a_square_compare_equal(self):
        # Both expand to the same flat sum, so equality needs no simplifier.
        assert normal_form_makes_equality_decidable()["equal"]

    def test_the_symbols_are_reported(self):
        assert multiply(symbol("n"), symbol("m")).symbols == ("m", "n")

    def test_an_unnamed_symbol_is_refused(self):
        with pytest.raises(ConfigError, match="needs a name"):
            symbol("")

    def test_a_term_with_no_symbols_is_refused(self):
        with pytest.raises(ConfigError, match="belongs in the constant"):
            Expr(terms=(((), 3),))

    def test_a_zero_coefficient_is_refused(self):
        with pytest.raises(ConfigError, match="zero coefficient"):
            Expr(terms=((("n",), 0),))

    def test_it_prints_readably(self):
        assert str(add(scale(symbol("n"), 2), constant(3))) == "2*n + 3"

    def test_a_bare_constant_prints_as_a_number(self):
        assert str(constant(7)) == "7"


class TestEvaluation:
    def test_the_symbolic_form_agrees_with_the_numbers(self):
        assert evaluation_agrees_with_the_symbols()["disagreements"] == 0

    def test_a_missing_value_is_refused(self):
        assert a_missing_value_is_refused()

    def test_a_quotient_evaluates_by_flooring(self):
        quotient = Quotient(numerator=symbol("n"), divisor=4)
        assert evaluate_quotient(quotient, {"n": 7}) == 1

    def test_or_by_ceiling(self):
        quotient = Quotient(numerator=symbol("n"), divisor=4, ceiling=True)
        assert evaluate_quotient(quotient, {"n": 7}) == 2

    def test_a_zero_divisor_is_refused(self):
        with pytest.raises(ConfigError, match="cannot divide by"):
            Quotient(numerator=symbol("n"), divisor=0)

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            evaluation_agrees_with_the_symbols(count=0)

    def test_evaluating_a_constant_needs_nothing(self):
        assert evaluate(constant(9), {}) == 9

    def test_a_quotient_prints_readably(self):
        assert str(Quotient(numerator=symbol("n"), divisor=4)) == "floordiv(n, 4)"


class TestDivisibility:
    def test_a_multiple_of_four_divides_by_four(self):
        assert divides(scale(symbol("n"), 4), 4)

    def test_and_by_two(self):
        assert divides(scale(symbol("n"), 4), 2)

    def test_but_not_by_eight(self):
        assert not divides(scale(symbol("n"), 4), 8)

    def test_a_fact_about_the_symbol_settles_it(self):
        facts = Facts(divisible_by={"n": 2})
        assert divides(scale(symbol("n"), 4), 8, facts)

    def test_a_fact_about_sixteen_covers_every_factor_of_sixteen(self):
        result = a_fact_about_sixteen_does_not_settle_thirty_two()
        assert result["multiple_of_eight"]
        assert result["multiple_of_sixteen"]

    def test_and_nothing_above_it(self):
        assert not a_fact_about_sixteen_does_not_settle_thirty_two()["multiple_of_thirty_two"]

    def test_a_sum_divides_term_by_term(self):
        result = divisibility_of_a_sum()
        assert result["symbol_plus_four"]
        assert not result["symbol_plus_two"]

    def test_the_rules_refuse_something_that_is_always_true(self):
        # The product of a number and the one after it is always even, and the term by term
        # test cannot see it.
        result = the_rules_are_incomplete()
        assert not result["proven_even"]
        assert result["actually_even_for_every_value"]

    def test_asking_about_multiples_of_zero_is_refused(self):
        with pytest.raises(ConfigError, match="cannot ask about"):
            divides(symbol("n"), 0)

    def test_a_fact_about_multiples_of_zero_is_refused(self):
        with pytest.raises(ConfigError, match="cannot ask about"):
            Facts().knows_multiple_of("n", 0)

    def test_facts_serialise(self):
        assert Facts(divisible_by={"n": 8}).as_dict()["divisible"] == {"n": 8}

    def test_a_symbol_with_no_bound_is_at_least_one(self):
        assert Facts().lower_bound("n") == 1


class TestDivision:
    def test_a_divisible_coefficient_is_carried_out(self):
        result = floor_divide(scale(symbol("n"), 8), 4)
        assert isinstance(result, Expr)
        assert equal(result, scale(symbol("n"), 2))

    def test_an_indivisible_one_is_kept(self):
        assert isinstance(floor_divide(symbol("n"), 4), Quotient)

    def test_even_when_the_value_is_known_to_divide(self):
        # The quotient of a symbol is not a polynomial in that symbol, whatever is known.
        facts = Facts(divisible_by={"n": 4})
        assert isinstance(floor_divide(symbol("n"), 4, facts), Quotient)

    def test_the_coefficient_test_is_stricter_than_the_value_test(self):
        facts = Facts(divisible_by={"n": 4})
        assert divides(symbol("n"), 4, facts)
        assert not coefficients_divide(symbol("n"), 4)

    def test_a_ceiling_division_is_kept_the_same_way(self):
        assert isinstance(ceil_divide(symbol("n"), 4), Quotient)

    def test_dividing_by_zero_is_refused(self):
        with pytest.raises(ConfigError, match="cannot divide by"):
            floor_divide(symbol("n"), 0)

    def test_and_so_is_a_ceiling_division_by_zero(self):
        with pytest.raises(ConfigError, match="cannot divide by"):
            ceil_divide(symbol("n"), 0)

    def test_asking_about_coefficients_of_zero_is_refused(self):
        with pytest.raises(ConfigError, match="cannot divide by"):
            coefficients_divide(symbol("n"), 0)


class TestTheIdentity:
    def test_dividing_and_multiplying_back_is_not_the_original(self):
        # It is the original rounded down, and the two agree only on the multiples.
        result = the_identity_that_is_false()
        assert not result["proven_equal"]
        assert result["counterexample"] == 1

    def test_and_the_difference_is_the_tail_a_tiling_pass_would_drop(self):
        assert the_identity_that_is_false()["short_by"] > 0

    def test_the_fact_buys_the_cancellation(self):
        result = the_identity_holds_when_it_divides()
        assert result["cancels"]

    def test_without_carrying_out_the_division(self):
        assert the_identity_holds_when_it_divides()["kept_as_a_quotient"]

    def test_and_without_the_fact_nothing_cancels(self):
        assert not the_identity_holds_when_it_divides()["cancels_without_the_fact"]

    def test_rebuilding_by_the_wrong_factor_is_refused(self):
        with pytest.raises(ConfigError, match="cancels nothing"):
            rebuild(Quotient(numerator=symbol("n"), divisor=4), 8)


class TestTiling:
    def test_a_dynamic_dimension_needs_a_tail(self):
        assert needs_a_tail(symbol("n"), 8)

    def test_and_a_known_multiple_does_not(self):
        assert not needs_a_tail(symbol("n"), 8, Facts(divisible_by={"n": 8}))

    def test_the_tile_count_is_a_ceiling_division(self):
        assert isinstance(tile_count(symbol("n"), 8), Quotient)

    def test_but_a_multiple_of_the_tile_divides_exactly(self):
        assert isinstance(tile_count(scale(symbol("n"), 8), 8), Expr)

    def test_one_fact_answers_every_tiling_question(self):
        result = one_fact_answers_every_question()
        assert result["unanswered_without_facts"] == result["questions"]
        assert result["unanswered_with_facts"] == 0

    def test_four_tiles_are_asked_about(self):
        assert len(tiling_questions()) == 4


class TestGuards:
    def test_a_guard_is_emitted_for_every_unproven_tiling(self):
        assert len(guards_for(symbol("n"), (2, 4, 8))) == 3

    def test_and_none_for_a_proven_one(self):
        facts = Facts(divisible_by={"n": 8})
        assert guards_for(symbol("n"), (2, 4, 8), facts) == []

    def test_one_fact_removes_four_of_five_guards(self):
        result = facts_remove_guards()
        assert result["without_facts"] == 5
        assert result["with_facts"] == 1

    def test_a_guard_passes_on_a_multiple(self):
        assert a_guard_that_fails()["passes_on_a_multiple"]

    def test_and_fails_otherwise(self):
        assert a_guard_that_fails()["fails_otherwise"]

    def test_a_guard_on_multiples_of_zero_is_refused(self):
        with pytest.raises(ConfigError, match="cannot guard"):
            Guard(expression=symbol("n"), factor=0)

    def test_it_serialises(self):
        assert Guard(expression=symbol("n"), factor=8).as_dict()["multiple_of"] == 8

    def test_guarding_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to guard"):
            guards_for(symbol("n"), ())


class TestPadding:
    def test_padding_a_dynamic_dimension_stays_a_quotient(self):
        assert isinstance(padding_instead_of_a_guard(symbol("n"), 8), Quotient)

    def test_padding_a_known_multiple_is_exact(self):
        result = padding_instead_of_a_guard(scale(symbol("n"), 8), 8)
        assert isinstance(result, Expr)

    def test_the_padded_length_always_divides(self):
        assert padding_always_divides()["all_divide"]

    def test_and_the_padding_is_never_more_than_a_tile(self):
        result = padding_always_divides(tile=8)
        assert result["largest_padding"] == 7

    def test_padding_to_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="cannot pad"):
            padding_instead_of_a_guard(symbol("n"), 0)

    def test_evaluating_a_missing_symbol_is_refused(self):
        with pytest.raises(TypeInferenceError, match="no value given"):
            evaluate(add(symbol("n"), symbol("m")), {"n": 4})
