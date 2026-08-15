from __future__ import annotations

import pytest

from tgc.analysis.dependence import (
    MAYBE_DEPENDENCE,
    NO_DEPENDENCE,
    DependenceResult,
    Distance,
    LoopSummary,
    Subscript,
    brute_force,
    classify,
    distance_between,
    gcd_never_says_no_when_there_is_one,
    gcd_test,
    imprecision_falls_with_extent,
    loop_is_parallel,
    worked_examples,
)
from tgc.errors import ConfigError, ScheduleError


class TestSubscript:
    def test_it_reaches_the_element_it_describes(self):
        assert Subscript(stride=2, offset=3).at(4) == 11

    def test_a_plain_index_prints_as_itself(self):
        assert str(Subscript(stride=1)) == "i"

    def test_a_stride_prints_without_an_offset(self):
        assert str(Subscript(stride=2)) == "2i"

    def test_a_negative_offset_prints_as_a_subtraction(self):
        assert str(Subscript(stride=1, offset=-1)) == "1i - 1"

    def test_a_positive_offset_prints_as_an_addition(self):
        assert str(Subscript(stride=2, offset=3)) == "2i + 3"


class TestGcdTest:
    def test_even_and_odd_indices_provably_never_meet(self):
        # The answer the test exists to give, and the only one it gives with certainty.
        result = gcd_test(Subscript(stride=2), Subscript(stride=2, offset=1))
        assert result.is_independent
        assert "does not divide" in result.reason

    def test_the_same_subscript_might_collide(self):
        assert not gcd_test(Subscript(stride=1), Subscript(stride=1)).is_independent

    def test_two_different_fixed_elements_never_collide(self):
        result = gcd_test(Subscript(stride=0, offset=1), Subscript(stride=0, offset=2))
        assert result.is_independent

    def test_the_same_fixed_element_does(self):
        result = gcd_test(Subscript(stride=0, offset=1), Subscript(stride=0, offset=1))
        assert not result.is_independent

    def test_maybe_blocks_a_transformation(self):
        # A compiler that treats maybe as no is not conservative, it is wrong.
        assert DependenceResult(verdict=MAYBE_DEPENDENCE).blocks_reordering

    def test_and_no_does_not(self):
        assert not DependenceResult(verdict=NO_DEPENDENCE).blocks_reordering

    def test_an_unknown_verdict_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown verdict"):
            DependenceResult(verdict="definitely")

    def test_it_serialises(self):
        assert gcd_test(Subscript(2), Subscript(2, 1)).as_dict()["independent"]


class TestSoundness:
    def test_the_test_never_says_no_when_a_collision_exists(self):
        # A single such case would make every transformation resting on it wrong.
        assert gcd_never_says_no_when_there_is_one()["unsound"] == 0

    def test_and_that_holds_at_every_loop_length(self):
        assert all(row["unsound"] == 0 for row in imprecision_falls_with_extent())

    def test_the_sweep_covers_thousands_of_pairs(self):
        assert gcd_never_says_no_when_there_is_one()["checked"] > 2000

    def test_a_zero_extent_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            gcd_never_says_no_when_there_is_one(extent=0)

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            imprecision_falls_with_extent(extents=())


class TestImprecision:
    def test_the_test_is_conservative_on_a_short_loop(self):
        # It ignores the loop bounds, so it is conservative exactly when the colliding
        # iterations exist as integers and lie outside the range the loop runs.
        rows = {row["extent"]: row for row in imprecision_falls_with_extent()}
        assert rows[2]["conservative_rate"] > 0.3

    def test_and_exact_on_a_long_one(self):
        rows = {row["extent"]: row for row in imprecision_falls_with_extent()}
        assert rows[24]["conservative_rate"] == 0.0

    def test_the_rate_falls_as_the_loop_grows(self):
        rates = [row["conservative_rate"] for row in imprecision_falls_with_extent()]
        assert rates == sorted(rates, reverse=True)

    def test_a_distant_offset_is_the_clearest_case(self):
        # A collision at iteration one hundred is real and unreachable in a loop of four.
        write, read = Subscript(stride=1), Subscript(stride=1, offset=-100)
        assert not gcd_test(write, read).is_independent
        assert brute_force(write, read, 4).is_independent


class TestBruteForce:
    def test_it_finds_a_colliding_pair(self):
        result = brute_force(Subscript(stride=1), Subscript(stride=1), 4)
        assert result.witness is not None

    def test_and_reports_none_when_there_is_none(self):
        result = brute_force(Subscript(stride=2), Subscript(stride=2, offset=1), 8)
        assert result.is_independent

    def test_a_zero_extent_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            brute_force(Subscript(1), Subscript(1), 0)


class TestDistance:
    def test_reading_the_same_element_is_distance_zero(self):
        assert distance_between(Subscript(1), Subscript(1), 32).value == 0

    def test_and_permits_parallelism(self):
        assert distance_between(Subscript(1), Subscript(1), 32).permits_parallelism

    def test_reading_one_step_behind_is_distance_one(self):
        assert distance_between(Subscript(1), Subscript(1, -1), 32).value == 1

    def test_and_is_loop_carried(self):
        # Iteration k reads what iteration k minus one wrote, so running them together gets
        # one of the two answers at random.
        assert distance_between(Subscript(1), Subscript(1, -1), 32).is_loop_carried

    def test_mismatched_strides_have_no_single_distance(self):
        # Calling a varying distance a distance is where dependence analysis usually goes
        # wrong.
        with pytest.raises(ScheduleError, match="strides match"):
            distance_between(Subscript(2), Subscript(3), 32)

    def test_a_zero_extent_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            distance_between(Subscript(1), Subscript(1), 0)

    def test_an_unknown_distance_permits_nothing(self):
        assert not Distance().permits_parallelism
        assert not Distance().is_loop_carried

    def test_it_serialises(self):
        assert Distance(value=0).as_dict()["parallel"]


class TestParallelism:
    def test_a_loop_reading_what_it_writes_is_parallel(self):
        assert loop_is_parallel(Subscript(1), Subscript(1), 32)

    def test_a_loop_reading_behind_itself_is_not(self):
        assert not loop_is_parallel(Subscript(1), Subscript(1, -1), 32)

    def test_provably_disjoint_accesses_are_parallel(self):
        assert loop_is_parallel(Subscript(2), Subscript(2, 1), 32)

    def test_mismatched_strides_are_treated_as_blocking(self):
        assert not loop_is_parallel(Subscript(2), Subscript(3), 32)

    def test_the_worked_examples_cover_both_answers(self):
        rows = worked_examples()
        assert any(row["parallel"] for row in rows)
        assert any(not row["parallel"] for row in rows)

    def test_classify_names_the_subscripts(self):
        row = classify(Subscript(2), Subscript(2, 1))
        assert row["write"] == "2i"
        assert row["read"] == "2i + 1"


class TestSummary:
    def test_a_loop_of_parallel_accesses_is_parallel(self):
        summary = LoopSummary(
            accesses=[(Subscript(1), Subscript(1)), (Subscript(2), Subscript(2, 1))]
        )
        assert summary.is_parallel
        assert summary.blocking_pairs == []

    def test_one_blocking_pair_is_enough_to_stop_it(self):
        summary = LoopSummary(
            accesses=[(Subscript(1), Subscript(1)), (Subscript(1), Subscript(1, -1))]
        )
        assert not summary.is_parallel
        assert len(summary.blocking_pairs) == 1

    def test_an_empty_loop_is_parallel(self):
        assert LoopSummary().is_parallel

    def test_a_zero_extent_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            LoopSummary(extent=0)

    def test_it_serialises(self):
        summary = LoopSummary(accesses=[(Subscript(1), Subscript(1))])
        assert summary.as_dict()["parallel"]
