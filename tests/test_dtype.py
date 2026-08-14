from __future__ import annotations

import pytest

from tgc.errors import ConfigError, TypeInferenceError
from tgc.ir.dtype import (
    BFLOAT16,
    BOOL,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT32,
    INT64,
    DType,
    accumulator_for,
    can_represent_exactly,
    from_name,
    promote,
    promote_all,
)


class TestType:
    def test_the_width_gives_the_storage(self):
        assert FLOAT32.bytes == 4
        assert FLOAT16.bytes == 2

    def test_a_boolean_still_takes_a_byte(self):
        assert BOOL.bytes == 1

    def test_floats_round_and_integers_do_not(self):
        assert FLOAT32.is_float
        assert INT32.is_integral
        assert not INT32.is_float

    def test_a_type_with_no_width_is_rejected(self):
        with pytest.raises(ConfigError, match="no width"):
            DType(name="nothing", bits=0, kind="float")

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown kind"):
            DType(name="odd", bits=8, kind="complex")

    def test_it_looks_up_by_name(self):
        assert from_name("float32") is FLOAT32

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown dtype"):
            from_name("float8")

    def test_it_prints_as_its_name(self):
        assert str(FLOAT32) == "float32"


class TestPromotion:
    def test_a_type_with_itself_is_itself(self):
        assert promote(FLOAT32, FLOAT32) is FLOAT32

    def test_the_wider_float_wins(self):
        assert promote(FLOAT32, FLOAT64) is FLOAT64

    def test_float_beats_integer_regardless_of_width(self):
        # The rule nobody expects: an int64 combined with a float16 gives float16, not
        # something wider. Every framework does this and none of them say so out loud.
        assert promote(INT64, FLOAT16) is FLOAT16

    def test_the_wider_integer_wins(self):
        assert promote(INT8, INT64) is INT64

    def test_a_boolean_promotes_to_anything(self):
        assert promote(BOOL, INT32) is INT32

    def test_the_two_half_widths_have_no_answer(self):
        # Same width, neither contains the other, so the graph has to say which it wants
        # rather than have the compiler pick one and be quietly wrong half the time.
        with pytest.raises(TypeInferenceError, match="same width"):
            promote(FLOAT16, BFLOAT16)

    def test_promotion_folds_across_several(self):
        assert promote_all([INT8, INT32, FLOAT32]) is FLOAT32

    def test_promoting_nothing_is_rejected(self):
        with pytest.raises(TypeInferenceError, match="nothing to promote"):
            promote_all([])

    def test_it_does_not_matter_which_way_round(self):
        for left in (BOOL, INT8, INT32, INT64, FLOAT32, FLOAT64):
            for right in (BOOL, INT8, INT32, INT64, FLOAT32, FLOAT64):
                assert promote(left, right) is promote(right, left)


class TestRepresentation:
    def test_a_small_integer_survives_a_half_precision_round_trip(self):
        assert can_represent_exactly(FLOAT16, 2048)

    def test_a_larger_one_does_not(self):
        # Which is what makes folding an index calculation into a half precision literal
        # produce a graph that was correct before it was compiled.
        assert not can_represent_exactly(FLOAT16, 2049)

    def test_float32_counts_much_further(self):
        assert can_represent_exactly(FLOAT32, 2**24)
        assert not can_represent_exactly(FLOAT32, 2**24 + 1)

    def test_an_integer_type_holds_its_range(self):
        assert can_represent_exactly(INT8, 127)
        assert not can_represent_exactly(INT8, 128)

    def test_a_boolean_holds_two_values(self):
        assert can_represent_exactly(BOOL, 1)
        assert not can_represent_exactly(BOOL, 2)

    def test_negative_values_are_bounded_too(self):
        assert can_represent_exactly(INT8, -128)
        assert not can_represent_exactly(INT8, -129)


class TestAccumulator:
    def test_a_narrow_float_accumulates_wider(self):
        # Summing ten thousand float16 values in float16 stalls once the running total is
        # large enough that each addend falls below its last bit.
        assert accumulator_for(FLOAT16) is FLOAT32
        assert accumulator_for(BFLOAT16) is FLOAT32

    def test_a_wide_float_accumulates_in_itself(self):
        assert accumulator_for(FLOAT32) is FLOAT32
        assert accumulator_for(FLOAT64) is FLOAT64

    def test_a_narrow_integer_accumulates_wider(self):
        assert accumulator_for(INT8) is INT32

    def test_a_wide_integer_accumulates_in_itself(self):
        assert accumulator_for(INT64) is INT64
