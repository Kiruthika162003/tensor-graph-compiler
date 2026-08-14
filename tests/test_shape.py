from __future__ import annotations

import pytest

from tgc.errors import ConfigError, TypeInferenceError
from tgc.ir.shape import (
    Dim,
    Shape,
    broadcast,
    broadcast_all,
    contiguous_strides,
    dim,
    dims_equal,
    is_broadcastable,
    matmul_shape,
    normalise_axes,
    reduce_shape,
    reshape_shape,
    shape,
    transpose_shape,
)


class TestDim:
    def test_a_number_is_static(self):
        assert dim(8).is_static

    def test_a_name_is_not(self):
        assert not dim("batch").is_static

    def test_a_one_is_the_broadcastable_size(self):
        assert dim(1).is_one

    def test_an_unknown_dimension_is_not_known_to_be_one(self):
        # It might be one at runtime and might not, and a compiler that guesses gets the
        # shape right on the first batch it sees.
        assert not dim("batch").is_one

    def test_a_dimension_needs_a_size_or_a_name(self):
        with pytest.raises(ConfigError, match="either a size or a name"):
            Dim()

    def test_it_cannot_be_both(self):
        with pytest.raises(ConfigError, match="cannot be both"):
            Dim(value=4, name="batch")

    def test_a_negative_size_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            Dim(value=-1)

    def test_two_numbers_compare_by_value(self):
        assert dims_equal(dim(8), dim(8))
        assert not dims_equal(dim(8), dim(4))

    def test_two_names_compare_by_name(self):
        assert dims_equal(dim("batch"), dim("batch"))
        assert not dims_equal(dim("batch"), dim("sequence"))

    def test_a_name_never_equals_a_number(self):
        # Certainly the same, not probably. Treating them as equal turns a shape error into
        # a wrong answer.
        assert not dims_equal(dim("batch"), dim(8))


class TestShape:
    def test_the_rank_is_the_dimension_count(self):
        assert shape(2, 3, 4).rank == 3

    def test_the_element_count_is_the_product(self):
        assert shape(2, 3, 4).elements == 24

    def test_a_scalar_holds_one_element(self):
        assert Shape().elements == 1

    def test_a_symbolic_shape_has_no_element_count(self):
        with pytest.raises(TypeInferenceError, match="not known until it runs"):
            _ = shape("batch", 4).elements

    def test_a_static_shape_knows_its_size(self):
        assert shape(2, 3).is_static
        assert not shape("batch", 3).is_static

    def test_the_storage_follows_the_element_size(self):
        assert shape(2, 3).bytes_for(4) == 24

    def test_a_zero_byte_element_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one byte"):
            shape(2).bytes_for(0)

    def test_it_prints_readably(self):
        assert str(shape(2, "batch")) == "[2, batch]"


class TestBroadcast:
    def test_equal_shapes_stay_equal(self):
        assert broadcast(shape(3, 4), shape(3, 4)) == shape(3, 4)

    def test_a_one_stretches(self):
        assert broadcast(shape(3, 1), shape(3, 4)) == shape(3, 4)

    def test_a_shorter_shape_aligns_from_the_right(self):
        assert broadcast(shape(4), shape(3, 4)) == shape(3, 4)

    def test_a_scalar_broadcasts_against_anything(self):
        assert broadcast(Shape(), shape(3, 4)) == shape(3, 4)

    def test_incompatible_sizes_are_refused(self):
        with pytest.raises(TypeInferenceError, match="cannot broadcast"):
            broadcast(shape(3), shape(4))

    def test_a_symbolic_dimension_against_a_number_is_refused(self):
        # Without knowing whether either is one there is no answer, and picking one is how a
        # compiled graph produces the wrong shape for its second batch.
        with pytest.raises(TypeInferenceError, match="whether either is one"):
            broadcast(shape("batch"), shape(4))

    def test_a_symbolic_dimension_against_itself_is_fine(self):
        assert broadcast(shape("batch", 4), shape("batch", 1)) == shape("batch", 4)

    def test_a_symbolic_dimension_against_a_one_is_fine(self):
        assert broadcast(shape("batch"), shape(1)) == shape("batch")

    def test_it_folds_across_several(self):
        assert broadcast_all([shape(1, 4), shape(3, 1), shape(3, 4)]) == shape(3, 4)

    def test_broadcasting_nothing_is_rejected(self):
        with pytest.raises(TypeInferenceError, match="nothing to broadcast"):
            broadcast_all([])

    def test_the_predicate_agrees_with_the_operation(self):
        assert is_broadcastable(shape(3, 1), shape(3, 4))
        assert not is_broadcastable(shape(3), shape(4))


class TestMatmul:
    def test_two_matrices_meet_in_the_middle(self):
        assert matmul_shape(shape(3, 4), shape(4, 5)) == shape(3, 5)

    def test_dimensions_that_do_not_meet_are_refused(self):
        with pytest.raises(TypeInferenceError, match="do not meet"):
            matmul_shape(shape(3, 4), shape(5, 6))

    def test_leading_dimensions_broadcast(self):
        assert matmul_shape(shape(2, 3, 4), shape(4, 5)) == shape(2, 3, 5)

    def test_a_vector_is_refused(self):
        with pytest.raises(TypeInferenceError, match="two matrices"):
            matmul_shape(shape(4), shape(4, 5))

    def test_a_symbolic_batch_survives(self):
        assert matmul_shape(shape("batch", 3, 4), shape(4, 5)) == shape("batch", 3, 5)


class TestReduce:
    def test_reducing_an_axis_removes_it(self):
        assert reduce_shape(shape(2, 3, 4), [1]) == shape(2, 4)

    def test_keeping_dimensions_leaves_a_one(self):
        assert reduce_shape(shape(2, 3, 4), [1], keepdims=True) == shape(2, 1, 4)

    def test_several_axes_go_at_once(self):
        assert reduce_shape(shape(2, 3, 4), [0, 2]) == shape(3)

    def test_reducing_everything_gives_a_scalar(self):
        assert reduce_shape(shape(2, 3), [0, 1]).rank == 0

    def test_a_negative_axis_counts_from_the_end(self):
        assert normalise_axes([-1], 3) == (2,)

    def test_an_axis_outside_the_rank_is_rejected(self):
        with pytest.raises(ConfigError, match="outside a tensor"):
            normalise_axes([3], 3)

    def test_a_repeated_axis_is_rejected(self):
        with pytest.raises(ConfigError, match="given twice"):
            normalise_axes([1, -2], 3)

    def test_the_axes_come_back_sorted(self):
        assert normalise_axes([2, 0], 3) == (0, 2)


class TestViews:
    def test_a_permutation_reorders_the_dimensions(self):
        assert transpose_shape(shape(2, 3, 4), [2, 0, 1]) == shape(4, 2, 3)

    def test_a_permutation_has_to_use_every_axis(self):
        with pytest.raises(ConfigError, match="each axis once"):
            transpose_shape(shape(2, 3), [0, 0])

    def test_a_reshape_keeps_the_element_count(self):
        assert reshape_shape(shape(2, 6), [3, 4]) == shape(3, 4)

    def test_a_reshape_that_loses_elements_is_refused(self):
        with pytest.raises(TypeInferenceError, match="cannot reshape"):
            reshape_shape(shape(2, 6), [5, 5])

    def test_one_dimension_can_be_inferred(self):
        assert reshape_shape(shape(2, 6), [3, -1]) == shape(3, 4)

    def test_two_inferred_dimensions_are_refused(self):
        with pytest.raises(ConfigError, match="at most one"):
            reshape_shape(shape(2, 6), [-1, -1])

    def test_a_zero_target_is_refused(self):
        with pytest.raises(ConfigError, match="positive or -1"):
            reshape_shape(shape(2, 6), [0, 12])

    def test_an_inferred_dimension_that_does_not_divide_is_refused(self):
        with pytest.raises(TypeInferenceError, match="cannot reshape"):
            reshape_shape(shape(2, 6), [5, -1])


class TestStrides:
    def test_a_packed_tensor_strides_by_the_trailing_sizes(self):
        assert contiguous_strides(shape(2, 3, 4)) == (12, 4, 1)

    def test_a_vector_strides_by_one(self):
        assert contiguous_strides(shape(5)) == (1,)

    def test_a_scalar_has_no_strides(self):
        assert contiguous_strides(Shape()) == ()

    def test_a_symbolic_shape_has_no_strides_yet(self):
        with pytest.raises(TypeInferenceError, match="not known until it runs"):
            contiguous_strides(shape("batch", 4))
