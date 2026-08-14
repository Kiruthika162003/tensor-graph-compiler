from __future__ import annotations

import pytest

from tgc.codegen.loops import (
    Loop,
    LoopNest,
    check_transformations,
    elementwise_nest,
    matmul_nest,
    peel,
    peeled_iteration_count,
    preserves_iterations,
    remainder_report,
    split,
    swap,
    tile_nest,
    unroll,
    vectorise,
    visits_the_same_points,
)
from tgc.errors import CodegenError, ConfigError

UNEVEN = LoopNest(
    loops=[
        Loop(variable="i", extent=2),
        Loop(variable="j", extent=3),
        Loop(variable="k", extent=4),
    ],
    body="work",
)


class TestLoop:
    def test_a_loop_runs_its_extent_by_default(self):
        assert Loop(variable="i", extent=16).iterations == 16

    def test_a_step_divides_the_iteration_count(self):
        assert Loop(variable="i", extent=16, step=4).iterations == 4

    def test_a_step_that_does_not_divide_rounds_up(self):
        assert Loop(variable="i", extent=13, step=4).iterations == 4

    def test_a_full_loop_has_no_remainder(self):
        assert Loop(variable="i", extent=16, step=4).is_full
        assert not Loop(variable="i", extent=13, step=4).is_full

    def test_the_indices_are_what_the_variable_takes(self):
        assert Loop(variable="i", extent=8, step=3).indices() == [0, 3, 6]

    def test_a_nameless_loop_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a variable"):
            Loop(variable="", extent=4)

    def test_a_negative_extent_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot run"):
            Loop(variable="i", extent=-1)

    def test_a_zero_step_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot step by"):
            Loop(variable="i", extent=4, step=0)

    def test_it_serialises(self):
        assert Loop(variable="i", extent=16, step=4).as_dict()["iterations"] == 4


class TestNest:
    def test_the_total_is_the_product(self):
        assert matmul_nest(4, 8, 16).total_iterations == 512

    def test_the_depth_is_the_loop_count(self):
        assert matmul_nest().depth == 3

    def test_the_variables_run_outermost_first(self):
        assert matmul_nest().variables() == ["i", "j", "k"]

    def test_a_repeated_variable_is_rejected(self):
        with pytest.raises(ConfigError, match="share a variable"):
            LoopNest(loops=[Loop(variable="i", extent=4), Loop(variable="i", extent=4)])

    def test_an_unknown_loop_is_rejected(self):
        with pytest.raises(CodegenError, match="no loop named"):
            matmul_nest().loop("z")

    def test_the_iteration_space_is_enumerated_in_order(self):
        nest = LoopNest(loops=[Loop(variable="i", extent=2), Loop(variable="j", extent=2)])
        assert nest.iteration_space() == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_it_prints_as_nested_loops(self):
        text = str(elementwise_nest(4, 4))
        assert text.startswith("for i in range(0, 4, 1):")
        assert "y[i, j] = f(x[i, j])" in text

    def test_it_serialises(self):
        assert matmul_nest().as_dict()["depth"] == 3


class TestSplit:
    def test_a_split_adds_a_loop(self):
        assert split(matmul_nest(), "k", 8).depth == 4

    def test_and_keeps_the_iteration_count(self):
        nest = matmul_nest(16, 16, 16)
        assert preserves_iterations(nest, split(nest, "k", 4))

    def test_the_outer_loop_steps_by_the_factor(self):
        nest = split(matmul_nest(16, 16, 16), "k", 4)
        assert nest.loop("k_outer").step == 4

    def test_the_inner_loop_covers_one_block(self):
        nest = split(matmul_nest(16, 16, 16), "k", 4)
        assert nest.loop("k_inner").extent == 4

    def test_a_factor_larger_than_the_extent_is_refused(self):
        with pytest.raises(CodegenError, match="cannot split"):
            split(matmul_nest(4, 4, 4), "k", 8)

    def test_a_zero_factor_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            split(matmul_nest(), "k", 0)


class TestSwap:
    def test_a_swap_keeps_the_depth(self):
        assert swap(matmul_nest(), "i", "k").depth == 3

    def test_and_visits_the_same_points(self):
        assert visits_the_same_points(UNEVEN, swap(UNEVEN, "i", "k"))

    def test_and_changes_the_order_they_are_visited_in(self):
        # Which is the entire content of the transformation, and is invisible to a count.
        assert UNEVEN.iteration_space() != swap(UNEVEN, "i", "k").iteration_space()

    def test_the_variables_are_exchanged(self):
        assert swap(UNEVEN, "i", "k").variables() == ["k", "j", "i"]

    def test_swapping_twice_returns_the_original(self):
        once = swap(UNEVEN, "i", "k")
        assert swap(once, "i", "k").variables() == UNEVEN.variables()

    def test_a_positional_comparison_would_have_missed_it(self):
        # Three loops of equal extent produce the same positional tuples after a swap, because
        # only the meaning of the positions changed. Keying the point by variable is what
        # makes the check about the swap rather than about the extents.
        square = matmul_nest(4, 4, 4)
        assert square.iteration_space() == swap(square, "i", "k").iteration_space()
        assert visits_the_same_points(square, swap(square, "i", "k"))

    def test_comparing_nests_over_different_variables_is_refused(self):
        with pytest.raises(CodegenError, match="different variables"):
            visits_the_same_points(UNEVEN, split(UNEVEN, "k", 2))


class TestUnroll:
    def test_unrolling_does_not_change_the_iteration_count(self):
        nest = matmul_nest(16, 16, 16)
        assert preserves_iterations(nest, unroll(nest, "k", 4))

    def test_it_records_the_factor(self):
        assert unroll(matmul_nest(), "k", 4).loop("k").unrolled == 4

    def test_an_unroll_past_the_iteration_count_is_refused(self):
        with pytest.raises(CodegenError, match="cannot unroll"):
            unroll(matmul_nest(4, 4, 4), "k", 8)

    def test_a_zero_factor_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            unroll(matmul_nest(), "k", 0)

    def test_it_shows_in_the_printed_form(self):
        assert "unroll 4" in str(unroll(matmul_nest(), "k", 4))


class TestVectorise:
    def test_the_innermost_loop_can_be_vectorised(self):
        assert vectorise(matmul_nest(), "k").loop("k").vectorised

    def test_an_outer_loop_cannot(self):
        with pytest.raises(CodegenError, match="not the innermost"):
            vectorise(matmul_nest(), "i")

    def test_a_zero_width_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            vectorise(matmul_nest(), "k", width=0)

    def test_it_shows_in_the_printed_form(self):
        assert "vector" in str(vectorise(matmul_nest(), "k"))


class TestPeel:
    def test_a_loop_that_divides_needs_no_peeling(self):
        nest = LoopNest(loops=[Loop(variable="i", extent=16, step=4)])
        main, tail = peel(nest, "i")
        assert tail is None
        assert main.total_iterations == 4

    def test_one_that_does_not_is_split_in_two(self):
        report = remainder_report(13, 4)
        assert report["main_iterations"] == 3
        assert report["tail_iterations"] == 1

    def test_the_two_halves_add_up_to_the_original(self):
        # Swept rather than sampled, and it found a real bug: the tail used to step by one,
        # so an extent of two stepping by three peeled into zero plus two where the original
        # ran once. A test at a size somebody picked would not have caught it.
        for extent in range(1, 60):
            for step in range(1, 12):
                nest = LoopNest(loops=[Loop(variable="i", extent=extent, step=step)])
                assert peeled_iteration_count(nest, "i") == nest.total_iterations, (
                    extent,
                    step,
                )

    def test_an_extent_smaller_than_the_step_is_all_tail(self):
        report = remainder_report(2, 3)
        assert report["main_iterations"] == 0
        assert report["tail_iterations"] == 1

    def test_the_tail_is_one_iteration_over_the_remainder(self):
        # Not one iteration per leftover element, which is the distinction the sweep found.
        nest = LoopNest(loops=[Loop(variable="i", extent=13, step=4)])
        _, tail = peel(nest, "i")
        assert tail is not None
        assert tail.total_iterations == 1

    def test_an_exact_multiple_leaves_no_tail(self):
        assert remainder_report(16, 4)["tail_iterations"] == 0

    def test_the_tail_covers_exactly_what_is_left(self):
        nest = LoopNest(loops=[Loop(variable="i", extent=13, step=4)])
        _, tail = peel(nest, "i")
        assert tail is not None
        assert tail.loop("i_tail").extent == 1


class TestTiling:
    def test_tiling_doubles_the_depth(self):
        assert tile_nest(matmul_nest(16, 16, 16), {"i": 4, "j": 4, "k": 4}).depth == 6

    def test_and_keeps_the_iteration_count(self):
        nest = matmul_nest(16, 16, 16)
        assert preserves_iterations(nest, tile_nest(nest, {"i": 4, "j": 4, "k": 4}))

    def test_every_block_loop_ends_up_outside_every_element_loop(self):
        # Which is what tiling is. Splitting alone leaves each block loop next to its own
        # element loop and buys nothing.
        tiled = tile_nest(matmul_nest(16, 16, 16), {"i": 4, "j": 4, "k": 4})
        variables = tiled.variables()
        last_outer = max(i for i, name in enumerate(variables) if name.endswith("_outer"))
        first_inner = min(i for i, name in enumerate(variables) if name.endswith("_inner"))
        assert last_outer < first_inner

    def test_tiling_one_axis_leaves_the_others_alone(self):
        tiled = tile_nest(matmul_nest(16, 16, 16), {"k": 4})
        assert "i" in tiled.variables()
        assert "j" in tiled.variables()

    def test_tiling_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to tile"):
            tile_nest(matmul_nest(), {})


class TestTransformationReport:
    def test_every_transformation_preserves_the_iterations(self):
        assert all(row["preserved"] for row in check_transformations())

    def test_a_split_deepens_the_nest(self):
        rows = {row["transformation"]: row for row in check_transformations()}
        assert rows["split k by 4"]["depth"] == 4

    def test_a_swap_does_not(self):
        rows = {row["transformation"]: row for row in check_transformations()}
        assert rows["swap i and k"]["depth"] == 3

    def test_an_unroll_does_not_either(self):
        rows = {row["transformation"]: row for row in check_transformations()}
        assert rows["unroll k by 4"]["depth"] == 3
