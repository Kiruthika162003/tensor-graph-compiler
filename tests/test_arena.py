from __future__ import annotations

import pytest

from tgc.analysis.liveness import compute_intervals
from tgc.errors import AllocationError, ConfigError
from tgc.ir.builder import elementwise_chain, softmax_graph
from tgc.memory.arena import (
    Arena,
    align_plan,
    align_up,
    alignment_sweep,
    arena_for,
    awkward_graph,
    check_alignment,
    compare_shapes,
    is_aligned,
    is_power_of_two,
    plan_survives_alignment,
    reuse_shares_padding,
    stacked_arena,
)
from tgc.memory.planner import Allocation, plan_largest_first
from tgc.schedule.order import depth_first_order


class TestAlignment:
    def test_a_power_of_two_is_recognised(self):
        assert is_power_of_two(64)
        assert not is_power_of_two(48)

    def test_zero_is_not_a_power_of_two(self):
        assert not is_power_of_two(0)

    def test_rounding_up_reaches_the_next_multiple(self):
        assert align_up(1, 64) == 64
        assert align_up(64, 64) == 64
        assert align_up(65, 64) == 128

    def test_an_alignment_of_one_changes_nothing(self):
        assert align_up(37, 1) == 37

    def test_a_non_power_of_two_alignment_is_rejected(self):
        with pytest.raises(ConfigError, match="power of two"):
            align_up(0, 48)

    def test_a_negative_offset_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            align_up(-1, 64)

    def test_the_predicate_agrees_with_the_rounding(self):
        for offset in range(0, 200):
            assert is_aligned(align_up(offset, 64), 64)


class TestArena:
    def test_a_misaligned_arena_is_refused(self):
        with pytest.raises(AllocationError, match="not aligned"):
            Arena(alignment=64, allocations=[Allocation(name="a", offset=8, size=16)])

    def test_a_bad_alignment_is_rejected(self):
        with pytest.raises(ConfigError, match="power of two"):
            Arena(alignment=48)

    def test_an_empty_arena_holds_nothing(self):
        assert Arena().size == 0
        assert Arena().used == 0
        assert Arena().padding_fraction == 0.0

    def test_the_size_reaches_the_last_byte(self):
        arena = Arena(
            alignment=64,
            allocations=[
                Allocation(name="a", offset=0, size=16),
                Allocation(name="b", offset=64, size=16),
            ],
        )
        assert arena.size == 80

    def test_shared_bytes_are_counted_once(self):
        # Summing the sizes counts them repeatedly and reports more used than the arena
        # holds, which is how the first version produced negative padding.
        arena = Arena(
            alignment=64,
            allocations=[
                Allocation(name="a", offset=0, size=16),
                Allocation(name="b", offset=0, size=16),
            ],
        )
        assert arena.used == 16

    def test_padding_is_never_negative(self):
        for graph in (awkward_graph(12), elementwise_chain(8), softmax_graph()):
            for alignment in (1, 16, 64, 256):
                assert arena_for(graph, alignment).padding >= 0

    def test_the_check_names_a_bad_offset(self):
        # Built at an alignment it satisfies, then held to a stricter one, which is what
        # happens when a plan built for one target is handed to another.
        arena = Arena(alignment=1, allocations=[Allocation(name="a", offset=8, size=16)])
        arena.alignment = 64
        with pytest.raises(AllocationError, match="not a multiple"):
            check_alignment(arena)

    def test_it_serialises(self):
        assert arena_for(awkward_graph(12), 64).as_dict()["alignment"] == 64


class TestSweep:
    def test_a_coarse_alignment_costs_more_padding(self):
        rows = {row["alignment"]: row for row in alignment_sweep(awkward_graph(12))}
        assert rows[4096]["padding"] > rows[64]["padding"]

    def test_an_alignment_of_one_costs_none(self):
        rows = {row["alignment"]: row for row in alignment_sweep(awkward_graph(12))}
        assert rows[1]["padding"] == 0

    def test_the_padding_fraction_grows_with_the_alignment(self):
        rows = alignment_sweep(awkward_graph(12))
        fractions = [row["padding_fraction"] for row in rows]
        assert fractions == sorted(fractions)

    def test_a_huge_alignment_makes_the_arena_almost_all_padding(self):
        rows = {row["alignment"]: row for row in alignment_sweep(awkward_graph(12))}
        assert rows[4096]["padding_fraction"] > 0.9

    def test_the_arena_never_drops_below_the_liveness_floor(self):
        for row in alignment_sweep(awkward_graph(12)):
            assert row["size"] >= row["floor"]

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            alignment_sweep(awkward_graph(4), alignments=())

    def test_power_of_two_shapes_are_aligned_already(self):
        # Every other fixture in this repository uses them, so an alignment sweep over those
        # measures nothing, which is why awkward_graph exists.
        rows = {row["alignment"]: row for row in alignment_sweep(elementwise_chain(8))}
        assert rows[64]["padding"] == 0


class TestShapes:
    def test_many_small_tensors_pay_a_larger_share(self):
        rows = {row["graph"]: row for row in compare_shapes()}
        assert rows["many small"]["padding_fraction"] > rows["few large"]["padding_fraction"]

    def test_few_large_ones_pay_almost_nothing(self):
        rows = {row["graph"]: row for row in compare_shapes()}
        assert rows["few large"]["padding_fraction"] == 0.0

    def test_the_awkward_fixture_is_not_a_multiple_of_the_alignment(self):
        graph = awkward_graph(4)
        assert graph.value("x").bytes % 64 != 0

    def test_an_empty_awkward_graph_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one node"):
            awkward_graph(0)


class TestReuse:
    def test_a_reusing_plan_pays_less_padding_than_a_stacked_one(self):
        # Two values sharing a slot share its rounding, so reuse gives back alignment waste
        # as well as tensor bytes.
        result = reuse_shares_padding(awkward_graph(12))
        assert result["reusing_padding"] < result["stacked_padding"]

    def test_and_is_far_smaller_overall(self):
        result = reuse_shares_padding(awkward_graph(12))
        assert result["reusing_size"] < result["stacked_size"] / 4

    def test_a_stacked_arena_places_everything_separately(self):
        graph = awkward_graph(6)
        intervals = compute_intervals(graph, depth_first_order(graph))
        arena = stacked_arena(intervals, 64)
        offsets = [allocation.offset for allocation in arena.allocations]
        assert len(set(offsets)) == len(offsets)

    def test_a_stacked_arena_is_aligned(self):
        graph = awkward_graph(6)
        intervals = compute_intervals(graph, depth_first_order(graph))
        check_alignment(stacked_arena(intervals, 64))

    def test_a_bad_alignment_is_rejected_when_stacking(self):
        graph = awkward_graph(4)
        intervals = compute_intervals(graph, depth_first_order(graph))
        with pytest.raises(ConfigError, match="power of two"):
            stacked_arena(intervals, 48)


class TestValidity:
    def test_aligning_keeps_live_values_apart(self):
        # Rounding offsets upward moves tensors, and moving tensors is exactly how a valid
        # plan stops being one. Re placing rather than nudging is what keeps this true.
        for graph in (awkward_graph(12), elementwise_chain(8), softmax_graph()):
            for alignment in (1, 16, 64, 256):
                assert plan_survives_alignment(graph, alignment), (graph, alignment)

    def test_every_placement_is_aligned(self):
        for alignment in (16, 64, 256):
            check_alignment(arena_for(awkward_graph(12), alignment))

    def test_a_bad_alignment_is_rejected_when_aligning_a_plan(self):
        graph = awkward_graph(4)
        intervals = compute_intervals(graph, depth_first_order(graph))
        with pytest.raises(ConfigError, match="power of two"):
            align_plan(plan_largest_first(intervals), 48)
