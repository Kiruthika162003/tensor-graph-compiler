from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.schedule.prefetch import (
    Pipeline,
    a_deeper_queue_absorbs_a_spike,
    a_single_tile_gains_nothing,
    asymptotic_speedup,
    balance_sweep,
    best_depth,
    buffer_cost,
    depth_beyond_one_buys_nothing,
    depth_pays_when_the_fetches_vary,
    depth_sweep,
    memory_against_gain,
    never_more_than_double,
    the_curve_is_symmetric,
    the_gain_peaks_when_the_stages_match,
    tile_count_sweep,
    tiles_needed_for,
    variable_fetch_time,
)


class TestPipeline:
    def test_a_serial_loop_pays_for_both_stages_every_tile(self):
        assert Pipeline(tiles=10, fetch=2.0, compute=3.0).serial_time == 50.0

    def test_an_overlapped_one_pays_for_the_slower_plus_a_prologue(self):
        assert Pipeline(tiles=10, fetch=2.0, compute=3.0).overlapped_time == 32.0

    def test_depth_zero_is_the_serial_loop(self):
        pipeline = Pipeline(tiles=10, fetch=2.0, compute=3.0, depth=0)
        assert pipeline.overlapped_time == pipeline.serial_time

    def test_the_slower_stage_is_named(self):
        assert Pipeline(tiles=4, fetch=2.0, compute=3.0).bound_by == "compute"
        assert Pipeline(tiles=4, fetch=3.0, compute=2.0).bound_by == "fetch"
        assert Pipeline(tiles=4, fetch=2.0, compute=2.0).bound_by == "balanced"

    def test_each_level_of_depth_costs_a_buffer(self):
        assert Pipeline(tiles=4, fetch=1.0, compute=1.0, depth=3).buffers == 4

    def test_a_loop_with_no_tiles_is_refused(self):
        with pytest.raises(ConfigError, match="at least one tile"):
            Pipeline(tiles=0, fetch=1.0, compute=1.0)

    def test_a_negative_stage_is_refused(self):
        with pytest.raises(ConfigError, match="negative time"):
            Pipeline(tiles=4, fetch=-1.0, compute=1.0)

    def test_a_negative_depth_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            Pipeline(tiles=4, fetch=1.0, compute=1.0, depth=-1)

    def test_a_loop_that_takes_no_time_gains_nothing(self):
        assert Pipeline(tiles=4, fetch=0.0, compute=0.0).speedup == 1.0

    def test_it_serialises(self):
        assert Pipeline(tiles=4, fetch=1.0, compute=1.0).as_dict()["bound_by"] == "balanced"


class TestBalance:
    def test_the_gain_peaks_where_the_stages_match(self):
        assert the_gain_peaks_when_the_stages_match()["best_ratio"] == 1.0

    def test_and_never_reaches_two(self):
        # Two stages can hide at most one of themselves behind the other.
        assert the_gain_peaks_when_the_stages_match()["best_speedup"] < 2.0

    def test_no_pair_of_stage_times_beats_a_factor_of_two(self):
        assert never_more_than_double()["under_two"]

    def test_a_lopsided_loop_gains_almost_nothing(self):
        rows = {row["ratio"]: row for row in balance_sweep()}
        assert rows[10.0]["speedup"] < 1.15

    def test_and_it_does_not_matter_which_way_it_leans(self):
        rows = {row["ratio"]: row for row in balance_sweep()}
        assert abs(rows[10.0]["speedup"] - rows[0.1]["speedup"]) < 0.02

    def test_the_curve_is_exactly_symmetric_in_the_limit(self):
        assert the_curve_is_symmetric()["symmetric_in_the_limit"]

    def test_but_not_over_a_finite_loop(self):
        # The stage that runs alone at the start is the fetch, so a fetch bound loop pays more
        # for its prologue.
        result = the_curve_is_symmetric()
        assert not result["symmetric_over_a_finite_loop"]
        assert result["largest_finite_gap"] < 0.01

    def test_equal_stages_double_in_the_limit(self):
        assert asymptotic_speedup(1.0, 1.0) == 2.0

    def test_a_loop_with_no_work_has_no_limit_to_reach(self):
        assert asymptotic_speedup(0.0, 0.0) == 1.0

    def test_a_negative_stage_is_refused(self):
        with pytest.raises(ConfigError, match="negative time"):
            asymptotic_speedup(-1.0, 1.0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            balance_sweep(ratios=())

    def test_a_zero_sample_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            never_more_than_double(samples=0)


class TestDepth:
    def test_the_first_level_of_depth_does_all_the_work(self):
        rows = {row["depth"]: row for row in depth_sweep()}
        assert rows[1]["speedup"] > rows[0]["speedup"]

    def test_and_every_level_after_it_does_none(self):
        # The steady state is set by the slower stage and running ahead does not change which
        # stage that is.
        assert depth_beyond_one_buys_nothing()["identical"]

    def test_while_costing_a_buffer_each(self):
        assert depth_beyond_one_buys_nothing()["extra_buffers"] == 7

    def test_unless_the_fetch_time_varies(self):
        # Then depth is exactly the queue that absorbs a spike.
        assert a_deeper_queue_absorbs_a_spike()["improved"]

    def test_and_then_it_saves_real_time(self):
        assert a_deeper_queue_absorbs_a_spike()["saving"] > 10.0

    def test_a_deeper_queue_never_takes_longer(self):
        rows = [row["time"] for row in depth_pays_when_the_fetches_vary()]
        assert rows == sorted(rows, reverse=True)

    def test_a_loop_with_no_fetches_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to run"):
            variable_fetch_time([])

    def test_a_negative_compute_is_refused(self):
        with pytest.raises(ConfigError, match="negative time"):
            variable_fetch_time([1.0], compute=-1.0)

    def test_a_negative_depth_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            variable_fetch_time([1.0], depth=-1)

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            depth_sweep(depths=())

    def test_an_empty_spike_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            depth_pays_when_the_fetches_vary(depths=())

    def test_a_spike_sweep_needs_tiles(self):
        with pytest.raises(ConfigError, match="at least one tile"):
            depth_pays_when_the_fetches_vary(tiles=0)


class TestTileCount:
    def test_one_tile_gains_nothing(self):
        # A tiling pass that leaves a handful of tiles has given away the overlap.
        assert a_single_tile_gains_nothing()["one_tile"] == 1.0

    def test_and_many_tiles_reach_the_limit(self):
        assert a_single_tile_gains_nothing()["many_tiles"] > 1.99

    def test_four_tiles_reach_four_fifths_of_it(self):
        rows = {row["tiles"]: row for row in tile_count_sweep()}
        assert rows[4]["share_of_the_limit"] == 0.8

    def test_nine_tiles_are_enough_for_nine_tenths(self):
        assert tiles_needed_for(0.9) == 9

    def test_and_the_last_percent_takes_a_thousand(self):
        assert tiles_needed_for(0.999) > 500

    def test_the_share_climbs_with_the_tile_count(self):
        shares = [row["share_of_the_limit"] for row in tile_count_sweep()]
        assert shares == sorted(shares)

    def test_a_share_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ConfigError, match="has to be in"):
            tiles_needed_for(1.5)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            tile_count_sweep(counts=())


class TestMemory:
    def test_depth_one_needs_two_buffers(self):
        assert buffer_cost(1024, 1) == 2048

    def test_a_negative_tile_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            buffer_cost(-1, 1)

    def test_a_negative_depth_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            buffer_cost(1024, -1)

    def test_the_table_shows_memory_rising_and_speedup_flat(self):
        rows = memory_against_gain()
        assert rows[-1]["bytes"] > rows[1]["bytes"]
        assert rows[-1]["speedup"] == rows[1]["speedup"]

    def test_so_the_best_depth_is_one_whatever_the_budget(self):
        assert best_depth(budget=10**9) == 1

    def test_a_budget_too_small_for_one_tile_is_refused(self):
        with pytest.raises(ConfigError, match="cannot hold one tile"):
            best_depth(tile_bytes=1024, budget=512)

    def test_an_empty_table_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            memory_against_gain(depths=())
