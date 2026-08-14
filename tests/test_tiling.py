from __future__ import annotations

import pytest

from tgc.errors import ConfigError, ScheduleError
from tgc.schedule.autotune import (
    NEAR_LIMIT_SIDES,
    SIDES,
    Candidate,
    compare_regimes,
    compare_strategies,
    cost_table,
    exhaustive,
    measured_cost,
    model_blind_spot,
    model_cost,
    model_only,
    model_ranking_agreement,
    model_then_measure,
    random_search,
    shortlist_sweep,
)
from tgc.schedule.tiling import (
    MatmulShape,
    Tile,
    arithmetic_per_byte,
    best_tile,
    doubling_halves_traffic,
    effective_traffic,
    largest_fitting_tile,
    square_tile,
    sweep_tiles,
    tiled_traffic,
    traffic_reduction,
    untiled_traffic,
)

CACHE = 256 * 1024
SHAPE = MatmulShape(500, 500, 500)


class TestTile:
    def test_the_working_set_counts_all_three_tiles(self):
        # Counting only two is the mistake that makes a tile look like it fits.
        assert square_tile(4).working_set_elements == 48

    def test_a_larger_tile_holds_more(self):
        assert square_tile(8).working_set_elements > square_tile(4).working_set_elements

    def test_a_small_tile_fits_a_small_cache(self):
        assert square_tile(4).fits_in(1024)

    def test_a_large_one_does_not(self):
        assert not square_tile(256).fits_in(1024)

    def test_a_zero_dimension_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            Tile(rows=0, columns=4, depth=4)

    def test_a_negative_cache_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            square_tile(4).fits_in(-1)

    def test_a_zero_byte_element_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one byte"):
            square_tile(4).working_set_bytes(0)

    def test_it_serialises(self):
        assert square_tile(8).as_dict()["working_set"] == 192


class TestTraffic:
    def test_blocking_moves_less_data(self):
        assert tiled_traffic(SHAPE, square_tile(64)) < untiled_traffic(SHAPE)

    def test_doubling_the_tile_halves_the_traffic(self):
        # What the n cubed over t term says, and worth seeing rather than deriving.
        rows = doubling_halves_traffic()
        ratios = [row["ratio_to_previous"] for row in rows if row["ratio_to_previous"]]
        assert all(1.9 < ratio <= 2.0 for ratio in ratios)

    def test_a_tile_of_one_saves_nothing(self):
        assert traffic_reduction(MatmulShape(), square_tile(1)) == pytest.approx(1.0, abs=0.01)

    def test_a_large_tile_saves_two_orders_of_magnitude(self):
        assert traffic_reduction(MatmulShape(), square_tile(128)) > 100

    def test_a_tile_that_does_not_fit_gets_no_reuse(self):
        # The discontinuity that makes tile selection a search rather than a formula.
        assert effective_traffic(MatmulShape(), square_tile(256), CACHE) == untiled_traffic(
            MatmulShape()
        )

    def test_the_cliff_is_a_single_step(self):
        rows = {row["side"]: row for row in sweep_tiles(cache_bytes=CACHE)}
        assert rows[128]["reduction"] > 100
        assert rows[256]["reduction"] == 1.0

    def test_a_tiling_that_moves_nothing_cannot_be_compared(self):
        with pytest.raises(ScheduleError, match="cannot be compared"):
            traffic_reduction(MatmulShape(rows=1, columns=1, depth=1), square_tile(1), 0)

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            sweep_tiles(sides=())


class TestFitting:
    def test_the_largest_fitting_tile_fits(self):
        side = largest_fitting_tile(CACHE)
        assert square_tile(side).fits_in(CACHE)

    def test_and_one_larger_does_not(self):
        side = largest_fitting_tile(CACHE)
        assert not square_tile(side + 1).fits_in(CACHE)

    def test_a_bigger_cache_takes_a_bigger_tile(self):
        assert largest_fitting_tile(1 << 20) > largest_fitting_tile(1 << 16)

    def test_the_best_tile_is_a_power_of_two_here_only_by_accident(self):
        # The search picks the largest fitting candidate offered, not the largest that fits.
        assert best_tile(cache_bytes=CACHE) == 128
        assert largest_fitting_tile(CACHE) > 128

    def test_a_cache_that_holds_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            largest_fitting_tile(0)

    def test_a_zero_limit_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            largest_fitting_tile(CACHE, limit=0)


class TestIntensity:
    def test_blocking_raises_the_work_per_byte(self):
        naive = MatmulShape().flops / untiled_traffic(MatmulShape())
        blocked = arithmetic_per_byte(MatmulShape(), square_tile(64))
        assert blocked > 50 * naive

    def test_a_tiling_that_moves_nothing_has_no_intensity(self):
        with pytest.raises(ScheduleError, match="no intensity"):
            arithmetic_per_byte(MatmulShape(rows=1, columns=1, depth=1), square_tile(1), 0)

    def test_a_zero_dimension_matmul_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            MatmulShape(rows=0)


class TestAutotuning:
    def test_the_model_and_the_measurement_disagree_on_the_penalised_tiles(self):
        tile = square_tile(125)
        assert measured_cost(SHAPE, tile, CACHE) > model_cost(SHAPE, tile, CACHE)

    def test_a_tile_that_divides_cleanly_pays_no_penalty(self):
        shape = MatmulShape(512, 512, 512)
        tile = square_tile(64)
        assert measured_cost(shape, tile, CACHE) == model_cost(shape, tile, CACHE)

    def test_a_zero_width_vector_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one element wide"):
            measured_cost(SHAPE, square_tile(64), CACHE, vector_width=0)

    def test_exhaustive_search_measures_everything(self):
        assert exhaustive(SHAPE, CACHE).measurements == len(SIDES)

    def test_and_finds_the_optimum_by_definition(self):
        assert exhaustive(SHAPE, CACHE).found_the_best

    def test_a_shortlist_finds_it_while_measuring_a_fraction(self):
        result = model_then_measure(SHAPE, CACHE)
        assert result.found_the_best
        assert result.measurements < len(SIDES) / 3

    def test_random_sampling_does_not(self):
        assert not random_search(SHAPE, CACHE).found_the_best

    def test_an_empty_candidate_set_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to tune over"):
            cost_table(SHAPE, CACHE, ())

    def test_a_zero_shortlist_is_rejected(self):
        with pytest.raises(ConfigError, match="must hold something"):
            model_then_measure(SHAPE, CACHE, shortlist=0)

    def test_a_zero_budget_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            random_search(SHAPE, CACHE, budget=0)

    def test_a_regret_against_an_uncovered_table_is_rejected(self):
        result = exhaustive(SHAPE, CACHE)
        with pytest.raises(ScheduleError, match="does not cover"):
            result.regret({})

    def test_it_serialises(self):
        assert Candidate(side=64, model_cost=10.0, measured_cost=20.0).as_dict()["error"] == 0.5


class TestRegimes:
    def test_over_powers_of_two_the_model_alone_is_right(self):
        # Traffic spans two orders of magnitude and the penalties the model misses are
        # bounded by a factor of two, so there is nothing for a measurement to change.
        rows = {(r["candidates"], r["strategy"]): r for r in compare_regimes()}
        assert rows[("coarse", "model only")]["regret"] == 0.0

    def test_near_the_cache_limit_it_is_not(self):
        rows = {(r["candidates"], r["strategy"]): r for r in compare_regimes()}
        assert rows[("near the limit", "model only")]["regret"] > 0.2

    def test_and_measuring_four_candidates_recovers_the_optimum(self):
        rows = {(r["candidates"], r["strategy"]): r for r in compare_regimes()}
        assert rows[("near the limit", "model then measure 4")]["regret"] == 0.0

    def test_the_blind_spot_is_the_vector_width(self):
        result = model_blind_spot()
        assert result["model_pick"] != result["measured_best"]
        assert result["measured_best"] % 8 == 0
        assert result["model_pick"] % 8 != 0

    def test_the_traffic_barely_moves_across_that_band(self):
        # Which is why something other than traffic decides there.
        assert model_blind_spot()["traffic_spread"] < 10

    def test_the_model_ranking_broadly_agrees_with_the_measured_one(self):
        assert model_ranking_agreement(SHAPE, CACHE) > 0.8

    def test_a_shortlist_of_one_is_the_model_alone(self):
        rows = shortlist_sweep(SHAPE, CACHE)
        assert rows[0]["shortlist"] == 1

    def test_every_strategy_is_compared(self):
        assert len(compare_strategies(SHAPE, CACHE)) == 4

    def test_the_narrow_band_holds_both_multiples_and_non_multiples_of_the_vector(self):
        assert any(side % 8 == 0 for side in NEAR_LIMIT_SIDES)
        assert any(side % 8 != 0 for side in NEAR_LIMIT_SIDES)

    def test_the_model_only_strategy_measures_nothing(self):
        assert model_only(SHAPE, CACHE).measurements == 0
