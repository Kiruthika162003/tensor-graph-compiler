from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.parallel.collective import (
    FAST,
    SLOW,
    Link,
    a_slow_link_changes_everything,
    bucket_sweep,
    bucket_time,
    bucketing_is_worth_a_factor_of_two,
    compare_methods,
    crossover_bytes,
    device_sweep,
    exposed_time,
    gradient_sizes,
    overlap_hides_almost_everything,
    overlap_is_worth_more_than_the_method,
    ring_time,
    ring_volume,
    size_sweep,
    the_crossover_is_below_any_real_tensor,
    the_floor_is_one_layer,
    the_ring_volume_has_a_ceiling,
    tree_time,
    tree_volume,
    where_the_overlap_runs_out,
)


class TestLinks:
    def test_a_link_charges_for_the_start_up(self):
        link = Link(name="test", bytes_per_second=1e9, latency_seconds=1e-6)
        assert link.time_for(0) == 1e-6

    def test_and_for_the_bytes(self):
        link = Link(name="test", bytes_per_second=1e9, latency_seconds=0.0)
        assert link.time_for(1e9) == 1.0

    def test_a_link_that_moves_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="move something"):
            Link(name="broken", bytes_per_second=0, latency_seconds=0)

    def test_a_negative_latency_is_refused(self):
        with pytest.raises(ConfigError, match="negative time"):
            Link(name="broken", bytes_per_second=1e9, latency_seconds=-1)

    def test_moving_negative_bytes_is_refused(self):
        with pytest.raises(ConfigError, match="cannot move"):
            FAST.time_for(-1)

    def test_it_serialises(self):
        assert FAST.as_dict()["link"] == "fast"


class TestMethods:
    def test_one_device_needs_no_reduction(self):
        assert ring_time(1 << 20, 1) == 0.0
        assert tree_time(1 << 20, 1) == 0.0

    def test_a_ring_moves_the_least_any_method_can(self):
        size = 1 << 20
        assert ring_volume(size, 8) < tree_volume(size, 8)

    def test_and_approaches_twice_the_tensor(self):
        rows = {row["devices"]: row for row in the_ring_volume_has_a_ceiling()}
        assert rows[2]["share_of_twice_the_tensor"] == 0.5
        assert rows[1024]["share_of_twice_the_tensor"] > 0.99

    def test_a_tree_takes_fewer_steps(self):
        # Six messages against fourteen at eight devices.
        assert tree_volume(1024, 8) / 1024 == 6.0

    def test_a_negative_tensor_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            ring_time(-1, 4)

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            tree_time(1024, 0)

    def test_both_methods_are_reported(self):
        assert len(compare_methods(1 << 20)) == 2

    def test_an_empty_volume_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_ring_volume_has_a_ceiling(counts=())


class TestCrossover:
    def test_the_tree_wins_on_small_tensors(self):
        rows = {row["bytes"]: row for row in size_sweep()}
        assert rows[1024]["winner"] == "tree"

    def test_and_the_ring_on_large_ones(self):
        rows = {row["bytes"]: row for row in size_sweep()}
        assert rows[67108864]["winner"] == "ring"

    def test_the_crossover_is_a_megabyte(self):
        assert crossover_bytes() == 1 << 20

    def test_which_is_in_the_middle_of_what_a_model_holds(self):
        # A small weight wants a tree and a large one wants a ring, so both are needed.
        result = the_crossover_is_below_any_real_tensor()
        assert result["tree_wins_on_the_small_one"]
        assert result["ring_wins_on_the_large_one"]

    def test_neither_method_gets_cheaper_with_more_devices(self):
        rows = device_sweep()
        assert rows[-1]["ring_seconds"] > rows[0]["ring_seconds"]
        assert rows[-1]["tree_seconds"] > rows[0]["tree_seconds"]

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            size_sweep(sizes=())

    def test_an_empty_device_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            device_sweep(counts=())


class TestBucketing:
    def test_grouping_the_tensors_is_worth_a_factor_of_two(self):
        assert bucketing_is_worth_a_factor_of_two()["saving"] > 2.0

    def test_because_there_are_a_hundred_and_forty_four_of_them(self):
        assert bucketing_is_worth_a_factor_of_two()["messages_before"] == 144

    def test_a_larger_bucket_keeps_helping(self):
        # There is no flat region, because the ring's per message cost is two start ups per
        # device and eight devices make that expensive enough to keep mattering.
        rows = [row["seconds"] for row in bucket_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_a_model_produces_three_gradients_per_layer(self):
        assert len(gradient_sizes(layers=4)) == 12

    def test_a_zero_layer_model_is_refused(self):
        with pytest.raises(ConfigError, match="needs layers"):
            gradient_sizes(layers=0)

    def test_reducing_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to reduce"):
            bucket_time([], 8)

    def test_a_negative_bucket_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            bucket_time([1024], 8, bucket_bytes=-1)

    def test_an_empty_bucket_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            bucket_sweep(buckets=())


class TestOverlap:
    def test_the_overlap_hides_almost_all_of_it(self):
        assert exposed_time()["hidden_share"] > 0.95

    def test_at_every_device_count_swept(self):
        assert all(row["hidden_share"] > 0.9 for row in overlap_hides_almost_everything())

    def test_though_the_share_falls_as_devices_are_added(self):
        rows = overlap_hides_almost_everything()
        assert rows[-1]["hidden_share"] < rows[0]["hidden_share"]

    def test_what_is_left_is_the_last_layer(self):
        # The gradient produced last has nothing behind it to hide under.
        assert the_floor_is_one_layer()["exposed_is_at_least_the_floor"]

    def test_and_it_is_exactly_that_floor(self):
        assert the_floor_is_one_layer()["ratio"] == 1.0

    def test_the_overlap_runs_out_sooner_on_a_slow_link(self):
        assert where_the_overlap_runs_out(link=SLOW) < where_the_overlap_runs_out(link=FAST)

    def test_a_slow_link_exposes_ten_times_as_much(self):
        assert a_slow_link_changes_everything()["exposed_ratio"] > 10.0

    def test_but_hides_the_same_share_of_it(self):
        # Overlap is just as effective on a slow link, there is simply more to overlap.
        result = a_slow_link_changes_everything()
        assert abs(result["slow_hidden_share"] - result["fast_hidden_share"]) < 0.05

    def test_overlap_is_worth_more_than_the_method_choice(self):
        result = overlap_is_worth_more_than_the_method()
        assert result["overlap_is_worth"] > result["method_choice_is_worth"]

    def test_though_the_method_choice_is_worth_a_lot_too(self):
        assert overlap_is_worth_more_than_the_method()["method_choice_is_worth"] > 0.5

    def test_a_backward_pass_of_no_duration_is_refused(self):
        with pytest.raises(ConfigError, match="takes time"):
            exposed_time(backward_seconds=0.0)

    def test_a_threshold_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ConfigError, match="has to be in"):
            where_the_overlap_runs_out(threshold=1.5)

    def test_an_empty_overlap_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            overlap_hides_almost_everything(counts=())
