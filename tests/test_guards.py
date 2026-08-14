from __future__ import annotations

import pytest

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import Builder, mlp_graph, softmax_graph
from tgc.ir.shape import shape
from tgc.runtime.guards import (
    CacheReport,
    Guard,
    GuardSet,
    bucket_for,
    bucket_guards,
    choose_bucket_count,
    compare_bucketing,
    crossover_moves_with_compile_time,
    exact_guards,
    geometric_buckets,
    guards_for,
    realistic_lengths,
    run_requests,
    short_request_padding,
    uniform_buckets,
)


class TestGuard:
    def test_an_exact_guard_admits_one_size(self):
        guard = Guard(dimension=0, exact=8)
        assert guard.admits(8)
        assert not guard.admits(9)

    def test_a_range_admits_everything_inside_it(self):
        guard = Guard(dimension=0, lower=4, upper=8)
        assert guard.admits(4) and guard.admits(8)
        assert not guard.admits(3) and not guard.admits(9)

    def test_an_open_range_admits_everything_above(self):
        assert Guard(dimension=0, lower=4).admits(1_000_000)

    def test_an_exact_guard_is_one_wide(self):
        assert Guard(dimension=0, exact=8).width == 1

    def test_a_range_is_as_wide_as_it_looks(self):
        assert Guard(dimension=0, lower=4, upper=8).width == 5

    def test_an_open_guard_has_no_width(self):
        with pytest.raises(ScheduleError, match="unboundedly many"):
            _ = Guard(dimension=0, lower=4).width

    def test_a_guard_that_constrains_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="has to constrain something"):
            Guard(dimension=0)

    def test_a_guard_cannot_be_both_exact_and_a_range(self):
        with pytest.raises(ConfigError, match="either exact or a range"):
            Guard(dimension=0, exact=4, lower=2)

    def test_an_empty_range_is_rejected(self):
        with pytest.raises(ConfigError, match="empty range"):
            Guard(dimension=0, lower=8, upper=4)

    def test_a_negative_dimension_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            Guard(dimension=-1, exact=4)

    def test_it_serialises(self):
        assert Guard(dimension=1, exact=8).as_dict()["exact"] == 8


class TestGuardSet:
    def test_exact_guards_admit_only_their_own_shape(self):
        guards = exact_guards(shape(8, 32))
        assert guards.admits(shape(8, 32))
        assert not guards.admits(shape(9, 32))

    def test_and_are_fully_static(self):
        assert exact_guards(shape(8, 32)).is_fully_static

    def test_bucket_guards_admit_a_range(self):
        guards = bucket_guards(shape(20, 32), 0, geometric_buckets())
        assert guards.admits(shape(20, 32))
        assert guards.admits(shape(30, 32))

    def test_and_refuse_the_next_bucket(self):
        guards = bucket_guards(shape(20, 32), 0, geometric_buckets())
        assert not guards.admits(shape(40, 32))

    def test_and_are_not_fully_static(self):
        assert not bucket_guards(shape(20, 32), 0, geometric_buckets()).is_fully_static

    def test_a_symbolic_shape_is_never_admitted(self):
        assert not exact_guards(shape(8, 32)).admits(shape("batch", 32))

    def test_a_shape_of_the_wrong_rank_is_never_admitted(self):
        assert not exact_guards(shape(8, 32)).admits(shape(8))

    def test_an_empty_guard_set_admits_anything(self):
        assert GuardSet().admits(shape(1, 2, 3))

    def test_a_dimension_outside_the_shape_is_rejected(self):
        with pytest.raises(ConfigError, match="outside a shape"):
            bucket_guards(shape(8), 3, geometric_buckets())

    def test_no_buckets_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one bucket"):
            bucket_guards(shape(8), 0, [])

    def test_a_symbolic_dimension_cannot_be_bucketed(self):
        with pytest.raises(ConfigError, match="concrete size"):
            bucket_guards(shape("batch", 32), 0, geometric_buckets())

    def test_a_graph_gets_exact_guards_by_default(self):
        assert guards_for(softmax_graph()).is_fully_static

    def test_and_bucketed_ones_when_asked(self):
        assert not guards_for(softmax_graph(), buckets=geometric_buckets()).is_fully_static

    def test_it_serialises(self):
        assert exact_guards(shape(8, 32)).as_dict()["guards"] == 2


class TestBuckets:
    def test_a_size_lands_in_the_smallest_bucket_that_holds_it(self):
        assert bucket_for(20, geometric_buckets()) == 32

    def test_an_exact_match_does_not_round_up(self):
        assert bucket_for(32, geometric_buckets()) == 32

    def test_a_size_past_the_largest_bucket_is_refused(self):
        with pytest.raises(ScheduleError, match="larger than the largest"):
            bucket_for(4096, geometric_buckets())

    def test_a_negative_size_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            bucket_for(-1, geometric_buckets())

    def test_geometric_buckets_grow_by_the_ratio(self):
        assert geometric_buckets(8, 128, 2.0) == [8, 16, 32, 64, 128]

    def test_a_ratio_that_does_not_grow_is_rejected(self):
        with pytest.raises(ConfigError, match="has to grow"):
            geometric_buckets(ratio=1.0)

    def test_uniform_buckets_are_evenly_spaced(self):
        assert uniform_buckets(8, 264, 128) == [8, 136, 264]

    def test_a_zero_step_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            uniform_buckets(step=0)

    def test_a_decreasing_range_is_rejected(self):
        with pytest.raises(ConfigError, match="positive and increasing"):
            geometric_buckets(smallest=64, largest=8)


class TestCache:
    def test_exact_specialisation_pads_nothing(self):
        report = run_requests(realistic_lengths())
        assert report.padding_overhead == 0.0

    def test_and_compiles_for_over_half_the_requests(self):
        report = run_requests(realistic_lengths())
        assert report.compiles > report.requests / 2

    def test_bucketing_compiles_a_handful_of_times(self):
        report = run_requests(realistic_lengths(), geometric_buckets())
        assert report.compiles <= len(geometric_buckets())

    def test_and_pays_for_it_in_padding(self):
        report = run_requests(realistic_lengths(), geometric_buckets())
        assert report.padding_overhead > 0.4

    def test_an_empty_stream_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to serve"):
            run_requests([])

    def test_an_empty_report_has_no_rates(self):
        assert CacheReport().hit_rate == 0.0
        assert CacheReport().padding_overhead == 0.0

    def test_negative_times_are_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            CacheReport().total_seconds(-1.0, 1.0)

    def test_a_zero_length_stream_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            realistic_lengths(count=0)

    def test_it_serialises(self):
        assert run_requests([8, 8, 16]).as_dict()["requests"] == 3


class TestComparison:
    def test_every_scheme_is_reported(self):
        assert len(compare_bucketing()) == 3

    def test_the_aggregate_padding_looks_like_a_tie(self):
        rows = {row["scheme"]: row for row in compare_bucketing()}
        geometric = rows["geometric"]["padding_overhead"]
        uniform = rows["uniform"]["padding_overhead"]
        assert abs(geometric - uniform) < 0.1

    def test_and_the_short_requests_say_otherwise(self):
        # The aggregate is dominated by long requests. Uniform buckets waste a constant
        # number of positions, which is a rounding error on a long request and several times
        # the work on a short one.
        rows = {row["scheme"]: row for row in short_request_padding()}
        assert rows["uniform"]["padding_overhead"] > 5 * rows["geometric"]["padding_overhead"]

    def test_a_threshold_nothing_falls_under_is_reported(self):
        with pytest.raises(ScheduleError, match="no request came in under"):
            short_request_padding(threshold=1)

    def test_fewer_buckets_means_more_padding(self):
        rows = choose_bucket_count()
        overheads = [row["padding_overhead"] for row in rows]
        assert overheads == sorted(overheads, reverse=True)

    def test_and_fewer_compiles(self):
        rows = choose_bucket_count()
        compiles = [row["compiles"] for row in rows]
        assert compiles == sorted(compiles)

    def test_the_best_bucket_count_moves_with_the_compile_time(self):
        # Which is the argument against shipping it as a constant.
        rows = crossover_moves_with_compile_time()
        assert len({row["buckets"] for row in rows}) > 1

    def test_a_slow_compiler_wants_fewer_buckets(self):
        rows = {row["compile_seconds"]: row for row in crossover_moves_with_compile_time()}
        assert rows[2.0]["buckets"] < rows[0.005]["buckets"]


class TestGraphs:
    def test_a_graph_with_no_inputs_needs_no_guards(self):
        builder = Builder()
        graph = builder.finish(builder.constant(1.0))
        with pytest.raises(ConfigError, match="needs no guards"):
            guards_for(graph)

    def test_an_mlp_gets_a_guard_per_dimension(self):
        assert len(guards_for(mlp_graph()).guards) == 2
