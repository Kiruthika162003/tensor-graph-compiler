from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.builder import elementwise_chain, layernorm_graph, mlp_graph, softmax_graph
from tgc.runtime.cache import (
    POLICIES,
    Cache,
    CacheStats,
    Entry,
    break_even_hit_rate,
    capacity_beats_policy,
    capacity_sweep,
    compare_policies,
    compile_cost_sweep,
    cyclic_workload,
    least_frequently_used_fails_on_a_phase_change,
    least_recently_used_fails_on_a_cycle,
    on_realistic_traffic,
    one_more_entry_fixes_it,
    phase_workload,
    run_workload,
    shape_signature,
    skewed_workload,
    time_with_cache,
)


class TestKeys:
    def test_the_same_graph_gets_the_same_key(self):
        assert shape_signature(softmax_graph()) == shape_signature(softmax_graph())

    def test_a_different_graph_gets_a_different_one(self):
        assert shape_signature(softmax_graph()) != shape_signature(mlp_graph())

    def test_the_same_graph_at_a_different_size_gets_a_different_one(self):
        # Every buffer offset in the generated module is a number rather than an expression.
        assert shape_signature(softmax_graph(8)) != shape_signature(softmax_graph(16))

    def test_the_shapes_are_part_of_the_key(self):
        assert "[8, 32]" in shape_signature(softmax_graph())


class TestPolicies:
    def test_every_policy_runs(self):
        rows = compare_policies(skewed_workload(count=200), 4)
        assert len(rows) == len(POLICIES)

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            run_workload(["a", "b"], 2, "magic")

    def test_a_cache_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            run_workload(["a"], 0)

    def test_an_empty_workload_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to serve"):
            compare_policies([], 4)

    def test_a_cache_larger_than_the_workload_never_evicts(self):
        assert run_workload(cyclic_workload(4), 8).evictions == 0

    def test_and_hits_everything_after_the_first_round(self):
        stats = run_workload(cyclic_workload(4, rounds=10), 8)
        assert stats.misses == 4

    def test_an_empty_run_has_no_hit_rate(self):
        assert CacheStats().hit_rate == 0.0

    def test_it_serialises(self):
        assert run_workload(cyclic_workload(4), 8).as_dict()["misses"] == 4


class TestFailureModes:
    def test_least_recently_used_hits_nothing_on_a_tight_cycle(self):
        # The entry evicted to make room is always the one wanted next.
        assert least_recently_used_fails_on_a_cycle()["least_recently_used"] == 0.0

    def test_and_neither_does_first_in(self):
        assert least_recently_used_fails_on_a_cycle()["first_in"] == 0.0

    def test_while_random_eviction_does_fine(self):
        assert least_recently_used_fails_on_a_cycle()["random"] > 0.5

    def test_the_optimum_does_better_still(self):
        result = least_recently_used_fails_on_a_cycle()
        assert result["optimal"] > result["random"]

    def test_one_more_cache_entry_fixes_the_cycle_entirely(self):
        result = one_more_entry_fixes_it()
        assert result["one_short"] == 0.0
        assert result["exactly_enough"] > 0.9

    def test_least_frequently_used_fails_on_a_phase_change(self):
        # An early shape keeps a count nothing later can beat, so it stays while the shapes
        # being asked for are evicted around it.
        result = least_frequently_used_fails_on_a_phase_change()
        assert result["least_frequently_used"] < 0.5

    def test_while_least_recently_used_does_almost_perfectly(self):
        assert least_frequently_used_fails_on_a_phase_change()["least_recently_used"] > 0.9

    def test_a_workload_with_no_shapes_is_refused(self):
        with pytest.raises(ConfigError, match="needs shapes"):
            cyclic_workload(0)

    def test_a_phase_workload_with_no_phases_is_refused(self):
        with pytest.raises(ConfigError, match="needs shapes"):
            phase_workload(phases=0)

    def test_a_skewed_workload_with_no_requests_is_refused(self):
        with pytest.raises(ConfigError, match="needs shapes"):
            skewed_workload(count=0)


class TestRealisticTraffic:
    def test_the_frequency_policy_wins_on_skewed_traffic(self):
        rates = on_realistic_traffic()["rates"]
        assert rates["least frequently used"] > rates["least recently used"]

    def test_but_none_of_them_reaches_the_optimum(self):
        assert on_realistic_traffic()["gap_to_optimal"] > 0.0

    def test_and_the_spread_between_them_is_real(self):
        assert on_realistic_traffic()["spread"] > 0.1

    def test_the_hit_rate_climbs_with_the_capacity(self):
        rates = [row["hit_rate"] for row in capacity_sweep()]
        assert rates == sorted(rates)

    def test_and_flattens_once_the_common_shapes_fit(self):
        rows = {row["capacity"]: row["hit_rate"] for row in capacity_sweep()}
        early = rows[8] - rows[4]
        late = rows[32] - rows[16]
        assert late < early

    def test_doubling_the_cache_is_worth_about_as_much_as_the_best_policy(self):
        result = capacity_beats_policy()
        assert result["capacity_wins"]

    def test_an_empty_capacity_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            capacity_sweep(capacities=())


class TestValue:
    def test_a_cache_is_faster_than_recompiling(self):
        assert time_with_cache(skewed_workload(count=200), 8)["speedup"] > 1.0

    def test_and_more_so_the_more_compiling_costs(self):
        rows = compile_cost_sweep()
        assert rows[-1]["speedup"] > rows[0]["speedup"]

    def test_though_the_hit_rate_caps_it(self):
        rows = compile_cost_sweep()
        assert rows[-1]["speedup"] < 1 / (1 - rows[-1]["hit_rate"])

    def test_the_break_even_hit_rate_is_nothing_on_this_model(self):
        # A miss costs what compiling always cost and a hit costs nothing.
        assert break_even_hit_rate() == 0.0

    def test_a_zero_cost_is_refused(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            break_even_hit_rate(compile_cost=0.0)

    def test_a_zero_run_cost_is_refused(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            time_with_cache(["a"], 2, run_cost=0.0)

    def test_an_empty_cost_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compile_cost_sweep(ratios=())


class TestRealCache:
    def test_a_repeated_graph_is_a_hit(self):
        cache = Cache(capacity=4)
        cache.get(softmax_graph())
        cache.get(softmax_graph())
        assert cache.stats.hits == 1

    def test_a_full_cache_evicts(self):
        cache = Cache(capacity=2)
        for graph in (softmax_graph(), mlp_graph(), layernorm_graph()):
            cache.get(graph)
        assert cache.stats.evictions == 1

    def test_and_the_oldest_one_is_gone(self):
        cache = Cache(capacity=2)
        for graph in (softmax_graph(), mlp_graph(), layernorm_graph()):
            cache.get(graph)
        assert not cache.holds(softmax_graph())

    def test_a_graph_that_was_never_asked_for_is_not_held(self):
        cache = Cache(capacity=4)
        cache.get(softmax_graph())
        assert not cache.holds(elementwise_chain(4))

    def test_a_cache_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            Cache(capacity=0)

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            Cache(capacity=4, policy="magic")

    def test_an_entry_records_what_it_cost(self):
        assert Entry(key="k", compile_cost=50.0).as_dict()["compile_cost"] == 50.0

    def test_the_cache_serialises(self):
        cache = Cache(capacity=4)
        cache.get(softmax_graph())
        assert cache.as_dict()["held"] == 1
