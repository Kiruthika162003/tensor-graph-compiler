from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import branching_graph, elementwise_chain, mlp_graph, softmax_graph
from tgc.schedule.order import (
    ORDERINGS,
    all_orders,
    arena_for_order,
    best_order,
    breadth_first_order,
    check_order,
    compare_orders,
    depth_first_order,
    greedy_min_peak_order,
    is_valid_order,
    order_versus_allocator,
    peak_for_order,
    ready_nodes,
    scheduled_arena,
    source_order,
    worst_order,
)
from tgc.verify.reference import random_feeds, run

WIDE = branching_graph(6, 2)


class TestValidity:
    def test_the_source_order_is_valid(self):
        graph = softmax_graph()
        assert is_valid_order(graph, source_order(graph))

    def test_every_heuristic_produces_a_valid_order(self):
        for graph in (WIDE, softmax_graph(), mlp_graph(), elementwise_chain(8)):
            for ordering in ORDERINGS.values():
                assert is_valid_order(graph, ordering(graph))

    def test_a_reversed_order_is_not(self):
        graph = elementwise_chain(4)
        assert not is_valid_order(graph, list(reversed(graph.nodes)))

    def test_a_short_order_is_not(self):
        graph = elementwise_chain(4)
        assert not is_valid_order(graph, graph.nodes[:2])

    def test_an_invalid_order_is_refused(self):
        graph = elementwise_chain(4)
        with pytest.raises(ScheduleError, match="before something it reads"):
            check_order(graph, list(reversed(graph.nodes)))

    def test_every_heuristic_runs_every_node_once(self):
        for ordering in ORDERINGS.values():
            names = [node.name for node in ordering(WIDE)]
            assert sorted(names) == sorted(node.name for node in WIDE.nodes)

    def test_the_ready_set_starts_at_the_inputs(self):
        graph = WIDE
        assert len(ready_nodes(graph, set())) == 6


class TestPeaks:
    def test_running_branches_one_at_a_time_beats_running_them_together(self):
        # No allocator can recover the difference, because the values genuinely do overlap.
        assert peak_for_order(WIDE, depth_first_order(WIDE)) < peak_for_order(
            WIDE, breadth_first_order(WIDE)
        )

    def test_a_wider_graph_widens_the_gap(self):
        def spread(graph):
            return peak_for_order(graph, breadth_first_order(graph)) / peak_for_order(
                graph, depth_first_order(graph)
            )

        assert spread(branching_graph(8, 2)) > spread(branching_graph(4, 2))

    def test_a_chain_has_only_one_order_worth_the_name(self):
        graph = elementwise_chain(8)
        peaks = {peak_for_order(graph, ordering(graph)) for ordering in ORDERINGS.values()}
        assert len(peaks) == 1

    def test_the_greedy_rule_is_not_optimal(self):
        # A local rule for a problem that is not local. Depth first beats it on a wide graph,
        # which is worth knowing before trusting it anywhere.
        assert peak_for_order(WIDE, greedy_min_peak_order(WIDE)) > peak_for_order(
            WIDE, depth_first_order(WIDE)
        )

    def test_but_it_beats_the_naive_worklist(self):
        assert peak_for_order(WIDE, greedy_min_peak_order(WIDE)) < peak_for_order(
            WIDE, breadth_first_order(WIDE)
        )

    def test_an_unknown_allocator_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            arena_for_order(WIDE, depth_first_order(WIDE), strategy="annealing")


class TestEnumeration:
    def test_a_chain_has_exactly_one_order(self):
        assert len(all_orders(elementwise_chain(5))) == 1

    def test_a_wide_graph_has_many(self):
        assert len(all_orders(branching_graph(3, 1))) > 1

    def test_every_enumerated_order_is_valid(self):
        graph = branching_graph(3, 1)
        assert all(is_valid_order(graph, order) for order in all_orders(graph))

    def test_the_best_order_is_no_worse_than_any_heuristic(self):
        graph = branching_graph(3, 2)
        floor = peak_for_order(graph, best_order(graph))
        for ordering in ORDERINGS.values():
            assert peak_for_order(graph, ordering(graph)) >= floor

    def test_depth_first_finds_the_optimum_on_a_small_wide_graph(self):
        graph = branching_graph(3, 2)
        assert peak_for_order(graph, depth_first_order(graph)) == peak_for_order(
            graph, best_order(graph)
        )

    def test_the_worst_order_is_no_better_than_any_heuristic(self):
        graph = branching_graph(3, 2)
        ceiling = peak_for_order(graph, worst_order(graph))
        for ordering in ORDERINGS.values():
            assert peak_for_order(graph, ordering(graph)) <= ceiling

    def test_the_limit_bounds_the_search(self):
        assert len(all_orders(branching_graph(4, 2), limit=10)) == 10

    def test_a_zero_limit_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            all_orders(elementwise_chain(3), limit=0)


class TestComparison:
    def test_it_reports_every_heuristic(self):
        assert len(compare_orders(WIDE)) == len(ORDERINGS)

    def test_the_allocator_adds_nothing_on_top_of_a_good_order(self):
        rows = {row["order"]: row for row in compare_orders(WIDE)}
        assert rows["depth first"]["allocator_overhead"] == 0.0

    def test_the_order_moves_the_peak_more_than_the_allocator_does(self):
        # Once any reusing allocator is in place. Comparing against a plan that never reuses
        # anything makes the allocator look decisive everywhere, because that comparison is
        # having an allocator against not having one.
        result = order_versus_allocator(WIDE)
        assert result["order_matters_more"]
        assert result["order_spread"] > 1.5
        assert result["allocator_spread"] == 1.0

    def test_and_the_no_reuse_baseline_dwarfs_both(self):
        result = order_versus_allocator(WIDE)
        assert result["no_reuse_arena"] > 3 * result["best_arena"]

    def test_neither_matters_on_a_chain(self):
        result = order_versus_allocator(elementwise_chain(8))
        assert result["order_spread"] == 1.0
        assert result["allocator_spread"] == 1.0

    def test_the_scheduled_arena_is_the_best_this_compiler_produces(self):
        assert scheduled_arena(WIDE) == peak_for_order(WIDE, depth_first_order(WIDE))


class TestSemantics:
    def test_every_order_computes_the_same_answer(self):
        # Which is the premise the whole file rests on, and is cheap to check.
        graph = branching_graph(3, 2, width=8)
        feeds = random_feeds(graph)
        expected = run(graph, feeds)[0]
        for ordering in ORDERINGS.values():
            reordered = graph.with_nodes(ordering(graph))
            assert torch.equal(run(reordered, feeds)[0], expected)

    def test_an_enumerated_order_does_too(self):
        graph = branching_graph(3, 1, width=8)
        feeds = random_feeds(graph)
        expected = run(graph, feeds)[0]
        for order in all_orders(graph, limit=20):
            assert torch.equal(run(graph.with_nodes(order), feeds)[0], expected)
