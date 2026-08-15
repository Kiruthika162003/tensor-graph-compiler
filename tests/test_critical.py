from __future__ import annotations

import pytest

from tgc.analysis.critical import (
    ParallelismReport,
    Span,
    a_chain_has_no_parallelism,
    analyse,
    branching_is_the_only_fixture_with_any,
    chain_share_by_graph,
    compare_graphs,
    cost_weighting_changes_the_answer,
    critical_path,
    diminishing_returns,
    level_widths,
    levels,
    longest_chain_by_nodes,
    node_cost,
    parallelism_on_generated_graphs,
    speedup_bounds,
    the_chain_is_the_whole_graph,
    total_work,
    unbalanced_graph,
    weighting_matters_somewhere,
    where_more_processors_stop_helping,
    widest_level,
)
from tgc.errors import ConfigError
from tgc.ir.builder import branching_graph, elementwise_chain, mlp_graph, softmax_graph


class TestSpan:
    def test_a_chain_is_its_own_critical_path(self):
        result = critical_path(elementwise_chain(8))
        assert result.span == result.work

    def test_so_its_parallelism_is_exactly_one(self):
        assert critical_path(elementwise_chain(8)).parallelism == 1.0

    def test_the_chain_is_reported_in_order(self):
        chain = critical_path(elementwise_chain(4)).longest_chain
        assert list(chain) == ["v0", "v1", "v2", "v3"]

    def test_a_branching_graph_has_more_work_than_span(self):
        result = critical_path(branching_graph())
        assert result.work > result.span

    def test_an_empty_span_reports_no_parallelism(self):
        assert Span(work=0.0, span=0.0).parallelism == 0.0

    def test_a_leaf_costs_nothing(self):
        # Giving one a cost puts a length on a chain that has no work in it.
        graph = softmax_graph()
        assert node_cost(graph, graph.inputs[0].name) == 0.0

    def test_the_work_is_the_sum_of_the_nodes(self):
        graph = elementwise_chain(4)
        assert total_work(graph) == sum(node_cost(graph, node.name) for node in graph.nodes)

    def test_it_serialises(self):
        assert critical_path(elementwise_chain(4)).as_dict()["depth"] == 4


class TestLevels:
    def test_a_chain_has_one_node_per_level(self):
        assert level_widths(elementwise_chain(8)) == [1] * 8

    def test_a_branching_graph_is_wider(self):
        assert widest_level(branching_graph()) > 1

    def test_every_node_lands_in_exactly_one_level(self):
        graph = branching_graph()
        placed = sum(len(level) for level in levels(graph))
        assert placed == len(graph.nodes)

    def test_nothing_in_a_level_reads_anything_else_in_it(self):
        graph = branching_graph()
        for level in levels(graph):
            names = set(level)
            for name in level:
                assert not names & set(graph.node(name).inputs)

    def test_an_empty_graph_has_no_widest_level(self):
        builder_free = softmax_graph().with_nodes([])
        assert widest_level(builder_free) == 0


class TestBounds:
    def test_the_bound_climbs_with_the_processor_count_and_then_stops(self):
        rows = speedup_bounds(branching_graph())
        assert rows[0]["bound"] < rows[2]["bound"]
        assert rows[-1]["bound"] == rows[-2]["bound"]

    def test_past_a_point_the_graph_is_the_limit(self):
        rows = speedup_bounds(branching_graph())
        assert rows[-1]["limited_by"] == "the graph"

    def test_one_processor_is_limited_by_the_processor(self):
        assert speedup_bounds(branching_graph())[0]["limited_by"] == "processors"

    def test_a_chain_gains_nothing_from_any_machine(self):
        assert all(row["bound"] == 1.0 for row in speedup_bounds(elementwise_chain(8)))

    def test_zero_processors_is_refused(self):
        with pytest.raises(ConfigError, match="at least one processor"):
            critical_path(softmax_graph()).speedup_bound(0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            speedup_bounds(softmax_graph(), counts=())

    def test_a_chain_uses_one_processor(self):
        assert where_more_processors_stop_helping(elementwise_chain(8)) == 1

    def test_a_branching_graph_uses_four(self):
        assert where_more_processors_stop_helping(branching_graph()) == 4

    def test_every_fixture_is_reported(self):
        assert len(diminishing_returns()) == 5


class TestFixtures:
    def test_almost_none_of_them_have_any_parallelism(self):
        # A compiler looking at one graph of one example is looking at something with a ratio
        # near one, and every claim about a clever schedule has to fit under that.
        rows = {row["graph"]: row["parallelism"] for row in compare_graphs()}
        near_one = [name for name, value in rows.items() if value < 1.2]
        assert len(near_one) == 5

    def test_only_the_branching_one_does(self):
        assert branching_is_the_only_fixture_with_any()["best"] == "branching"

    def test_and_the_next_best_is_barely_above_one(self):
        assert branching_is_the_only_fixture_with_any()["next_best"] < 1.2

    def test_a_chain_is_all_critical_path(self):
        assert a_chain_has_no_parallelism()["parallelism"] == 1.0

    def test_and_every_node_is_on_it(self):
        result = a_chain_has_no_parallelism()
        assert result["depth"] == result["nodes"]

    def test_most_of_a_layer_is_on_its_critical_path(self):
        assert the_chain_is_the_whole_graph(softmax_graph())["share"] == 1.0

    def test_and_less_than_half_of_a_branching_graph(self):
        rows = {row["graph"]: row for row in chain_share_by_graph()}
        assert rows["branching"]["share"] < 0.5


class TestWeighting:
    def test_the_ordinary_fixtures_do_not_distinguish_the_two_measures(self):
        # They agree because those graphs have one long path and no choice to get wrong.
        rows = {row["graph"]: row for row in weighting_matters_somewhere()}
        assert rows["chain"]["same_path"]
        assert rows["softmax"]["same_path"]

    def test_the_unbalanced_one_does(self):
        # Two matrix products are shorter in nodes and far longer in time than ten elementwise
        # operations, and a scheduler that counts nodes shortens the wrong branch.
        assert not cost_weighting_changes_the_answer(unbalanced_graph())["same_path"]

    def test_the_expensive_branch_is_the_shorter_one(self):
        result = cost_weighting_changes_the_answer(unbalanced_graph())
        assert result["weighted_length"] < result["counted_length"]

    def test_the_cheap_branch_wins_on_node_count(self):
        assert len(longest_chain_by_nodes(unbalanced_graph())) == 11

    def test_a_cheap_branch_needs_some_length(self):
        with pytest.raises(ConfigError, match="needs some length"):
            unbalanced_graph(cheap=0)

    def test_six_fixtures_are_compared(self):
        assert len(weighting_matters_somewhere()) == 6


class TestReports:
    def test_a_report_records_the_shape_over_time(self):
        report = analyse(branching_graph(), "branching")
        assert sum(report.widths) == len(branching_graph().nodes)

    def test_an_empty_report_has_no_parallelism(self):
        assert ParallelismReport(label="nothing").parallelism == 0.0

    def test_it_serialises(self):
        assert analyse(mlp_graph(), "mlp").as_dict()["graph"] == "mlp"

    def test_generated_graphs_are_wider_than_the_fixtures(self):
        # Which says something about the fuzzer rather than about models. It builds nodes from
        # whatever is available rather than from what a person would write next.
        result = parallelism_on_generated_graphs()
        assert result["generated_mean"] > result["fixture_mean"]

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            parallelism_on_generated_graphs(count=0)
