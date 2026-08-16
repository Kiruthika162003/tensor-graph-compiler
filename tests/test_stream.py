from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.builder import branching_graph, elementwise_chain
from tgc.runtime.stream import (
    ASSIGNMENTS,
    Schedule,
    Task,
    a_queue_runs_one_task_at_a_time,
    assign,
    compare_assignments,
    compare_graphs,
    durations,
    every_task_runs_after_its_inputs,
    expensive_signals_undo_the_parallelism,
    list_scheduling_reaches_the_bound,
    occupancy_falls_as_queues_are_added,
    one_queue_is_the_serial_time,
    only_the_branching_graph_gains,
    round_robin_pays_for_ignoring_the_graph,
    signal_cost_sweep,
    simulate,
    speedup,
    stream_sweep,
    the_simulation_agrees_with_the_analysis,
    the_speedup_stops_at_the_parallelism,
)


class TestSimulation:
    def test_one_queue_takes_the_whole_work(self):
        assert one_queue_is_the_serial_time()["equal"]

    def test_no_task_starts_before_its_inputs_finish(self):
        # Starting early would shorten the makespan, which is the direction a bug is welcomed
        # in.
        assert every_task_runs_after_its_inputs()["violations"] == 0

    def test_and_no_queue_runs_two_things_at_once(self):
        assert a_queue_runs_one_task_at_a_time()["overlaps"] == 0

    def test_every_node_gets_a_task(self):
        graph = branching_graph()
        assert len(simulate(graph).tasks) == len(graph.nodes)

    def test_a_queue_count_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="at least one queue"):
            simulate(branching_graph(), streams=0)

    def test_an_unknown_assignment_is_refused(self):
        with pytest.raises(ConfigError, match="unknown assignment"):
            simulate(branching_graph(), policy="guesswork")

    def test_and_when_asked_for_a_placement(self):
        with pytest.raises(ConfigError, match="unknown assignment"):
            assign(branching_graph(), 4, "guesswork")

    def test_list_scheduling_has_no_static_placement(self):
        assert assign(branching_graph(), 4, "list scheduling") == {}

    def test_every_node_has_a_duration(self):
        graph = branching_graph()
        assert len(durations(graph)) == len(graph.nodes)

    def test_nothing_takes_less_than_a_unit(self):
        assert min(durations(branching_graph()).values()) >= 1.0

    def test_a_task_knows_when_it_finished(self):
        assert Task(name="a", stream=0, start=2.0, duration=3.0).finish == 5.0

    def test_an_empty_schedule_has_no_makespan(self):
        assert Schedule().makespan == 0.0

    def test_and_no_occupancy(self):
        assert Schedule().occupancy == 0.0

    def test_it_serialises(self):
        assert simulate(branching_graph()).as_dict()["streams"] == 4


class TestBound:
    def test_the_speedup_never_beats_the_parallelism(self):
        # A schedule that beat it would be running tasks before their inputs existed.
        assert the_speedup_stops_at_the_parallelism()["within_the_bound"]

    def test_and_reaches_it(self):
        result = the_speedup_stops_at_the_parallelism()
        assert result["best_speedup"] == result["bound"]

    def test_list_scheduling_reaches_the_bound(self):
        result = list_scheduling_reaches_the_bound()
        assert result["list_scheduling"] == result["bound"]

    def test_and_so_does_level_based_placement(self):
        result = list_scheduling_reaches_the_bound()
        assert result["by_level"] == result["bound"]

    def test_round_robin_does_not(self):
        result = list_scheduling_reaches_the_bound()
        assert result["round_robin"] < result["bound"]

    def test_the_speedup_flattens_once_the_graph_runs_out(self):
        rows = {row["streams"]: row for row in stream_sweep()}
        assert rows[8]["speedup"] == rows[16]["speedup"]

    def test_and_the_occupancy_falls_with_it(self):
        assert occupancy_falls_as_queues_are_added()["falling"]

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            stream_sweep(counts=())


class TestPlacement:
    def test_round_robin_signals_far_more(self):
        # It puts a value and its consumer on different queues about as often as not.
        result = round_robin_pays_for_ignoring_the_graph()
        assert result["round_robin"] > 2 * result["list_scheduling"]

    def test_and_the_two_graph_aware_policies_tie(self):
        result = round_robin_pays_for_ignoring_the_graph()
        assert result["by_level"] == result["list_scheduling"]

    def test_every_assignment_is_compared(self):
        assert len(compare_assignments()) == len(ASSIGNMENTS)

    def test_the_worst_placement_is_also_the_slowest(self):
        rows = {row["policy"]: row for row in compare_assignments()}
        assert rows["round robin"]["makespan"] > rows["list scheduling"]["makespan"]

    def test_a_placement_covers_every_node(self):
        graph = branching_graph()
        assert len(assign(graph, 4, "round robin")) == len(graph.nodes)

    def test_and_stays_inside_the_queue_count(self):
        placement = assign(branching_graph(), 4, "by level")
        assert max(placement.values()) < 4


class TestSignals:
    def test_free_signals_leave_the_speedup_alone(self):
        rows = signal_cost_sweep()
        assert rows[0]["worth_it"]

    def test_expensive_ones_undo_it(self):
        assert not signal_cost_sweep()[-1]["worth_it"]

    def test_the_break_even_is_a_long_way_out(self):
        # Seven signals for fifteen nodes, so it takes a lot to matter.
        result = expensive_signals_undo_the_parallelism()
        assert result["still_worth_it"] == result["of"] - 1

    def test_a_single_queue_signals_only_from_its_inputs(self):
        assert one_queue_is_the_serial_time()["signals"] == 4

    def test_an_empty_cost_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            signal_cost_sweep(costs=())


class TestGraphs:
    def test_a_chain_gains_nothing_from_more_queues(self):
        assert speedup(elementwise_chain(8), 8) == 1.0

    def test_only_some_fixtures_gain_anything(self):
        assert len(only_the_branching_graph_gains()["gained"]) < 4

    def test_and_the_branching_one_is_among_them(self):
        assert "branching" in only_the_branching_graph_gains()["gained"]

    def test_the_simulation_mostly_agrees_with_the_analysis(self):
        assert the_simulation_agrees_with_the_analysis()["agreeing"] == 3

    def test_and_the_one_disagreement_is_the_duration_floor(self):
        # This file floors a task at one unit and the analysis does not, so a graph of tiny
        # nodes measures differently.
        rows = {row["graph"]: row for row in the_simulation_agrees_with_the_analysis()["rows"]}
        assert not rows["layernorm"]["agree"]
        assert rows["layernorm"]["measured"] > rows["layernorm"]["predicted"]

    def test_four_graphs_are_compared(self):
        assert len(compare_graphs()) == 4

    def test_the_measured_speedup_never_exceeds_the_bound_by_much(self):
        for row in compare_graphs():
            assert row["speedup"] <= row["parallelism"] + 0.2
