from __future__ import annotations

import pytest

from tgc.analysis.liveness import (
    Interval,
    LivenessReport,
    analyse,
    bytes_live_at,
    compute_intervals,
    conflict_graph,
    live_at,
    max_simultaneous,
    peak_bytes,
    peak_step,
    total_bytes,
)
from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import Builder, branching_graph, elementwise_chain, softmax_graph


class TestInterval:
    def test_the_length_counts_both_ends(self):
        assert Interval(name="a", start=2, end=4, size=8).length == 3

    def test_a_value_alive_for_one_step_has_length_one(self):
        assert Interval(name="a", start=2, end=2, size=8).length == 1

    def test_two_touching_intervals_overlap(self):
        # A value produced at the step another is last read at does overlap it, because the
        # reading step needs both present.
        first = Interval(name="a", start=0, end=2, size=8)
        second = Interval(name="b", start=2, end=4, size=8)
        assert first.overlaps(second)

    def test_two_separated_intervals_do_not(self):
        first = Interval(name="a", start=0, end=1, size=8)
        second = Interval(name="b", start=2, end=4, size=8)
        assert not first.overlaps(second)

    def test_overlap_is_symmetric(self):
        first = Interval(name="a", start=0, end=3, size=8)
        second = Interval(name="b", start=2, end=4, size=8)
        assert first.overlaps(second) == second.overlaps(first)

    def test_it_knows_which_steps_it_covers(self):
        interval = Interval(name="a", start=1, end=3, size=8)
        assert interval.contains(2)
        assert not interval.contains(4)

    def test_an_interval_that_ends_before_it_starts_is_rejected(self):
        with pytest.raises(ScheduleError, match="before it starts"):
            Interval(name="a", start=4, end=2, size=8)

    def test_a_nameless_interval_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Interval(name="", start=0, end=0, size=8)

    def test_a_negative_size_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot occupy"):
            Interval(name="a", start=0, end=0, size=-1)

    def test_it_serialises(self):
        assert Interval(name="a", start=2, end=4, size=8).as_dict()["length"] == 3


class TestComputation:
    def test_every_value_gets_an_interval(self):
        graph = softmax_graph()
        intervals = compute_intervals(graph)
        assert {interval.name for interval in intervals} == graph.value_names

    def test_an_input_is_alive_before_the_first_step(self):
        intervals = {i.name: i for i in compute_intervals(softmax_graph())}
        assert intervals["x"].start == -1

    def test_an_output_is_alive_after_the_last_step(self):
        graph = softmax_graph()
        intervals = {i.name: i for i in compute_intervals(graph)}
        assert intervals[graph.outputs[0]].end == len(graph.nodes) - 1

    def test_an_intermediate_dies_at_its_last_reader(self):
        graph = elementwise_chain(4)
        intervals = {i.name: i for i in compute_intervals(graph)}
        assert intervals["v0"].end == 1

    def test_a_value_nothing_reads_still_occupies_its_own_step(self):
        # It is written before anybody could have decided it was pointless.
        builder = Builder()
        x = builder.input([4], name="x")
        dead = builder.exp(x)
        graph = builder.finish(builder.neg(x))
        intervals = {i.name: i for i in compute_intervals(graph)}
        assert intervals[dead].length == 1

    def test_an_input_that_is_also_an_output_lives_throughout(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.neg(x), "x")
        intervals = {i.name: i for i in compute_intervals(graph)}
        assert intervals["x"].end == len(graph.nodes) - 1

    def test_a_different_order_gives_different_intervals(self):
        graph = branching_graph(3, 2)
        forward = compute_intervals(graph)
        reversed_order = compute_intervals(graph, list(reversed(graph.nodes)))
        assert forward != reversed_order

    def test_running_a_node_twice_is_rejected(self):
        graph = elementwise_chain(3)
        with pytest.raises(ScheduleError, match="same node twice"):
            compute_intervals(graph, [graph.nodes[0], graph.nodes[0], graph.nodes[1]])


class TestPeak:
    def test_a_chain_holds_two_tensors_at_once(self):
        # The one being read and the one being written, which is the whole reason a chain is
        # cheap and a wide graph is not.
        assert max_simultaneous(compute_intervals(elementwise_chain(8))) == 2

    def test_the_peak_is_the_worst_moment(self):
        intervals = compute_intervals(elementwise_chain(8))
        step = peak_step(intervals)
        assert bytes_live_at(intervals, step) == peak_bytes(intervals)

    def test_the_peak_is_below_the_naive_total(self):
        intervals = compute_intervals(elementwise_chain(8))
        assert peak_bytes(intervals) < total_bytes(intervals)

    def test_an_empty_schedule_has_no_peak(self):
        assert peak_bytes([]) == 0
        with pytest.raises(ScheduleError, match="no peak"):
            peak_step([])

    def test_it_counts_what_is_alive_at_a_step(self):
        intervals = compute_intervals(elementwise_chain(4))
        assert len(live_at(intervals, 0)) == 2

    def test_the_headroom_is_what_reuse_could_remove(self):
        report = analyse(elementwise_chain(8))
        assert report.reuse_headroom > 0.7

    def test_a_graph_where_everything_is_live_has_no_headroom(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.neg(x), "x")
        assert analyse(graph).reuse_headroom == 0.0

    def test_the_longest_lived_value_is_reported(self):
        report = analyse(elementwise_chain(8))
        assert report.longest_lived is not None
        assert report.longest_lived.name == "x"

    def test_an_empty_report_has_no_longest(self):
        assert LivenessReport().longest_lived is None

    def test_it_serialises(self):
        assert analyse(elementwise_chain(8)).as_dict()["max_simultaneous"] == 2


class TestConflicts:
    def test_overlapping_values_conflict(self):
        intervals = compute_intervals(elementwise_chain(4))
        conflicts = conflict_graph(intervals)
        assert "v1" in conflicts["v0"]

    def test_separated_values_do_not(self):
        intervals = compute_intervals(elementwise_chain(4))
        conflicts = conflict_graph(intervals)
        assert "v3" not in conflicts["v0"]

    def test_conflict_is_symmetric(self):
        conflicts = conflict_graph(compute_intervals(branching_graph(3, 2)))
        for name, others in conflicts.items():
            for other in others:
                assert name in conflicts[other]

    def test_a_wide_graph_conflicts_more_than_a_chain(self):
        wide = conflict_graph(compute_intervals(branching_graph(4, 2)))
        narrow = conflict_graph(compute_intervals(elementwise_chain(8)))
        assert max(len(v) for v in wide.values()) > max(len(v) for v in narrow.values())

    def test_nothing_conflicts_with_itself(self):
        conflicts = conflict_graph(compute_intervals(elementwise_chain(4)))
        assert all(name not in others for name, others in conflicts.items())
