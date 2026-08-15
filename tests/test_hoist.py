from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.graph import validate
from tgc.passes.hoist import (
    SinkReport,
    broadcast_bytes,
    can_sink,
    check_sinking,
    compare_fixtures,
    contracting_broadcast_graph,
    elementwise_broadcast_graph,
    is_broadcast,
    measure_sinking,
    reduced_broadcast_graph,
    report_sinking,
    saving_fraction,
    sink_broadcasts,
)
from tgc.runtime.executor import compile_graph
from tgc.verify.reference import outputs_agree, random_feeds, run

FIXTURES = {
    "elementwise": elementwise_broadcast_graph(16),
    "matmul": contracting_broadcast_graph(16),
    "reduction": reduced_broadcast_graph(16),
}


class TestRecognition:
    def test_a_broadcast_is_recognised(self):
        graph = elementwise_broadcast_graph(16)
        assert any(is_broadcast(node) for node in graph.nodes)

    def test_a_reduction_is_not(self):
        graph = softmax_graph()
        assert not any(is_broadcast(node) for node in graph.nodes)

    def test_a_broadcast_read_only_by_elementwise_ops_can_be_sunk(self):
        graph = elementwise_broadcast_graph(16)
        broadcast = next(node.name for node in graph.nodes if is_broadcast(node))
        allowed, _ = can_sink(graph, broadcast)
        assert allowed

    def test_one_read_by_a_matrix_product_cannot(self):
        # A matmul contracts over the broadcast axis and reads every copy.
        graph = contracting_broadcast_graph(16)
        broadcast = next(node.name for node in graph.nodes if is_broadcast(node))
        allowed, reason = can_sink(graph, broadcast)
        assert not allowed
        assert "matmul" in reason

    def test_nor_one_read_by_a_reduction(self):
        graph = reduced_broadcast_graph(16)
        broadcast = next(node.name for node in graph.nodes if is_broadcast(node))
        allowed, reason = can_sink(graph, broadcast)
        assert not allowed
        assert "sum" in reason

    def test_a_broadcast_that_is_an_output_cannot(self):
        # The caller asked for the wide tensor.
        builder = Builder()
        x = builder.input([1, 8], name="x")
        graph = builder.finish(builder.broadcast_to(x, [4, 8]))
        allowed, reason = can_sink(graph, graph.outputs[0])
        assert not allowed
        assert "graph output" in reason

    def test_something_that_is_not_a_broadcast_is_rejected(self):
        allowed, reason = can_sink(softmax_graph(), "v0")
        assert not allowed
        assert "not a broadcast" in reason

    def test_the_check_raises_with_the_reason(self):
        graph = contracting_broadcast_graph(16)
        broadcast = next(node.name for node in graph.nodes if is_broadcast(node))
        with pytest.raises(PassError, match="needs the widened shape"):
            check_sinking(graph, broadcast)


class TestSinking:
    def test_the_broadcast_disappears(self):
        graph = elementwise_broadcast_graph(16)
        assert broadcast_bytes(sink_broadcasts(graph)) == 0

    def test_and_took_real_bytes_before(self):
        assert broadcast_bytes(elementwise_broadcast_graph(16)) > 0

    def test_the_graph_gets_shorter(self):
        result = measure_sinking()
        assert result["nodes_after"] == result["nodes_before"] - 1

    def test_the_peak_falls_by_about_a_third(self):
        assert saving_fraction() > 0.3

    def test_the_answer_is_bit_identical(self):
        # An elementwise operation broadcasts mismatched operands anyway, so reading the
        # narrow tensor performs the same arithmetic on the same values in the same order.
        for name, graph in FIXTURES.items():
            feeds = random_feeds(graph, positive=True)
            assert outputs_agree(run(graph, feeds), run(sink_broadcasts(graph), feeds)), name

    def test_the_result_still_validates(self):
        for graph in FIXTURES.values():
            validate(sink_broadcasts(graph))

    def test_the_output_shape_is_unchanged(self):
        # The other operand still carries the width.
        for graph in FIXTURES.values():
            sunk = sink_broadcasts(graph)
            assert graph.value(graph.outputs[0]).shape == sunk.value(sunk.outputs[0]).shape

    def test_a_graph_that_needs_its_broadcast_keeps_it(self):
        for name in ("matmul", "reduction"):
            graph = FIXTURES[name]
            assert broadcast_bytes(sink_broadcasts(graph)) == broadcast_bytes(graph)

    def test_running_it_twice_changes_nothing_the_second_time(self):
        once = sink_broadcasts(elementwise_broadcast_graph(16))
        assert len(sink_broadcasts(once).nodes) == len(once.nodes)

    def test_a_graph_with_no_broadcasts_is_left_alone(self):
        graph = softmax_graph()
        assert len(sink_broadcasts(graph).nodes) == len(graph.nodes)

    def test_the_sunk_graph_compiles_and_matches(self):
        graph = elementwise_broadcast_graph(16)
        sunk = sink_broadcasts(graph)
        feeds = random_feeds(graph, positive=True)
        assert outputs_agree(compile_graph(sunk)(feeds), run(graph, feeds))


class TestReporting:
    def test_it_names_what_it_sank(self):
        assert report_sinking(elementwise_broadcast_graph(16)).count == 1

    def test_and_why_it_kept_the_rest(self):
        report = report_sinking(contracting_broadcast_graph(16))
        assert report.count == 0
        assert report.kept

    def test_every_fixture_is_compared(self):
        assert len(compare_fixtures()) == 3

    def test_the_two_refusals_have_different_reasons(self):
        rows = {row["graph"]: row for row in compare_fixtures()}
        assert rows["matmul reader"]["reason"] != rows["reduction reader"]["reason"]

    def test_a_graph_with_no_broadcasts_reports_nothing(self):
        assert report_sinking(softmax_graph()).count == 0

    def test_an_empty_report_sinks_nothing(self):
        assert SinkReport().count == 0

    def test_it_serialises(self):
        assert report_sinking(elementwise_broadcast_graph(16)).as_dict()["sunk"] == 1


class TestFixtures:
    def test_more_readers_do_not_change_how_many_broadcasts_there_are(self):
        assert report_sinking(elementwise_broadcast_graph(16, readers=5)).count == 1

    def test_a_degenerate_width_is_rejected(self):
        with pytest.raises(ConfigError, match="width above one"):
            elementwise_broadcast_graph(1)

    def test_a_fixture_with_no_readers_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one reader"):
            elementwise_broadcast_graph(16, readers=0)

    def test_the_contracting_fixture_rejects_a_degenerate_width(self):
        with pytest.raises(ConfigError, match="width above one"):
            contracting_broadcast_graph(1)

    def test_the_reducing_fixture_does_too(self):
        with pytest.raises(ConfigError, match="width above one"):
            reduced_broadcast_graph(1)
