from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, elementwise_chain, softmax_graph
from tgc.ir.graph import Graph, Node, Value
from tgc.ir.shape import shape
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.manager import (
    Pass,
    PassResult,
    Pipeline,
    PipelineReport,
    compose,
    count_ops,
    fixed_point,
    graphs_equal,
    identity,
    run_once,
)


def drop_last(graph: Graph) -> Graph:
    """A pass that removes the last node, which usually breaks the graph."""
    return graph.with_nodes(graph.nodes[:-1])


def not_a_graph(graph: Graph) -> object:
    """A pass that forgets to return a graph."""
    return len(graph.nodes)


class TestPass:
    def test_a_pass_reports_what_it_removed(self):
        graph = elementwise_chain(4)
        _, result = Pass(name="drop", transform=drop_last).run(graph, check=False)
        assert result.removed == 1

    def test_a_pass_that_changes_nothing_says_so(self):
        _, result = Pass(name="nothing", transform=identity).run(elementwise_chain(4))
        assert not result.changed

    def test_a_pass_that_breaks_the_graph_fails_at_that_pass(self):
        # Rather than during code generation four passes later.
        with pytest.raises(PassError, match="produced an invalid graph"):
            Pass(name="drop", transform=drop_last).run(elementwise_chain(4))

    def test_a_pass_that_returns_the_wrong_thing_is_caught(self):
        with pytest.raises(PassError, match="rather than a graph"):
            Pass(name="odd", transform=not_a_graph).run(elementwise_chain(4))

    def test_a_nameless_pass_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Pass(name="", transform=identity)

    def test_something_uncallable_is_rejected(self):
        with pytest.raises(ConfigError, match="not callable"):
            Pass(name="odd", transform=42)

    def test_it_serialises(self):
        assert (
            PassResult(name="p", nodes_before=4, nodes_after=2, changed=True).as_dict()[
                "removed"
            ]
            == 2
        )


class TestEquality:
    def test_a_graph_equals_itself(self):
        assert graphs_equal(softmax_graph(), softmax_graph())

    def test_a_rebuilt_graph_that_changed_nothing_compares_equal(self):
        # Otherwise a pass that rebuilds every node while changing nothing reports a change,
        # and the fixed point loop runs forever.
        graph = softmax_graph()
        assert graphs_equal(graph, graph.clone())

    def test_different_node_counts_are_not_equal(self):
        graph = elementwise_chain(4)
        assert not graphs_equal(graph, graph.with_nodes(graph.nodes[:-1]))

    def test_different_outputs_are_not_equal(self):
        graph = elementwise_chain(4)
        other = Graph(nodes=list(graph.nodes), inputs=list(graph.inputs), outputs=["v0"])
        assert not graphs_equal(graph, other)

    def test_different_operands_are_not_equal(self):
        builder = Builder()
        x = builder.input([4], name="x")
        y = builder.input([4], name="y")
        first = builder.finish(builder.add(x, y))
        second = first.with_nodes(
            [Node(op=ops.ADD, inputs=("y", "x"), output=first.nodes[0].output)]
        )
        assert not graphs_equal(first, second)


class TestPipeline:
    def test_it_runs_until_nothing_changes(self):
        pipeline = Pipeline(passes=[Pass(name="dce", transform=eliminate_dead_code)])
        _, report = pipeline.run(softmax_graph())
        assert report.rounds == 1

    def test_a_pass_that_fires_forces_another_round(self):
        builder = Builder()
        x = builder.input([4], name="x")
        builder.exp(x)
        graph = builder.finish(builder.neg(x))
        pipeline = Pipeline(passes=[Pass(name="dce", transform=eliminate_dead_code)])
        _, report = pipeline.run(graph)
        assert report.rounds == 2
        assert report.changed

    def test_the_report_names_what_fired(self):
        builder = Builder()
        x = builder.input([4], name="x")
        builder.exp(x)
        graph = builder.finish(builder.neg(x))
        pipeline = Pipeline(
            passes=[
                Pass(name="dce", transform=eliminate_dead_code),
                Pass(name="nothing", transform=identity),
            ]
        )
        _, report = pipeline.run(graph)
        assert report.passes_that_fired() == ["dce"]

    def test_two_passes_undoing_each_other_are_reported(self):
        # Rather than looping forever, which is what a plain fixed point does the first time
        # somebody canonicalises in the opposite direction to an existing rule.
        def grow(graph: Graph) -> Graph:
            grown = Value(
                name=f"g{len(graph.nodes)}", shape=shape(4), dtype=graph.value("x").dtype
            )
            return graph.with_nodes(
                [*graph.nodes, Node(op=ops.NEG, inputs=("x",), output=grown)]
            )

        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.neg(x))
        pipeline = Pipeline(
            passes=[
                Pass(name="grow", transform=grow),
                Pass(name="dce", transform=eliminate_dead_code),
            ],
            max_rounds=3,
        )
        with pytest.raises(PassError, match="undoing each other"):
            pipeline.run(graph)

    def test_a_pass_can_be_turned_off(self):
        pipeline = Pipeline(
            passes=[
                Pass(name="dce", transform=eliminate_dead_code),
                Pass(name="nothing", transform=identity),
            ]
        )
        assert len(pipeline.without("dce").enabled_passes()) == 1

    def test_only_keeps_one(self):
        pipeline = Pipeline(
            passes=[
                Pass(name="dce", transform=eliminate_dead_code),
                Pass(name="nothing", transform=identity),
            ]
        )
        assert [item.name for item in pipeline.only("dce").enabled_passes()] == ["dce"]

    def test_turning_off_an_unknown_pass_is_rejected(self):
        pipeline = Pipeline(passes=[Pass(name="dce", transform=eliminate_dead_code)])
        with pytest.raises(ConfigError, match="no such pass"):
            pipeline.without("fusion")

    def test_two_passes_with_the_same_name_are_rejected(self):
        with pytest.raises(ConfigError, match="share a name"):
            Pipeline(
                passes=[
                    Pass(name="dce", transform=identity),
                    Pass(name="dce", transform=identity),
                ]
            )

    def test_a_pipeline_that_never_runs_is_rejected(self):
        with pytest.raises(ConfigError, match="at least once"):
            Pipeline(passes=[], max_rounds=0)

    def test_an_empty_pipeline_changes_nothing(self):
        result, report = Pipeline(passes=[]).run(softmax_graph())
        assert not report.changed
        assert graphs_equal(result, softmax_graph())

    def test_it_serialises(self):
        assert PipelineReport().as_dict()["rounds"] == 0


class TestHelpers:
    def test_running_once_applies_the_transformation(self):
        builder = Builder()
        x = builder.input([4], name="x")
        builder.exp(x)
        graph = builder.finish(builder.neg(x))
        assert len(run_once(graph, eliminate_dead_code).nodes) == 1

    def test_a_fixed_point_stops_when_nothing_changes(self):
        assert graphs_equal(fixed_point(softmax_graph(), identity), softmax_graph())

    def test_a_transformation_that_never_settles_is_reported(self):
        def grow(graph: Graph) -> Graph:
            output = Value(
                name=f"g{len(graph.nodes)}", shape=shape(4), dtype=graph.value("x").dtype
            )
            return graph.with_nodes(
                [*graph.nodes, Node(op=ops.NEG, inputs=("x",), output=output)]
            )

        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.neg(x))
        with pytest.raises(PassError, match="still changing"):
            fixed_point(graph, grow, limit=3)

    def test_a_zero_limit_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            fixed_point(softmax_graph(), identity, limit=0)

    def test_composition_runs_in_order(self):
        combined = compose(eliminate_dead_code, identity)
        builder = Builder()
        x = builder.input([4], name="x")
        builder.exp(x)
        graph = builder.finish(builder.neg(x))
        assert len(combined(graph).nodes) == 1

    def test_composing_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compose"):
            compose()

    def test_the_identity_pass_is_a_control(self):
        # A measurement that swaps one pass for this one is measuring that pass. A
        # measurement that compares two different pipelines is measuring something harder.
        assert graphs_equal(identity(softmax_graph()), softmax_graph())

    def test_ops_can_be_counted_by_name(self):
        assert count_ops(softmax_graph(), ["sum", "max"]) == 2

    def test_counting_something_absent_gives_zero(self):
        assert count_ops(softmax_graph(), ["matmul"]) == 0
