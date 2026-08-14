from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError, VerificationError
from tgc.ir.builder import (
    Builder,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.passes.fusion import (
    FusedNode,
    FusionGroup,
    arithmetic_intensity,
    can_extend_group,
    can_start_group,
    find_groups,
    fused_bytes,
    fused_intermediates,
    peak_intermediates,
    report_fusion,
    to_fused_nodes,
    traffic_ratio,
    unfused_bytes,
)
from tgc.verify.fused import (
    every_group_is_equivalent,
    groups_are_equivalent,
    run_group_elementwise,
)
from tgc.verify.reference import random_feeds


def gentle_chain(length: int = 8):
    """A chain that does not overflow, so the comparison is about fusion and not about inf."""
    builder = Builder()
    current = builder.input([8, 8], name="x")
    for index in range(length):
        current = builder.tanh(current) if index % 2 else builder.relu(current)
    return builder.finish(current)


class TestGrouping:
    def test_a_chain_becomes_one_group(self):
        groups = find_groups(elementwise_chain(8))
        assert len(groups) == 1
        assert groups[0].size == 8

    def test_a_reduction_ends_a_group(self):
        # A sum reads every element to produce one, so there is no element i to carry.
        assert report_fusion(softmax_graph()).largest_group == 2

    def test_a_matmul_ends_one_too(self):
        assert report_fusion(mlp_graph()).largest_group == 2

    def test_a_value_read_twice_is_not_fused_over(self):
        # Fusing into both readers computes it twice, which is sometimes right and is never
        # free, so the pass will not do it silently.
        graph = diamond_graph()
        assert all(group.size <= 2 for group in find_groups(graph))

    def test_a_graph_output_is_not_fused_over(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        middle = builder.relu(x)
        graph = builder.finish(builder.exp(middle), middle)
        assert all(group.size == 1 for group in find_groups(graph))

    def test_an_elementwise_node_can_start_a_group(self):
        assert can_start_group(softmax_graph().node("v1"))

    def test_a_reduction_cannot(self):
        assert not can_start_group(softmax_graph().node("v0"))

    def test_extending_needs_the_previous_value_read_once(self):
        graph = diamond_graph()
        counts = graph.use_counts()
        assert not can_extend_group(graph.node("v1"), "v0", counts, set())

    def test_extending_needs_the_node_to_read_it(self):
        graph = elementwise_chain(4)
        counts = graph.use_counts()
        assert not can_extend_group(graph.node("v3"), "v0", counts, set())

    def test_every_node_belongs_to_at_most_one_group(self):
        for graph in (elementwise_chain(8), softmax_graph(), layernorm_graph(), mlp_graph()):
            members = [name for group in find_groups(graph) for name in group.members]
            assert len(members) == len(set(members))

    def test_a_group_names_what_it_reads_from_outside(self):
        group = find_groups(elementwise_chain(4))[0]
        assert group.inputs == ["x"]

    def test_a_single_node_group_is_trivial(self):
        assert FusionGroup(members=["a"]).is_trivial

    def test_the_last_member_is_the_output(self):
        assert FusionGroup(members=["a", "b"]).output == "b"

    def test_the_rest_are_intermediates(self):
        assert FusionGroup(members=["a", "b", "c"]).intermediates == ["a", "b"]

    def test_it_serialises(self):
        assert FusionGroup(members=["a", "b"]).as_dict()["size"] == 2


class TestTraffic:
    def test_a_chain_of_eight_moves_an_eighth_of_the_bytes(self):
        # An unfused chain writes eight tensors and reads eight. Fused it writes one and
        # reads one, and the arithmetic is identical.
        assert traffic_ratio(elementwise_chain(8)) == pytest.approx(8.0)

    def test_a_longer_chain_saves_more(self):
        assert traffic_ratio(elementwise_chain(16)) > traffic_ratio(elementwise_chain(4))

    def test_a_softmax_saves_something(self):
        assert traffic_ratio(softmax_graph()) > 1.0

    def test_a_graph_with_nothing_to_fuse_saves_nothing(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        graph = builder.finish(builder.sum(x, axes=[1]))
        assert traffic_ratio(graph) == 1.0

    def test_fusing_never_moves_more(self):
        for graph in (elementwise_chain(8), softmax_graph(), layernorm_graph(), mlp_graph()):
            assert fused_bytes(graph) <= unfused_bytes(graph)

    def test_an_empty_tensor_moves_nothing_and_cannot_be_compared(self):
        builder = Builder()
        x = builder.input([0, 4], name="x")
        graph = builder.finish(builder.relu(x))
        with pytest.raises(PassError, match="cannot be compared"):
            traffic_ratio(graph)


class TestIntensity:
    def test_fusing_raises_the_work_done_per_byte(self):
        # An elementwise chain is memory bound at every length until it is fused, at which
        # point the same arithmetic is spread over a fraction of the traffic.
        graph = elementwise_chain(8)
        assert arithmetic_intensity(graph) > arithmetic_intensity(graph, fused=False)

    def test_by_the_same_factor_the_traffic_fell(self):
        graph = elementwise_chain(8)
        raised = arithmetic_intensity(graph) / arithmetic_intensity(graph, fused=False)
        assert raised == pytest.approx(traffic_ratio(graph))

    def test_an_unfusable_graph_has_the_same_intensity_either_way(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        graph = builder.finish(builder.sum(x, axes=[1]))
        assert arithmetic_intensity(graph) == arithmetic_intensity(graph, fused=False)


class TestBuffers:
    def test_fusing_removes_the_intermediates(self):
        graph = elementwise_chain(8)
        assert peak_intermediates(graph) == 7
        assert fused_intermediates(graph) == 0

    def test_an_output_is_not_an_intermediate(self):
        assert peak_intermediates(elementwise_chain(1)) == 0

    def test_the_report_counts_what_was_removed(self):
        assert report_fusion(elementwise_chain(8)).buffers_removed == 7

    def test_a_trivial_group_removes_nothing(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        graph = builder.finish(builder.sum(x, axes=[1]))
        assert report_fusion(graph).buffers_removed == 0

    def test_it_serialises(self):
        assert report_fusion(elementwise_chain(8)).as_dict()["nodes_fused"] == 8


class TestFusedNodes:
    def test_a_group_becomes_one_node(self):
        assert len(to_fused_nodes(elementwise_chain(8))) == 1

    def test_the_loop_body_keeps_its_operations(self):
        # Kept rather than collapsed into an opaque kernel name, because the cost model, the
        # numerics checker and a debugger all still need to know what is inside.
        node = to_fused_nodes(elementwise_chain(4))[0]
        assert node.op_names == ["exp", "relu", "exp", "relu"]

    def test_the_cost_is_the_sum_of_the_chain(self):
        node = to_fused_nodes(elementwise_chain(4))[0]
        assert node.flops_per_element() == pytest.approx(18.0)

    def test_a_transcendental_in_the_body_is_reported(self):
        assert to_fused_nodes(elementwise_chain(4))[0].has_transcendental()

    def test_a_cheap_body_is_not(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        y = builder.input([8, 8], name="y")
        graph = builder.finish(builder.neg(builder.add(x, y)))
        assert not to_fused_nodes(graph)[0].has_transcendental()

    def test_a_trivial_group_is_not_emitted(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        graph = builder.finish(builder.sum(x, axes=[1]))
        assert to_fused_nodes(graph) == []

    def test_an_empty_body_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one operation"):
            FusedNode(name="f", op_names=[], inputs=["x"], output="f")

    def test_it_serialises(self):
        assert to_fused_nodes(elementwise_chain(4))[0].as_dict()["length"] == 4


class TestEquivalence:
    def test_a_fused_chain_matches_the_reference_bit_for_bit(self):
        # Fusion moves memory traffic and leaves the arithmetic alone, so anything less than
        # exact equality means the premise is wrong somewhere.
        graph = gentle_chain(8)
        assert every_group_is_equivalent(graph, random_feeds(graph))

    def test_a_softmax_group_does_too(self):
        graph = softmax_graph()
        assert every_group_is_equivalent(graph, random_feeds(graph))

    def test_a_layernorm_group_does_too(self):
        graph = layernorm_graph()
        assert every_group_is_equivalent(graph, random_feeds(graph))

    def test_broadcast_operands_are_indexed_correctly(self):
        # The subtraction in a softmax reads a column vector against a matrix. Indexing it
        # linearly rather than through the broadcast reads the wrong row and still produces
        # a plausible looking tensor.
        graph = softmax_graph()
        rows = groups_are_equivalent(graph, random_feeds(graph))
        assert all(row["largest_gap"] == 0.0 for row in rows)

    def test_the_mlp_group_does_too(self):
        graph = mlp_graph()
        assert every_group_is_equivalent(graph, random_feeds(graph))

    def test_a_graph_with_no_groups_reports_nothing(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        graph = builder.finish(builder.sum(x, axes=[1]))
        assert groups_are_equivalent(graph, random_feeds(graph)) == []

    def test_asking_for_a_group_that_is_not_there_is_rejected(self):
        graph = softmax_graph()
        stranger = FusedNode(name="nope", op_names=["neg"], inputs=["x"], output="nope")
        with pytest.raises(VerificationError, match="no group produces"):
            run_group_elementwise(graph, stranger, random_feeds(graph))
