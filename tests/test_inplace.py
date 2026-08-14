from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import (
    Builder,
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16
from tgc.memory.planner import validate_plan
from tgc.passes.inplace import (
    Donation,
    InplaceReport,
    aliased_plan,
    can_donate,
    check_donation,
    compare_graphs,
    donated_matches_reference,
    donation_chains,
    donation_is_safe,
    donation_saving,
    find_donations,
    merged_intervals,
    refusal_reasons,
)
from tgc.verify.fuzz import generate_many

GRAPHS = {
    "chain": elementwise_chain(8, sizes=(8, 8)),
    "softmax": softmax_graph(),
    "layernorm": layernorm_graph(),
    "mlp": mlp_graph(),
    "diamond": diamond_graph(sizes=(8, 8)),
    "branching": branching_graph(4, 2, width=8),
}


class TestConditions:
    def test_an_elementwise_node_can_take_its_input_buffer(self):
        graph = elementwise_chain(4)
        allowed, _ = can_donate(graph, "v0", "v1")
        assert allowed

    def test_a_graph_input_is_never_donated(self):
        # The caller owns that memory and will be surprised to find it changed.
        graph = elementwise_chain(4)
        allowed, reason = can_donate(graph, "x", "v0")
        assert not allowed
        assert "caller owns it" in reason

    def test_a_value_with_two_readers_is_not_donated(self):
        # The other reader would get the overwritten version.
        graph = diamond_graph()
        allowed, reason = can_donate(graph, "v0", "v1")
        assert not allowed
        assert "another reader" in reason

    def test_a_node_reading_the_same_value_twice_is_not_donated(self):
        # The second read happens after the first write.
        builder = Builder()
        x = builder.input([4, 4], name="x")
        doubled = builder.relu(x)
        graph = builder.finish(builder.mul(doubled, doubled))
        allowed, reason = can_donate(graph, doubled, graph.outputs[0])
        assert not allowed
        assert "twice" in reason or "another reader" in reason

    def test_a_reduction_is_not_donated_to(self):
        graph = softmax_graph()
        allowed, reason = can_donate(graph, "v2", "v3")
        assert not allowed
        assert "not elementwise" in reason

    def test_a_broadcast_is_not_donated_to(self):
        # A broadcast writes more elements than it read.
        graph = softmax_graph()
        allowed, reason = can_donate(graph, "v0", "v1")
        assert not allowed
        assert "shapes differ" in reason or "not elementwise" in reason

    def test_a_type_change_is_not_donated_to(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        wide = builder.relu(x)
        graph = builder.finish(builder.cast(wide, FLOAT16))
        allowed, reason = can_donate(graph, wide, graph.outputs[0])
        assert not allowed
        assert "different widths" in reason or "graph output" in reason

    def test_a_node_that_does_not_read_the_donor_is_rejected(self):
        graph = elementwise_chain(4)
        allowed, reason = can_donate(graph, "v0", "v3")
        assert not allowed
        assert "does not read" in reason

    def test_an_input_cannot_receive(self):
        graph = elementwise_chain(4)
        allowed, reason = can_donate(graph, "v0", "x")
        assert not allowed
        assert "receiver is a graph input" in reason

    def test_the_check_raises_with_the_reason(self):
        graph = elementwise_chain(4)
        with pytest.raises(PassError, match="caller owns it"):
            check_donation(graph, "x", "v0")


class TestFinding:
    def test_a_chain_donates_at_every_step_but_the_first(self):
        assert find_donations(elementwise_chain(8)).count == 7

    def test_a_softmax_donates_once(self):
        assert find_donations(softmax_graph()).count == 1

    def test_the_refusals_are_named(self):
        assert "the donor has another reader" in refusal_reasons(diamond_graph())

    def test_no_buffer_is_donated_twice(self):
        for graph in GRAPHS.values():
            donors = [d.donor for d in find_donations(graph).donations]
            assert len(donors) == len(set(donors))

    def test_it_serialises(self):
        assert find_donations(elementwise_chain(8)).as_dict()["donations"] == 7

    def test_a_donation_serialises(self):
        assert Donation(donor="a", receiver="b").as_dict()["donor"] == "a"

    def test_an_empty_report_donates_nothing(self):
        assert InplaceReport().count == 0


class TestChains:
    def test_a_chain_resolves_to_its_head(self):
        heads = donation_chains(elementwise_chain(8))
        assert heads["v7"] == "v0"

    def test_a_value_that_donates_nothing_is_its_own_head(self):
        assert donation_chains(softmax_graph())["x"] == "x"

    def test_a_chain_occupies_one_merged_interval(self):
        merged = merged_intervals(elementwise_chain(8))
        assert len(merged) < len(elementwise_chain(8).value_names)

    def test_the_merged_interval_spans_the_whole_chain(self):
        merged = {
            interval.name: interval for interval in merged_intervals(elementwise_chain(8))
        }
        assert merged["v0"].end == 7


class TestPlanning:
    def test_the_donated_plan_places_every_value(self):
        for graph in GRAPHS.values():
            placements = aliased_plan(graph).by_name()
            assert set(placements) == graph.value_names

    def test_a_donated_pair_shares_an_offset(self):
        graph = elementwise_chain(8)
        placements = aliased_plan(graph).by_name()
        assert placements["v0"].offset == placements["v1"].offset

    def test_the_merged_plan_is_valid_against_the_merged_intervals(self):
        for graph in GRAPHS.values():
            merged = merged_intervals(graph)
            validate_plan(merged, aliased_plan(graph))


class TestSaving:
    def test_a_chain_has_donations_and_saves_nothing(self):
        # The allocator had already reached the floor and there was nothing left to take.
        result = donation_saving(elementwise_chain(8))
        assert result["donations"] == 7
        assert result["saved"] == 0

    def test_a_wide_graph_saves_a_quarter_of_the_arena(self):
        result = donation_saving(branching_graph(4, 2))
        assert result["saved"] > 0
        assert result["arena_donated"] < result["arena_plain"]

    def test_and_goes_below_the_simultaneously_live_bytes(self):
        # Only possible because donation changes what simultaneously live means.
        result = donation_saving(branching_graph(4, 2))
        assert result["arena_donated"] < result["peak_bytes"]

    def test_every_graph_is_compared(self):
        assert len(compare_graphs(GRAPHS)) == len(GRAPHS)

    def test_comparing_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            compare_graphs({})


class TestSafety:
    def test_every_fixture_runs_correctly_under_its_donated_plan(self):
        # The plan places values the liveness analysis calls overlapping into the same bytes,
        # so validate_plan cannot be the safety argument. Running it is.
        for row in donation_is_safe(GRAPHS):
            assert row["matches"], row["graph"]

    def test_a_wide_graph_survives_several_inputs(self):
        graph = branching_graph(6, 3, width=8)
        assert all(donated_matches_reference(graph, seed=seed) for seed in range(4))

    def test_generated_graphs_survive_too(self):
        for graph in generate_many(20):
            assert donated_matches_reference(graph)

    def test_a_zero_seed_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            donation_is_safe(GRAPHS, seeds=0)

    def test_checking_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to check"):
            donation_is_safe({})
