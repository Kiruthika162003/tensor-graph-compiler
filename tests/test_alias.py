from __future__ import annotations

import pytest

from tgc.analysis.alias import (
    AliasReport,
    AliasSet,
    alias_roots,
    alias_sets,
    aliased_names,
    analyse,
    compare_graphs,
    is_materialising,
    is_view,
    materialised_buffers,
    materialising_graph,
    may_alias,
    named_values,
    refused_by_elementwise_alone,
    refused_by_shape_alone,
    transpose_pair_aliases,
    unsafe_donations,
    view_chain_graph,
    which_check_refuses,
)
from tgc.errors import ConfigError, GraphError
from tgc.ir.builder import Builder, layernorm_graph, softmax_graph
from tgc.passes.inplace import can_donate


class TestRecognition:
    def test_a_transpose_is_a_view(self):
        assert any(is_view(node) for node in view_chain_graph().nodes)

    def test_a_rectifier_is_not(self):
        assert not any(is_view(node) for node in materialising_graph().nodes)

    def test_a_reshape_is_a_view(self):
        builder = Builder()
        x = builder.input([4, 8], name="x")
        graph = builder.finish(builder.reshape(x, [32]))
        assert is_view(graph.nodes[0])

    def test_a_broadcast_is_a_view(self):
        builder = Builder()
        x = builder.input([1, 8], name="x")
        graph = builder.finish(builder.broadcast_to(x, [4, 8]))
        assert is_view(graph.nodes[0])

    def test_a_real_operation_materialises(self):
        assert is_materialising(softmax_graph().node("v2"))

    def test_a_view_does_not(self):
        assert not is_materialising(view_chain_graph().nodes[-1])


class TestRoots:
    def test_a_chain_of_views_collapses_to_one_buffer(self):
        # A pass that resolves one level at a time sees the first and the last as unrelated.
        graph = view_chain_graph(length=4)
        assert alias_roots(graph)[graph.outputs[0]] == "v0"

    def test_an_input_is_its_own_root(self):
        assert alias_roots(view_chain_graph())["x"] == "x"

    def test_a_materialising_graph_has_a_root_per_name(self):
        graph = materialising_graph()
        assert len(set(alias_roots(graph).values())) == len(graph.value_names)

    def test_two_names_in_one_chain_alias(self):
        graph = view_chain_graph()
        assert may_alias(graph, "v0", graph.outputs[0])

    def test_a_materialised_value_does_not_alias_its_source(self):
        assert not may_alias(view_chain_graph(), "x", "v0")

    def test_comparing_something_undefined_is_rejected(self):
        with pytest.raises(GraphError, match="not defined"):
            may_alias(view_chain_graph(), "x", "nothing")

    def test_the_aliased_names_are_listed(self):
        assert len(aliased_names(view_chain_graph())) > 1

    def test_a_materialising_graph_has_none(self):
        assert aliased_names(materialising_graph()) == []


class TestSets:
    def test_a_view_chain_forms_one_large_set(self):
        sets = {item.root: item for item in alias_sets(view_chain_graph())}
        assert sets["v0"].size == 5

    def test_a_set_always_contains_its_root(self):
        assert AliasSet(root="a").size == 1

    def test_a_nameless_set_is_rejected(self):
        with pytest.raises(ConfigError, match="needs a root"):
            AliasSet(root="")

    def test_it_serialises(self):
        assert AliasSet(root="a", members={"a", "b"}).as_dict()["size"] == 2


class TestMeasurement:
    def test_a_view_chain_needs_far_fewer_buffers_than_names(self):
        report = analyse(view_chain_graph())
        assert report.buffers < report.names / 2

    def test_a_materialising_chain_needs_one_per_name(self):
        report = analyse(materialising_graph())
        assert report.buffers == report.names

    def test_the_view_fraction_separates_them(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["view chain"]["view_fraction"] > 0.6
        assert rows["materialising chain"]["view_fraction"] == 0.0

    def test_the_fixtures_hold_the_same_number_of_names(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["view chain"]["names"] == rows["materialising chain"]["names"]

    def test_a_graph_with_no_views_saves_nothing(self):
        assert analyse(softmax_graph()).saved_buffers == 0
        assert analyse(layernorm_graph()).saved_buffers == 0

    def test_an_empty_report_has_no_fraction(self):
        assert AliasReport().view_fraction == 0.0

    def test_the_counts_agree_with_the_helpers(self):
        graph = view_chain_graph()
        report = analyse(graph)
        assert report.names == named_values(graph)
        assert report.buffers == materialised_buffers(graph)

    def test_it_serialises(self):
        assert analyse(view_chain_graph()).as_dict()["view_nodes"] == 4

    def test_a_degenerate_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="width above one"):
            view_chain_graph(width=1)

    def test_and_for_the_control_too(self):
        with pytest.raises(ConfigError, match="width above one"):
            materialising_graph(width=1)


class TestDonationInteraction:
    def test_a_view_chain_holds_unsafe_donation_pairs(self):
        assert len(unsafe_donations(view_chain_graph())) == 4

    def test_a_materialising_chain_holds_none(self):
        assert unsafe_donations(materialising_graph()) == []

    def test_the_shape_check_catches_none_of_them(self):
        # A transpose of a square matrix has exactly the shape it started with.
        assert refused_by_shape_alone(view_chain_graph()) == 0

    def test_the_elementwise_check_catches_all_of_them(self):
        # Which makes the donation pass correct for a reason it does not state.
        graph = view_chain_graph()
        assert refused_by_elementwise_alone(graph) == len(unsafe_donations(graph))

    def test_the_donation_pass_does_refuse_every_unsafe_pair(self):
        graph = view_chain_graph()
        for donor, receiver in unsafe_donations(graph):
            allowed, _ = can_donate(graph, donor, receiver)
            assert not allowed

    def test_each_pair_reports_which_condition_stopped_it(self):
        rows = which_check_refuses(view_chain_graph())
        assert all(not row["receiver_is_elementwise"] for row in rows)


class TestTransposePair:
    def test_a_double_transpose_is_the_same_buffer(self):
        assert transpose_pair_aliases()["x_aliases_twice"]

    def test_and_the_operation_after_it_is_not(self):
        assert not transpose_pair_aliases()["x_aliases_the_output"]

    def test_four_names_sit_in_two_buffers(self):
        result = transpose_pair_aliases()
        assert result["names"] == 4
        assert result["buffers"] == 2

    def test_a_degenerate_width_is_rejected(self):
        with pytest.raises(ConfigError, match="width above one"):
            transpose_pair_aliases(width=1)
