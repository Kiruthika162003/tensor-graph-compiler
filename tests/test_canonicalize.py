from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError
from tgc.ir.builder import Builder, layernorm_graph, softmax_graph
from tgc.ir.graph import validate
from tgc.passes.canonicalize import (
    CanonicalReport,
    Rewrite,
    aligned_pairs_graph,
    canonicalisation_alone_changes_nothing,
    canonicalise,
    check_idempotent,
    commuted_pairs_graph,
    is_canonical,
    is_constant,
    literal_on_the_left_graph,
    measure_interaction,
    measure_one_sided_matcher,
    node_counts,
    one_sided_constant_matches,
    report_canonicalisation,
    sort_key,
)
from tgc.verify.fuzz import generate_many
from tgc.verify.reference import outputs_agree, random_feeds, run


class TestOrdering:
    def test_a_literal_sorts_after_a_computed_value(self):
        graph = literal_on_the_left_graph(2)
        constant = next(node.name for node in graph.nodes if node.op.name == "constant")
        assert sort_key(graph, "x") < sort_key(graph, constant)

    def test_computed_values_sort_by_name(self):
        graph = commuted_pairs_graph(2)
        assert sort_key(graph, "in0") < sort_key(graph, "in1")

    def test_a_literal_is_recognised(self):
        graph = literal_on_the_left_graph(1)
        constant = next(node.name for node in graph.nodes if node.op.name == "constant")
        assert is_constant(graph, constant)

    def test_an_input_is_not(self):
        assert not is_constant(literal_on_the_left_graph(1), "x")


class TestRewriting:
    def test_a_commuted_node_is_reordered(self):
        graph = commuted_pairs_graph(2)
        assert report_canonicalisation(graph).count > 0

    def test_an_aligned_graph_is_already_canonical(self):
        # Its operands happen to sort the way they were written.
        assert is_canonical(canonicalise(aligned_pairs_graph(2)))

    def test_a_non_commutative_node_is_never_reordered(self):
        graph = softmax_graph()
        subtraction = graph.node("v1")
        assert canonicalise(graph).node("v1").inputs == subtraction.inputs

    def test_canonicalising_is_idempotent(self):
        # A canonical form that is not a fixed point is not a canonical form, and it is what
        # makes a pass pipeline fail to terminate.
        for graph in (commuted_pairs_graph(3), softmax_graph(), layernorm_graph()):
            check_idempotent(graph)

    def test_the_result_still_validates(self):
        validate(canonicalise(commuted_pairs_graph(3)))

    def test_the_answer_is_bit_identical(self):
        # Swapping two reads and performing the same arithmetic on them is exact for every
        # value, nan included.
        for graph in (commuted_pairs_graph(3), softmax_graph(), layernorm_graph()):
            feeds = random_feeds(graph, positive=True)
            assert outputs_agree(run(graph, feeds), run(canonicalise(graph), feeds))

    def test_generated_graphs_survive_it(self):
        for graph in generate_many(20):
            feeds = random_feeds(graph, positive=True)
            assert outputs_agree(run(graph, feeds), run(canonicalise(graph), feeds))

    def test_it_serialises(self):
        assert report_canonicalisation(commuted_pairs_graph(2)).as_dict()["rewrites"] > 0

    def test_a_rewrite_serialises(self):
        assert Rewrite(node="a", rule="r").as_dict()["rule"] == "r"

    def test_an_empty_report_rewrote_nothing(self):
        assert CanonicalReport().count == 0


class TestInteraction:
    def test_canonicalising_alone_removes_no_nodes(self):
        assert canonicalisation_alone_changes_nothing()

    def test_it_gains_the_subexpression_pass_nothing(self):
        # Which is the answer and not a disappointment: that pass already sorts commutative
        # operands when it builds a signature, so the two orders were never different to it.
        assert all(row["gained"] == 0 for row in measure_interaction())

    def test_the_commuted_and_aligned_graphs_merge_equally(self):
        rows = {row["graph"]: row for row in measure_interaction()}
        assert (
            rows["commuted"]["merges_without_canonicalising"]
            == rows["aligned"]["merges_without_canonicalising"]
        )

    def test_the_pipeline_node_count_is_the_same_either_way(self):
        rows = {row["pipeline"]: row for row in node_counts()}
        assert (
            rows["subexpressions only"]["nodes"]
            == rows["canonicalise then subexpressions"]["nodes"]
        )

    def test_an_empty_pair_count_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one pair"):
            commuted_pairs_graph(0)

    def test_and_for_the_aligned_fixture_too(self):
        with pytest.raises(ConfigError, match="at least one pair"):
            aligned_pairs_graph(0)


class TestOneSidedMatcher:
    def test_a_one_sided_rule_fires_on_none_of_a_raw_graph(self):
        # Users write the literal first constantly, and a rule that only checks the right
        # operand sees none of it.
        assert measure_one_sided_matcher()["matches_before"] == 0

    def test_and_on_all_of_a_canonicalised_one(self):
        result = measure_one_sided_matcher()
        assert result["matches_after"] == result["expressions"]

    def test_which_is_where_the_benefit_lives(self):
        # A benefit to the matchers rather than to the graph, which is why it never shows up
        # in a node count.
        graph = literal_on_the_left_graph(5)
        assert len(canonicalise(graph).nodes) == len(graph.nodes)

    def test_the_matcher_counts_only_commutative_nodes(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.sub(x, builder.constant(2.0)))
        assert one_sided_constant_matches(graph) == 0

    def test_an_empty_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one expression"):
            literal_on_the_left_graph(0)

    def test_the_answer_survives_the_rewrite(self):
        graph = literal_on_the_left_graph(5)
        feeds = random_feeds(graph, positive=True)
        assert torch.equal(run(graph, feeds)[0], run(canonicalise(graph), feeds)[0])
