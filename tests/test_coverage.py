from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.verify.coverage import (
    ALL_NAMES,
    LEAF_NAMES,
    NEVER_A_NODE,
    REACHABLE,
    CategoryReport,
    CoverageReport,
    an_empty_corpus_is_refused,
    an_unknown_corpus_is_refused,
    audit,
    binary_graph,
    builder_methods,
    by_category,
    categories_in,
    checked_graph,
    compare_corpora,
    corpus_named,
    coverage_of,
    covered_by,
    every_category_appears_somewhere,
    extended_corpus,
    four_operations_cannot_be_written_down,
    node_counts,
    node_weighted_coverage,
    one_operation_can_never_be_a_node,
    ops_in,
    ops_without_a_builder_method,
    per_graph_coverage,
    reachable_coverage_of,
    report_for,
    shared_corpus,
    the_common_operations_dominate_the_node_count,
    the_extended_corpus_closes_it,
    the_fixtures_are_not_redundant,
    the_gap_is_not_what_it_looks_like,
    the_shared_fixtures_leave_half_the_table,
    unary_graph,
    uncovered_by,
    view_graph,
)


class TestCounting:
    def test_the_table_has_thirty_operations(self):
        assert len(ALL_NAMES) == len(ops.ALL_OPS)

    def test_one_of_them_can_never_be_a_node(self):
        assert {"input"} == NEVER_A_NODE

    def test_so_the_reachable_table_is_one_smaller(self):
        assert len(REACHABLE) == len(ALL_NAMES) - 1

    def test_the_other_leaf_is_reachable(self):
        assert one_operation_can_never_be_a_node()["reachable_leaves"] == ["constant"]

    def test_which_the_category_does_not_tell_you(self):
        assert len(LEAF_NAMES) == 2

    def test_a_graph_reports_the_operations_in_it(self):
        assert "matmul" in ops_in(corpus_named("shared")["mlp"])

    def test_and_the_categories(self):
        assert ops.CONTRACTION in categories_in(corpus_named("shared")["mlp"])

    def test_and_how_many_of_each(self):
        counts = node_counts(corpus_named("shared")["branching"])
        assert sum(counts.values()) == len(corpus_named("shared")["branching"].nodes)

    def test_an_unknown_corpus_is_refused(self):
        assert an_unknown_corpus_is_refused()

    def test_an_empty_corpus_is_refused(self):
        assert an_empty_corpus_is_refused()

    def test_and_so_is_weighting_one(self):
        with pytest.raises(ConfigError, match="covers nothing"):
            covered_by({})


class TestTheGap:
    def test_the_shared_corpus_covers_under_half(self):
        assert the_shared_fixtures_leave_half_the_table()["coverage"] < 0.5

    def test_on_six_graphs(self):
        assert the_shared_fixtures_leave_half_the_table()["graphs"] == 6

    def test_fifteen_reachable_operations_are_missing(self):
        assert len(uncovered_by(shared_corpus()) - NEVER_A_NODE) == 15

    def test_three_of_them_were_added_for_a_pass(self):
        result = the_gap_is_not_what_it_looks_like()
        assert result["added_for_a_pass"] == ["concat", "slice", "step"]

    def test_two_are_only_ever_inserted(self):
        assert the_gap_is_not_what_it_looks_like()["inserted_by_a_pass"] == [
            "assert_finite",
            "print",
        ]

    def test_but_ten_have_no_excuse_at_all(self):
        # These have been in the table since the start and no fixture ever needed one.
        assert len(the_gap_is_not_what_it_looks_like()["just_never_used"]) == 10

    def test_which_is_the_largest_group(self):
        result = the_gap_is_not_what_it_looks_like()
        assert len(result["just_never_used"]) > len(result["added_for_a_pass"]) + len(
            result["inserted_by_a_pass"]
        )

    def test_two_whole_categories_are_empty(self):
        assert every_category_appears_somewhere()["empty"] == ["side_effect", "view"]

    def test_no_fixture_reshapes_or_transposes_anything(self):
        covered = covered_by(shared_corpus())
        assert not covered & {"reshape", "transpose", "broadcast_to", "cast"}

    def test_the_reductions_are_completely_covered(self):
        rows = {row["category"]: row for row in by_category(shared_corpus())}
        assert rows[ops.REDUCTION]["share"] == 1.0

    def test_and_the_elementwise_ones_are_about_half(self):
        rows = {row["category"]: row for row in by_category(shared_corpus())}
        assert 0.5 <= rows[ops.ELEMENTWISE]["share"] < 0.6

    def test_a_category_with_no_operations_is_refused(self):
        empty = CategoryReport(category="ghost", total=0, covered=0)
        with pytest.raises(ConfigError, match="no operations in it"):
            empty.as_dict()

    def test_an_empty_report_covers_nothing(self):
        assert CoverageReport(corpus="none").share == 0.0


class TestBuilderMethods:
    def test_four_operations_have_no_builder_method(self):
        assert four_operations_cannot_be_written_down()["count"] == 4

    def test_and_they_are_the_ones_a_pass_inserts(self):
        assert ops_without_a_builder_method() == {"abs", "assert_finite", "minimum", "print"}

    def test_none_of_them_reach_the_shared_corpus(self):
        assert four_operations_cannot_be_written_down()["in_the_shared_corpus"] == []

    def test_all_of_them_reach_the_extended_one(self):
        assert len(four_operations_cannot_be_written_down()["in_the_extended_corpus"]) == 4

    def test_the_builder_covers_everything_else(self):
        assert not (REACHABLE - builder_methods() - ops_without_a_builder_method())


class TestTheExtendedCorpus:
    def test_it_reaches_every_operation_that_can_be_a_node(self):
        assert the_extended_corpus_closes_it()["complete"]

    def test_with_nothing_left_over(self):
        assert the_extended_corpus_closes_it()["still_missing"] == []

    def test_at_the_cost_of_four_graphs(self):
        assert the_extended_corpus_closes_it()["graphs_added"] == 4

    def test_no_category_is_empty_any_more(self):
        assert every_category_appears_somewhere(extended_corpus())["empty"] == []

    def test_the_unary_graph_reaches_the_transcendentals(self):
        assert {"exp", "log", "sigmoid", "tanh", "sqrt"} <= ops_in(unary_graph())

    def test_the_binary_graph_reaches_both_comparisons(self):
        assert {"maximum", "minimum"} <= ops_in(binary_graph())

    def test_the_view_graph_reaches_every_view(self):
        views = {operation.name for operation in ops.ALL_OPS if operation.category == ops.VIEW}
        assert views <= ops_in(view_graph())

    def test_the_checked_graph_carries_the_side_effects(self):
        assert {"assert_finite", "print"} <= ops_in(checked_graph())

    def test_the_extended_corpus_still_cannot_reach_the_input(self):
        assert uncovered_by(extended_corpus()) == NEVER_A_NODE


class TestWeighting:
    def test_the_weighted_coverage_is_always_one(self):
        # A corpus covers everything it contains, so weighting answers nothing.
        assert node_weighted_coverage(shared_corpus()) == 1.0

    def test_including_the_extended_corpus(self):
        assert node_weighted_coverage(extended_corpus()) == 1.0

    def test_three_operations_are_over_half_the_nodes(self):
        assert the_common_operations_dominate_the_node_count()["top_three_share"] > 0.5

    def test_out_of_fourteen_distinct_ones(self):
        assert the_common_operations_dominate_the_node_count()["distinct_operations"] == 14

    def test_the_corpus_is_small(self):
        assert the_common_operations_dominate_the_node_count()["nodes"] < 100


class TestPerGraph:
    def test_five_of_the_six_fixtures_contribute_something(self):
        assert len(the_fixtures_are_not_redundant()["contributing"]) == 5

    def test_only_the_chain_is_redundant(self):
        assert the_fixtures_are_not_redundant()["redundant"] == ["chain"]

    def test_so_the_corpus_is_not_padded(self):
        assert the_fixtures_are_not_redundant()["share_redundant"] < 0.2

    def test_every_graph_appears_in_the_breakdown(self):
        assert len(per_graph_coverage()) == len(shared_corpus())

    def test_the_layernorm_fixture_is_the_only_source_of_the_constant(self):
        rows = {row["graph"]: row for row in per_graph_coverage()}
        assert "constant" in rows["layernorm"]["unique"]

    def test_the_matmul_comes_only_from_the_mlp(self):
        rows = {row["graph"]: row for row in per_graph_coverage()}
        assert rows["mlp"]["unique"] == ["matmul"]


class TestReports:
    def test_both_corpora_are_compared(self):
        assert len(compare_corpora()) == 2

    def test_the_extended_one_misses_nothing(self):
        rows = {row["corpus"]: row for row in compare_corpora()}
        assert rows["extended"]["missing"] == 0

    def test_where_the_shared_one_misses_fifteen(self):
        rows = {row["corpus"]: row for row in compare_corpora()}
        assert rows["shared"]["missing"] == 15

    def test_the_audit_reports_both(self):
        assert [row["corpus"] for row in audit()] == ["shared", "extended"]

    def test_an_empty_audit_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to audit"):
            audit(names=())

    def test_a_report_carries_the_missing_names(self):
        assert "transpose" in report_for("shared").as_dict()["missing"]

    def test_and_the_share(self):
        assert report_for("extended").share == 1.0

    def test_coverage_against_the_whole_table_never_reaches_one(self):
        assert coverage_of(extended_corpus()) < 1.0

    def test_but_reachable_coverage_does(self):
        assert reachable_coverage_of(extended_corpus()) == 1.0
