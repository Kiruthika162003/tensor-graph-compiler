from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import softmax_graph
from tgc.passes.pipeline import (
    DEFAULT_ORDER,
    EXACT_PASSES,
    PipelineComparison,
    compare_orders,
    confluence_on_generated_graphs,
    default_is_competitive,
    every_order_preserves_the_answer,
    exhaustive_orders,
    messy_graph,
    order_spread,
    pipeline_from,
    run_order,
    single_pass_is_not_enough,
)


class TestPipelineBuilding:
    def test_a_named_pipeline_holds_those_passes(self):
        pipeline = pipeline_from(["dead code", "algebraic"])
        assert [item.name for item in pipeline.passes] == ["dead code", "algebraic"]

    def test_an_unknown_pass_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown passes"):
            pipeline_from(["vectorise"])

    def test_an_empty_pipeline_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one pass"):
            pipeline_from([])

    def test_the_default_order_uses_every_exact_pass(self):
        assert set(DEFAULT_ORDER) == set(EXACT_PASSES)

    def test_every_pass_is_callable(self):
        assert all(callable(transform) for transform in EXACT_PASSES.values())


class TestSinglePasses:
    def test_no_single_pass_finishes_the_job(self):
        # Which is the condition that makes ordering a question at all.
        rows = {row["pipeline"]: row["nodes"] for row in single_pass_is_not_enough()}
        combined = rows["all of them"]
        for name in DEFAULT_ORDER:
            assert rows[name] > combined

    def test_the_whole_pipeline_removes_half_the_graph(self):
        rows = {row["pipeline"]: row["nodes"] for row in single_pass_is_not_enough()}
        assert rows["all of them"] < rows["nothing"] / 2

    def test_subexpression_elimination_does_the_most_alone(self):
        rows = {row["pipeline"]: row["nodes"] for row in single_pass_is_not_enough()}
        singles = {name: rows[name] for name in DEFAULT_ORDER}
        assert min(singles, key=lambda name: singles[name]) == "subexpressions"

    def test_the_fixture_needs_some_depth(self):
        with pytest.raises(ConfigError, match="needs some depth"):
            messy_graph(0)


class TestOrdering:
    def test_every_ordering_reaches_the_same_node_count(self):
        # The pipeline runs to a fixed point and these passes are confluent over this graph.
        assert order_spread()["same_result"]

    def test_but_not_in_the_same_number_of_rounds(self):
        # The cost a bad order carries is the work, not the result.
        assert order_spread()["round_spread"] > 1.0

    def test_the_worst_order_takes_twice_the_rounds_of_the_best(self):
        result = order_spread()
        assert result["most_rounds"] == 2 * result["fewest_rounds"]

    def test_the_default_order_is_not_the_best_one(self):
        # Written to find out rather than to confirm, and the answer was no: the reasoning
        # behind the default was a story and this is the correction.
        assert not default_is_competitive()["default_is_best"]

    def test_and_costs_exactly_one_extra_round(self):
        assert default_is_competitive()["extra_rounds"] == 1

    def test_a_zero_sample_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            compare_orders(sample=0)

    def test_the_default_ordering_is_sampled_first(self):
        rows = compare_orders(sample=4)
        assert rows[0]["order"] == " then ".join(DEFAULT_ORDER)


class TestExhaustive:
    def test_folding_before_simplifying_settles_soonest(self):
        # Folding exposes identities that only appear once the literals are known.
        rows = exhaustive_orders(["constant folding", "algebraic", "dead code"], messy_graph())
        best = min(rows, key=lambda row: row["rounds"])
        assert best["order"].startswith("constant folding")

    def test_dead_code_first_settles_last(self):
        rows = exhaustive_orders(["constant folding", "algebraic", "dead code"], messy_graph())
        worst = max(rows, key=lambda row: row["rounds"])
        assert worst["order"].startswith("dead code")

    def test_every_permutation_is_enumerated(self):
        rows = exhaustive_orders(["constant folding", "algebraic", "dead code"], messy_graph())
        assert len(rows) == 6

    def test_and_all_reach_the_same_result(self):
        rows = exhaustive_orders(["constant folding", "algebraic", "dead code"], messy_graph())
        assert len({row["nodes"] for row in rows}) == 1

    def test_too_many_passes_to_enumerate_is_refused(self):
        with pytest.raises(PassError, match="too many orderings"):
            exhaustive_orders(list(DEFAULT_ORDER), messy_graph())


class TestSemantics:
    def test_every_ordering_computes_the_same_thing(self):
        # Bit equality, because every pass here is exact. An ordering that changed the answer
        # would mean one of them is not, and the confluence result would be describing a bug.
        assert every_order_preserves_the_answer()

    def test_confluence_holds_on_graphs_nobody_wrote(self):
        # The fixture was built to give every pass something to do, which is exactly where a
        # confluence claim is most likely to be an accident of the fixture.
        result = confluence_on_generated_graphs()
        assert result["graphs_where_order_changed_the_result"] == 0

    def test_a_zero_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            confluence_on_generated_graphs(count=0)

    def test_an_ordinary_graph_survives_every_ordering(self):
        assert every_order_preserves_the_answer(softmax_graph(), sample=6)


class TestResults:
    def test_a_result_records_what_it_took(self):
        result = run_order(messy_graph(), DEFAULT_ORDER)
        assert result.rounds > 0
        assert result.passes_run >= len(DEFAULT_ORDER)

    def test_it_serialises(self):
        assert "then" in run_order(messy_graph(), DEFAULT_ORDER).as_dict()["order"]

    def test_an_empty_comparison_reports_nothing(self):
        assert PipelineComparison().best_rounds == 0
        assert PipelineComparison().worst_rounds == 0

    def test_a_comparison_serialises(self):
        comparison = PipelineComparison(rows=compare_orders(sample=4))
        assert comparison.as_dict()["orderings"] == 4
