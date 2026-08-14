from __future__ import annotations

import pytest

from tgc.errors import ConfigError, VerificationError
from tgc.ir.graph import validate
from tgc.passes.algebraic import simplify
from tgc.passes.constfold import fold_constants
from tgc.passes.cse import eliminate_common_subexpressions
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.layout import cancel_transposes
from tgc.passes.manager import identity
from tgc.verify.fuzz import (
    Failure,
    FuzzReport,
    GeneratorConfig,
    broken_transform,
    check_against_reference,
    differential,
    feeds_for,
    fuzz_allocation,
    fuzz_compiler,
    fuzz_transform,
    generate,
    generate_many,
    generated_graph_statistics,
    preserves_semantics,
    shrink,
)

PASSES = {
    "dead code": eliminate_dead_code,
    "subexpressions": eliminate_common_subexpressions,
    "algebraic": simplify,
    "constant folding": fold_constants,
    "transposes": cancel_transposes,
}


class TestGenerator:
    def test_the_same_seed_gives_the_same_graph(self):
        assert len(generate(seed=3).nodes) == len(generate(seed=3).nodes)

    def test_a_different_seed_gives_a_different_graph(self):
        first = [node.op.name for node in generate(seed=1).nodes]
        second = [node.op.name for node in generate(seed=2).nodes]
        assert first != second

    def test_every_generated_graph_validates(self):
        for graph in generate_many(30):
            validate(graph)

    def test_the_node_count_is_what_was_asked_for(self):
        config = GeneratorConfig(nodes=20)
        assert len(generate(config, seed=0).nodes) == 20

    def test_it_produces_reductions(self):
        assert generated_graph_statistics()["reductions"] > 0

    def test_and_values_read_more_than_once(self):
        # Which no hand written fixture in this repository does more than a handful of times,
        # and which several passes are only interesting because of.
        assert generated_graph_statistics()["values_read_more_than_once"] > 100

    def test_a_graph_of_no_nodes_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one node"):
            GeneratorConfig(nodes=0)

    def test_a_graph_of_no_inputs_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one input"):
            GeneratorConfig(inputs=0)

    def test_an_impossible_probability_is_rejected(self):
        with pytest.raises(ConfigError, match="has to be a probability"):
            GeneratorConfig(reduction_chance=1.5)

    def test_generating_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            generate_many(0)

    def test_the_feeds_are_positive(self):
        graph = generate(seed=0)
        assert all((tensor > 0).all() for tensor in feeds_for(graph).values())

    def test_it_serialises(self):
        assert GeneratorConfig().as_dict()["nodes"] == 12


class TestPasses:
    def test_every_pass_survives_generated_graphs_bit_for_bit(self):
        for name, transform in PASSES.items():
            report = fuzz_transform(transform, count=40)
            assert report.clean, f"{name} failed on {report.as_dict()}"

    def test_the_identity_pass_survives_too(self):
        assert fuzz_transform(identity, count=20).clean

    def test_the_deliberately_broken_pass_does_not(self):
        # Its shape is a real bug: a rewrite rule that fires on a pattern it should not,
        # produces a graph that validates perfectly, and quietly drops half the computation.
        report = fuzz_transform(broken_transform, count=40)
        assert not report.clean
        assert len(report.failures) > 5

    def test_a_zero_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            fuzz_transform(identity, count=0)

    def test_the_report_counts_what_passed(self):
        report = fuzz_transform(broken_transform, count=40)
        assert report.passed == report.checked - len(report.failures)

    def test_a_pass_that_raises_counts_as_a_failure(self):
        def explodes(_graph):
            raise RuntimeError("no")

        assert not fuzz_transform(explodes, count=3).clean

    def test_it_serialises(self):
        assert fuzz_transform(identity, count=5).as_dict()["checked"] == 5


class TestShrinking:
    def test_a_failure_shrinks_to_something_small(self):
        # A forty node counterexample says nothing and a one node one says everything.
        report = fuzz_transform(broken_transform, count=40)
        failure = report.failures[0]
        assert len(shrink(broken_transform, failure.graph).nodes) < failure.size

    def test_the_shrunk_graph_still_fails(self):
        report = fuzz_transform(broken_transform, count=40)
        smaller = shrink(broken_transform, report.failures[0].graph)
        assert not preserves_semantics(broken_transform, smaller)

    def test_and_still_validates(self):
        report = fuzz_transform(broken_transform, count=40)
        validate(shrink(broken_transform, report.failures[0].graph))

    def test_it_reaches_the_minimal_case(self):
        # A single binary operation whose result the broken pass throws away.
        report = fuzz_transform(broken_transform, count=40)
        smaller = shrink(broken_transform, report.failures[0].graph)
        assert len(smaller.nodes) == 1

    def test_shrinking_a_graph_that_passes_is_rejected(self):
        with pytest.raises(VerificationError, match="nothing to shrink"):
            shrink(identity, generate(seed=0))

    def test_a_zero_round_budget_is_rejected(self):
        report = fuzz_transform(broken_transform, count=40)
        with pytest.raises(ConfigError, match="must be positive"):
            shrink(broken_transform, report.failures[0].graph, rounds=0)

    def test_a_failure_serialises(self):
        graph = generate(seed=0)
        assert Failure(seed=1, graph=graph).as_dict()["seed"] == 1


class TestDifferential:
    def test_two_correct_passes_agree(self):
        assert differential(eliminate_dead_code, identity, count=20).clean

    def test_a_broken_pass_disagrees_with_a_correct_one(self):
        assert not differential(broken_transform, identity, count=20).clean

    def test_a_zero_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            differential(identity, identity, count=0)

    def test_the_disagreement_is_labelled(self):
        report = differential(broken_transform, identity, count=20)
        assert report.failures[0].detail == "disagreed"


class TestEndToEnd:
    def test_the_whole_compiler_matches_the_interpreter_on_generated_graphs(self):
        # A pass can be correct on its own and wrong after the scheduler has reordered around
        # it, and neither shows up when passes are fuzzed one at a time.
        report = fuzz_compiler(30)
        assert report.clean, report.as_dict()

    def test_every_allocator_places_generated_graphs_validly(self):
        # A plan that overlaps two live values does not raise, it produces wrong numbers.
        assert fuzz_allocation(40).clean

    def test_a_zero_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            fuzz_compiler(0)

    def test_and_for_allocation_too(self):
        with pytest.raises(ConfigError, match="must be positive"):
            fuzz_allocation(0)

    def test_the_per_graph_report_covers_every_seed(self):
        rows = check_against_reference(eliminate_dead_code, seeds=range(10))
        assert len(rows) == 10
        assert all(row["preserved"] for row in rows)

    def test_checking_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to check"):
            check_against_reference(identity, seeds=[])

    def test_an_empty_report_is_clean(self):
        assert FuzzReport().clean
