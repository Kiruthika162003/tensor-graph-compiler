from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, softmax_graph
from tgc.ir.graph import validate
from tgc.passes.strength import (
    Reduction,
    StrengthReport,
    compare_divisors,
    compare_rewrite_by_divisor,
    constant_value,
    divides_by_literal,
    division_graph,
    is_constant,
    is_self_multiply,
    measure_division_rewrite,
    measure_safe_rewrite,
    reciprocal_disagreement_rate,
    reduce_divisions,
    reduce_safe_divisions,
    report_strength,
    safe_to_reduce,
    squaring_graph,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


class TestRecognition:
    def test_a_division_by_a_literal_is_found(self):
        graph = division_graph()
        assert any(divides_by_literal(graph, node) for node in graph.nodes)

    def test_a_division_by_a_value_is_not(self):
        assert not divides_by_literal(softmax_graph(), softmax_graph().node("v4"))

    def test_a_self_multiplication_is_found(self):
        graph = squaring_graph()
        assert any(is_self_multiply(node) for node in graph.nodes)

    def test_a_multiplication_of_two_things_is_not(self):
        graph = softmax_graph()
        assert not any(is_self_multiply(node) for node in graph.nodes)

    def test_a_literal_is_recognised(self):
        graph = division_graph()
        constant = next(node.name for node in graph.nodes if node.op is ops.CONSTANT)
        assert is_constant(graph, constant)

    def test_asking_a_non_literal_for_its_value_is_rejected(self):
        with pytest.raises(PassError, match="not a literal"):
            constant_value(softmax_graph(), "x")


class TestSafety:
    def test_a_power_of_two_reciprocal_is_exact(self):
        # The multiply performs the same single rounding the divide did.
        assert safe_to_reduce(2.0)
        assert safe_to_reduce(4.0)
        assert safe_to_reduce(8.0)

    def test_a_third_is_not(self):
        assert not safe_to_reduce(3.0)

    def test_nor_a_tenth(self):
        assert not safe_to_reduce(10.0)

    def test_dividing_by_zero_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be zero"):
            safe_to_reduce(0.0)


class TestDisagreement:
    def test_a_power_of_two_never_disagrees(self):
        assert reciprocal_disagreement_rate(2.0) == 0.0
        assert reciprocal_disagreement_rate(8.0) == 0.0

    def test_dividing_by_three_disagrees_on_a_third_of_inputs(self):
        # Not a contrived few. One over three is not representable, so the multiply performs
        # two roundings where the divide performed one.
        assert reciprocal_disagreement_rate(3.0) > 0.3

    def test_dividing_by_ten_disagrees_on_a_fifth(self):
        assert 0.15 < reciprocal_disagreement_rate(10.0) < 0.25

    def test_the_exact_divisors_are_exactly_the_powers_of_two(self):
        rows = compare_divisors()
        for row in rows:
            assert row["reciprocal_is_exact"] == (row["disagreement_rate"] == 0.0)

    def test_a_zero_sample_count_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            reciprocal_disagreement_rate(3.0, samples=0)

    def test_a_zero_divisor_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be zero"):
            reciprocal_disagreement_rate(0.0)

    def test_an_empty_comparison_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            compare_divisors(divisors=())


class TestRewriting:
    def test_the_divisions_are_gone(self):
        assert measure_division_rewrite(3.0)["divisions_after"] == 0

    def test_a_power_of_two_survives_the_rewrite_exactly(self):
        assert measure_division_rewrite(2.0)["identical"]

    def test_a_third_does_not(self):
        assert not measure_division_rewrite(3.0)["identical"]

    def test_but_only_by_a_last_bit(self):
        # Which is the argument for the flag being available rather than for it being on.
        assert measure_division_rewrite(3.0)["largest_gap"] < 1e-6

    def test_every_divisor_loses_its_divisions(self):
        assert all(row["divisions_after"] == 0 for row in compare_rewrite_by_divisor())

    def test_only_the_powers_of_two_stay_identical(self):
        rows = {row["divisor"]: row for row in compare_rewrite_by_divisor()}
        assert rows[2.0]["identical"] and rows[4.0]["identical"]
        assert not rows[3.0]["identical"] and not rows[10.0]["identical"]

    def test_the_result_still_validates(self):
        validate(reduce_divisions(division_graph()))

    def test_dividing_by_zero_cannot_be_reduced(self):
        builder = Builder()
        x = builder.input([4], name="x")
        graph = builder.finish(builder.div(x, builder.constant(0.0)))
        with pytest.raises(PassError, match="divides by zero"):
            reduce_divisions(graph)

    def test_a_graph_with_no_divisions_is_left_alone(self):
        graph = squaring_graph()
        feeds = random_feeds(graph, positive=True)
        assert outputs_agree(run(graph, feeds), run(reduce_divisions(graph), feeds))


class TestSafeRewriting:
    def test_it_never_changes_an_answer(self):
        # The version a compiler can turn on without a flag.
        assert all(row["identical"] for row in measure_safe_rewrite())

    def test_it_reduces_the_powers_of_two(self):
        rows = {row["divisor"]: row for row in measure_safe_rewrite()}
        assert rows[2.0]["divisions_after"] == 0
        assert rows[4.0]["divisions_after"] == 0

    def test_and_leaves_the_rest_alone(self):
        rows = {row["divisor"]: row for row in measure_safe_rewrite()}
        assert rows[3.0]["divisions_after"] == 4
        assert rows[10.0]["divisions_after"] == 4

    def test_it_fires_less_often_than_the_unconditional_version(self):
        safe = sum(row["divisions_after"] for row in measure_safe_rewrite())
        unconditional = sum(row["divisions_after"] for row in compare_rewrite_by_divisor())
        assert safe > unconditional

    def test_the_result_still_validates(self):
        for divisor in (2.0, 3.0):
            validate(reduce_safe_divisions(division_graph(divisor)))

    def test_an_empty_comparison_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            measure_safe_rewrite(divisors=())


class TestReporting:
    def test_a_division_graph_reports_inexact_reductions(self):
        report = report_strength(division_graph())
        assert report.count == 4
        assert report.inexact_count == 4

    def test_a_squaring_graph_reports_exact_ones(self):
        # x times x is the definition, and any pow that disagrees is the one that is wrong.
        report = report_strength(squaring_graph())
        assert report.exact_count == 4

    def test_a_graph_with_neither_reports_nothing(self):
        assert report_strength(softmax_graph()).count == 0

    def test_an_empty_report_counts_nothing(self):
        assert StrengthReport().count == 0

    def test_a_reduction_serialises(self):
        assert (
            Reduction(node="a", original="div", replacement="mul", exact=False).as_dict()[
                "exact"
            ]
            is False
        )

    def test_it_serialises(self):
        assert report_strength(division_graph()).as_dict()["inexact"] == 4

    def test_an_empty_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one division"):
            division_graph(count=0)

    def test_a_zero_divisor_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be zero"):
            division_graph(divisor=0.0)

    def test_an_empty_squaring_fixture_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one squaring"):
            squaring_graph(count=0)
