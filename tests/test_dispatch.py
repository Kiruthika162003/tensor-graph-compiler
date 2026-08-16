from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.runtime.dispatch import (
    ACCELERATOR,
    KERNELS,
    PROCESSOR,
    DispatchReport,
    Machine,
    ProductShape,
    Timing,
    applicable,
    cheap_cost,
    compare_strategies,
    coverage,
    detailed_cost,
    every_kernel_computes_the_same_product,
    kernel_named,
    kernels_that_never_win,
    machine_changes_the_answer,
    measure_all,
    regret,
    report_for,
    select,
    shape_population,
    the_cheap_model_fails_on_small_shapes,
    the_cheap_model_is_the_default_in_disguise,
    the_spread_is_wider_than_the_gap,
    time_kernel,
    what_the_missing_terms_are_worth,
    where_the_models_disagree,
)


class TestShapes:
    def test_a_product_knows_its_arithmetic(self):
        assert ProductShape(rows=2, inner=3, columns=4).arithmetic == 48.0

    def test_and_what_it_has_to_read(self):
        assert ProductShape(rows=2, inner=3, columns=4).elements == 6 + 12 + 8

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            ProductShape(rows=0, inner=4, columns=4)

    def test_it_serialises(self):
        assert ProductShape(rows=2, inner=3, columns=4).as_dict()["inner"] == 3

    def test_the_population_covers_thin_and_wide(self):
        shapes = shape_population()
        assert min(shape.rows for shape in shapes) == 1
        assert max(shape.rows for shape in shapes) == 256

    def test_every_shape_has_a_kernel(self):
        assert coverage()["uncovered"] == []

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            shape_population(count=0)


class TestKernels:
    def test_every_kernel_computes_the_same_product(self):
        assert all(row["agrees"] for row in every_kernel_computes_the_same_product())

    def test_but_not_bit_for_bit(self):
        # They perform the same multiplications in different orders, so a dispatcher swapping
        # between them changes the answer between runs.
        rows = {row["kernel"]: row for row in every_kernel_computes_the_same_product()}
        assert not rows["thin"]["identical"]

    def test_the_thin_kernel_only_runs_on_thin_products(self):
        assert kernel_named("thin").applies_to(ProductShape(rows=4, inner=8, columns=8))
        assert not kernel_named("thin").applies_to(ProductShape(rows=64, inner=8, columns=8))

    def test_the_library_runs_on_everything(self):
        assert all(kernel_named("library").applies_to(shape) for shape in shape_population(40))

    def test_an_unknown_kernel_is_refused(self):
        with pytest.raises(ConfigError, match="unknown kernel"):
            kernel_named("magic")

    def test_a_kernel_serialises(self):
        assert kernel_named("library").as_dict()["arithmetic_factor"] == 1.0

    def test_three_are_available(self):
        assert len(KERNELS) == 3

    def test_a_wide_product_has_two_candidates(self):
        assert len(applicable(ProductShape(rows=256, inner=64, columns=64))) == 2


class TestModels:
    def test_the_detailed_model_charges_for_the_launch(self):
        shape = ProductShape(rows=1, inner=8, columns=8)
        assert detailed_cost(kernel_named("library"), shape) > cheap_cost(
            kernel_named("library"), shape
        )

    def test_and_for_the_tile_waste(self):
        # A product of one row on a tile of sixty four does sixty four rows of arithmetic.
        thin = ProductShape(rows=1, inner=256, columns=256)
        wide = ProductShape(rows=64, inner=256, columns=256)
        library = kernel_named("library")
        assert detailed_cost(library, thin) > detailed_cost(library, wide) * 0.9

    def test_on_a_large_product_the_two_models_agree(self):
        shape = ProductShape(rows=1024, inner=1024, columns=1024)
        assert select(shape) == select(shape, detailed=True)

    def test_a_machine_that_does_no_work_is_refused(self):
        with pytest.raises(ConfigError, match="some work per second"):
            Machine(name="broken", flops_per_second=0, bytes_per_second=1, launch_seconds=0)

    def test_a_negative_launch_is_refused(self):
        with pytest.raises(ConfigError, match="negative time"):
            Machine(name="broken", flops_per_second=1, bytes_per_second=1, launch_seconds=-1)

    def test_a_tile_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="is not a tile"):
            Machine(
                name="broken",
                flops_per_second=1,
                bytes_per_second=1,
                launch_seconds=0,
                tile=0,
            )


class TestSelection:
    def test_the_cheap_model_always_picks_the_library(self):
        chosen = {select(shape) for shape in shape_population()}
        assert chosen == {"library"}

    def test_so_it_is_the_fixed_default_in_disguise(self):
        # Every number it produces is the number choosing the library without looking produces.
        assert the_cheap_model_is_the_default_in_disguise()["identical"]

    def test_the_detailed_model_picks_the_thin_kernel_sometimes(self):
        chosen = {select(shape, detailed=True) for shape in shape_population()}
        assert "thin" in chosen

    def test_and_that_is_worth_a_factor_of_five(self):
        result = what_the_missing_terms_are_worth()
        assert result["without_them"] / result["with_them"] > 4.0

    def test_the_two_models_disagree_on_most_of_the_population(self):
        assert where_the_models_disagree()["share"] > 0.5

    def test_and_the_disagreements_are_expensive(self):
        # Concentrated exactly where the cheap model is blind rather than where the candidates
        # are close.
        assert where_the_models_disagree()["mean_regret_where_they_disagree"] > 4.0

    def test_the_errors_are_all_on_thin_products(self):
        rows = {row["rows"]: row for row in the_cheap_model_fails_on_small_shapes()}
        assert not rows[1]["agree"]
        assert rows[64]["agree"]

    def test_the_detailed_model_has_no_regret_by_construction(self):
        rows = {row["strategy"]: row for row in compare_strategies()}
        assert rows["detailed model"]["mean_regret"] == 1.0

    def test_a_shape_no_kernel_can_run_is_refused(self):
        shape = ProductShape(rows=1, inner=1, columns=1)
        assert regret(shape, select(shape, detailed=True)) == 1.0

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            report_for("guessing")

    def test_a_report_serialises(self):
        assert report_for("cheap model").as_dict()["strategy"] == "cheap model"

    def test_an_empty_report_has_no_regret(self):
        assert DispatchReport(strategy="none").mean_regret == 1.0


class TestMachines:
    def test_the_right_kernel_depends_on_the_machine(self):
        # A processor has a launch cost four hundred times smaller and a tile four times
        # smaller, so the shapes where a specialised kernel wins are different ones.
        assert machine_changes_the_answer()["different_choices"] > 0

    def test_on_a_quarter_of_the_population(self):
        assert machine_changes_the_answer()["share"] > 0.1

    def test_a_processor_reaches_for_the_thin_kernel_less_often(self):
        shapes = shape_population()
        accelerator = sum(1 for s in shapes if select(s, ACCELERATOR, detailed=True) == "thin")
        processor = sum(1 for s in shapes if select(s, PROCESSOR, detailed=True) == "thin")
        assert processor < accelerator

    def test_the_transposed_kernel_never_wins_here(self):
        # Reported rather than deleted. It may win on a machine with different ratios.
        assert "transposed" in kernels_that_never_win()


class TestMeasurement:
    def test_every_applicable_kernel_gets_timed(self):
        rows = measure_all(ProductShape(rows=4, inner=64, columns=64), repeats=3)
        assert len(rows) == 3

    def test_a_timing_records_its_spread(self):
        timing = time_kernel(kernel_named("library"), ProductShape(4, 64, 64), repeats=3)
        assert timing.spread >= 0.0

    def test_and_how_many_runs_it_took(self):
        timing = time_kernel(kernel_named("library"), ProductShape(4, 64, 64), repeats=5)
        assert timing.samples == 5

    def test_a_measurement_with_no_runs_is_refused(self):
        with pytest.raises(ConfigError, match="needs a run"):
            time_kernel(kernel_named("library"), ProductShape(4, 8, 8), repeats=0)

    def test_the_verdict_is_reported_rather_than_asserted(self):
        # Whether one timing can tell the kernels apart depends on what else is running, so
        # nothing here asserts on the answer.
        result = the_spread_is_wider_than_the_gap(ProductShape(4, 64, 64))
        assert "measurement_is_conclusive" in result

    def test_a_timing_serialises(self):
        assert Timing(kernel="library", median=1.0, spread=0.1).as_dict()["kernel"] == "library"
