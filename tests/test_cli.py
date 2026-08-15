from __future__ import annotations

import json

import pytest

from tgc.cli.main import COMMANDS, GRAPHS, build_graph, build_parser, emit, main
from tgc.errors import CompilerError


def run(capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


class TestParser:
    def test_every_command_has_a_handler(self):
        parser = build_parser()
        for name in COMMANDS:
            args = parser.parse_args([name])
            assert callable(args.handler)

    def test_every_command_has_help_text(self):
        assert all(help_text for _, help_text in COMMANDS.values())

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_an_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["convolve"])

    def test_the_graph_option_only_accepts_fixtures(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["show", "--graph", "nothing"])

    def test_json_is_available_everywhere(self):
        assert build_parser().parse_args(["--json", "fusion"]).json


class TestGraphs:
    def test_every_fixture_builds(self):
        for name in GRAPHS:
            assert build_graph(name).nodes

    def test_an_unknown_fixture_is_rejected(self):
        with pytest.raises(CompilerError, match="unknown graph"):
            build_graph("nothing")


class TestCommands:
    def test_show_prints_a_graph(self, capsys):
        code, output = run(capsys, "show", "--graph", "softmax")
        assert code == 0
        assert output.startswith("graph(")

    def test_source_prints_generated_python(self, capsys):
        code, output = run(capsys, "source", "--graph", "softmax")
        assert code == 0
        assert "def compiled(arena, inputs):" in output

    def test_fusion_reports_every_fixture(self, capsys):
        code, output = run(capsys, "fusion")
        assert code == 0
        for name in GRAPHS:
            assert name in output

    def test_memory_reports_the_floor(self, capsys):
        code, output = run(capsys, "memory", "--graph", "chain")
        assert code == 0
        assert "floor" in output

    def test_schedule_reports_both_comparisons(self, capsys):
        code, output = run(capsys, "schedule", "--graph", "branching")
        assert code == 0
        assert "depth first" in output
        assert "order_spread" in output

    def test_tiling_reports_the_cliff(self, capsys):
        code, output = run(capsys, "tiling")
        assert code == 0
        assert "reduction" in output

    def test_autotune_reports_both_regimes(self, capsys):
        code, output = run(capsys, "autotune")
        assert code == 0
        assert "near the limit" in output

    def test_numerics_reports_three_tables(self, capsys):
        code, output = run(capsys, "numerics")
        assert code == 0
        assert "cancelling" in output
        assert "float16" in output

    def test_divergence_reports_every_inexact_rule(self, capsys):
        code, output = run(capsys, "divergence")
        assert code == 0
        assert "mul_by_zero" in output

    def test_partition_reports_every_placement(self, capsys):
        code, output = run(capsys, "partition", "--graph", "chain")
        assert code == 0
        assert "round robin" in output

    def test_align_reports_the_padding(self, capsys):
        code, output = run(capsys, "align")
        assert code == 0
        assert "padding_fraction" in output

    def test_reduction_reports_the_error(self, capsys):
        code, output = run(capsys, "reduction", "--length", "2000")
        assert code == 0
        assert "serial_depth" in output

    def test_buckets_reports_every_scheme(self, capsys):
        code, output = run(capsys, "buckets")
        assert code == 0
        assert "geometric" in output

    def test_roofline_reports_the_ridge(self, capsys):
        code, output = run(capsys, "roofline", "--graph", "mlp")
        assert code == 0
        assert "memory_bound" in output

    def test_broadcast_reports_the_refusals(self, capsys):
        code, output = run(capsys, "broadcast")
        assert code == 0
        assert "matmul reader" in output

    def test_donate_reports_every_fixture(self, capsys):
        code, output = run(capsys, "donate")
        assert code == 0
        assert "donations" in output

    def test_alias_reports_names_against_buffers(self, capsys):
        code, output = run(capsys, "alias")
        assert code == 0
        assert "view chain" in output


class TestVerification:
    def test_verify_passes_on_every_fixture(self, capsys):
        code, output = run(capsys, "verify")
        assert code == 0
        assert "False" not in output

    def test_fuzz_passes_on_generated_graphs(self, capsys):
        code, output = run(capsys, "fuzz", "--count", "8")
        assert code == 0
        assert "checked" in output

    def test_the_exit_code_reflects_the_result(self, capsys):
        # What a build would check, so it has to carry the answer rather than merely print it.
        assert run(capsys, "verify")[0] == 0
        assert run(capsys, "fuzz", "--count", "4")[0] == 0


class TestOutput:
    def test_json_is_parseable(self, capsys):
        code, output = run(capsys, "--json", "divergence")
        assert code == 0
        assert isinstance(json.loads(output), list)

    def test_a_table_lines_up(self, capsys):
        _, output = run(capsys, "align")
        lines = output.splitlines()
        assert set(lines[1]) <= {"-", " "}

    def test_an_empty_result_says_so(self, capsys):
        emit([], as_json=False)
        assert "nothing to report" in capsys.readouterr().out

    def test_a_single_mapping_prints_as_one_row(self, capsys):
        emit({"a": 1, "b": 2}, as_json=False)
        output = capsys.readouterr().out
        assert "a" in output and "b" in output

    def test_json_of_a_mapping_is_parseable(self, capsys):
        emit({"a": 1}, as_json=True)
        assert json.loads(capsys.readouterr().out) == {"a": 1}
