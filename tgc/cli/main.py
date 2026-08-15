from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tgc.analysis.alias import compare_graphs as compare_alias
from tgc.analysis.cost import compare_machines, size_sweep
from tgc.analysis.liveness import compute_intervals
from tgc.analysis.numerics import (
    accumulator_comparison,
    compare_conditioning,
    compare_narrow_types,
)
from tgc.errors import CompilerError
from tgc.ir.builder import (
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.graph import Graph
from tgc.ir.serialize import dumps
from tgc.memory.arena import alignment_sweep, awkward_graph
from tgc.memory.planner import compare_strategies as compare_allocators
from tgc.parallel.partition import compare_strategies as compare_partitions
from tgc.passes.divergence import measure_divergence
from tgc.passes.fusion import report_fusion, traffic_ratio
from tgc.passes.hoist import compare_fixtures as compare_broadcasts
from tgc.passes.inplace import compare_graphs as compare_donation
from tgc.passes.reduction import compare_part_counts
from tgc.runtime.executor import compiled_matches_reference, source_of
from tgc.runtime.guards import compare_bucketing
from tgc.schedule.autotune import compare_regimes
from tgc.schedule.order import compare_orders, order_versus_allocator
from tgc.schedule.tiling import sweep_tiles
from tgc.verify.fuzz import fuzz_compiler

# The command line, which exists so the measurements can be run without importing anything.
#
# Every subcommand prints a table of the same rows the tests assert on. That is deliberate:
# a number in a README that nobody can reproduce is a number nobody should believe, and the
# quickest way to make one reproducible is to make it one command.

GRAPHS = {
    "chain": lambda: elementwise_chain(8),
    "softmax": softmax_graph,
    "layernorm": layernorm_graph,
    "mlp": mlp_graph,
    "diamond": diamond_graph,
    "branching": lambda: branching_graph(6, 3),
}


def build_graph(name: str) -> Graph:
    """One of the named fixtures."""
    if name not in GRAPHS:
        raise CompilerError(f"unknown graph {name!r}, expected one of {sorted(GRAPHS)}")
    return GRAPHS[name]()


def emit(rows, as_json: bool) -> None:
    """Print a list of mappings as JSON or as an aligned table."""
    if as_json:
        print(json.dumps(rows, indent=2, default=str))
        return
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        print("(nothing to report)")
        return

    columns = list(rows[0].keys())
    widths = {
        column: max(len(str(column)), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(str(column).ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def command_show(args: argparse.Namespace) -> int:
    """Print a fixture graph in the text form."""
    print(dumps(build_graph(args.graph)), end="")
    return 0


def command_source(args: argparse.Namespace) -> int:
    """Print the Python the backend generates for a fixture."""
    print(source_of(build_graph(args.graph)), end="")
    return 0


def command_fusion(args: argparse.Namespace) -> int:
    """What fusion finds in each fixture, and what it saves in traffic."""
    rows = []
    for name in sorted(GRAPHS):
        graph = build_graph(name)
        row = report_fusion(graph).as_dict()
        row["graph"] = name
        row["traffic_ratio"] = round(traffic_ratio(graph), 3)
        rows.append(row)
    emit(rows, args.json)
    return 0


def command_memory(args: argparse.Namespace) -> int:
    """Allocators against the theoretical floor on one graph."""
    graph = build_graph(args.graph)
    emit(compare_allocators(compute_intervals(graph)), args.json)
    return 0


def command_schedule(args: argparse.Namespace) -> int:
    """Execution orders on one graph, and how they compare to the allocator choice."""
    graph = build_graph(args.graph)
    emit(compare_orders(graph), args.json)
    print()
    emit(order_versus_allocator(graph), args.json)
    return 0


def command_tiling(args: argparse.Namespace) -> int:
    """Traffic across tile sizes, with the cache limit applied."""
    emit(sweep_tiles(cache_bytes=args.cache), args.json)
    return 0


def command_autotune(args: argparse.Namespace) -> int:
    """Where the cost model is enough and where a measurement is needed."""
    emit(compare_regimes(), args.json)
    return 0


def command_numerics(args: argparse.Namespace) -> int:
    """Conditioning, accumulator width and precision narrowing."""
    emit(compare_conditioning(), args.json)
    print()
    emit(accumulator_comparison(), args.json)
    print()
    emit(compare_narrow_types(), args.json)
    return 0


def command_divergence(args: argparse.Namespace) -> int:
    """What each inexact algebraic rule costs on its worked example."""
    emit(measure_divergence(), args.json)
    return 0


def command_partition(args: argparse.Namespace) -> int:
    """Device placements on one graph."""
    emit(compare_partitions(build_graph(args.graph), args.devices), args.json)
    return 0


def command_align(args: argparse.Namespace) -> int:
    """Arena size across a range of alignments."""
    emit(alignment_sweep(awkward_graph(12)), args.json)
    return 0


def command_reduction(args: argparse.Namespace) -> int:
    """Accuracy of a split reduction across part counts."""
    emit(compare_part_counts(length=args.length), args.json)
    return 0


def command_buckets(args: argparse.Namespace) -> int:
    """Shape specialisation against bucketing on realistic traffic."""
    emit(compare_bucketing(), args.json)
    return 0


def command_roofline(args: argparse.Namespace) -> int:
    """Where fusion buys time and where it only buys bytes."""
    emit(size_sweep(), args.json)
    print()
    emit(compare_machines(build_graph(args.graph)), args.json)
    return 0


def command_broadcast(args: argparse.Namespace) -> int:
    """Which broadcasts can be sunk and why the rest cannot."""
    emit(compare_broadcasts(), args.json)
    return 0


def command_donate(args: argparse.Namespace) -> int:
    """Buffer donation across the fixtures."""
    emit(compare_donation({name: build_graph(name) for name in sorted(GRAPHS)}), args.json)
    return 0


def command_alias(args: argparse.Namespace) -> int:
    """Names against buffers, for a graph of views and one without."""
    emit(compare_alias(), args.json)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    """Compile every fixture and check it against the interpreter."""
    rows = []
    for name in sorted(GRAPHS):
        result = compiled_matches_reference(build_graph(name))
        result["graph"] = name
        rows.append(result)
    emit(rows, args.json)
    return 0 if all(row["identical"] for row in rows) else 1


def command_fuzz(args: argparse.Namespace) -> int:
    """Compile generated graphs and check each one against the interpreter."""
    report = fuzz_compiler(args.count)
    emit(report.as_dict(), args.json)
    return 0 if report.clean else 1


COMMANDS = {
    "show": (command_show, "print a fixture graph"),
    "source": (command_source, "print the generated Python for a fixture"),
    "fusion": (command_fusion, "what fusion finds and what it saves"),
    "memory": (command_memory, "allocators against the theoretical floor"),
    "schedule": (command_schedule, "execution orders against allocator choice"),
    "tiling": (command_tiling, "traffic across tile sizes"),
    "autotune": (command_autotune, "where the cost model is enough"),
    "numerics": (command_numerics, "conditioning and precision"),
    "divergence": (command_divergence, "what each inexact rule costs"),
    "partition": (command_partition, "device placements"),
    "align": (command_align, "arena size across alignments"),
    "reduction": (command_reduction, "accuracy of a split reduction"),
    "buckets": (command_buckets, "shape specialisation against bucketing"),
    "roofline": (command_roofline, "where fusion buys time"),
    "broadcast": (command_broadcast, "which broadcasts can be sunk"),
    "donate": (command_donate, "buffer donation across the fixtures"),
    "alias": (command_alias, "names against buffers"),
    "verify": (command_verify, "compile every fixture and check it"),
    "fuzz": (command_fuzz, "compile generated graphs and check them"),
}


def build_parser() -> argparse.ArgumentParser:
    """Every subcommand this tool offers."""
    parser = argparse.ArgumentParser(
        prog="tgc", description="measurements from a tensor graph compiler"
    )
    parser.add_argument("--json", action="store_true", help="print json rather than a table")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, (handler, help_text) in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.set_defaults(handler=handler)
        if name in ("show", "source", "memory", "schedule", "partition", "roofline"):
            subparser.add_argument("--graph", default="softmax", choices=sorted(GRAPHS))
        if name == "tiling":
            subparser.add_argument("--cache", type=int, default=256 * 1024)
        if name == "partition":
            subparser.add_argument("--devices", type=int, default=4)
        if name == "reduction":
            subparser.add_argument("--length", type=int, default=20_000)
        if name == "fuzz":
            subparser.add_argument("--count", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.handler(args)
    except CompilerError as failure:
        print(f"error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
