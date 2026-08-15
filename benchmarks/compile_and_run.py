"""Time the compiler and the program it produces, and report the spread alongside.

Run with `python -m benchmarks.compile_and_run`. Every ratio is printed next to the spread of
the samples it came from, because a ratio without one is a number nobody can check.
"""

from __future__ import annotations

import time

from tgc.ir.builder import branching_graph, elementwise_chain, mlp_graph, softmax_graph
from tgc.runtime.executor import compile_graph
from tgc.runtime.profile import (
    arena_allocation_cost,
    compare_execution,
    hot_node_share,
    model_against_measurement,
    ranking_agreement,
    warmup_matters,
)

GRAPHS = {
    "softmax": softmax_graph(),
    "chain": elementwise_chain(16, sizes=(64, 64)),
    "mlp": mlp_graph(batch=64, hidden=128),
    "branching": branching_graph(6, 3),
}

SIZES = [
    ("small", mlp_graph(batch=8, hidden=32)),
    ("medium", mlp_graph(batch=64, hidden=128)),
    ("large", mlp_graph(batch=256, hidden=256)),
]


def print_table(title: str, rows: list[dict]) -> None:
    """Print a list of mappings with aligned columns."""
    print(title)
    if not rows:
        print("  (nothing)")
        return
    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns
    }
    print("  " + "  ".join(column.ljust(widths[column]) for column in columns))
    print("  " + "  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  " + "  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def compile_times() -> list[dict]:
    """How long each graph takes to compile."""
    rows = []
    for name, graph in GRAPHS.items():
        start = time.perf_counter()
        compiled = compile_graph(graph)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "graph": name,
                "nodes": len(graph.nodes),
                "compile_seconds": round(elapsed, 6),
                "generated_lines": compiled.module.lines,
                "arena_bytes": compiled.arena_bytes,
            }
        )
    return rows


def execution_rows() -> list[dict]:
    """The interpreter against the compiled program, spread included."""
    rows = []
    for name, graph in GRAPHS.items():
        row = compare_execution(graph, repeats=15)
        row["graph"] = name
        rows.append(row)
    return rows


def main() -> None:
    print_table("compilation", compile_times())
    print()
    print_table("execution, with the spread of the samples", execution_rows())
    print()
    print("  the two paths are not distinguishable on these graphs, and the spread column")
    print("  says so. this backend emits Python calling torch and the interpreter is Python")
    print("  calling the same torch, so the same kernels run either way. what the compiled")
    print("  path buys is the memory plan, which a stopwatch cannot see.")
    print()

    print_table(
        "arena allocation, fresh against reused",
        [
            {"graph": name, **arena_allocation_cost(graph, repeats=15)}
            for name, graph in GRAPHS.items()
        ],
    )
    print()

    print_table(
        "first call against steady state",
        [{"graph": name, **warmup_matters(graph, repeats=9)} for name, graph in GRAPHS.items()],
    )
    print()

    print_table("cost model against measurement", model_against_measurement(SIZES, repeats=9))
    print(f"\n  ranking agreement {ranking_agreement(SIZES, repeats=9)}")
    print("  the ranking is the only thing the model claims. its predicted seconds are for")
    print("  a machine this is not running on, so a stopwatch cannot check them.")
    print()

    print_table(
        "share of the run taken by the single hottest node",
        [
            {"graph": name, "hot_share": round(hot_node_share(graph, repeats=5), 4)}
            for name, graph in GRAPHS.items()
        ],
    )


if __name__ == "__main__":
    main()
