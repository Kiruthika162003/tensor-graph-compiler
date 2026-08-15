"""Once any reusing allocator is in place, the execution order is what moves the peak.

Run with `python -m examples.the_schedule_decides`.
"""

from __future__ import annotations

from tgc.analysis.liveness import analyse, compute_intervals
from tgc.ir.builder import branching_graph, elementwise_chain
from tgc.memory.planner import compare_strategies
from tgc.schedule.order import compare_orders, order_versus_allocator


def main() -> None:
    wide = branching_graph(6, 2)

    print("a graph six branches wide")
    print(f"  liveness  {analyse(wide).as_dict()}")
    print()

    print("execution orders, allocator held fixed")
    for row in compare_orders(wide):
        print(f"  {row['order']:14} peak {row['peak_bytes']:8}  arena {row['arena_bytes']:8}")
    print()

    print("allocators, order held fixed")
    for row in compare_strategies(compute_intervals(wide)):
        print(
            f"  {row['strategy']:20} arena {row['arena_bytes']:8}"
            f"  reaches floor {row['reaches_floor']}"
        )
    print()

    result = order_versus_allocator(wide)
    print("the two spreads")
    print(f"  ordering    {result['order_spread']}")
    print(f"  allocator   {result['allocator_spread']}")
    print(f"  order wins  {result['order_matters_more']}")
    print()
    print("  the allocator spread excludes the plan that never reuses anything, which")
    print("  would otherwise be comparing having an allocator against not having one.")
    print(f"  that plan needs {result['no_reuse_arena']} bytes.")
    print()

    narrow = elementwise_chain(8)
    control = order_versus_allocator(narrow)
    print("on a chain neither matters, which is the control")
    print(f"  ordering    {control['order_spread']}")
    print(f"  allocator   {control['allocator_spread']}")


if __name__ == "__main__":
    main()
