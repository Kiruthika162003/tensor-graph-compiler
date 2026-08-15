"""Fusion moves memory traffic and leaves the arithmetic alone.

Every fused group is run one element at a time and compared against the tensor at a time
reference, bit for bit. Run with `python -m examples.fusion_is_exact`.
"""

from __future__ import annotations

from tgc.ir.builder import Builder, elementwise_chain, layernorm_graph, mlp_graph, softmax_graph
from tgc.passes.fusion import (
    arithmetic_intensity,
    fused_bytes,
    report_fusion,
    to_fused_nodes,
    traffic_ratio,
    unfused_bytes,
)
from tgc.verify.fused import groups_are_equivalent
from tgc.verify.reference import random_feeds


def gentle_chain(length: int = 8) -> object:
    """A chain that does not overflow, so the comparison is about fusion and not about inf."""
    builder = Builder()
    current = builder.input([8, 8], name="x")
    for index in range(length):
        current = builder.tanh(current) if index % 2 else builder.relu(current)
    return builder.finish(current)


def main() -> None:
    chain = elementwise_chain(8, sizes=(8, 8))

    print("a chain of eight elementwise operations")
    report = report_fusion(chain)
    print(f"  groups found        {report.fused_groups and report.fused_groups[0].size}")
    print(f"  buffers removed     {report.buffers_removed}")
    print(f"  bytes unfused       {unfused_bytes(chain)}")
    print(f"  bytes fused         {fused_bytes(chain)}")
    print(f"  traffic ratio       {traffic_ratio(chain):.1f}")
    print(
        f"  intensity           {arithmetic_intensity(chain, fused=False):.3f}"
        f" to {arithmetic_intensity(chain):.3f}"
    )
    print()

    print("the loop body")
    for node in to_fused_nodes(chain):
        print(f"  {node.op_names}")
    print()

    print("every group, one element at a time against the reference")
    for name, graph in (
        ("chain", gentle_chain()),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        rows = groups_are_equivalent(graph, random_feeds(graph, positive=True))
        for row in rows:
            print(
                f"  {name:10} group of {row['length']}  identical {row['identical']}"
                f"  gap {row['largest_gap']}"
            )


if __name__ == "__main__":
    main()
