"""Trace a Python function, compile it, and check the answer against the function.

Run with `python -m examples.compile_a_function` from the repository root.
"""

from __future__ import annotations

import torch

from tgc.frontend.trace import softmax, trace
from tgc.ir.serialize import dumps
from tgc.runtime.executor import compile_graph, source_of
from tgc.verify.reference import random_feeds


def main() -> None:
    graph = trace(softmax, [[8, 32]], names=["x"])

    print("the function")
    print("  shifted = x - x.amax(dim=-1, keepdim=True)")
    print("  return shifted.exp() / shifted.exp().sum(dim=-1, keepdim=True)")
    print()

    print("traced into")
    print(dumps(graph), end="")
    print()

    compiled = compile_graph(graph)
    print("generated")
    print(source_of(graph), end="")
    print()

    feeds = random_feeds(graph, positive=True)
    produced = compiled(feeds)[0]
    expected = softmax(feeds["x"])

    print("checked against the function it came from")
    print(f"  rows sum to        {produced.sum(dim=1).tolist()}")
    print(f"  bit identical      {torch.equal(produced, expected)}")
    print(f"  arena bytes        {compiled.arena_bytes}")
    print(f"  values in the graph {len(graph.value_names)}")


if __name__ == "__main__":
    main()
