from __future__ import annotations

import torch

from tgc.errors import VerificationError
from tgc.ir.graph import Graph
from tgc.passes.fusion import FusedNode, find_groups, to_fused_nodes
from tgc.verify.reference import evaluate_node, interpret

# Checking that a fused loop computes what the chain of tensor operations computed.
#
# The premise of elementwise fusion is that element i of the output depends only on element i
# of the inputs, so the whole chain can be carried through one element at a time. That is a
# claim about the operation set, and this file checks it the only way worth checking: by
# running the chain one element at a time and comparing against the tensor at a time
# reference, bit for bit.
#
# Bit for bit is the right bar here. Fusion reorders memory traffic and does not touch the
# arithmetic, so anything less than exact equality means the premise is wrong somewhere
# rather than that a tolerance needs widening.


def run_group_elementwise(
    graph: Graph, fused: FusedNode, feeds: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute a fused group one element at a time.

    Deliberately slow. Every element goes through the whole chain on its own, which is what a
    fused loop does and what the tensor at a time reference does not, so agreement between
    the two is evidence about the transformation rather than about the interpreter.
    """
    environment = interpret(graph, feeds)
    producers = {node.name: node for node in graph.nodes}
    group = next((item for item in find_groups(graph) if item.output == fused.output), None)
    if group is None:
        raise VerificationError(f"no group produces {fused.output}")

    shape = environment[fused.output].shape
    flat_size = environment[fused.output].numel()
    result = torch.empty(flat_size, dtype=environment[fused.output].dtype)

    # Broadcast every input up to the output shape first. A fused loop computes a broadcast
    # index per operand rather than a linear one, and indexing a [8, 1] tensor linearly
    # against an [8, 32] output silently reads the wrong row.
    expanded = {
        name: environment[name].broadcast_to(shape).reshape(-1) for name in group.inputs
    }

    for index in range(flat_size):
        local: dict[str, torch.Tensor] = {
            name: values[index] for name, values in expanded.items()
        }
        for name in group.members:
            node = producers[name]
            operands = [local[operand] for operand in node.inputs]
            local[name] = evaluate_node(node, operands)
        result[index] = local[fused.output]
    return result.reshape(shape)


def groups_are_equivalent(graph: Graph, feeds: dict[str, torch.Tensor]) -> list[dict]:
    """Every fused group in a graph, run both ways and compared.

    The comparison is exact. Fusion moves memory traffic and leaves the arithmetic alone, so
    a disagreement of any size means the premise is wrong somewhere.
    """
    environment = interpret(graph, feeds)
    rows = []
    for fused in to_fused_nodes(graph):
        elementwise = run_group_elementwise(graph, fused, feeds)
        reference = environment[fused.output]
        rows.append(
            {
                "group": fused.output,
                "length": fused.length,
                "identical": bool(torch.equal(elementwise, reference)),
                "largest_gap": float((elementwise - reference).abs().max()),
            }
        )
    return rows


def every_group_is_equivalent(graph: Graph, feeds: dict[str, torch.Tensor]) -> bool:
    """Whether fusing changed any answer at all."""
    rows = groups_are_equivalent(graph, feeds)
    return all(row["identical"] for row in rows)
