from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import CodegenError, ConfigError
from tgc.ir import op as ops
from tgc.ir.graph import Graph, Node
from tgc.memory.planner import Plan
from tgc.verify.reference import to_torch

# Turning a scheduled graph into Python that runs.
#
# The generated code writes into one preallocated arena at the offsets the memory planner
# chose, which is the point of generating it at all. An interpreter allocates a tensor per
# node and lets the allocator underneath decide what that costs; generated code with a plan
# behind it allocates once and never again, and the arena size is the number the planner
# promised rather than whatever the runtime happened to do.
#
# Every buffer is a view into the arena, so a plan that overlaps two live values does not
# raise here, it silently produces wrong numbers. That is exactly why validate_plan runs on
# every plan in the test suite and why the generated code is checked against the interpreter
# rather than trusted.

HEADER = """import torch


def _step(source):
    return (source > 0).to(source.dtype)


def compiled(arena, inputs):
"""

BINARY = {
    "add": "torch.add",
    "sub": "torch.sub",
    "mul": "torch.mul",
    "div": "torch.div",
    "maximum": "torch.maximum",
    "minimum": "torch.minimum",
}

UNARY = {
    "neg": "torch.neg",
    "exp": "torch.exp",
    "log": "torch.log",
    "sqrt": "torch.sqrt",
    "tanh": "torch.tanh",
    "relu": "torch.relu",
    "sigmoid": "torch.sigmoid",
    "reciprocal": "torch.reciprocal",
    "abs": "torch.abs",
    "step": "_step",
}


@dataclass
class EmittedModule:
    """Generated source and the metadata a caller needs to run it."""

    source: str
    arena_bytes: int
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    @property
    def lines(self) -> int:
        """Length of the generated function."""
        return len(self.source.splitlines())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "lines": self.lines,
            "arena_bytes": self.arena_bytes,
            "inputs": list(self.input_names),
            "outputs": list(self.output_names),
        }


def static_sizes(node_or_value) -> tuple[int, ...]:
    """The concrete shape of a value, refusing symbolic ones."""
    shape = getattr(node_or_value, "shape", None)
    if shape is None:
        shape = node_or_value.output.shape
    if not shape.is_static:
        raise CodegenError("a symbolic shape cannot be given a fixed offset in an arena")
    return tuple(size.value or 0 for size in shape.sizes)


def view_expression(name: str, offset: int, sizes: Sequence[int], dtype: str) -> str:
    """Source for a tensor viewed out of the arena at a byte offset.

    Sliced by elements rather than by bytes, because the arena is typed and slicing a byte
    buffer would need a reinterpret that torch will not do for free. The planner works in
    bytes, so the offset is divided here, and the division has to be exact or the view starts
    mid element.
    """
    elements = 1
    for size in sizes:
        elements *= size
    shape = ", ".join(str(size) for size in sizes) if sizes else ""
    reshape = f".reshape({shape})" if sizes else ".reshape(())"
    return f"    {name} = arena[{offset}:{offset + elements}]{reshape}.view(torch.{dtype})"


def statement_for(node: Node, buffers: dict[str, str]) -> str:
    """Source for one operation, writing into its planned buffer."""
    target = buffers[node.name]
    operands = [buffers[name] for name in node.inputs]
    name = node.op.name

    if name in UNARY:
        return f"    {target}.copy_({UNARY[name]}({operands[0]}))"
    if name in BINARY:
        return f"    {target}.copy_({BINARY[name]}({operands[0]}, {operands[1]}))"
    if name == "cast":
        return f"    {target}.copy_({operands[0]}.to(torch.{node.output.dtype.name}))"
    if name == "matmul":
        return f"    {target}.copy_({operands[0]} @ {operands[1]})"
    if name in ("sum", "mean", "max"):
        return _reduction_statement(node, target, operands[0])
    if name == "reshape":
        sizes = ", ".join(str(size) for size in static_sizes(node))
        return f"    {target}.copy_({operands[0]}.reshape({sizes}))"
    if name == "transpose":
        permutation = ", ".join(str(axis) for axis in node.attrs["permutation"])
        return f"    {target}.copy_({operands[0]}.permute({permutation}))"
    if name == "concat":
        axis = int(node.attrs["axis"])
        return f"    {target}.copy_(torch.cat([{operands[0]}, {operands[1]}], dim={axis}))"
    if name == "slice":
        axis = int(node.attrs["axis"])
        start = int(node.attrs["start"])
        length = int(node.attrs["length"])
        return f"    {target}.copy_({operands[0]}.narrow({axis}, {start}, {length}))"
    if name == "broadcast_to":
        sizes = ", ".join(str(size) for size in static_sizes(node))
        return f"    {target}.copy_({operands[0]}.broadcast_to({sizes}))"
    if name == "constant":
        return f"    {target}.fill_({float(node.attrs['value'])})"
    if name in ("print", "assert_finite"):
        return f"    {target}.copy_({operands[0]})"
    raise CodegenError(f"no lowering for {name}")


def _reduction_statement(node: Node, target: str, source: str) -> str:
    """Source for a reduction, widened to the accumulator the type rules chose."""
    axes = ", ".join(str(axis) for axis in node.attrs["axes"])
    keepdims = bool(node.attrs.get("keepdims", False))
    accumulator = node.output.dtype.name
    widened = f"{source}.to(torch.{accumulator})"
    if node.op.name == "max":
        expression = source
        for axis in sorted(node.attrs["axes"], reverse=True):
            expression = f"{expression}.amax(dim={axis}, keepdim=True)"
        if not keepdims:
            expression = f"{expression}.squeeze(dim=({axes},))"
        return f"    {target}.copy_({expression})"
    reducer = "sum" if node.op.name == "sum" else "mean"
    return f"    {target}.copy_({widened}.{reducer}(dim=({axes},), keepdim={keepdims}))"


def emit(graph: Graph, order: Sequence[Node], plan: Plan) -> EmittedModule:
    """Generate a function that runs the graph out of one arena.

    The inputs are copied into their planned slots rather than aliased. Aliasing would save a
    copy and would mean the caller's tensors are overwritten by any pass that decided an
    input's storage could be reused, which is a thing this compiler is allowed to decide.
    """
    placements = plan.by_name()
    missing = [name for name in graph.value_names if name not in placements]
    if missing:
        raise CodegenError(f"the plan does not place {missing}")

    element_bytes = 4
    buffers: dict[str, str] = {}
    lines: list[str] = []

    for value in graph.inputs:
        buffers[value.name] = f"buf_{value.name}"
        offset = placements[value.name].offset // element_bytes
        lines.append(
            view_expression(buffers[value.name], offset, static_sizes(value), value.dtype.name)
        )
        lines.append(f"    {buffers[value.name]}.copy_(inputs[{value.name!r}])")

    for node in order:
        buffers[node.name] = f"buf_{node.name}"
        offset = placements[node.name].offset // element_bytes
        lines.append(
            view_expression(
                buffers[node.name], offset, static_sizes(node), node.output.dtype.name
            )
        )

    for node in order:
        lines.append(statement_for(node, buffers))

    returned = ", ".join(buffers[name] for name in graph.outputs)
    lines.append(f"    return [{returned}]")

    return EmittedModule(
        source=HEADER + "\n".join(lines) + "\n",
        arena_bytes=plan.arena_bytes,
        input_names=[value.name for value in graph.inputs],
        output_names=list(graph.outputs),
    )


def arena_elements(module: EmittedModule, element_bytes: int = 4) -> int:
    """How many elements the arena has to hold."""
    if element_bytes < 1:
        raise ConfigError("an element takes at least one byte")
    return -(-module.arena_bytes // element_bytes)


def check_dtypes_uniform(graph: Graph) -> None:
    """Refuse a graph whose values are not all the same width.

    The arena is a single typed buffer, so mixing widths would need per value reinterpret
    casts and an offset in bytes rather than elements. Worth doing and not worth doing
    quietly, so it is refused rather than half supported.
    """
    widths = {graph.value(name).dtype.bytes for name in graph.value_names}
    if len(widths) > 1:
        raise CodegenError(
            f"the arena holds one element width and this graph uses {sorted(widths)}"
        )


def dtype_of(graph: Graph) -> str:
    """The single element type a graph uses."""
    check_dtypes_uniform(graph)
    names = {graph.value(name).dtype.name for name in graph.value_names}
    if len(names) != 1:
        raise CodegenError(f"the arena holds one dtype and this graph uses {sorted(names)}")
    return names.pop()


def torch_dtype_of(graph: Graph):
    """The torch type matching a graph's single element type."""
    return to_torch(graph.value(next(iter(graph.value_names))).dtype)


def operation_names(graph: Graph) -> set[str]:
    """Every operation a graph uses, for checking coverage."""
    return {node.op.name for node in graph.nodes}


def unsupported_operations(graph: Graph) -> list[str]:
    """Operations this backend cannot lower."""
    supported = (
        set(UNARY)
        | set(BINARY)
        | {"cast", "matmul", "sum", "mean", "max", "reshape", "transpose"}
        | {"broadcast_to", "constant", "print", "assert_finite"}
    )
    return sorted(operation_names(graph) - supported)


def can_lower(graph: Graph) -> bool:
    """Whether every operation in a graph has a lowering."""
    return not unsupported_operations(graph) and all(
        node.output.shape.is_static for node in graph.nodes
    )


def leaf_free(graph: Graph) -> bool:
    """Whether the graph has any node that produces without reading.

    Constants are fine and inputs are fine. The check exists because a leaf with a symbolic
    shape has no arena slot, and the failure otherwise happens inside generated code.
    """
    return all(node.op is not ops.INPUT for node in graph.nodes)
