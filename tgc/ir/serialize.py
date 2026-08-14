from __future__ import annotations

import re

from tgc.errors import ConfigError, GraphError
from tgc.ir import op as ops
from tgc.ir.dtype import DType, from_name
from tgc.ir.graph import Graph, Node, Value, validate
from tgc.ir.shape import Shape, dim

# A text form of the IR that can be read back.
#
# The printer already exists on Graph, and a printer without a parser is a debugging
# convenience rather than a format. Being able to read the text back is what makes it useful
# for anything else: a failing graph from a fuzzer can be pasted into a bug report, a
# reduction can be checked in as a regression test, and two builds of the compiler can be
# compared on the same input without either of them running.
#
# The property that matters is the round trip. Printing a graph and parsing the result has to
# give back a graph that is equal node for node, and equality here is structural rather than
# by identity, because a parsed graph shares nothing with the one it came from. The test is
# run over generated graphs rather than fixtures, so the format has to survive shapes nobody
# chose.

HEADER = re.compile(r"^graph\((.*)\)\s*\{$")
INPUT = re.compile(r"^\s*(\w+):\s*(\w+)\[([^\]]*)\]\s*$")
ASSIGNMENT = re.compile(r"^\s*(\w+)\s*=\s*(\w+)\((.*?)\)\s*(?:\{(.*)\})?\s*$")
RETURN = re.compile(r"^\s*return\s+(.*)$")


class ParseError(GraphError):
    """Raised when text does not describe a graph."""


def format_shape(shape: Shape) -> str:
    """The dimensions of a value as text."""
    return ", ".join(str(size) for size in shape.sizes)


def parse_shape(text: str) -> Shape:
    """Dimensions from the text form, symbolic names included."""
    stripped = text.strip()
    if not stripped:
        return Shape()
    sizes = []
    for piece in stripped.split(","):
        token = piece.strip()
        if not token:
            raise ParseError(f"an empty dimension in {text!r}")
        sizes.append(dim(int(token)) if token.isdigit() else dim(token))
    return Shape(sizes=tuple(sizes))


def format_attrs(attrs: dict) -> str:
    """Node attributes as text, in a stable order."""
    if not attrs:
        return ""
    pieces = []
    for key in sorted(attrs):
        pieces.append(f"{key}={format_value(attrs[key])}")
    return " {" + ", ".join(pieces) + "}"


def format_value(value: object) -> str:
    """One attribute value as text."""
    if isinstance(value, DType):
        return f"dtype:{value.name}"
    if isinstance(value, Shape):
        return f"shape:[{format_shape(value)}]"
    if isinstance(value, tuple | list):
        return "[" + ",".join(format_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_value(text: str) -> object:
    """One attribute value from text."""
    token = text.strip()
    if token in ("true", "false"):
        return token == "true"
    if token.startswith("dtype:"):
        return from_name(token[6:])
    if token.startswith("shape:["):
        return parse_shape(token[7:-1])
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return ()
        return tuple(parse_value(piece) for piece in inner.split(","))
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def dumps(graph: Graph) -> str:
    """A graph as text.

    The dtype of every value goes on the line that defines it rather than being inferred on
    the way back in. Inference is available and would usually agree, and usually is the wrong
    standard for a format: a graph whose types were chosen by a pass has to round trip as
    itself and not as whatever inference would have picked.
    """
    lines = []
    header = ", ".join(
        f"{value.name}: {value.dtype.name}[{format_shape(value.shape)}]"
        for value in graph.inputs
    )
    lines.append(f"graph({header}) {{")
    for node in graph.nodes:
        arguments = ", ".join(node.inputs)
        typing = f": {node.output.dtype.name}[{format_shape(node.output.shape)}]"
        lines.append(
            f"  {node.name}{typing} = {node.op.name}({arguments}){format_attrs(node.attrs)}"
        )
    lines.append(f"  return {', '.join(graph.outputs)}")
    lines.append("}")
    return "\n".join(lines) + "\n"


TYPED_ASSIGNMENT = re.compile(
    r"^\s*(\w+):\s*(\w+)\[([^\]]*)\]\s*=\s*(\w+)\(([^)]*)\)\s*(?:\{(.*)\})?\s*$"
)


def loads(text: str) -> Graph:
    """A graph from text."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ParseError("there is nothing to parse")

    header = HEADER.match(lines[0].strip())
    if header is None:
        raise ParseError(f"the first line is not a graph header: {lines[0]!r}")

    inputs = []
    declared = header.group(1).strip()
    if declared:
        for piece in _split_top_level(declared):
            match = INPUT.match(piece)
            if match is None:
                raise ParseError(f"cannot read the input declaration {piece!r}")
            inputs.append(
                Value(
                    name=match.group(1),
                    shape=parse_shape(match.group(3)),
                    dtype=from_name(match.group(2)),
                )
            )

    nodes: list[Node] = []
    outputs: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "}":
            continue
        returning = RETURN.match(stripped)
        if returning is not None:
            outputs = [name.strip() for name in returning.group(1).split(",") if name.strip()]
            continue
        nodes.append(_parse_node(stripped))

    graph = Graph(nodes=nodes, inputs=inputs, outputs=outputs)
    validate(graph)
    return graph


def _parse_node(line: str) -> Node:
    """One assignment line."""
    match = TYPED_ASSIGNMENT.match(line)
    if match is None:
        raise ParseError(f"cannot read the assignment {line!r}")
    name, dtype_name, shape_text, op_name, argument_text, attribute_text = match.groups()

    arguments = tuple(piece.strip() for piece in argument_text.split(",") if piece.strip())
    attrs = _parse_attrs(attribute_text or "")
    return Node(
        op=ops.get_op(op_name),
        inputs=arguments,
        output=Value(name=name, shape=parse_shape(shape_text), dtype=from_name(dtype_name)),
        attrs=attrs,
    )


def _parse_attrs(text: str) -> dict:
    """The attribute block of an assignment."""
    if not text.strip():
        return {}
    attrs: dict = {}
    for piece in _split_top_level(text):
        if "=" not in piece:
            raise ParseError(f"cannot read the attribute {piece!r}")
        key, _, value = piece.partition("=")
        attrs[key.strip()] = parse_value(value)
    return attrs


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets."""
    pieces = []
    depth = 0
    current = []
    for character in text:
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
        if character == "," and depth == 0:
            pieces.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    if current:
        pieces.append("".join(current).strip())
    return [piece for piece in pieces if piece]


def round_trips(graph: Graph) -> bool:
    """Whether printing and parsing gives back the same graph."""
    try:
        parsed = loads(dumps(graph))
    except (ParseError, ConfigError, GraphError):
        return False
    return graphs_identical(graph, parsed)


def graphs_identical(left: Graph, right: Graph) -> bool:
    """Structural equality, since a parsed graph shares nothing with its source."""
    if left.outputs != right.outputs:
        return False
    if [value.name for value in left.inputs] != [value.name for value in right.inputs]:
        return False
    for first, second in zip(left.inputs, right.inputs, strict=True):
        if first.shape != second.shape or first.dtype is not second.dtype:
            return False
    if len(left.nodes) != len(right.nodes):
        return False
    for first, second in zip(left.nodes, right.nodes, strict=True):
        if first.op is not second.op or first.inputs != second.inputs:
            return False
        if first.output != second.output:
            return False
        if _normalise(first.attrs) != _normalise(second.attrs):
            return False
    return True


def _normalise(attrs: dict) -> dict:
    """Attributes with lists turned into tuples, so a round trip compares equal."""
    return {
        key: tuple(value) if isinstance(value, list) else value for key, value in attrs.items()
    }


def round_trip_report(graphs) -> dict:
    """How many of a batch of graphs survive printing and parsing."""
    checked = 0
    survived = 0
    for graph in graphs:
        checked += 1
        survived += 1 if round_trips(graph) else 0
    if checked == 0:
        raise ConfigError("there is nothing to check")
    return {"checked": checked, "survived": survived, "clean": survived == checked}
