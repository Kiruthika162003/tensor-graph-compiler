from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.verify.reference import random_feeds, run

# The operation whose intermediate is larger than its inputs, and how to avoid writing it down.
#
# Attention multiplies a query against every key, softmaxes the result and multiplies by the
# values. The scores in the middle are square in the sequence length, so at a length of four
# thousand they are sixteen million numbers per head against a quarter of a million of input.
# The whole difficulty of the operation is that the thing nobody wants is the thing the obvious
# implementation materialises.
#
# It does not have to be materialised, because a softmax can be computed in one pass over its
# input if the running maximum and the running total are rescaled as the maximum moves. That is
# the entire trick, it is four lines, and this file implements it and checks it against the
# obvious version rather than describing it.
#
# Three things the measurements say.
#
# The saving is asymptotic, which means it is nothing at a short sequence and everything at a
# long one. At a sequence of a hundred and twenty eight with a block of the same size the
# streaming version holds slightly more than the naive one, because it keeps three running
# values on top of a block that is the whole matrix. At eight thousand it holds twenty six times
# less, and the ratio keeps climbing after that.
#
# The extra arithmetic is real and it is tiny. Rescaling costs a multiply per block per element
# of the output, which at a block of a hundred and twenty eight is two tenths of a percent of
# the products the operation was doing anyway.
#
# And the running maximum is not an optimisation, it is the reason the thing works. Without it
# the exponentials overflow at a score of about eighty nine, and a head of width sixty four
# reaches that between an input scale of four and one of eight, which is not a strange place for
# a model to be. The answer past that point is nan rather than inaccurate.


@dataclass(frozen=True)
class AttentionShape:
    """One head of attention."""

    sequence: int
    width: int

    def __post_init__(self) -> None:
        if min(self.sequence, self.width) < 1:
            raise ConfigError(f"a head cannot be {self.sequence} by {self.width}")

    @property
    def input_elements(self) -> int:
        """Numbers in the queries, keys and values together."""
        return 3 * self.sequence * self.width

    @property
    def score_elements(self) -> int:
        """Numbers in the matrix nobody wants."""
        return self.sequence * self.sequence

    @property
    def arithmetic(self) -> float:
        """Multiply adds in the two products."""
        return 4.0 * self.sequence * self.sequence * self.width

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "sequence": self.sequence,
            "width": self.width,
            "inputs": self.input_elements,
            "scores": self.score_elements,
        }


def naive_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Attention as it is written, with the score matrix built.

    The reference. Everything else in this file is checked against it, and it is here in full
    rather than as a call to a library so that the thing being checked and the thing checking it
    do not share an implementation.
    """
    scale = 1.0 / math.sqrt(queries.shape[-1])
    scores = (queries @ keys.transpose(-2, -1)) * scale
    peak = scores.amax(dim=-1, keepdim=True)
    weights = (scores - peak).exp()
    return (weights / weights.sum(dim=-1, keepdim=True)) @ values


def streaming_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block: int = 128,
) -> torch.Tensor:
    """Attention in one pass over the keys, never holding the scores.

    The running maximum moves as blocks arrive, and everything accumulated so far has to be
    rescaled by the exponential of how far it moved. That rescaling is the whole algorithm: it
    is what lets a softmax that is defined over the whole row be computed a piece at a time
    without ever seeing the whole row.
    """
    if block < 1:
        raise ConfigError(f"a block has to hold something, got {block}")
    scale = 1.0 / math.sqrt(queries.shape[-1])
    rows = queries.shape[0]

    running_max = torch.full((rows, 1), float("-inf"), dtype=queries.dtype)
    running_sum = torch.zeros((rows, 1), dtype=queries.dtype)
    accumulator = torch.zeros((rows, values.shape[-1]), dtype=queries.dtype)

    for start in range(0, keys.shape[0], block):
        key_block = keys[start : start + block]
        value_block = values[start : start + block]
        scores = (queries @ key_block.transpose(-2, -1)) * scale

        block_max = scores.amax(dim=-1, keepdim=True)
        updated_max = torch.maximum(running_max, block_max)
        correction = (running_max - updated_max).exp()
        correction = torch.where(
            torch.isfinite(correction), correction, torch.zeros_like(correction)
        )

        weights = (scores - updated_max).exp()
        running_sum = running_sum * correction + weights.sum(dim=-1, keepdim=True)
        accumulator = accumulator * correction + weights @ value_block
        running_max = updated_max

    return accumulator / running_sum


def unstable_streaming_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block: int = 128,
) -> torch.Tensor:
    """The same thing without the running maximum, kept so the failure can be measured.

    Exponentiating the raw scores. Correct in exact arithmetic and useless in floating point,
    because the exponential of anything above about eighty nine is an infinity in float32 and
    the sum of infinities divided by an infinity is a nan.
    """
    if block < 1:
        raise ConfigError(f"a block has to hold something, got {block}")
    scale = 1.0 / math.sqrt(queries.shape[-1])
    rows = queries.shape[0]

    running_sum = torch.zeros((rows, 1), dtype=queries.dtype)
    accumulator = torch.zeros((rows, values.shape[-1]), dtype=queries.dtype)
    for start in range(0, keys.shape[0], block):
        scores = (queries @ keys[start : start + block].transpose(-2, -1)) * scale
        weights = scores.exp()
        running_sum = running_sum + weights.sum(dim=-1, keepdim=True)
        accumulator = accumulator + weights @ values[start : start + block]
    return accumulator / running_sum


def random_head(
    shape: AttentionShape, *, seed: int = 0, scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries, keys and values for one head."""
    generator = torch.Generator().manual_seed(seed)
    queries = torch.randn(shape.sequence, shape.width, generator=generator) * scale
    keys = torch.randn(shape.sequence, shape.width, generator=generator) * scale
    values = torch.randn(shape.sequence, shape.width, generator=generator)
    return queries, keys, values


def streaming_matches_the_reference(
    shape: AttentionShape | None = None, block: int = 128, *, seed: int = 0
) -> dict:
    """The streaming version against the one that builds the scores.

    Not bit identical and it cannot be: the streaming version adds the contributions in blocks
    and rescales between them, which is a different order of the same additions. It agrees to
    about a millionth in float32, which is the rounding of the type rather than a property of
    the algorithm.
    """
    target = shape if shape is not None else AttentionShape(sequence=512, width=64)
    queries, keys, values = random_head(target, seed=seed)
    reference = naive_attention(queries, keys, values)
    streamed = streaming_attention(queries, keys, values, block)
    gap = float((streamed - reference).abs().max())
    scale = float(reference.abs().max())
    return {
        "identical": bool(torch.equal(streamed, reference)),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def block_size_sweep(
    blocks: Sequence[int] = (1, 8, 64, 128, 512), shape: AttentionShape | None = None
) -> list[dict]:
    """Agreement against how much of the sequence is processed at a time.

    A block of one is the extreme case, a rescale per key, and it agrees to five ten millionths.
    A block as large as the sequence rescales once. The error moves by half between the two,
    which is nothing next to the two hundred and fifty six times more rescaling the small block
    does, and that is the property that makes the algorithm usable at whatever block size the
    hardware wants.
    """
    if not blocks:
        raise ConfigError("there is nothing to sweep")
    target = shape if shape is not None else AttentionShape(sequence=256, width=32)
    queries, keys, values = random_head(target)
    reference = naive_attention(queries, keys, values)
    rows = []
    for block in blocks:
        streamed = streaming_attention(queries, keys, values, block)
        gap = float((streamed - reference).abs().max())
        rows.append({"block": block, "largest_gap": gap})
    return rows


def a_small_block_costs_almost_no_accuracy() -> dict:
    """What a small block costs in accuracy, which is half again rather than a factor.

    Written as a ratio because the interesting thing is not the size of the error, it is that
    two hundred and fifty six times as many rescalings produce an error only half again as
    large. The rescaling is stable, which is why the algorithm survives being run at whatever
    block size a machine happens to want.
    """
    rows = {row["block"]: row["largest_gap"] for row in block_size_sweep()}
    return {
        "at_one": rows[1],
        "at_five_hundred_and_twelve": rows[512],
        "ratio": round(rows[1] / rows[512], 3) if rows[512] else float("inf"),
    }


def the_running_maximum_is_not_optional(
    shape: AttentionShape | None = None, scale: float = 4.0
) -> dict:
    """What happens without it, on inputs large enough to matter.

    The exponential of a score above about eighty nine is an infinity in float32, and a dot
    product of sixty four terms with a scale of four reaches that easily. The version without
    the running maximum returns nan; the version with it returns the same answer it always did.
    """
    target = shape if shape is not None else AttentionShape(sequence=256, width=64)
    queries, keys, values = random_head(target, scale=scale)
    reference = naive_attention(queries, keys, values)
    stable = streaming_attention(queries, keys, values)
    unstable = unstable_streaming_attention(queries, keys, values)
    return {
        "largest_score": float(
            ((queries @ keys.transpose(-2, -1)) / math.sqrt(target.width)).abs().max()
        ),
        "stable_is_finite": bool(torch.isfinite(stable).all()),
        "unstable_is_finite": bool(torch.isfinite(unstable).all()),
        "stable_gap": float((stable - reference).abs().max()),
    }


def where_the_exponential_overflows(
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> list[dict]:
    """The input scale at which the unstable version stops returning numbers.

    Between four and eight for a head of sixty four, which is not an extreme input. The dot
    product of two vectors of that width grows like the scale squared, so a factor of two in the
    inputs is a factor of four in the scores and the boundary is two doublings away from an
    ordinary setting rather than somewhere safely far off.
    """
    if not scales:
        raise ConfigError("there is nothing to sweep")
    shape = AttentionShape(sequence=128, width=64)
    rows = []
    for scale in scales:
        queries, keys, values = random_head(shape, scale=scale)
        unstable = unstable_streaming_attention(queries, keys, values)
        rows.append(
            {
                "scale": scale,
                "finite": bool(torch.isfinite(unstable).all()),
                "largest_score": round(
                    float(((queries @ keys.transpose(-2, -1)) / math.sqrt(64)).abs().max()), 2
                ),
            }
        )
    return rows


def memory_for(shape: AttentionShape, block: int = 128, *, element_bytes: int = 4) -> dict:
    """Bytes each version has to hold at its peak.

    The naive one holds the whole score matrix. The streaming one holds a block of scores and
    the three running values, which is linear in the sequence rather than square, and that is
    the entire argument for the algorithm stated as two numbers.
    """
    if block < 1:
        raise ConfigError(f"a block has to hold something, got {block}")
    naive = (shape.input_elements + shape.score_elements) * element_bytes
    streamed = (
        shape.input_elements + shape.sequence * min(block, shape.sequence) + 2 * shape.sequence
    ) * element_bytes
    return {
        "naive": naive,
        "streaming": streamed,
        "ratio": round(naive / streamed, 3) if streamed else 0.0,
    }


def sequence_sweep(
    lengths: Sequence[int] = (128, 512, 2048, 8192), width: int = 64, block: int = 128
) -> list[dict]:
    """How the memory saving grows with the sequence.

    Linearly, because one side is square in the length and the other is not. At a sequence of a
    hundred and twenty eight with a block of the same size there is no saving at all and a
    slight loss, since the streaming version holds the whole matrix as its block plus three
    running values on top. By eight thousand it is a factor of twenty six and still climbing.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    return [
        {"sequence": length, **memory_for(AttentionShape(length, width), block)}
        for length in lengths
    ]


def the_saving_grows_with_the_sequence() -> dict:
    """Whether the ratio really climbs rather than settling."""
    rows = sequence_sweep()
    return {
        "at_the_shortest": rows[0]["ratio"],
        "at_the_longest": rows[-1]["ratio"],
        "grew": rows[-1]["ratio"] > rows[0]["ratio"],
    }


def extra_arithmetic(shape: AttentionShape, block: int = 128) -> dict:
    """What the rescaling costs, against what the operation was doing anyway.

    One multiply per block per element of the accumulator, plus one for the running sum. Under
    one percent at a block of a hundred and twenty eight, which is the answer to whether the
    memory saving is bought with time: it is not, it is close to free.
    """
    if block < 1:
        raise ConfigError(f"a block has to hold something, got {block}")
    blocks = math.ceil(shape.sequence / block)
    rescaling = blocks * shape.sequence * (shape.width + 1)
    return {
        "products": shape.arithmetic,
        "rescaling": rescaling,
        "share": round(rescaling / shape.arithmetic, 6) if shape.arithmetic else 0.0,
    }


def block_size_changes_the_overhead(
    blocks: Sequence[int] = (1, 16, 128, 1024), shape: AttentionShape | None = None
) -> list[dict]:
    """The rescaling cost against the block size.

    Falls like one over the block, because the rescaling happens once per block and the products
    happen once per element. A block of one pays a rescale for every key and costs a quarter
    more than the naive version; a block of a hundred and twenty eight pays two tenths of a
    percent.
    """
    if not blocks:
        raise ConfigError("there is nothing to sweep")
    target = shape if shape is not None else AttentionShape(sequence=2048, width=64)
    return [{"block": block, **extra_arithmetic(target, block)} for block in blocks]


def the_block_size_trade(shape: AttentionShape | None = None) -> dict:
    """Memory against arithmetic, which move in opposite directions with the block size.

    A larger block holds more scores and rescales less often. The two are not symmetric: the
    memory grows linearly in the block and the overhead falls like one over it, so there is a
    wide flat region in the middle where neither matters and that is where a real
    implementation sits.
    """
    target = shape if shape is not None else AttentionShape(sequence=2048, width=64)
    rows = []
    for block in (16, 64, 128, 512, 2048):
        rows.append(
            {
                "block": block,
                "memory": memory_for(target, block)["streaming"],
                "overhead": extra_arithmetic(target, block)["share"],
            }
        )
    return {
        "rows": rows,
        "memory_grows": rows[-1]["memory"] > rows[0]["memory"],
        "overhead_falls": rows[-1]["overhead"] < rows[0]["overhead"],
    }


def attention_graph(sequence: int = 32, width: int = 16) -> Graph:
    """One head written out in the IR, scores and all.

    Built so the rest of the compiler has something to look at that has a square intermediate in
    it. Every fixture until now has intermediates the same size as its inputs, which is exactly
    the case where a memory planner has nothing interesting to do.
    """
    if min(sequence, width) < 1:
        raise ConfigError(f"a head cannot be {sequence} by {width}")
    builder = Builder()
    queries = builder.input([sequence, width], name="q")
    keys = builder.input([sequence, width], name="k")
    values = builder.input([sequence, width], name="v")

    scores = builder.matmul(queries, builder.transpose(keys, [1, 0]))
    scaled = builder.mul(scores, builder.constant(1.0 / math.sqrt(width)))
    peak = builder.max(scaled, axes=[1], keepdims=True)
    shifted = builder.sub(scaled, builder.broadcast_to(peak, [sequence, sequence]))
    weights = builder.exp(shifted)
    total = builder.sum(weights, axes=[1], keepdims=True)
    normalised = builder.div(weights, builder.broadcast_to(total, [sequence, sequence]))
    return builder.finish(builder.matmul(normalised, values))


def the_graph_has_a_square_intermediate(sequence: int = 32, width: int = 16) -> dict:
    """How much larger the middle of the graph is than its ends.

    The number that makes attention different from everything else in the fixture set. At a
    sequence of thirty two and a width of sixteen the largest intermediate is twice the size of
    an input; at the sizes a model runs at it is hundreds of times.
    """
    graph = attention_graph(sequence, width)
    largest_input = max(value.shape.elements for value in graph.inputs)
    largest_value = max(node.output.shape.elements for node in graph.nodes)
    return {
        "largest_input": largest_input,
        "largest_intermediate": largest_value,
        "ratio": round(largest_value / largest_input, 3) if largest_input else 0.0,
    }


def the_graph_agrees_with_the_reference(sequence: int = 32, width: int = 16) -> dict:
    """The IR version against the torch one, so the fixture is not quietly wrong.

    Worth checking because the fixture is written twice, once as a graph and once as a function,
    and a fixture that computes something other than what it is named after would make every
    measurement taken on it meaningless in a way nothing else would catch.
    """
    graph = attention_graph(sequence, width)
    feeds = random_feeds(graph, seed=3)
    from_graph = run(graph, feeds)[0]
    from_torch = naive_attention(feeds["q"], feeds["k"], feeds["v"])
    gap = float((from_graph - from_torch).abs().max())
    scale = float(from_torch.abs().max())
    return {"largest_gap": gap, "relative_gap": gap / scale if scale else gap}
