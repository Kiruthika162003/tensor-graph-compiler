from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError

# Looking rows out of a table, and the gradient that comes back.
#
# An embedding is a gather. The forward pass reads one row per token and does no arithmetic at
# all, so it is pure traffic, and the traffic is tiny next to the table it reads from. The
# backward pass is where it gets interesting: the derivative with respect to the table is zero
# everywhere except the rows that were read, and writing it out as a dense tensor produces
# something the size of the whole table to hold a few hundred nonzero rows.
#
# The numbers are not close. A vocabulary of fifty thousand and a batch of two thousand tokens
# gives a dense gradient twenty five times larger than the rows it describes, and that ratio is
# the vocabulary over the batch, so it gets worse as the vocabulary grows and better as the
# batch does.
#
# The part that is easy to get wrong is duplicates. A token appearing twice in a batch
# contributes twice, so the scatter has to add rather than assign, and a scatter that assigns
# produces a gradient that is silently wrong by however many times the commonest token appeared.
# On a skewed batch of two thousand that is a lot: the commonest token appears two hundred and
# sixty times, and the error on its row is most of that row's gradient.
#
# And the optimiser does not cooperate. Adam's moments decay every step whether or not a row was
# touched, so a sparse update either applies the decay to rows it did not read, which is the
# whole table again, or skips it, which makes the moment of a rare row depend on how long ago it
# was last seen. Both are wrong and everybody picks the second.


@dataclass(frozen=True)
class TableShape:
    """One embedding table and the batch that reads it."""

    vocabulary: int
    width: int
    tokens: int

    def __post_init__(self) -> None:
        if min(self.vocabulary, self.width, self.tokens) < 1:
            raise ConfigError("every dimension has to be positive")

    @property
    def table_elements(self) -> int:
        """Numbers in the table."""
        return self.vocabulary * self.width

    @property
    def read_elements(self) -> int:
        """Numbers the forward pass reads out."""
        return self.tokens * self.width

    @property
    def dense_gradient_elements(self) -> int:
        """Numbers a dense gradient holds."""
        return self.table_elements

    @property
    def sparse_gradient_elements(self) -> int:
        """Numbers a sparse gradient holds, before duplicates are merged."""
        return self.read_elements

    @property
    def density(self) -> float:
        """Share of the dense gradient that is not zero, at most."""
        return min(self.tokens / self.vocabulary, 1.0)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "vocabulary": self.vocabulary,
            "width": self.width,
            "tokens": self.tokens,
            "density": round(self.density, 6),
        }


def lookup(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The rows a batch of tokens names."""
    if table.dim() != 2:
        raise ConfigError(f"a table is a matrix, got rank {table.dim()}")
    if int(indices.max()) >= table.shape[0]:
        raise ConfigError("a token names a row the table does not have")
    return table[indices]


def dense_gradient(
    table: torch.Tensor, indices: torch.Tensor, cotangent: torch.Tensor
) -> torch.Tensor:
    """The gradient with respect to the whole table.

    Built by adding each row's contribution into a zero tensor the size of the table, which is
    what an autograd system does when nothing has told it the operation is sparse. Correct and
    enormous.
    """
    result = torch.zeros_like(table)
    result.index_add_(0, indices, cotangent)
    return result


def sparse_gradient(
    indices: torch.Tensor, cotangent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same gradient as the rows that were touched and their values.

    Duplicates are merged here rather than left for the caller, because a caller that scattered
    them without adding would get a wrong answer and nothing about the shapes would say so.
    """
    unique, inverse = torch.unique(indices, return_inverse=True)
    values = torch.zeros(unique.shape[0], cotangent.shape[1], dtype=cotangent.dtype)
    values.index_add_(0, inverse, cotangent)
    return unique, values


def scatter_without_adding(
    indices: torch.Tensor, cotangent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The version that assigns instead of accumulating, kept so the error can be measured.

    A token appearing three times contributes three times and this keeps one of them. The
    result has the right shape, the right indices and the wrong values, which is the shape of a
    bug that survives a shape check.
    """
    unique, inverse = torch.unique(indices, return_inverse=True)
    values = torch.zeros(unique.shape[0], cotangent.shape[1], dtype=cotangent.dtype)
    values[inverse] = cotangent
    return unique, values


def random_batch(
    shape: TableShape, *, seed: int = 0, skewed: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A table, a batch of tokens and a cotangent for the rows they read.

    The skewed option draws tokens from a distribution whose weight falls off with the rank,
    which is what text does. Uniform tokens almost never repeat at these sizes and the
    duplicate handling would look unnecessary, which is exactly the wrong impression to leave.
    """
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(shape.vocabulary, shape.width, generator=generator)
    if skewed:
        weights = 1.0 / torch.arange(1, shape.vocabulary + 1, dtype=torch.float32)
        indices = torch.multinomial(
            weights, shape.tokens, replacement=True, generator=generator
        )
    else:
        indices = torch.randint(
            0, shape.vocabulary, (shape.tokens,), generator=generator, dtype=torch.int64
        )
    cotangent = torch.randn(shape.tokens, shape.width, generator=generator)
    return table, indices, cotangent


def the_gradient_matches_autograd(shape: TableShape | None = None, *, seed: int = 0) -> dict:
    """The dense gradient against the one torch produces.

    The reference interpreter cannot help here, because this compiler has no gather operation
    and the lookup is written in torch. So the check is against torch's own autograd, which is
    the same arrangement grad/reverse.py uses and is a comparison between two implementations
    rather than one implementation and a restatement.
    """
    target = shape if shape is not None else TableShape(1000, 16, 200)
    table, indices, cotangent = random_batch(target, seed=seed)

    tracked = table.clone().detach().requires_grad_(True)
    torch.autograd.backward(lookup(tracked, indices), cotangent)
    mine = dense_gradient(table, indices, cotangent)
    gap = float((mine - tracked.grad).abs().max())
    return {"largest_gap": gap, "identical": bool(torch.equal(mine, tracked.grad))}


def the_sparse_form_is_the_same_gradient(
    shape: TableShape | None = None, *, seed: int = 0
) -> dict:
    """The sparse rows and values against the dense tensor.

    Expanding the sparse form back out has to give the dense one exactly. Anything else means
    the merge lost a contribution, and a lost contribution is a row that trains slower than it
    should for reasons nothing will report.
    """
    target = shape if shape is not None else TableShape(1000, 16, 200)
    table, indices, cotangent = random_batch(target, seed=seed, skewed=True)
    dense = dense_gradient(table, indices, cotangent)
    rows, values = sparse_gradient(indices, cotangent)

    rebuilt = torch.zeros_like(table)
    rebuilt[rows] = values
    return {
        "touched_rows": int(rows.shape[0]),
        "largest_gap": float((rebuilt - dense).abs().max()),
        "identical": bool(torch.equal(rebuilt, dense)),
    }


def assigning_instead_of_adding_is_wrong(
    shape: TableShape | None = None, *, seed: int = 0
) -> dict:
    """What the scatter that assigns produces.

    A gradient of the right shape with the wrong values, wrong by exactly the contributions it
    dropped. On a skewed batch the commonest token appears many times, so the error on that row
    is most of its gradient.
    """
    target = shape if shape is not None else TableShape(1000, 16, 2000)
    _, indices, cotangent = random_batch(target, seed=seed, skewed=True)
    rows, right = sparse_gradient(indices, cotangent)
    _, wrong = scatter_without_adding(indices, cotangent)

    counts = torch.bincount(indices, minlength=target.vocabulary)
    return {
        "rows": int(rows.shape[0]),
        "same_shape": list(right.shape) == list(wrong.shape),
        "largest_gap": float((right - wrong).abs().max()),
        "commonest_token_appears": int(counts.max()),
    }


def duplicates_are_common_in_real_text(
    shape: TableShape | None = None, *, seed: int = 0
) -> dict:
    """How often a batch repeats a token, on a uniform draw and a skewed one.

    Rarely under a uniform draw at these sizes and constantly under a skewed one. That is why
    the duplicate handling has to be tested against skewed tokens: a test written against a
    uniform batch would pass with the wrong scatter in it.
    """
    target = shape if shape is not None else TableShape(1000, 16, 2000)
    rows = {}
    for label, skewed in (("uniform", False), ("skewed", True)):
        _, indices, _ = random_batch(target, seed=seed, skewed=skewed)
        counts = torch.bincount(indices, minlength=target.vocabulary)
        rows[label] = {
            "distinct": int((counts > 0).sum()),
            "commonest": int(counts.max()),
            "repeated_rows": int((counts > 1).sum()),
        }
    return rows


def memory_comparison(shape: TableShape | None = None, *, element_bytes: int = 4) -> dict:
    """What each form of the gradient costs to hold."""
    target = shape if shape is not None else TableShape(50000, 512, 2000)
    dense = target.dense_gradient_elements * element_bytes
    sparse = target.sparse_gradient_elements * element_bytes
    return {
        "dense": dense,
        "sparse": sparse,
        "ratio": round(dense / sparse, 3) if sparse else 0.0,
        "density": round(target.density, 6),
    }


def vocabulary_sweep(
    sizes: Sequence[int] = (1000, 10000, 50000, 250000), tokens: int = 2000, width: int = 512
) -> list[dict]:
    """The saving against how large the vocabulary is.

    Linear in the vocabulary, because the dense form is the table and the sparse form is not.
    At a thousand words the dense form is the smaller of the two and at a quarter of a million
    it is a hundred and twenty five times larger, which is where real vocabularies live.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    return [
        {"vocabulary": size, **memory_comparison(TableShape(size, width, tokens))}
        for size in sizes
    ]


def the_saving_is_the_vocabulary_over_the_batch() -> dict:
    """Whether the ratio really is that simple."""
    rows = vocabulary_sweep()
    checks = [abs(row["ratio"] - row["vocabulary"] / 2000) < 0.01 for row in rows]
    return {"points": len(rows), "matching": sum(checks), "all_match": all(checks)}


def batch_sweep(
    sizes: Sequence[int] = (128, 512, 2048, 8192), vocabulary: int = 50000, width: int = 512
) -> list[dict]:
    """The saving against how many tokens the batch holds.

    Falls as the batch grows, and it does not fall to one. A batch of eight thousand tokens
    against a vocabulary of fifty thousand still holds a gradient six times smaller than the
    dense one, because most of a large vocabulary goes untouched in any single step.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    return [
        {"tokens": size, **memory_comparison(TableShape(vocabulary, width, size))}
        for size in sizes
    ]


def the_forward_pass_does_no_arithmetic(shape: TableShape | None = None) -> dict:
    """How much work a lookup is, which is none.

    A gather reads and writes and computes nothing, so it is bound by memory at any size and
    there is nothing a compiler can fuse into it. That is why an embedding never appears in a
    fusion group and why the interesting question about it is entirely about the gradient.
    """
    target = shape if shape is not None else TableShape(50000, 512, 2000)
    return {
        "multiplies": 0,
        "elements_read": target.read_elements,
        "elements_written": target.read_elements,
        "arithmetic_intensity": 0.0,
    }


def rows_touched_per_step(
    steps: int = 100, shape: TableShape | None = None, *, seed: int = 0
) -> dict:
    """How much of the table one step of training reaches.

    Two percent per step, and the cumulative coverage grows slowly because the same common
    tokens keep coming back. After a hundred steps of a skewed batch forty three percent of the
    vocabulary has still never been read once, which is what makes a rare embedding rare.
    """
    if steps < 1:
        raise ConfigError(f"the step count must be positive, got {steps}")
    target = shape if shape is not None else TableShape(50000, 64, 2000)
    seen = torch.zeros(target.vocabulary, dtype=torch.bool)
    per_step = []
    for step in range(steps):
        _, indices, _ = random_batch(target, seed=seed + step, skewed=True)
        per_step.append(int(torch.unique(indices).shape[0]))
        seen[indices] = True
    return {
        "steps": steps,
        "mean_rows_per_step": round(sum(per_step) / steps, 1),
        "share_per_step": round(sum(per_step) / steps / target.vocabulary, 6),
        "ever_touched": int(seen.sum()),
        "never_touched": int((~seen).sum()),
    }


def most_of_the_table_is_never_read(steps: int = 100) -> dict:
    """Whether that really leaves most of the vocabulary untouched."""
    result = rows_touched_per_step(steps=steps)
    total = result["ever_touched"] + result["never_touched"]
    return {
        "steps": steps,
        "share_never_touched": round(result["never_touched"] / total, 4) if total else 0.0,
    }


def optimiser_state_for(shape: TableShape, *, sparse: bool, element_bytes: int = 4) -> int:
    """Bytes the moments cost for one step of an embedding update.

    A dense update reads and writes both moments for the whole table; a sparse one does it for
    the rows it touched. The saving is the same ratio as the gradient's, and the correctness
    question underneath it is not the same at all.
    """
    rows = shape.tokens if sparse else shape.vocabulary
    return 2 * rows * shape.width * element_bytes


def sparse_updates_change_the_answer(
    steps: int = 200, decay: float = 0.999, gap: int = 50
) -> dict:
    """What skipping the decay on untouched rows does to a moment.

    A row read at step zero and again at step fifty has a second moment that decayed fifty times
    under a dense update and not at all under a sparse one, so the two updates divide by numbers
    that differ by a factor. Neither is a rounding difference: the sparse rule is a different
    optimiser and it is the one everybody runs.
    """
    if steps < 1 or gap < 1:
        raise ConfigError("the step count and the gap both have to be positive")
    gradient_square = 1.0
    dense = 0.0
    sparse = 0.0
    for step in range(steps):
        dense = decay * dense
        if step % gap == 0:
            dense = dense + (1 - decay) * gradient_square
            sparse = sparse + (1 - decay) * gradient_square
    return {
        "dense_moment": round(dense, 8),
        "sparse_moment": round(sparse, 8),
        "ratio": round(sparse / dense, 4) if dense else 0.0,
    }


def state_memory_comparison(shape: TableShape | None = None) -> dict:
    """The optimiser state for a dense and a sparse update, side by side."""
    target = shape if shape is not None else TableShape(50000, 512, 2000)
    dense = optimiser_state_for(target, sparse=False)
    sparse = optimiser_state_for(target, sparse=True)
    return {
        "dense": dense,
        "sparse": sparse,
        "ratio": round(dense / sparse, 3) if sparse else 0.0,
    }


def the_table_is_most_of_a_small_model(
    vocabulary: int = 50000, width: int = 512, layers: int = 12
) -> dict:
    """How much of a model's parameters the table is.

    Two fifths of it, at these sizes. Twenty five million in the table against thirty eight
    million in twelve layers of a width of five hundred and twelve, so a decision about the
    table is a decision about most of the parameters.
    """
    if min(vocabulary, width, layers) < 1:
        raise ConfigError("every dimension has to be positive")
    table = vocabulary * width
    body = layers * 12 * width * width
    return {
        "table": table,
        "body": body,
        "share": round(table / (table + body), 4),
    }


def a_token_outside_the_table_is_refused() -> bool:
    """Whether a token naming a row the table does not have is caught.

    It is, and the reason is that torch will not. Indexing a table with an out of range value
    raises in some paths and reads adjacent memory in others, and a compiler that passes the
    index through without checking is relying on which path it happened to take.
    """
    table = torch.randn(8, 4)
    try:
        lookup(table, torch.tensor([0, 9]))
    except ConfigError:
        return True
    return False
