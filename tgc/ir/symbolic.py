from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, TypeInferenceError

# Arithmetic on dimensions that are not numbers yet.
#
# ir/shape.py treats a named dimension as an opaque token: two names are equal only if they are
# the same name, and anything else is refused. That is the right default and it is not enough
# for a compiler that wants to tile a dynamic dimension, because tiling asks questions like
# whether the length divides by four, and a token cannot answer.
#
# So this file gives a dimension an expression rather than a name. A sum, a product, a floor
# division and a ceiling division, over symbols and integers, with the simplifications that hold
# unconditionally and none that do not. Everything it can prove it proves; everything else it
# reports as unproven, and the guard that would settle it is what a runtime check is generated
# from.
#
# Two results worth stating up front.
#
# The identity everyone reaches for is false. A floor division by four, multiplied by four, is
# not the original value: it is the original value rounded down to a multiple of four, and the
# two agree only when the value was already a multiple. A compiler that tiles a dynamic
# dimension by rewriting the loop that way is correct on exactly the inputs where the tile
# divides and silently drops the tail everywhere else.
#
# And divisibility is worth tracking separately from value. Knowing that a length is a multiple
# of sixteen is much less information than knowing the length, and it answers every question a
# tiling pass asks about tiles of two, four, eight and sixteen. One fact removes four of the
# five runtime guards in the sweep at the end.
#
# One correction the first version of this file needed. A symbol known to be a multiple of four
# makes the value divide by four and does not make the quotient expressible: n over four is not
# a polynomial in n, whatever is known about n. So the division stays a quotient and the fact is
# spent on the cancellation instead, where multiplying that quotient back by four returns n
# rather than n rounded down. Proving something about a value and being able to compute it are
# different questions and the code has to keep them apart.


@dataclass(frozen=True)
class Expr:
    """A dimension expression in a normal form.

    Held as a constant plus a mapping from a sorted tuple of symbol names to a coefficient, so a
    sum of products is stored flat and two expressions that are equal after expansion compare
    equal without a separate simplifier. Division is not distributive over that form, so a
    quotient is stored as an opaque term keyed by its numerator and divisor.
    """

    constant: int = 0
    terms: tuple[tuple[tuple[str, ...], int], ...] = ()

    def __post_init__(self) -> None:
        for symbols, coefficient in self.terms:
            if not symbols:
                raise ConfigError("a term with no symbols belongs in the constant")
            if coefficient == 0:
                raise ConfigError("a term with a zero coefficient should not be stored")

    @property
    def is_constant(self) -> bool:
        """Whether the expression is a plain number."""
        return not self.terms

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every symbol appearing anywhere in the expression."""
        found: set[str] = set()
        for names, _ in self.terms:
            found.update(names)
        return tuple(sorted(found))

    def __str__(self) -> str:
        parts = []
        for names, coefficient in self.terms:
            body = "*".join(names)
            parts.append(body if coefficient == 1 else f"{coefficient}*{body}")
        if self.constant or not parts:
            parts.append(str(self.constant))
        return " + ".join(parts)


def constant(value: int) -> Expr:
    """A dimension that is already known."""
    return Expr(constant=int(value))


def symbol(name: str) -> Expr:
    """A dimension that is not known yet."""
    if not name:
        raise ConfigError("a symbol needs a name")
    return Expr(terms=(((name,), 1),))


def _normalise(constant_part: int, terms: dict[tuple[str, ...], int]) -> Expr:
    """Drop zero coefficients and sort, so equal expressions compare equal."""
    kept = tuple(sorted((names, value) for names, value in terms.items() if value))
    return Expr(constant=constant_part, terms=kept)


def add(left: Expr, right: Expr) -> Expr:
    """The sum of two expressions."""
    terms: dict[tuple[str, ...], int] = {}
    for names, value in left.terms + right.terms:
        terms[names] = terms.get(names, 0) + value
    return _normalise(left.constant + right.constant, terms)


def multiply(left: Expr, right: Expr) -> Expr:
    """The product of two expressions.

    Expanded rather than kept as a product, which is what makes equality decidable here. Two
    expressions written differently and equal after expansion have the same normal form, and
    anything not expandable is not representable in this form at all.
    """
    terms: dict[tuple[str, ...], int] = {}
    for names, value in left.terms:
        terms[names] = terms.get(names, 0) + value * right.constant
    for names, value in right.terms:
        terms[names] = terms.get(names, 0) + value * left.constant
    for left_names, left_value in left.terms:
        for right_names, right_value in right.terms:
            joined = tuple(sorted(left_names + right_names))
            terms[joined] = terms.get(joined, 0) + left_value * right_value
    return _normalise(left.constant * right.constant, terms)


def scale(expression: Expr, factor: int) -> Expr:
    """An expression multiplied by a number."""
    return multiply(expression, constant(factor))


@dataclass(frozen=True)
class Quotient:
    """A division that could not be carried out.

    Kept as a pair rather than folded into an expression, because a floor division is not a
    polynomial and pretending it is would let the normal form claim two unequal things are
    equal. Everything a caller can ask about one is answered by the divisibility rules below.
    """

    numerator: Expr
    divisor: int
    ceiling: bool = False

    def __post_init__(self) -> None:
        if self.divisor < 1:
            raise ConfigError(f"cannot divide by {self.divisor}")

    def __str__(self) -> str:
        operator = "ceildiv" if self.ceiling else "floordiv"
        return f"{operator}({self.numerator}, {self.divisor})"


@dataclass
class Facts:
    """What is known about the symbols in an expression.

    Divisibility rather than value, because divisibility is what a tiling pass asks about and a
    range is what a bounds check asks about, and neither of them needs the number. Storing the
    number would be storing something the compiler does not have.
    """

    divisible_by: dict[str, int] = field(default_factory=dict)
    at_least: dict[str, int] = field(default_factory=dict)

    def knows_multiple_of(self, name: str, factor: int) -> bool:
        """Whether a symbol is known to be a multiple of a number."""
        if factor < 1:
            raise ConfigError(f"cannot ask about multiples of {factor}")
        known = self.divisible_by.get(name)
        return known is not None and known % factor == 0

    def lower_bound(self, name: str) -> int:
        """The smallest value a symbol is known to take."""
        return self.at_least.get(name, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "divisible": dict(sorted(self.divisible_by.items())),
            "at_least": dict(sorted(self.at_least.items())),
        }


def divides(expression: Expr, factor: int, facts: Facts | None = None) -> bool:
    """Whether an expression is certainly a multiple of a number.

    Certainly, not probably. Every term has to be a multiple on its own, which is sufficient and
    not necessary: a sum of two terms neither of which divides can still divide, and proving
    that needs more than this form carries. Reporting unproven where the answer is yes costs a
    guard; reporting proven where the answer is no costs an incorrect program.
    """
    if factor < 1:
        raise ConfigError(f"cannot ask about multiples of {factor}")
    known = facts if facts is not None else Facts()
    if expression.constant % factor:
        return False
    for names, coefficient in expression.terms:
        if coefficient % factor == 0:
            continue
        remaining = factor // math.gcd(factor, coefficient)
        if not any(known.knows_multiple_of(name, remaining) for name in names):
            return False
    return True


def coefficients_divide(expression: Expr, divisor: int) -> bool:
    """Whether a division can be carried out inside this normal form.

    A stricter question than whether the value divides, and the difference is the thing that
    took a correction to get right. A symbol known to be a multiple of four makes the value
    divide by four; it does not let the division be performed, because the quotient of a symbol
    is not a polynomial in that symbol. Only a division that hits the coefficients can be done
    here, and everything else stays a quotient however much is known about it.
    """
    if divisor < 1:
        raise ConfigError(f"cannot divide by {divisor}")
    if expression.constant % divisor:
        return False
    return all(coefficient % divisor == 0 for _, coefficient in expression.terms)


def floor_divide(
    expression: Expr, divisor: int, _facts: Facts | None = None
) -> Expr | Quotient:
    """A floor division, carried out when the coefficients allow and kept when they do not.

    The facts are accepted and not used, which is deliberate rather than an oversight. Knowing
    that a symbol is a multiple of the divisor makes the value divide and does not make the
    quotient expressible, so nothing here can act on it. The cancellation in rebuild is where
    the fact is spent.
    """
    if divisor < 1:
        raise ConfigError(f"cannot divide by {divisor}")
    if coefficients_divide(expression, divisor):
        terms = {names: value // divisor for names, value in expression.terms}
        return _normalise(expression.constant // divisor, terms)
    return Quotient(numerator=expression, divisor=divisor)


def ceil_divide(expression: Expr, divisor: int, _facts: Facts | None = None) -> Expr | Quotient:
    """A ceiling division, carried out when the coefficients allow and kept when they do not."""
    if divisor < 1:
        raise ConfigError(f"cannot divide by {divisor}")
    if coefficients_divide(expression, divisor):
        terms = {names: value // divisor for names, value in expression.terms}
        return _normalise(expression.constant // divisor, terms)
    return Quotient(numerator=expression, divisor=divisor, ceiling=True)


def rebuild(quotient: Quotient, factor: int, facts: Facts | None = None) -> Expr | Quotient:
    """A kept quotient multiplied back by its divisor.

    Returns the numerator when the value is known to divide, which is the only case where the
    two operations cancel. Otherwise the quotient stays a quotient, because multiplying a floor
    division by its divisor gives the numerator rounded down rather than the numerator.
    """
    if factor != quotient.divisor:
        raise ConfigError(
            f"multiplying a division by {quotient.divisor} back by {factor} cancels nothing"
        )
    if divides(quotient.numerator, quotient.divisor, facts):
        return quotient.numerator
    return quotient


def equal(left: Expr, right: Expr) -> bool:
    """Whether two expressions are certainly the same value."""
    return left == right


def evaluate(expression: Expr, values: dict[str, int]) -> int:
    """The number an expression takes, given values for its symbols."""
    missing = [name for name in expression.symbols if name not in values]
    if missing:
        raise TypeInferenceError(f"no value given for {missing}")
    total = expression.constant
    for names, coefficient in expression.terms:
        product = coefficient
        for name in names:
            product *= values[name]
        total += product
    return total


def evaluate_quotient(quotient: Quotient, values: dict[str, int]) -> int:
    """The number a kept division takes, given values for its symbols."""
    numerator = evaluate(quotient.numerator, values)
    if quotient.ceiling:
        return -(-numerator // quotient.divisor)
    return numerator // quotient.divisor


def the_identity_that_is_false(length: str = "n", tile: int = 4) -> dict:
    """Whether dividing by a tile and multiplying back gives the original.

    It does not, and the failure is not exotic. The rewritten expression is the original rounded
    down to a multiple of the tile, so it agrees on the multiples and is short by the remainder
    everywhere else, which is exactly the tail a tiling pass would be dropping.
    """
    dimension = symbol(length)
    divided = floor_divide(dimension, tile)
    if isinstance(divided, Expr):
        return {"proven_equal": True, "counterexample": None}

    for value in range(1, 4 * tile + 1):
        rebuilt = evaluate_quotient(divided, {length: value}) * tile
        if rebuilt != value:
            return {
                "proven_equal": False,
                "counterexample": value,
                "rebuilt_as": rebuilt,
                "short_by": value - rebuilt,
            }
    return {"proven_equal": False, "counterexample": None}


def the_identity_holds_when_it_divides(length: str = "n", tile: int = 4) -> dict:
    """The same question with the divisibility known.

    Told that the length is a multiple of the tile, the division still cannot be performed,
    because the quotient of a symbol is not a polynomial in that symbol. What the fact does buy
    is the cancellation: multiplying the kept quotient back by its divisor returns the numerator
    rather than the numerator rounded down. One fact turns an unsound rewrite into an exact one
    without ever computing the quotient, which is the distinction the first version of this file
    got wrong.
    """
    facts = Facts(divisible_by={length: tile})
    dimension = symbol(length)
    divided = floor_divide(dimension, tile, facts)
    if isinstance(divided, Expr):
        return {"kept_as_a_quotient": False, "cancels": False}
    rebuilt = rebuild(divided, tile, facts)
    without_facts = rebuild(floor_divide(dimension, tile), tile)
    return {
        "kept_as_a_quotient": True,
        "cancels": isinstance(rebuilt, Expr) and equal(rebuilt, dimension),
        "cancels_without_the_fact": isinstance(without_facts, Expr),
    }


def tile_count(length: Expr, tile: int, facts: Facts | None = None) -> Expr | Quotient:
    """How many tiles cover a dimension.

    A ceiling division, because the last tile is partial and still has to run. Exact when the
    tile divides, which is the case a tiling pass wants because it means no tail loop, and kept
    otherwise so the caller can see there is one.
    """
    return ceil_divide(length, tile, facts)


def needs_a_tail(length: Expr, tile: int, facts: Facts | None = None) -> bool:
    """Whether a tiled loop needs a partial iteration at the end."""
    return not divides(length, tile, facts)


def tiling_questions(length: str = "n") -> list[dict]:
    """Every question a tiling pass asks about a dynamic dimension, with and without facts.

    All of them are answered by divisibility alone. None of them needs the value of the
    dimension, which is the reason a compiler can tile a dynamic shape at all: what it has to
    prove is a much weaker statement than what it does not know.
    """
    dimension = symbol(length)
    rows = []
    for tile in (2, 4, 8, 16):
        without = needs_a_tail(dimension, tile)
        with_facts = needs_a_tail(dimension, tile, Facts(divisible_by={length: 16}))
        rows.append(
            {
                "tile": tile,
                "tail_without_facts": without,
                "tail_with_facts": with_facts,
            }
        )
    return rows


def one_fact_answers_every_question(length: str = "n") -> dict:
    """How many of those questions the single divisibility fact settles."""
    rows = tiling_questions(length)
    return {
        "questions": len(rows),
        "unanswered_without_facts": sum(1 for row in rows if row["tail_without_facts"]),
        "unanswered_with_facts": sum(1 for row in rows if row["tail_with_facts"]),
    }


def a_fact_about_sixteen_does_not_settle_thirty_two(length: str = "n") -> dict:
    """Where the divisibility fact stops helping.

    A multiple of sixteen is a multiple of every factor of sixteen and of nothing above it. So
    one fact answers four questions and the fifth needs a different fact, which is the shape of
    every proof system: the facts are cheap and the useful ones are specific.
    """
    facts = Facts(divisible_by={length: 16})
    dimension = symbol(length)
    return {
        "multiple_of_eight": divides(dimension, 8, facts),
        "multiple_of_sixteen": divides(dimension, 16, facts),
        "multiple_of_thirty_two": divides(dimension, 32, facts),
    }


def divisibility_of_a_sum(length: str = "n") -> dict:
    """What the rules can and cannot prove about a sum.

    A symbol known to be a multiple of four plus a constant four is a multiple of four, and the
    same symbol plus two is not. Both of those are decided term by term, which is sound and
    incomplete: two terms that individually do not divide can sum to something that does, and
    nothing here will notice.
    """
    facts = Facts(divisible_by={length: 4})
    dimension = symbol(length)
    return {
        "symbol_plus_four": divides(add(dimension, constant(4)), 4, facts),
        "symbol_plus_two": divides(add(dimension, constant(2)), 4, facts),
        "twice_the_symbol": divides(scale(dimension, 2), 8, facts),
    }


def the_rules_are_incomplete(length: str = "n") -> dict:
    """A case the rules refuse that is true for every value.

    The product of a number and the one after it is always even, because one of the two is. The
    term by term test looks at a squared term with coefficient one and a linear term with
    coefficient one, finds neither divisible by two, and refuses. So the compiler emits a guard
    that can never fail.

    That is the right direction to be wrong in. Refusing something true costs a comparison;
    accepting something false costs the answer.
    """
    dimension = symbol(length)
    product = multiply(dimension, add(dimension, constant(1)))
    return {
        "expression": str(product),
        "proven_even": divides(product, 2),
        "actually_even_for_every_value": all(
            evaluate(product, {length: value}) % 2 == 0 for value in range(1, 33)
        ),
    }


@dataclass
class Guard:
    """A condition a runtime check has to confirm."""

    expression: Expr
    factor: int

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ConfigError(f"cannot guard on multiples of {self.factor}")

    def holds_for(self, values: dict[str, int]) -> bool:
        """Whether the condition is true for a set of values."""
        return evaluate(self.expression, values) % self.factor == 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"expression": str(self.expression), "multiple_of": self.factor}


def guards_for(length: Expr, tiles: Sequence[int], facts: Facts | None = None) -> list[Guard]:
    """A check for every tiling that could not be proven.

    Only the unproven ones. A guard for something already known is a branch the compiler can
    prove is never taken, and emitting it wastes a comparison per call and, worse, suggests to
    whoever reads the generated code that the compiler did not know.
    """
    if not tiles:
        raise ConfigError("there is nothing to guard")
    return [
        Guard(expression=length, factor=tile)
        for tile in tiles
        if needs_a_tail(length, tile, facts)
    ]


def facts_remove_guards(length: str = "n") -> dict:
    """How many runtime checks one compile time fact removes."""
    dimension = symbol(length)
    tiles = (2, 4, 8, 16, 32)
    return {
        "without_facts": len(guards_for(dimension, tiles)),
        "with_facts": len(guards_for(dimension, tiles, Facts(divisible_by={length: 16}))),
    }


def a_guard_that_fails(length: str = "n", tile: int = 8) -> dict:
    """What a guard is checking, on values that pass and values that do not."""
    guard = Guard(expression=symbol(length), factor=tile)
    return {
        "passes_on_a_multiple": guard.holds_for({length: tile * 3}),
        "fails_otherwise": not guard.holds_for({length: tile * 3 + 1}),
    }


def padding_instead_of_a_guard(length: Expr, tile: int) -> Expr | Quotient:
    """The other way to make a dimension divide: round it up.

    Rounding a dynamic dimension up to a multiple of the tile removes the tail and adds work on
    the padding, which is the same trade runtime/guards.py measures between recompiling and
    padding. It is expressed here as an expression rather than as a policy, so the two files are
    talking about the same thing.
    """
    if tile < 1:
        raise ConfigError(f"cannot pad to multiples of {tile}")
    tiles = ceil_divide(length, tile)
    if isinstance(tiles, Expr):
        return scale(tiles, tile)
    return Quotient(numerator=length, divisor=tile, ceiling=True)


def padding_always_divides(tile: int = 8) -> dict:
    """Whether the padded dimension is a multiple by construction.

    It is, and that is the point: padding buys the divisibility that the guard was checking for,
    at the cost of doing arithmetic on elements nobody asked about. Measured over a range of
    lengths so the claim is about the construction rather than about one value.
    """
    rows = []
    for value in range(1, 4 * tile + 1):
        padded = -(-value // tile) * tile
        rows.append(padded % tile == 0)
    return {
        "lengths": len(rows),
        "all_divide": all(rows),
        "largest_padding": max(
            -(-value // tile) * tile - value for value in range(1, 4 * tile + 1)
        ),
    }


def evaluation_agrees_with_the_symbols(count: int = 64) -> dict:
    """Whether the symbolic arithmetic gives the same answers as the numbers.

    The check that keeps the whole file honest. Every expression built here has a value once its
    symbols do, and if the normal form ever disagreed with plain arithmetic it would be a
    compiler that reasons correctly about something other than the program.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    first = symbol("n")
    second = symbol("m")
    built = add(multiply(add(first, constant(2)), second), scale(first, 3))

    disagreements = 0
    for n in range(1, count + 1):
        for m in range(1, 5):
            expected = (n + 2) * m + 3 * n
            if evaluate(built, {"n": n, "m": m}) != expected:
                disagreements += 1
    return {"checked": count * 4, "disagreements": disagreements}


def a_missing_value_is_refused() -> bool:
    """Whether evaluating with a symbol left out is caught."""
    try:
        evaluate(symbol("n"), {})
    except TypeInferenceError:
        return True
    return False


def normal_form_makes_equality_decidable() -> dict:
    """Two expressions written differently that are the same after expansion.

    Both spellings expand to the same flat sum, so they compare equal without anything having to
    call a simplifier, and the compiler can decide that two shapes match rather than reporting
    that it does not know.
    """
    first = symbol("n")
    second = symbol("m")
    left = multiply(add(first, second), add(first, second))
    right = add(
        add(multiply(first, first), scale(multiply(first, second), 2)),
        multiply(second, second),
    )
    return {"left": str(left), "right": str(right), "equal": equal(left, right)}
