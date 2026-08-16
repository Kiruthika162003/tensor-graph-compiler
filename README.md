# tensor-graph-compiler

An optimising compiler for tensor graphs, written to be measured rather than believed.

It takes a graph of tensor operations, runs it through a pass pipeline to a fixed point, plans
its memory, assigns its layouts, schedules it, and generates Python that executes against an
arena. That part is ordinary. The part that is not ordinary is that every optimisation in here
is checked against something that cannot be wrong in the same way it is: a reference
interpreter, a second implementation, an exhaustive search, or a deliberately broken variant
behind a flag. Where a claim and a measurement disagreed, the measurement won and the claim was
rewritten, and the corrections are in the module docstrings and the commit messages rather than
quietly removed.

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest tests/ -q
```

```bash
tgc fusion --json
```

## What it does

```
tgc/ir          single assignment graph, symbolic shapes, explicit promotion, text format
tgc/passes      dce, cse, constant folding, algebraic rules, fusion, layout, rematerialisation
tgc/analysis    cost model, liveness, dependence, aliasing, numerics, quantisation, attention
tgc/schedule    execution order, tiling, autotuning, prefetch, pipelining, fusion search
tgc/memory      buffer planning against the theoretical floor, arena allocation
tgc/codegen     kernel emission, loop nests, vectorisation, register numbering
tgc/grad        reverse mode, forward mode, and the dot product identity between them
tgc/parallel    sharding, collectives, pipeline schedules
tgc/runtime     executor, caching allocator, compilation cache, dispatch, shape guards
tgc/verify      reference interpreter, differential testing, fuzzing with shrinking, mutation
tgc/frontend    tracing, a module frontend, an importer for a foreign graph format
```

Four invariants hold everywhere. The node list is topologically sorted and every pass has to
leave it that way. A value is assigned once. A shape is symbolic or it is concrete, and a
broadcast that cannot be resolved is refused rather than guessed. And an optimisation that
changes the answer, however slightly, sits behind an inexact flag with its divergence measured
rather than described.

## The measurements

The findings are the point, so a few of them:

**Fusion is a decision about recomputation.** A value read by two reductions is read by two
kernels, so it is either written to memory or computed twice. Searching every subset of a cone
feeding three reductions says the rule every compiler uses, keep anything with more than one
reader, is right on two of three shapes and pays half again on the third. The crossover sits
between eight and sixteen flops per element with nothing about the graph changing, which makes
it a property of the machine.

**The scheduler is choosing the accuracy of every reduction.** Nothing in the IR says how a sum
is bracketed. On sixty five thousand values with one large term the sequential order is wrong
in the fifth significant figure and the pairwise tree is wrong in the ninth, for the same number
of additions. More partitions is not monotonically better either: the error bottoms out at the
square root of the length and climbs back to exactly the sequential error at one value per
partition.

**A pass is worth what it does to the step, not to its category.** Ranking the passes by the
factor each one reports gives a different order than ranking them by what they save. Deleting
the launch overhead entirely is worth five percent and loses to fusion by a factor of three.
The credit for a pass also depends on when it was measured, in the direction that flatters
whoever measures last.

**The corpus reaches under half the operation table.** Every pass in here was verified on six
builder fixtures, and counting what they contain says fourteen of thirty operations, with two
whole categories at zero. The extended corpus in `tgc/verify/coverage.py` closes it.

Each of these lives in a module with the code that produced it, and every number above is
recomputed by the test suite on every run.

## Building and checking

```bash
python -m ruff check tgc tests examples benchmarks
```

```bash
python -m pytest tests/ -q -p no:cacheprovider
```

```bash
python examples/fusion_is_exact.py
```

3,095 tests, 30,446 lines of code counting neither comments, docstrings nor blank lines, of
which 18,431 are the compiler and 11,801 are the tests. Requires Python 3.10 or later and
PyTorch, which is used as the ground truth the reference interpreter is written against and
nowhere else.

## Attribution

Written by Kiruthika Subramani in collaboration with Claude, Anthropic's AI assistant, which
co-authored the implementation, the tests and the analysis under direction. Licensed under
Apache 2.0.
