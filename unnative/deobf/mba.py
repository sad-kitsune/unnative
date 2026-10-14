"""MBA / constant evaluation engine (plan.md Phase 4 — the core).

JVMTP's `obfuscation.constants.mba` hides an int/long literal behind a
mixed-boolean-arithmetic expression: arithmetic and bitwise operations over other
(runtime) values that, by an algebraic identity, always collapse to the same
constant. `x ^ x` is 0 for every `x`; `(x ^ y) + 2*(x & y)` is `x + y` for every
`x, y`, so `C ^ ((x ^ y) + 2*(x & y)) ^ (x + y)` is `C` for every `x, y`.

This module is the engine that recovers those constants. It is a small integer
expression IR with exact two's-complement wrapping at a fixed width, plus:

- `evaluate` — concrete evaluation (the plan's primary path: constants are constant
  per site, so folding an expression with no free variables yields the literal).
- `fold` — the symbolic-input fallback: probe the expression over many assignments
  of its free variables (every edge value and a deterministic spread of random
  ones); if the result never varies, it is an opaque constant and we record it.

No external solver: for a genuine MBA identity (true for *all* inputs) the probe set
settles it, and a non-constant expression almost certainly differs on some probe.
The engine is width-explicit (32- or 64-bit) and models sub-register writes as
explicit masks, so a lifter can keep everything in one space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# -- expression IR ------------------------------------------------------------


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Un:
    op: str          # 'neg' | 'not'
    a: object


@dataclass(frozen=True)
class Bin:
    op: str          # + - * & | ^ << shr sar
    a: object
    b: object


Expr = object        # Const | Var | Un | Bin


def _mask(width: int) -> int:
    return (1 << width) - 1


def _signed(u: int, width: int) -> int:
    u &= _mask(width)
    return u - (1 << width) if u >> (width - 1) else u


def evaluate(expr: Expr, env: dict[str, int], width: int = 64) -> int:
    """Evaluate `expr` under variable assignment `env`, as a signed `width`-bit int.

    Unassigned variables raise KeyError — callers that mean "unknown" use `fold`,
    which supplies the assignments itself."""
    m = _mask(width)
    if isinstance(expr, Const):
        return _signed(expr.value, width)
    if isinstance(expr, Var):
        return _signed(env[expr.name], width)
    if isinstance(expr, Un):
        a = evaluate(expr.a, env, width) & m
        if expr.op == "neg":
            return _signed(-a, width)
        if expr.op == "not":
            return _signed(~a, width)
        raise ValueError(f"unknown unary op {expr.op!r}")
    if isinstance(expr, Bin):
        a = evaluate(expr.a, env, width) & m
        b = evaluate(expr.b, env, width) & m
        shift = (width - 1)
        if expr.op == "+":
            return _signed(a + b, width)
        if expr.op == "-":
            return _signed(a - b, width)
        if expr.op == "*":
            return _signed(a * b, width)
        if expr.op == "&":
            return _signed(a & b, width)
        if expr.op == "|":
            return _signed(a | b, width)
        if expr.op == "^":
            return _signed(a ^ b, width)
        if expr.op == "<<":
            return _signed(a << (b & shift), width)
        if expr.op == "shr":                       # logical (unsigned) right shift
            return _signed(a >> (b & shift), width)
        if expr.op == "sar":                       # arithmetic (signed) right shift
            return _signed(_signed(a, width) >> (b & shift), width)
        raise ValueError(f"unknown binary op {expr.op!r}")
    raise TypeError(f"not an expression: {expr!r}")


def variables(expr: Expr) -> set[str]:
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Un):
        return variables(expr.a)
    if isinstance(expr, Bin):
        return variables(expr.a) | variables(expr.b)
    return set()


def fold(expr: Expr, width: int = 64, *, n_random: int = 256, seed: int = 0x5eed) -> int | None:
    """Return the constant `expr` always evaluates to, or None if it varies.

    With no free variables this is exact. With free variables it is a probe: every
    variable is driven through the edge values (cheap, and enough to reject the
    overwhelming majority of non-constant expressions) and then, only if those all
    agree, a deterministic spread of random assignments for confidence. The probing
    is seeded, so a given expression always folds the same way."""
    vs = sorted(variables(expr))
    if not vs:
        return evaluate(expr, {}, width)
    hi = 1 << width
    edges = [0, 1, hi - 1, 2, hi >> 1, (hi >> 1) - 1,
             0x5555555555555555 & _mask(width), 0xAAAAAAAAAAAAAAAA & _mask(width)]
    first = None
    for e in edges:                                 # all-vars-equal rows, no allocation
        r = evaluate(expr, {v: e for v in vs}, width)
        if first is None:
            first = r
        elif r != first:
            return None                             # varies: the common, cheap exit
    rng = random.Random(seed)                       # only reached for edge-constant exprs
    for _ in range(n_random):
        env = {v: rng.randrange(hi) for v in vs}    # fully independent rows
        if evaluate(expr, env, width) != first:
            return None
    return first
