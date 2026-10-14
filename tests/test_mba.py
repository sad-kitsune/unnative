"""Tests for the MBA / constant engine (unnative/deobf/mba).

The identities below are the real ones MBA obfuscators use; the engine must fold
each to its constant, and must *not* fold a near-miss that only looks like one.
No external tools.
"""

from __future__ import annotations

import unittest

from unnative.deobf import mba
from unnative.deobf.mba import Bin, Const, Un, Var

X, Y = Var("x"), Var("y")


def _mba_sum(x, y):
    """The classic (x ^ y) + 2*(x & y), which equals x + y for all x, y."""
    return Bin("+", Bin("^", x, y), Bin("*", Const(2), Bin("&", x, y)))


class EvaluateTests(unittest.TestCase):
    def test_wrapping_arithmetic(self):
        self.assertEqual(mba.evaluate(Bin("+", Const(1), Const(2)), {}, 32), 3)
        # 0x7fffffff + 1 overflows to the most-negative int.
        self.assertEqual(mba.evaluate(Bin("+", Const(0x7FFFFFFF), Const(1)), {}, 32),
                         -0x80000000)

    def test_not_and_neg(self):
        self.assertEqual(mba.evaluate(Un("not", Const(0)), {}, 32), -1)
        self.assertEqual(mba.evaluate(Un("neg", Const(5)), {}, 64), -5)

    def test_shifts_signed_vs_unsigned(self):
        self.assertEqual(mba.evaluate(Bin("sar", Const(-8), Const(1)), {}, 32), -4)
        self.assertEqual(mba.evaluate(Bin("shr", Const(-8), Const(1)), {}, 32),
                         (0xFFFFFFF8 >> 1))          # logical: large positive


class FoldConstantTests(unittest.TestCase):
    def test_no_variables_is_exact(self):
        self.assertEqual(mba.fold(Bin("+", Const(40), Const(2)), 32), 42)

    def test_mba_sum_identity_folds_to_zero(self):
        # (x + y) - mba_sum(x, y) == 0 for all x, y.
        expr = Bin("-", Bin("+", X, Y), _mba_sum(X, Y))
        self.assertEqual(mba.fold(expr, 64), 0)
        self.assertEqual(mba.fold(expr, 32), 0)

    def test_hidden_constant_through_cancelling_vars(self):
        # 42 ^ x ^ x  ==  42, the x's cancel.
        expr = Bin("^", Bin("^", Const(42), X), X)
        self.assertEqual(mba.fold(expr, 32), 42)

    def test_xor_minus_or_and_identities(self):
        self.assertEqual(mba.fold(Bin("^", X, X), 32), 0)                 # x ^ x
        self.assertEqual(mba.fold(Bin("&", X, Un("not", X)), 32), 0)      # x & ~x
        self.assertEqual(mba.fold(Bin("|", X, Un("not", X)), 32), -1)     # x | ~x
        # (x | y) - (x & y) == x ^ y, so the difference with x ^ y is 0.
        expr = Bin("-", Bin("-", Bin("|", X, Y), Bin("&", X, Y)), Bin("^", X, Y))
        self.assertEqual(mba.fold(expr, 64), 0)

    def test_non_constant_returns_none(self):
        self.assertIsNone(mba.fold(Bin("+", X, Y), 64))
        self.assertIsNone(mba.fold(X, 32))

    def test_near_miss_is_not_folded(self):
        # (x ^ y) + (x & y) is NOT x + y (missing the factor of two): must stay None.
        near = Bin("+", Bin("^", X, Y), Bin("&", X, Y))
        self.assertIsNone(mba.fold(Bin("-", Bin("+", X, Y), near), 64))


if __name__ == "__main__":
    unittest.main()
