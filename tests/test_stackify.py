"""Tests for operand-stack reconstruction (unnative/lift/stackify).

Streams are built through the real lifter (so the data-flow track is produced the
same way the pipeline produces it), then stackified. No emulator or capstone needed.
"""

from __future__ import annotations

import unittest

from unnative.lift import idioms_jvmtp as idioms
from unnative.lift import stackify


def _throw_new_stream(err_class, ctor_desc="()V", ctor_args=None):
    C, G, MID, OBJ = 100, 101, 102, 103
    calls = [
        {"fn": "FindClass", "ret": C, "info": {"class": err_class, "known": False}},
        {"fn": "NewGlobalRef", "ret": G, "args": [C]},
        {"fn": "GetMethodID", "ret": MID, "info": {"member": "method", "owner": err_class,
         "name": "<init>", "desc": ctor_desc, "static": False, "resolved": False}},
        {"fn": "AllocObject", "ret": OBJ, "args": [G]},
        {"fn": "CallNonvirtualVoidMethod", "args": [OBJ, G, MID] + (ctor_args or [])},
        {"fn": "Throw", "args": [OBJ]},
    ]
    return idioms.lift_method("x/Y.f()V", calls)


class StackifyThrowNewTests(unittest.TestCase):
    def test_throw_new_no_arg(self):
        m = _throw_new_stream("java/lang/LinkageError")
        r = stackify.stackify(m)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.bytecode, [
            {"bc": "new", "owner": "java/lang/LinkageError"},
            {"bc": "dup"},
            {"bc": "invokespecial", "owner": "java/lang/LinkageError",
             "name": "<init>", "desc": "()V"},
            {"bc": "athrow"},
        ])

    def test_constructor_with_args_is_refused(self):
        # <init>(String) carries an operand we have not recovered: refuse, honestly.
        m = _throw_new_stream("java/lang/IllegalStateException",
                              ctor_desc="(Ljava/lang/String;)V", ctor_args=[0])
        r = stackify.stackify(m)
        self.assertFalse(r.ok)
        self.assertIn("opaque", r.reason)


class StackifyRefusalTests(unittest.TestCase):
    def test_pure_computation_refused(self):
        m = idioms.lift_method("x/Y.add(II)I", [])
        r = stackify.stackify(m)
        self.assertFalse(r.ok)
        self.assertIsNotNone(r.reason)

    def test_field_access_refused(self):
        # getfield's receiver is an untracked reference -> opaque -> refused
        calls = [
            {"fn": "FindClass", "ret": 1, "info": {"class": "x/Y", "known": True}},
            {"fn": "GetFieldID", "ret": 2, "info": {"member": "field", "owner": "x/Y",
             "name": "v", "desc": "I", "static": False, "resolved": True}},
            {"fn": "GetIntField", "args": [9, 2]},
        ]
        m = idioms.lift_method("x/Y.f()I", calls)
        r = stackify.stackify(m)
        self.assertFalse(r.ok)
        self.assertIn("opaque", r.reason)


if __name__ == "__main__":
    unittest.main()
