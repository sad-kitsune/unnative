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


def _println_stream(text):
    C, F, OUT, S, MID = 10, 11, 12, 13, 14
    calls = [
        {"fn": "FindClass", "ret": C, "info": {"class": "java/lang/System"}},
        {"fn": "GetStaticFieldID", "ret": F, "info": {"member": "field",
         "owner": "java/lang/System", "name": "out", "desc": "Ljava/io/PrintStream;",
         "static": True}},
        {"fn": "GetStaticObjectField", "ret": OUT, "args": [C, F]},
        {"fn": "NewString", "ret": S, "info": {"string": text}},
        {"fn": "GetMethodID", "ret": MID, "info": {"member": "method",
         "owner": "java/io/PrintStream", "name": "println",
         "desc": "(Ljava/lang/String;)V", "static": False}},
        {"fn": "CallVoidMethod", "args": [OUT, MID, S]},
    ]
    return idioms.lift_method("dev/jnic/HelloWorld.main([Ljava/lang/String;)V", calls)


def _invoke_stream(owner, name, desc, args=None, addrs=None):
    """A single virtual call on a tracked `new` object, with per-arg call-site
    proof (`addrs` maps args-index -> proven constant at the call's retaddr)."""
    calls = [
        {"fn": "FindClass", "ret": 1, "info": {"class": owner, "known": True}},
        {"fn": "NewGlobalRef", "ret": 2, "args": [1]},
        {"fn": "AllocObject", "ret": 3, "args": [2]},
        {"fn": "GetMethodID", "ret": 4, "info": {"member": "method", "owner": owner,
         "name": name, "desc": desc, "static": False, "resolved": False}},
        {"fn": "CallVoidMethod", "args": [3, 4] + list(args or []), "retaddr": 0x5555},
    ]
    sites = {0x5555: addrs} if addrs else None
    return idioms.lift_method("x/Y.f()V", calls, call_sites=sites)


class StackifyInvokeTests(unittest.TestCase):
    def test_system_out_println(self):
        # The JNIC HelloWorld shape: System.out.println("Hello, world!").
        m = _println_stream("Hello, world!")
        r = stackify.stackify(m)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.bytecode, [
            {"bc": "getstatic", "owner": "java/lang/System", "name": "out",
             "desc": "Ljava/io/PrintStream;"},
            {"bc": "ldc", "ctype": "String", "value": "Hello, world!"},
            {"bc": "invokevirtual", "owner": "java/io/PrintStream", "name": "println",
             "desc": "(Ljava/lang/String;)V"},
            {"bc": "return"},
        ])

    def test_int_const_args_materialise_as_ldc(self):
        # setSize(300, 300): both int args are statically-proven immediates.
        m = _invoke_stream("x/Window", "setSize", "(II)V", args=[300, 300],
                           addrs={2: 300, 3: 300})
        r = stackify.stackify(m)
        self.assertTrue(r.ok, r.reason)
        ldcs = [bc for bc in r.bytecode if bc["bc"] == "ldc"]
        self.assertEqual([(b["ctype"], b["value"]) for b in ldcs],
                         [("I", 300), ("I", 300)])
        self.assertEqual(r.bytecode[-1], {"bc": "return"})

    def test_multi_use_construct_spills_to_local(self):
        # A construct (new+<init>) consumed by two calls must be built once and
        # stored to a local (new;dup;<init>;astore), then aloaded per use —
        # javac's shape. The spill applies to constructs only.
        base = [
            {"fn": "FindClass", "ret": 1, "info": {"class": "x/W", "known": True}},
            {"fn": "NewGlobalRef", "ret": 2, "args": [1]},
            {"fn": "GetMethodID", "ret": 9, "info": {"member": "method", "owner": "x/W",
             "name": "<init>", "desc": "()V", "static": False, "resolved": False}},
            {"fn": "AllocObject", "ret": 3, "args": [2]},
            {"fn": "CallNonvirtualVoidMethod", "args": [3, 2, 9]},
        ]
        mids = []
        for k, mname in enumerate(("f", "g")):
            mids.append({"fn": "GetMethodID", "ret": 10 + k,
                         "info": {"member": "method", "owner": "x/W", "name": mname,
                                  "desc": "()V", "static": False, "resolved": False}})
        calls = base + mids + [
            {"fn": "CallVoidMethod", "args": [3, 10]},
            {"fn": "CallVoidMethod", "args": [3, 11]},
        ]
        m = idioms.lift_method("x/Y.f()V", calls)
        r = stackify.stackify(m)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.bytecode[:5], [
            {"bc": "new", "owner": "x/W"},
            {"bc": "dup"},
            {"bc": "invokespecial", "owner": "x/W", "name": "<init>", "desc": "()V"},
            {"bc": "astore", "slot": 0},
            {"bc": "aload", "slot": 0},
        ])
        self.assertEqual(r.bytecode[-1], {"bc": "return"})


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
