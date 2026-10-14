"""Tests for the Phase 6 idiom lifter (unnative/lift).

These drive the lifter with synthetic JNI-call streams shaped exactly like the
enriched trace records (fn / ret / args / info), so they need no emulator and run
under the plain system Python. They pin the handle-provenance resolution: a
`Call*Method` must recover its target from the `Get*MethodID` that produced the
method handle it consumes.
"""

from __future__ import annotations

import unittest

from unnative.lift import idioms_jvmtp as idioms


def _find(fn, ret=None, args=None, info=None):
    e = {"fn": fn}
    if ret is not None:
        e["ret"] = ret
    if args is not None:
        e["args"] = args
    if info is not None:
        e["info"] = info
    return e


class ThrowNewIdiomTests(unittest.TestCase):
    """The `throw new LinkageError()` shape from the real UB.clear() method:
    FindClass -> NewGlobalRef (aliases the class) -> GetMethodID(<init>) ->
    AllocObject -> CallNonvirtualVoidMethod -> Throw."""

    def test_new_init_throw(self):
        C, G, MID, OBJ = 100, 101, 102, 103
        calls = [
            _find("FindClass", ret=C, info={"class": "java/lang/LinkageError", "known": False}),
            _find("NewGlobalRef", ret=G, args=[C]),
            _find("GetMethodID", ret=MID, info={"member": "method",
                  "owner": "java/lang/LinkageError", "name": "<init>", "desc": "()V",
                  "static": False, "resolved": False}),
            _find("AllocObject", ret=OBJ, args=[G]),
            _find("ExceptionCheck"),
            _find("CallNonvirtualVoidMethod", args=[OBJ, G, MID]),
            _find("ExceptionCheck"),
            _find("Throw", args=[OBJ]),
        ]
        m = idioms.lift_method("x/Y.clear()V", calls)
        self.assertEqual(m.ops, [
            {"op": "new", "owner": "java/lang/LinkageError"},
            {"op": "invoke", "kind": "special", "owner": "java/lang/LinkageError",
             "name": "<init>", "desc": "()V"},
            {"op": "athrow", "class": "java/lang/LinkageError"},
        ])
        self.assertEqual(m.unlifted, 0)
        self.assertEqual(m.deferred, 0)
        self.assertEqual(m.ignored, 5)   # FindClass, NewGlobalRef, GetMethodID, 2x ExceptionCheck
        self.assertEqual(m.coverage, 1.0)


class InvokeIdiomTests(unittest.TestCase):
    def test_static_methodid_at_arg1(self):
        C, MID = 1, 2
        calls = [
            _find("FindClass", ret=C, info={"class": "java/lang/Math", "known": False}),
            _find("GetStaticMethodID", ret=MID, info={"member": "method",
                  "owner": "java/lang/Math", "name": "max", "desc": "(II)I",
                  "static": True, "resolved": False}),
            _find("CallStaticIntMethod", args=[C, MID, 0, 0]),
        ]
        m = idioms.lift_method("x/Y.f()I", calls)
        self.assertEqual(m.ops, [{"op": "invoke", "kind": "static",
                                  "owner": "java/lang/Math", "name": "max", "desc": "(II)I"}])

    def test_virtual_methodid_at_arg1(self):
        C, MID, OBJ = 1, 2, 3
        calls = [
            _find("FindClass", ret=C, info={"class": "java/lang/String", "known": False}),
            _find("GetMethodID", ret=MID, info={"member": "method",
                  "owner": "java/lang/String", "name": "length", "desc": "()I",
                  "static": False, "resolved": False}),
            _find("CallIntMethod", args=[OBJ, MID]),
        ]
        m = idioms.lift_method("x/Y.f()I", calls)
        self.assertEqual(m.ops[0]["kind"], "virtual")
        self.assertEqual(m.ops[0]["name"], "length")

    def test_unresolved_methodid_is_unlifted(self):
        calls = [_find("CallStaticIntMethod", args=[1, 2])]   # no Get*MethodID before
        m = idioms.lift_method("x/Y.f()I", calls)
        self.assertEqual(m.ops[0]["op"], "unlifted")
        self.assertEqual(m.unlifted, 1)


class FieldIdiomTests(unittest.TestCase):
    def test_getfield(self):
        C, FID, OBJ = 1, 2, 3
        calls = [
            _find("FindClass", ret=C, info={"class": "x/Y", "known": True}),
            _find("GetFieldID", ret=FID, info={"member": "field", "owner": "x/Y",
                  "name": "v", "desc": "I", "static": False, "resolved": True}),
            _find("GetIntField", args=[OBJ, FID]),
        ]
        m = idioms.lift_method("x/Y.f()I", calls)
        self.assertEqual(m.ops, [{"op": "field", "kind": "getfield",
                                  "owner": "x/Y", "name": "v", "desc": "I"}])

    def test_putstatic(self):
        C, FID = 1, 2
        calls = [
            _find("FindClass", ret=C, info={"class": "x/Y", "known": True}),
            _find("GetStaticFieldID", ret=FID, info={"member": "field", "owner": "x/Y",
                  "name": "s", "desc": "J", "static": True, "resolved": True}),
            _find("SetStaticLongField", args=[C, FID, 7]),
        ]
        m = idioms.lift_method("x/Y.f()V", calls)
        self.assertEqual(m.ops[0], {"op": "field", "kind": "putstatic",
                                    "owner": "x/Y", "name": "s", "desc": "J"})


class ArrayIdiomTests(unittest.TestCase):
    def test_length_and_new_and_region(self):
        calls = [
            _find("GetArrayLength", args=[9]),
            _find("NewIntArray", ret=10, args=[5]),
            _find("GetByteArrayRegion", args=[9, 0, 4, 0]),
        ]
        m = idioms.lift_method("x/Y.f()V", calls)
        self.assertEqual(m.ops[0], {"op": "arraylength"})
        self.assertEqual(m.ops[1], {"op": "newarray", "type": "int"})
        self.assertEqual(m.ops[2]["op"], "deferred")
        self.assertEqual(m.deferred, 1)


class PureComputationTests(unittest.TestCase):
    def test_no_jni_calls(self):
        m = idioms.lift_method("x/Y.add(II)I", [])
        self.assertEqual(m.ops, [])
        self.assertIsNotNone(m.note)
        self.assertEqual(m.coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
