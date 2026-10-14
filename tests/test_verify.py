"""Tests for Phase 8 — differential verification.

The behaviour derivations and the verdict logic run with no external tools. The stage
test drives lift-shaped inputs through emit and then verify end-to-end, and the live
harness test (gated on a JDK) runs recovered bytecode on a real JVM.
"""

from __future__ import annotations

import json
import os
import unittest

from _fixtures import NATIVE_WITH_LOADER, JavaFixtureCase, compile_java

from unnative.classfile import ClassFile
from unnative.emit import run_stage as emit_stage
from unnative.emit.writer import ClassWriter
from unnative.verify import differential as diff
from unnative.verify import harness
from unnative.verify import run_stage as verify_stage
from unnative.workdir import StageNotImplemented, WorkDir

ERR = "java/lang/LinkageError"

THROW_NEW = [
    {"bc": "new", "owner": ERR},
    {"bc": "dup"},
    {"bc": "invokespecial", "owner": ERR, "name": "<init>", "desc": "()V"},
    {"bc": "athrow"},
]
THROW_OPS = [
    {"op": "new", "owner": ERR},
    {"op": "invoke", "kind": "special", "owner": ERR, "name": "<init>", "desc": "()V"},
    {"op": "athrow", "class": ERR},
]
# A trace whose thrown handle chains Throw -> AllocObject -> NewGlobalRef -> FindClass.
THROW_TRACE = [
    {"fn": "FindClass", "ret": 1, "info": {"class": ERR}},
    {"fn": "NewGlobalRef", "ret": 2, "args": [1]},
    {"fn": "AllocObject", "ret": 3, "args": [2]},
    {"fn": "Throw", "args": [3]},
]


class DerivationTests(unittest.TestCase):
    def test_from_jir_ops(self):
        self.assertEqual(diff.from_jir_ops(THROW_OPS), diff.Behaviour("throw", ERR))
        self.assertEqual(diff.from_jir_ops([]), diff.UNKNOWN)

    def test_from_jir_ops_non_throw_is_return(self):
        ops = [{"op": "field", "kind": "getstatic"}, {"op": "invoke", "kind": "virtual"}]
        self.assertEqual(diff.from_jir_ops(ops), diff.Behaviour("return"))

    def test_from_trace_chases_handles(self):
        self.assertEqual(diff.from_trace(THROW_TRACE), diff.Behaviour("throw", ERR))

    def test_from_trace_thrownew(self):
        calls = [{"fn": "FindClass", "ret": 5, "info": {"class": ERR}},
                 {"fn": "ThrowNew", "args": [5, 0]}]
        self.assertEqual(diff.from_trace(calls), diff.Behaviour("throw", ERR))

    def test_from_trace_no_throw_is_return(self):
        # No Throw in the stream means the emulated method completed normally.
        self.assertEqual(diff.from_trace([{"fn": "GetVersion", "ret": 8}]),
                         diff.Behaviour("return"))

    def test_from_trace_multiple_refs_indeterminate_but_still_a_throw(self):
        # Throw of an object whose class can't be followed -> throw with no class.
        calls = [{"fn": "Throw", "args": [99]}]
        self.assertEqual(diff.from_trace(calls), diff.Behaviour("throw", None))


class VerdictTests(unittest.TestCase):
    T = diff.Behaviour("throw", ERR)

    def test_full_agreement_is_confirmed_parity(self):
        self.assertEqual(diff.verdict(self.T, self.T, self.T), ("parity", True))

    def test_emit_divergence_is_mismatch(self):
        other = diff.Behaviour("throw", "java/lang/IllegalStateException")
        self.assertEqual(diff.verdict(self.T, other, self.T)[0], "mismatch")

    def test_trace_disagreement_is_mismatch(self):
        other = diff.Behaviour("throw", "java/lang/Error")
        self.assertEqual(diff.verdict(self.T, self.T, other)[0], "mismatch")

    def test_trace_indeterminate_is_unconfirmed_parity(self):
        self.assertEqual(diff.verdict(self.T, self.T, diff.UNKNOWN), ("parity", False))

    def test_unknown_recovered_is_unsupported(self):
        self.assertEqual(diff.verdict(diff.UNKNOWN, diff.UNKNOWN, diff.UNKNOWN)[0],
                         "unsupported")

    def test_return_parity(self):
        R = diff.Behaviour("return")
        self.assertEqual(diff.verdict(R, R, R), ("parity", True))

    def test_return_vs_throw_trace_is_mismatch(self):
        R = diff.Behaviour("return")
        self.assertEqual(diff.verdict(R, R, self.T)[0], "mismatch")


class FromBytecodeTests(JavaFixtureCase):
    def test_reads_emitted_throw_shape(self):
        classes = compile_java(NATIVE_WITH_LOADER, self.tmp)
        with open(os.path.join(classes, "Demo.class"), "rb") as f:
            raw = f.read()
        w = ClassWriter(ClassFile("Demo.class", raw))
        w.set_method_code("add", "(II)I", THROW_NEW)
        cf = ClassFile("Demo.class", w.to_bytes())
        add = next(m for m in cf.methods if m["name"] == "add")
        insns, _ = cf.insns(add)
        self.assertEqual(diff.from_bytecode(insns, cf), diff.Behaviour("throw", ERR))


class VerifyStageTests(JavaFixtureCase):
    def _setup(self, with_trace=True):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        os.makedirs(os.path.join(wd.path("jir"), "methods"), exist_ok=True)
        doc = {"owner": "Demo", "name": "add", "desc": "(II)I",
               "method": "Demo.add(II)I", "ops": THROW_OPS, "dataflow": [],
               "stackify": {"ok": True, "bytecode": THROW_NEW, "reason": None}}
        with open(os.path.join(wd.path("jir"), "methods", "m0000.json"), "w") as f:
            json.dump(doc, f)
        if with_trace:
            with open(wd.path("trace.json"), "w") as f:
                json.dump({"traces": [{"method": "Demo.add(II)I",
                                       "jni_calls": THROW_TRACE}]}, f)
        emit_stage(wd)
        return wd

    def test_parity_with_trace_confirmation(self):
        wd = self._setup(with_trace=True)
        verify_stage(wd)
        with open(wd.path("verify.json")) as f:
            v = json.load(f)
        self.assertEqual(v["checked"], 1)
        self.assertEqual(v["parity"], 1)
        self.assertEqual(v["mismatch"], 0)
        self.assertEqual(v["trace_confirmed"], 1)
        m = v["methods"][0]
        self.assertEqual(m["verdict"], "parity")
        self.assertEqual(m["recovered"], f"throw {ERR}")
        self.assertEqual(m["rebuilt"], f"throw {ERR}")
        # If a JDK is present the live run must have agreed; otherwise it's skipped.
        if m["live"].get("status") == "ran":
            self.assertTrue(m["live"]["agreed"])

    def test_parity_unconfirmed_without_trace(self):
        wd = self._setup(with_trace=False)
        with open(wd.path("trace.json"), "w") as f:
            json.dump({"traces": []}, f)
        verify_stage(wd)
        with open(wd.path("verify.json")) as f:
            v = json.load(f)
        self.assertEqual(v["parity"], 1)
        self.assertEqual(v["trace_confirmed"], 0)     # no trace to confirm against

    def test_stops_without_emit(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        with self.assertRaises(StageNotImplemented):
            verify_stage(wd)


class LiveHarnessTests(unittest.TestCase):
    def test_runs_recovered_bytecode_on_jvm(self):
        if not harness.available():
            self.skipTest("javac/java not on PATH")
        res = harness.run_bytecode(THROW_NEW)
        self.assertEqual(res["status"], "ran", res.get("reason"))
        self.assertEqual(res["behaviour"], {"kind": "throw", "exc": ERR})


if __name__ == "__main__":
    unittest.main()
