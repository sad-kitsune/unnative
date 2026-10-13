"""End-to-end driver tests: unpack -> bind -> trace, and honest deferral on blobs."""

from __future__ import annotations

import json
import os
import unittest

from _fixtures import TRACE_C, JavaFixtureCase, compile_shared, have_unicorn

from unnative.cli import build_pipeline
from unnative.workdir import StageNotImplemented, WorkDir
from unnative.unpack import run_stage as unpack_stage

_COMPUTE_JAR = {"Demo.java": "public class Demo { public static native int compute(int x); }"}


class UnpackTests(JavaFixtureCase):
    def test_blob_only_defers(self):
        # A jar with a native method but only an opaque (cafebabe, non-.class) blob
        # must defer — that is the protector-packed case we can't unpack yet.
        jar = self.build_jar(_COMPUTE_JAR, name="demo.jar",
                             extra={"payload.bin": b"\xca\xfe\xba\xbe" + b"\x00" * 64})
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        with self.assertRaises(StageNotImplemented) as cm:
            unpack_stage(wd)
        self.assertEqual(cm.exception.phase, "Phase 2")

    def test_extracts_native_lib(self):
        so = compile_shared(TRACE_C, self.tmp, "libdemo.so", jni=True)
        with open(so, "rb") as f:
            lib_bytes = f.read()
        jar = self.build_jar(_COMPUTE_JAR, name="demo.jar",
                             extra={"natives/linux-x86_64/libdemo.so": lib_bytes})
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        unpack_stage(wd)
        extracted = os.path.join(wd.path("libs"), "natives", "linux-x86_64", "libdemo.so")
        self.assertTrue(os.path.isfile(extracted))
        self.assertTrue(os.path.isfile(os.path.join(wd.path("libs"), "manifest.json")))


@unittest.skipUnless(have_unicorn(), "unicorn not installed")
class FullPipelineTests(JavaFixtureCase):
    def test_run_through_trace(self):
        so = compile_shared(TRACE_C, self.tmp, "libdemo.so", jni=True)
        with open(so, "rb") as f:
            lib_bytes = f.read()
        jar = self.build_jar(_COMPUTE_JAR, name="demo.jar",
                             extra={"natives/linux-x86_64/libdemo.so": lib_bytes})
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work"))
        results = build_pipeline().run(wd)
        status = {r.name: r.status for r in results}
        self.assertEqual(status["triage"], "ran")
        self.assertEqual(status["unpack"], "ran")
        self.assertEqual(status["bind"], "ran")
        self.assertEqual(status["trace"], "ran")
        self.assertEqual(status["deobf"], "stopped")       # first unimplemented stage

        with open(wd.path("bindings.json")) as f:
            bindings = json.load(f)
        resolved = [b for lib in bindings["libraries"] for b in lib.get("bindings", [])]
        self.assertTrue(any(b["name"] == "compute" for b in resolved))

        with open(wd.path("trace.json")) as f:
            trace = json.load(f)
        compute = next(t for t in trace["traces"] if t["method"].startswith("Demo.compute"))
        self.assertEqual([c["fn"] for c in compute["jni_calls"]],
                         ["GetVersion", "FindClass", "NewStringUTF", "ExceptionCheck"])


if __name__ == "__main__":
    unittest.main()
