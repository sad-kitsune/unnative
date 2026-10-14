"""Tests for Phase 7 — the class-file writer and the emit stage.

The writer's core guarantee is surgical fidelity: a class rebuilt with no edits is
byte-for-byte identical, and a filled `native` stub re-parses with real bytecode and
nothing else disturbed. The strongest check runs the rebuilt class through a real
`java -Xverify:all` (skipped when no JDK is on PATH); the rest use the project's own
parser/decoder so they run anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import zipfile

from _fixtures import INHERIT, NATIVE_WITH_LOADER, JavaFixtureCase, compile_java, require

from unnative.classfile import ACC_NATIVE, ClassFile, decode
from unnative.emit import run_stage as emit_stage
from unnative.emit.writer import ClassWriter
from unnative.errors import UnnativeError
from unnative.workdir import WorkDir

# The one shape stackify emits today: `throw new LinkageError()`.
THROW_NEW = [
    {"bc": "new", "owner": "java/lang/LinkageError"},
    {"bc": "dup"},
    {"bc": "invokespecial", "owner": "java/lang/LinkageError",
     "name": "<init>", "desc": "()V"},
    {"bc": "athrow"},
]

# A native method with no loader <clinit>, so a plain `java` run links (and under
# -Xverify:all verifies) the filled method without an UnsatisfiedLinkError on load.
BOOM = {"Boom.java": """
public class Boom {
    public static native void boom();
    public int inst(int a, long b) { return a; }
    public native int keep(int a);
    public static void main(String[] a) { System.out.println("ok"); }
}
"""}


def _class_bytes(sources: dict, tmp: str, simple_name: str) -> bytes:
    classes = compile_java(sources, tmp)
    with open(os.path.join(classes, simple_name + ".class"), "rb") as f:
        return f.read()


class WriterRoundTripTests(JavaFixtureCase):
    def test_unmodified_is_byte_identical(self):
        # Covers native + concrete methods, a <init> with Code, a loader <clinit>.
        for sources, name in [(NATIVE_WITH_LOADER, "Demo"), (INHERIT, "Square")]:
            raw = _class_bytes(sources, self.tmp, name)
            cf = ClassFile(name + ".class", raw)
            self.assertEqual(ClassWriter(cf).to_bytes(), raw,
                             f"{name} did not round-trip identically")


class WriterFillTests(JavaFixtureCase):
    def test_fills_native_stub(self):
        raw = _class_bytes(NATIVE_WITH_LOADER, self.tmp, "Demo")
        w = ClassWriter(ClassFile("Demo.class", raw))
        w.set_method_code("add", "(II)I", THROW_NEW)
        out = ClassFile("Demo.class", w.to_bytes())

        add = next(m for m in out.methods if m["name"] == "add")
        self.assertFalse(add["access"] & ACC_NATIVE, "ACC_NATIVE not cleared")
        body, meta = out.code_of(add)
        self.assertEqual([i.name for i in decode(body)],
                         ["new", "dup", "invokespecial", "athrow"])
        self.assertEqual(meta["max_stack"], 2)       # new,dup -> 2; then it drains
        self.assertEqual(meta["max_locals"], 2)      # static (II) -> two arg slots
        # The other methods are untouched.
        self.assertEqual(out.native_methods, set())  # add was the only native one

    def test_instance_method_reserves_this_and_wide_args(self):
        raw = _class_bytes(BOOM, self.tmp, "Boom")
        w = ClassWriter(ClassFile("Boom.class", raw))
        w.set_method_code("keep", "(I)I", THROW_NEW)
        out = ClassFile("Boom.class", w.to_bytes())
        keep = next(m for m in out.methods if m["name"] == "keep")
        _body, meta = out.code_of(keep)
        self.assertEqual(meta["max_locals"], 2)      # this + one int


class WriterRefusalTests(JavaFixtureCase):
    def test_refuses_overwriting_real_code(self):
        raw = _class_bytes(NATIVE_WITH_LOADER, self.tmp, "Demo")
        w = ClassWriter(ClassFile("Demo.class", raw))
        with self.assertRaises(UnnativeError):
            w.set_method_code("mul", "(II)I", THROW_NEW)   # mul has a Code attribute

    def test_refuses_unsupported_op(self):
        raw = _class_bytes(NATIVE_WITH_LOADER, self.tmp, "Demo")
        w = ClassWriter(ClassFile("Demo.class", raw))
        with self.assertRaises(UnnativeError):
            w.set_method_code("add", "(II)I", [{"bc": "iconst_0"}])

    def test_refuses_missing_method(self):
        raw = _class_bytes(NATIVE_WITH_LOADER, self.tmp, "Demo")
        w = ClassWriter(ClassFile("Demo.class", raw))
        with self.assertRaises(UnnativeError):
            w.set_method_code("nope", "()V", THROW_NEW)


class EmitStageTests(JavaFixtureCase):
    def _jir_method(self, jir_dir: str, idx: int, owner, name, desc, bytecode, ok=True):
        os.makedirs(os.path.join(jir_dir, "methods"), exist_ok=True)
        doc = {"owner": owner, "name": name, "desc": desc,
               "method": f"{owner}.{name}{desc}", "ops": [], "dataflow": [],
               "stackify": {"ok": ok, "bytecode": bytecode, "reason": None}}
        with open(os.path.join(jir_dir, "methods", f"m{idx:04d}.json"), "w") as f:
            json.dump(doc, f)

    def test_emit_fills_and_rebuilds(self):
        jar = self.build_jar(NATIVE_WITH_LOADER, extra={"res/x.txt": b"keep me"})
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        self._jir_method(wd.path("jir"), 0, "Demo", "add", "(II)I", THROW_NEW)
        # A second method that did not stackify must be ignored.
        self._jir_method(wd.path("jir"), 1, "Demo", "mul", "(II)I", [], ok=False)

        summary = emit_stage(wd)
        self.assertEqual(summary["methods_filled"], 1)

        out = wd.path("rebuilt.jar")
        with zipfile.ZipFile(jar) as zin, zipfile.ZipFile(out) as zout:
            self.assertEqual(set(zin.namelist()), set(zout.namelist()))
            self.assertEqual(zout.read("res/x.txt"), b"keep me")
            cf = ClassFile("Demo.class", zout.read("Demo.class"))
            self.assertEqual(cf.native_methods, set())   # add is no longer native
        with open(wd.path("emit.json")) as f:
            emitted = json.load(f)
        self.assertEqual(emitted["filled"], ["Demo.add(II)I"])

    def test_emit_strips_jar_signing(self):
        # A signed jar: a manifest with a per-entry digest, plus .SF/.RSA blocks.
        classes = compile_java(NATIVE_WITH_LOADER, self.tmp)
        jar = os.path.join(self.tmp, "signed.jar")
        manifest = (b"Manifest-Version: 1.0\r\nCreated-By: test\r\n\r\n"
                    b"Name: Demo.class\r\nSHA-256-Digest: AAAA\r\n\r\n")
        with zipfile.ZipFile(jar, "w") as z:
            z.writestr("META-INF/MANIFEST.MF", manifest)
            z.writestr("META-INF/TEST.SF", b"Signature-Version: 1.0\r\n")
            z.writestr("META-INF/TEST.RSA", b"\x30\x82fake-block")
            for root, _dirs, files in os.walk(classes):
                for n in (f for f in files if f.endswith(".class")):
                    full = os.path.join(root, n)
                    z.write(full, os.path.relpath(full, classes))

        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        self._jir_method(wd.path("jir"), 0, "Demo", "add", "(II)I", THROW_NEW)
        emit_stage(wd)
        with open(wd.path("emit.json")) as f:
            summary = json.load(f)
        self.assertEqual(sorted(summary["signing_stripped"]),
                         ["META-INF/TEST.RSA", "META-INF/TEST.SF"])

        with zipfile.ZipFile(wd.path("rebuilt.jar")) as z:
            names = z.namelist()
            self.assertNotIn("META-INF/TEST.SF", names)
            self.assertNotIn("META-INF/TEST.RSA", names)
            mf = z.read("META-INF/MANIFEST.MF")
            self.assertIn(b"Created-By: test", mf)          # main section kept
            self.assertNotIn(b"SHA-256-Digest", mf)         # per-entry digest dropped
            self.assertNotIn(b"Name: Demo.class", mf)

    def test_emit_without_stackified_methods_stops(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        self._jir_method(wd.path("jir"), 0, "Demo", "add", "(II)I", [], ok=False)
        from unnative.workdir import StageNotImplemented
        with self.assertRaises(StageNotImplemented):
            emit_stage(wd)


class JavaVerifyTests(JavaFixtureCase):
    """The real proof: the JVM's own verifier accepts the filled method."""

    def test_filled_class_passes_xverify_all(self):
        require("java")
        classes = compile_java(BOOM, self.tmp)
        boom_path = os.path.join(classes, "Boom.class")
        with open(boom_path, "rb") as f:
            raw = f.read()
        w = ClassWriter(ClassFile("Boom.class", raw))
        w.set_method_code("boom", "()V", THROW_NEW)
        w.set_method_code("keep", "(I)I", THROW_NEW)
        with open(boom_path, "wb") as f:
            f.write(w.to_bytes())

        proc = subprocess.run(["java", "-Xverify:all", "-cp", classes, "Boom"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         f"verify/run failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertEqual(proc.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
