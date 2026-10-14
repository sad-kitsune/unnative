"""Optional live check — run the recovered bytecode on a real JVM.

The strongest confirmation that an emitted method behaves as recovered is to let the
JVM run it. The original native method can't be run in place (initialising its class
triggers the protector's loader), so instead we drop the *recovered bytecode* into a
clean stub method of a throwaway harness class and run that under `-Xverify:all`,
reporting what it observably did. This needs a JDK; without one the stage falls back
to the static differential and records the live check as skipped.

The bytecode run here is our own `throw new X()` recovery, not any sample native
code — nothing from the protected jar executes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ..classfile import ClassFile
from ..emit.writer import ClassWriter
from ..errors import UnnativeError

_HARNESS_SRC = """public class H {
    public static native void t();
    public static void main(String[] a) {
        try { t(); System.out.println("RETURN"); }
        catch (Throwable e) { System.out.println("THROW " + e.getClass().getName()); }
    }
}
"""


def available() -> bool:
    return bool(shutil.which("javac") and shutil.which("java"))


def run_bytecode(bytecode: list[dict]) -> dict:
    """Fill `H.t()` with `bytecode`, run it, and report the observed behaviour.

    -> {"status": "ran", "behaviour": {"kind", "exc"}} on success, or
       {"status": "skipped"|"error", "reason": ...}. `exc` is an internal name."""
    if not available():
        return {"status": "skipped", "reason": "javac/java not on PATH"}
    tmp = tempfile.mkdtemp(prefix="unnative-verify-")
    try:
        src = os.path.join(tmp, "H.java")
        with open(src, "w") as f:
            f.write(_HARNESS_SRC)
        jc = subprocess.run(["javac", "-d", tmp, src], capture_output=True, text=True)
        if jc.returncode != 0:
            return {"status": "skipped", "reason": f"javac failed: {jc.stderr.strip()}"}

        cls = os.path.join(tmp, "H.class")
        with open(cls, "rb") as f:
            raw = f.read()
        try:
            w = ClassWriter(ClassFile("H.class", raw))
            w.set_method_code("t", "()V", bytecode)
        except UnnativeError as ex:
            return {"status": "error", "reason": str(ex)}
        with open(cls, "wb") as f:
            f.write(w.to_bytes())

        run = subprocess.run(["java", "-Xverify:all", "-cp", tmp, "H"],
                             capture_output=True, text=True)
        out = run.stdout.strip()
        if out.startswith("THROW "):
            name = out[len("THROW "):].strip().replace(".", "/")
            return {"status": "ran", "behaviour": {"kind": "throw", "exc": name}}
        if out == "RETURN":
            return {"status": "ran", "behaviour": {"kind": "return", "exc": None}}
        detail = (run.stderr.strip() or out)[:200]
        return {"status": "error", "reason": f"harness did not run cleanly: {detail}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
