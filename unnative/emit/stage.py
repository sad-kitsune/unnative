"""Phase 7 stage — emit bytecode for the stackified methods and rebuild the jar.

Reads the JIR directory (Phase 6) and, for every method `stackify` lowered to
bytecode, fills the `native` stub: clears `ACC_NATIVE` and attaches a Code
attribute built by `ClassWriter`. Every other jar entry — unlifted methods, the
loader, the native blob, resources — is copied through unchanged, so the result
is a *partial* recovery that still runs (plan.md Phase 7): the loader stays, the
few recovered methods no longer dispatch to native code.

A method the writer cannot encode is skipped with a reason rather than failing the
stage, and the reasons are recorded in `emit.json` next to the rebuilt jar.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zipfile

from ..classfile import ClassFile
from ..errors import UnnativeError
from ..workdir import StageNotImplemented, WorkDir
from .writer import ClassWriter

_SIG_EXTS = (".SF", ".RSA", ".DSA", ".EC")


def _is_signature_file(name: str) -> bool:
    """A JAR signature block (`META-INF/<name>.SF/.RSA/.DSA/.EC`)."""
    return name.startswith("META-INF/") and name.upper().endswith(_SIG_EXTS)


def _strip_manifest_digests(manifest: bytes) -> bytes:
    """Drop every per-entry section of a manifest, keeping only the main section.

    A signed jar records a `*-Digest` for each entry in `META-INF/MANIFEST.MF`;
    once we rewrite a class those digests no longer match and the JVM refuses the
    jar with a SecurityException *before* the class even loads. The per-entry
    sections follow the main section after a blank line, so we keep everything up
    to the first blank line (the global attributes — Manifest-Version, any
    TweakClass — stay) and drop the rest."""
    for sep in (b"\r\n\r\n", b"\n\n"):
        idx = manifest.find(sep)
        if idx != -1:
            return manifest[:idx] + sep[: len(sep) // 2]
    return manifest


def _stackified_methods(jir_dir: str) -> list[dict]:
    """Every JIR method whose stackify succeeded, as {owner,name,desc,bytecode}."""
    out = []
    for path in sorted(glob.glob(os.path.join(jir_dir, "methods", "m*.json"))):
        with open(path) as f:
            doc = json.load(f)
        sk = doc.get("stackify") or {}
        if sk.get("ok") and sk.get("bytecode"):
            out.append({"owner": doc["owner"], "name": doc["name"],
                        "desc": doc["desc"], "bytecode": sk["bytecode"]})
    return out


def run_stage(wd: WorkDir) -> dict:
    jir_dir = wd.path("jir")
    if not os.path.isdir(os.path.join(jir_dir, "methods")):
        raise StageNotImplemented("emit", "Phase 7", "no jir/ — run lift first")

    methods = _stackified_methods(jir_dir)
    if not methods:
        raise StageNotImplemented("emit", "Phase 7",
                                  "no stackified methods to emit (lift produced none)")

    # Group the methods to fill by their owning class.
    by_owner: dict[str, list[dict]] = {}
    for m in methods:
        by_owner.setdefault(m["owner"], []).append(m)

    src = wd.source
    zin = zipfile.ZipFile(src)
    try:
        names = set(zin.namelist())
        rebuilt: dict[str, bytes] = {}            # jar entry -> new class bytes
        filled, skipped = [], []
        for owner, ms in sorted(by_owner.items()):
            entry = owner + ".class"
            if entry not in names:
                for m in ms:
                    skipped.append({"method": f"{owner}.{m['name']}{m['desc']}",
                                    "reason": f"class entry {entry!r} not in jar"})
                continue
            cf = ClassFile(entry, zin.read(entry))
            w = ClassWriter(cf)
            applied = 0
            for m in ms:
                sig = f"{owner}.{m['name']}{m['desc']}"
                try:
                    w.set_method_code(m["name"], m["desc"], m["bytecode"])
                    filled.append(sig)
                    applied += 1
                except UnnativeError as ex:
                    skipped.append({"method": sig, "reason": str(ex)})
            if applied:
                rebuilt[entry] = w.to_bytes()

        # A signed jar must have its signing stripped: rewriting a class invalidates
        # that class's digest, and the JVM rejects the whole jar on the mismatch.
        manifest = zin.read("META-INF/MANIFEST.MF") if "META-INF/MANIFEST.MF" in names else None
        sig_files = sorted(n for n in names if _is_signature_file(n))
        signed = bool(sig_files) or (manifest is not None and b"-Digest" in manifest)
        stripped_signing = sig_files if signed else []

        out_jar = wd.path("rebuilt.jar")
        with zipfile.ZipFile(out_jar, "w") as zout:
            for info in zin.infolist():
                name = info.filename
                if signed and _is_signature_file(name):
                    continue                              # drop signature blocks
                data = rebuilt.get(name)
                if data is None:
                    data = zin.read(name)
                    if signed and name == "META-INF/MANIFEST.MF":
                        data = _strip_manifest_digests(data)
                # Preserve each entry's original compression choice.
                zout.writestr(info, data)
    finally:
        zin.close()

    summary = {
        "stage": "emit",
        "source": src,
        "classes_modified": len(rebuilt),
        "methods_filled": len(filled),
        "methods_skipped": len(skipped),
        "signing_stripped": stripped_signing,
        "filled": filled,
        "skipped": skipped,
    }
    with open(wd.path("emit.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"emit: {len(filled)} method(s) filled across {len(rebuilt)} class(es)"
          f"{f', {len(skipped)} skipped' if skipped else ''} -> rebuilt.jar",
          file=sys.stderr)
    return {"methods_filled": len(filled), "classes_modified": len(rebuilt)}
