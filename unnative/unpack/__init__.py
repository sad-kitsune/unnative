"""Phase 2 — unpack: get the native libraries out of the jar onto disk.

What is implemented now is the straightforward case: libraries shipped as plain jar
resources (an ELF/PE/Mach-O under some path), which the census already recognises.
They are extracted under `libs/` preserving their in-jar path so a per-triple layout
survives.

The protector-specific case — a single compressed/encrypted blob that JNIC or JVMTP
streams out at startup (plan.md §1.1/§1.2) — is deferred: re-implementing those
loaders needs the sample corpus. When a jar carries only an opaque blob and no plain
library, this stage stops with a `StageNotImplemented` naming that gap, and the
fallback (running the loader in the sandbox to collect what it writes) is future work.
"""

from __future__ import annotations

import json
import os
import sys

from ..classfile import ClassSource
from ..triage.census import take_census
from ..workdir import StageNotImplemented, WorkDir


def _safe_rel(name: str) -> str:
    # Keep the in-jar path (for the per-triple layout) but never escape libs/.
    parts = [p for p in name.split("/") if p not in ("", ".", "..")]
    return os.path.join(*parts) if parts else "lib.bin"


def _use_provided_libs(wd: WorkDir, provided: list[str]) -> dict:
    """Copy externally-extracted libraries into libs/ and write the manifest, so
    bind/trace can run on them when the protector-packed blob can't be unpacked yet
    (the `--lib` path)."""
    import shutil

    libs_dir = wd.path("libs")
    os.makedirs(libs_dir, exist_ok=True)
    extracted = []
    for src in provided:
        dest = os.path.join(libs_dir, os.path.basename(src))
        shutil.copyfile(src, dest)
        extracted.append({"entry": src, "path": os.path.basename(dest),
                          "size": os.path.getsize(dest)})
    with open(os.path.join(libs_dir, "manifest.json"), "w") as f:
        json.dump({"libraries": extracted, "source": "provided"}, f, indent=2)
    print(f"unpack: using {len(extracted)} provided librar(y/ies)", file=sys.stderr)
    return {"libraries": len(extracted), "provided": True}


def run_stage(wd: WorkDir) -> dict:
    provided = getattr(wd, "provided_libs", None)
    if provided:
        return _use_provided_libs(wd, provided)

    census = take_census(wd.source)
    native_libs = [r for r in census.resources if r.kind == "nativelib"]
    blobs = [r for r in census.resources if r.kind == "blob"]

    if not native_libs:
        if blobs:
            raise StageNotImplemented(
                "unpack", "Phase 2",
                "jar carries only an opaque blob (likely a protector-packed library); "
                "re-implementing the JNIC/JVMTP loader needs the sample corpus")
        raise StageNotImplemented(
            "unpack", "Phase 2",
            "no plain native library resource found in this jar")

    libs_dir = wd.path("libs")
    os.makedirs(libs_dir, exist_ok=True)
    extracted = []
    with ClassSource(wd.source) as src:
        for r in native_libs:
            rel = _safe_rel(r.name)
            dest = os.path.join(libs_dir, rel)
            os.makedirs(os.path.dirname(dest) or libs_dir, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(src.read_name(r.name))
            extracted.append({"entry": r.name, "path": os.path.relpath(dest, libs_dir),
                              "size": r.size})

    with open(os.path.join(libs_dir, "manifest.json"), "w") as f:
        json.dump({"libraries": extracted}, f, indent=2)
    print(f"unpack: extracted {len(extracted)} native librar(y/ies) to {libs_dir}",
          file=sys.stderr)
    return {"libraries": len(extracted)}
