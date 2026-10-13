"""Phase 3 stage — build the native binding map from the unpacked libraries.

Per library:
  * resolve `native` methods statically via `Java_*` symbol mangling (any arch);
  * when Unicorn is available and the library is x86_64, run the dynamic binders
    under the emulator — `JNI_OnLoad` if present, plus each **per-class binder hook**
    (a statically-resolved `static ()V` native, e.g. JVMTP's per-class
    batch-resolve), calling it with its class as receiver to capture the
    `RegisterNatives` table it installs.

The two paths are fused by `native.binder` (decoys and unresolved methods flagged).
Each hook runs under a time/instruction bound so one misbehaving hook cannot hang
the batch; a hook that faults is counted, not fatal. Without Unicorn the stage
still produces the static half and says so.
"""

from __future__ import annotations

import json
import os
import sys

from .native.binder import bind
from .native.jvm_model import JvmModel
from .native.loader_elf import ElfImage
from .errors import UnnativeError
from .workdir import StageNotImplemented, WorkDir


def _have_unicorn() -> bool:
    try:
        import unicorn  # noqa: F401
        return True
    except ImportError:
        return False


# Per-hook execution bound, so a hook that loops or hits anti-debug can't hang the
# batch (the RegisterNatives call it makes happens early, so we still capture it).
_HOOK_TIMEOUT_US = 5_000_000
_HOOK_INSN_COUNT = 50_000_000


def _dynamic_bindings(elf: ElfImage, model: JvmModel, static_bindings):
    """Drive the library's dynamic binders under the emulator and return
    `((owner,name,desc,vaddr) list, stats)`.

    The binders are `JNI_OnLoad` (if present) and every per-class hook — a
    statically-resolved `static ()V` native — invoked with its class as receiver.
    """
    from .native.emu import UnicornEmu
    emu = UnicornEmu(elf, model)
    try:
        emu.call_jni_onload()
    except UnnativeError:
        pass

    hooks = [b for b in static_bindings if b.desc == "()V"]
    faults = 0
    for b in hooks:
        inv = emu.trace_method(b.vaddr, b.owner, b.name, b.desc,
                               timeout_us=_HOOK_TIMEOUT_US, count=_HOOK_INSN_COUNT)
        if inv.fault:
            faults += 1
    rn = [(b.owner, b.name, b.signature, b.fn_ptr - emu.load_bias)
          for b in emu.env.bindings]
    stats = {"hooks_driven": len(hooks), "hook_faults": faults,
             "jni_onload": emu.elf.jni_onload() is not None}
    return rn, stats


def run_stage(wd: WorkDir) -> dict:
    libs_dir = wd.path("libs")
    manifest_path = os.path.join(libs_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise StageNotImplemented("bind", "Phase 3",
                                  "no libs/manifest.json — run unpack first")
    with open(manifest_path) as f:
        manifest = json.load(f)

    model = JvmModel.from_source(wd.source)
    have_uc = _have_unicorn()
    lib_reports = []
    for lib in manifest["libraries"]:
        lib_path = os.path.join(libs_dir, lib["path"])
        entry = {"lib": lib["path"]}
        try:
            elf = ElfImage.from_path(lib_path)
        except UnnativeError as ex:
            entry["error"] = f"not an ELF (PE/Mach-O loader pending): {ex}"
            lib_reports.append(entry)
            continue

        # First pass: static Java_* resolution (gives the per-class hooks).
        static = bind(elf, model)
        register_natives = []
        used_emulator = False
        if elf.arch == "x86_64" and have_uc:
            try:
                register_natives, dyn_stats = _dynamic_bindings(elf, model,
                                                                static.bindings)
                used_emulator = True
                entry["dynamic"] = dyn_stats
            except UnnativeError as ex:
                entry["emulator_error"] = str(ex)

        # Second pass: fuse static + harvested RegisterNatives.
        result = bind(elf, model, register_natives=register_natives)
        entry.update(result.to_json())
        entry["used_emulator"] = used_emulator
        lib_reports.append(entry)

    report = {
        "stage": "bind",
        "unicorn_available": have_uc,
        "note": None if have_uc else
                "Unicorn not installed: only the static Java_* path ran; "
                "RegisterNatives bindings were not captured.",
        "libraries": lib_reports,
    }
    with open(wd.path("bindings.json"), "w") as f:
        json.dump(report, f, indent=2)

    resolved = sum(lr.get("counts", {}).get("resolved", 0) for lr in lib_reports)
    decoys = sum(lr.get("counts", {}).get("decoys", 0) for lr in lib_reports)
    print(f"bind: {resolved} method(s) resolved across {len(lib_reports)} librar(y/ies)"
          + (f", {decoys} decoy(s)" if decoys else "")
          + ("" if have_uc else "  [static Java_* only; Unicorn absent]"),
          file=sys.stderr)
    return {"resolved": resolved, "decoys": decoys, "unicorn": have_uc}
