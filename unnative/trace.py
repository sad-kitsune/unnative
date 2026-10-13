"""Phase 5 stage — trace each bound native method's JNI-call stream.

This is the behaviour half of M1 (plan.md §6: readable behaviour per method without
lifting). For every resolved binding in an x86_64 library we run the method under
the emulator — after `JNI_OnLoad`, so the library is initialised — with default
(zero/null) arguments, and record the ordered JNI calls it makes plus its return.

Needs the emulator (Unicorn); without it the stage reports the gap rather than
producing an empty, misleading trace.
"""

from __future__ import annotations

import json
import os
import sys

from .errors import UnnativeError
from .native.jvm_model import JvmModel
from .workdir import StageNotImplemented, WorkDir


def _require_unicorn():
    try:
        import unicorn  # noqa: F401
    except ImportError as ex:
        raise StageNotImplemented("trace", "Phase 5",
                                  "needs Unicorn (pip install unicorn)") from ex


def run_stage(wd: WorkDir) -> dict:
    _require_unicorn()
    from .native.emu import UnicornEmu
    from .native.loader_elf import ElfImage

    bindings_path = wd.path("bindings.json")
    if not os.path.isfile(bindings_path):
        raise StageNotImplemented("trace", "Phase 5",
                                  "no bindings.json — run bind first")
    with open(bindings_path) as f:
        bindings = json.load(f)

    model = JvmModel.from_source(wd.source)
    libs_dir = wd.path("libs")
    traced = []
    n_methods = 0
    for lib in bindings["libraries"]:
        if lib.get("arch") != "x86_64" or not lib.get("bindings"):
            continue
        elf = ElfImage.from_path(os.path.join(libs_dir, lib["lib"]))
        emu = UnicornEmu(elf, model)
        try:
            emu.call_jni_onload()
        except UnnativeError:
            pass                        # some libs have no JNI_OnLoad; trace anyway
        for b in lib["bindings"]:
            # Bounded: real translated methods may loop or hit anti-debug; capture
            # whatever JNI calls they make before the bound and move on.
            inv = emu.trace_method(b["vaddr"], b["owner"], b["name"], b["desc"],
                                   timeout_us=2_000_000, count=5_000_000)
            n_methods += 1
            traced.append({
                "lib": lib["lib"],
                "method": f"{b['owner']}.{b['name']}{b['desc']}",
                **inv.to_json(),
            })

    report = {"stage": "trace", "methods": n_methods, "traces": traced}
    with open(wd.path("trace.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"trace: recorded JNI-call streams for {n_methods} method(s)",
          file=sys.stderr)
    return {"methods": n_methods}
