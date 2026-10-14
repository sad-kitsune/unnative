"""Phase 4 stage — per-site deobfuscation, focused on MBA / hidden constants.

The plan's Phase 4 has several sub-passes; this stage implements the two the project
owns end-to-end — **MBA and hidden constants** — and explicitly delegates the string
layer to the sibling tool `unseed`, which already cracks the caller-keyed Java string
encryption these jars use (there is no value in reimplementing it here).

For constants it walks every bound native function, recovers each function's CFG
(capstone), and lifts each basic block to register expressions (`x86expr`). A
register that reads as runtime-dependent but always folds to one value is an opaque /
MBA-hidden constant; those are recorded per method. The lift is sound over complete,
so a recovered constant is real. Writes `deobf.json`.

capstone is required for this stage (it reads native machine code); without it the
stage stops cleanly, like the emulation stages do without Unicorn.
"""

from __future__ import annotations

import json
import os
import sys

from ..lift import cfg as cfgmod
from ..workdir import StageNotImplemented, WorkDir
from . import x86expr

# A function can disassemble into very many blocks (the lazy-init spinlocks show as
# hundreds of tiny ones); cap the per-function scan so one pathological method can't
# dominate the pass. The cap is generous and revisable.
_BLOCK_CAP = 512


def _load_vaddr_map(bindings_path: str) -> dict[str, tuple[str, int]]:
    """method string -> (lib filename, function vaddr) for x86_64 bindings.

    Same shape the lift stage reads; kept local so deobf stands alone."""
    out: dict[str, tuple[str, int]] = {}
    with open(bindings_path) as f:
        bindings = json.load(f)
    for lib in bindings.get("libraries", []):
        if lib.get("arch") != "x86_64":
            continue
        for b in lib.get("bindings", []):
            out[f"{b['owner']}.{b['name']}{b['desc']}"] = (lib["lib"], b["vaddr"])
    return out


def _readers(libs_dir: str):
    cache: dict[str, object] = {}

    def get(lib: str):
        if lib not in cache:
            from ..native.loader_elf import ElfImage
            elf = ElfImage.from_path(os.path.join(libs_dir, lib))
            cache[lib] = cfgmod.elf_reader(elf)
        return cache[lib]
    return get


def _recover_constants(md, reader, entry: int) -> dict[int, list[str]]:
    """{constant value -> [block hex addresses it was materialised in]} for one function.

    Each block is lifted independently (no cross-block assumptions), so this finds
    the intra-block opaque constants — the common MBA shape — and never guesses a
    value across a branch it didn't follow."""
    cfg = cfgmod.build_cfg(reader, entry)
    found: dict[int, list[str]] = {}
    for block in list(cfg.blocks.values())[:_BLOCK_CAP]:
        data = reader(block.start, block.end - block.start)
        if not data:
            continue
        insns = list(md.disasm(data, block.start))
        for _reg, value in x86expr.constants_in(md, insns, 64).items():
            found.setdefault(value, []).append(hex(block.start))
    return found


def run_stage(wd: WorkDir) -> dict:
    if not cfgmod.CAPSTONE_AVAILABLE:
        raise StageNotImplemented("deobf", "Phase 4",
                                  "needs capstone for native MBA/constant recovery")
    bindings_path = wd.path("bindings.json")
    libs_dir = wd.path("libs")
    if not (os.path.isfile(bindings_path) and os.path.isdir(libs_dir)):
        raise StageNotImplemented("deobf", "Phase 4", "no bindings.json / libs — run bind first")

    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True

    vaddrs = _load_vaddr_map(bindings_path)
    get_reader = _readers(libs_dir)

    methods = []
    total_constants = 0
    for method, (lib, vaddr) in sorted(vaddrs.items()):
        try:
            found = _recover_constants(md, get_reader(lib), vaddr)
        except Exception as ex:                        # a bad decode must not sink the pass
            methods.append({"method": method, "error": f"{type(ex).__name__}: {ex}"})
            continue
        if found:
            consts = [{"value": v, "hex": hex(v & 0xFFFFFFFFFFFFFFFF),
                       "signed32": v if -0x80000000 <= v < 0x80000000 else None,
                       "sites": sites}
                      for v, sites in sorted(found.items())]
            total_constants += len(consts)
            methods.append({"method": method, "constants": consts})

    summary = {
        "stage": "deobf",
        "focus": "mba+constants",
        "strings": {
            "status": "delegated",
            "recommended_tool": "unseed",
            "note": "Java string encryption (caller-keyed) is recovered by unseed; "
                    "run `unseed patch <jar>` — unnative does not reimplement it.",
        },
        "functions_scanned": len(vaddrs),
        "functions_with_constants": sum(1 for m in methods if m.get("constants")),
        "constants_recovered": total_constants,
        "methods": methods,
    }
    with open(wd.path("deobf.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"deobf: scanned {len(vaddrs)} native function(s); recovered "
          f"{total_constants} opaque/MBA constant(s) in "
          f"{summary['functions_with_constants']} function(s); "
          f"strings delegated to unseed", file=sys.stderr)
    return {"functions_scanned": len(vaddrs), "constants_recovered": total_constants}
