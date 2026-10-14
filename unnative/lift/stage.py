"""Phase 6 stage — lift every traced method's JNI-call stream to JIR.

Reads `trace.json` (Phase 5) and writes a `jir/` directory: one `m<NNNN>.json` per
method plus an `index.json` and a `summary.json`.

The linear idiom lift (`idioms_jvmtp`) needs no dependencies and runs under any
Python. When capstone is installed *and* the binding map and native libraries are on
disk, each method's JIR is additionally enriched with its recovered control-flow
graph (`cfg`): basic blocks, edges, loops and unresolved indirect exits. That half
is optional — without it the linear JIR is still produced and the summary says so.
"""

from __future__ import annotations

import json
import os
import sys

from . import cfg as cfgmod
from . import idioms_jvmtp
from . import stackify as stackifymod
from ..workdir import StageNotImplemented, WorkDir


def _load_vaddr_map(bindings_path: str) -> dict[str, tuple[str, int]]:
    """method string -> (lib filename, function vaddr) for x86_64 bindings."""
    out: dict[str, tuple[str, int]] = {}
    with open(bindings_path) as f:
        bindings = json.load(f)
    for lib in bindings.get("libraries", []):
        if lib.get("arch") != "x86_64":
            continue
        for b in lib.get("bindings", []):
            key = f"{b['owner']}.{b['name']}{b['desc']}"
            out[key] = (lib["lib"], b["vaddr"])
    return out


def _call_site_constants(md, reader, entry: int) -> dict[int, dict[int, int]]:
    """{retaddr: {args_index: constant}} for every call in the function at `entry`.

    Builds the function's CFG, disassembles each basic block, and lifts it block by
    block (no cross-block guessing): wherever an argument register (RCX/R8/R9 =
    args indices 2/3/4) provably holds a constant at a `call`, the value is recorded
    against that call's return address. The lifter only accepts a primitive operand
    that *equals* one of these — the literal is proven by the machine code, never
    assumed from the traced number."""
    from ..deobf import x86expr          # local: deobf imports lift.cfg, avoid a cycle
    graph = cfgmod.build_cfg(reader, entry)
    out: dict[int, dict[int, int]] = {}
    for block in graph.blocks.values():
        data = reader(block.start, block.end - block.start)
        if not data:
            continue
        insns = list(md.disasm(data, block.start))
        out.update(x86expr.call_site_constants(md, insns))
    return out


class _ElfCache:
    """Lazily load and reuse ElfImages by filename, with capstone-backed readers."""

    def __init__(self, libs_dir: str):
        self.libs_dir = libs_dir
        self._readers: dict[str, object] = {}

    def reader(self, lib: str):
        if lib not in self._readers:
            from ..native.loader_elf import ElfImage
            elf = ElfImage.from_path(os.path.join(self.libs_dir, lib))
            self._readers[lib] = cfgmod.elf_reader(elf)
        return self._readers[lib]


def run_stage(wd: WorkDir) -> dict:
    trace_path = wd.path("trace.json")
    if not os.path.isfile(trace_path):
        raise StageNotImplemented("lift", "Phase 6", "no trace.json — run trace first")
    with open(trace_path) as f:
        trace = json.load(f)

    # Optional CFG enrichment: needs capstone + the binding map + the libraries.
    bindings_path = wd.path("bindings.json")
    libs_dir = wd.path("libs")
    cfg_enabled = (cfgmod.CAPSTONE_AVAILABLE and os.path.isfile(bindings_path)
                   and os.path.isdir(libs_dir))
    vaddrs: dict[str, tuple[str, int]] = {}
    elves: _ElfCache | None = None
    cfg_skip_reason = None
    if cfg_enabled:
        vaddrs = _load_vaddr_map(bindings_path)
        elves = _ElfCache(libs_dir)
    elif not cfgmod.CAPSTONE_AVAILABLE:
        cfg_skip_reason = "capstone not installed"
    else:
        cfg_skip_reason = "no bindings.json / libs"

    out_dir = wd.path("jir")
    methods_dir = os.path.join(out_dir, "methods")
    os.makedirs(methods_dir, exist_ok=True)

    index = []
    totals = {"methods": 0, "ops": 0, "lifted": 0, "deferred": 0, "unlifted": 0,
              "fully_lifted": 0, "pure_computation": 0, "faulted": 0,
              "with_cfg": 0, "cfg_with_loops": 0, "cfg_indirect_exits": 0,
              "cfg_truncated": 0, "stackified": 0}
    md = None
    if cfg_enabled:
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
    for i, t in enumerate(trace.get("traces", [])):
        method = t["method"]
        call_sites = None
        cfg_json = None
        if cfg_enabled and method in vaddrs:
            lib, vaddr = vaddrs[method]
            reader = elves.reader(lib)
            try:
                graph = cfgmod.build_cfg(reader, vaddr)
                cfg_json = graph.to_json()
                sites = _call_site_constants(md, reader, vaddr)
                call_sites = sites or None
            except Exception as ex:                   # a bad decode must not sink lift
                cfg_json = {"error": f"{type(ex).__name__}: {ex}"}
        jm = idioms_jvmtp.lift_method(method, t.get("jni_calls", []),
                                      fault=t.get("fault"), locals=t.get("locals"),
                                      call_sites=call_sites)
        doc = jm.to_json()
        doc["cfg"] = cfg_json

        sk = stackifymod.stackify(jm)
        doc["stackify"] = sk.to_json()
        if sk.ok:
            totals["stackified"] += 1

        fname = f"m{i:04d}.json"
        with open(os.path.join(methods_dir, fname), "w") as f:
            json.dump(doc, f, indent=2)

        totals["methods"] += 1
        totals["ops"] += len(jm.ops)
        totals["lifted"] += jm.lifted
        totals["deferred"] += jm.deferred
        totals["unlifted"] += jm.unlifted
        if jm.fault:
            totals["faulted"] += 1
        if jm.note:
            totals["pure_computation"] += 1
        elif jm.unlifted == 0 and jm.deferred == 0 and jm.jni_calls > jm.ignored:
            totals["fully_lifted"] += 1
        cfg_stats = cfg_json.get("stats") if cfg_json and "stats" in cfg_json else None
        if cfg_stats:
            totals["with_cfg"] += 1
            totals["cfg_with_loops"] += int(cfg_stats["has_loops"])
            totals["cfg_indirect_exits"] += cfg_stats["indirect_exits"]
            totals["cfg_truncated"] += int(cfg_stats["truncated"])

        index.append({
            "id": fname,
            "method": method,
            "ops": len(jm.ops),
            "lifted": jm.lifted,
            "unlifted": jm.unlifted,
            "coverage": jm.coverage,
            "blocks": cfg_stats["blocks"] if cfg_stats else None,
            "has_loops": cfg_stats["has_loops"] if cfg_stats else None,
            "stackified": sk.ok,
            "note": jm.note,
            "fault": jm.fault,
        })

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"stage": "lift", "methods": index}, f, indent=2)
    summary = {"stage": "lift", "cfg": cfg_skip_reason or "enabled", **totals}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    cfg_line = (f"; CFG for {totals['with_cfg']} method(s) "
                f"({totals['cfg_with_loops']} with loops)" if cfg_enabled
                else f"; CFG skipped ({cfg_skip_reason})")
    print(f"lift: {totals['methods']} method(s) -> JIR; "
          f"{totals['fully_lifted']} fully lifted, "
          f"{totals['pure_computation']} pure-computation (no JNI); "
          f"{totals['lifted']} op(s) lifted, {totals['deferred']} deferred, "
          f"{totals['unlifted']} unlifted{cfg_line}; "
          f"{totals['stackified']} stackified to bytecode", file=sys.stderr)
    return totals
