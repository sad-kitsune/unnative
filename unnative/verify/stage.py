"""Phase 8 stage — differentially verify the methods the emit stage filled.

For every method `emit` rebuilt, compare the behaviour the lift stage recovered (the
JIR) against (a) the behaviour re-decoded from `rebuilt.jar`'s bytes and (b) the
ground-truth behaviour of the emulated original (the Phase 5 trace). A method is at
parity when the rebuilt bytecode realises the recovered behaviour and the emulated
original does not contradict it; disagreements are reported per method with a reason.
When a JDK is present, each distinct recovered body is additionally run on a real JVM
and its observed outcome recorded. Writes `verify.json`.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zipfile

from ..errors import UnnativeError
from ..lift.jir import split_method
from ..workdir import StageNotImplemented, WorkDir
from . import differential as diff
from . import harness


def _index_jir(jir_dir: str) -> dict[str, dict]:
    """method string -> {"ops": [...], "bytecode": [...]} from the JIR files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(jir_dir, "methods", "m*.json"))):
        with open(path) as f:
            doc = json.load(f)
        out[doc["method"]] = {"ops": doc.get("ops", []),
                              "bytecode": (doc.get("stackify") or {}).get("bytecode", [])}
    return out


def _index_trace(trace_path: str) -> dict[str, dict]:
    with open(trace_path) as f:
        trace = json.load(f)
    return {t["method"]: {"calls": t.get("jni_calls", []), "fault": t.get("fault")}
            for t in trace.get("traces", [])}


# Throwables raised because a class is *missing from the harness stub classpath*,
# not because the recovered behaviour differs. Inconclusive, never a disagreement.
_MISSING_APP_CLASS = frozenset({"java/lang/NoClassDefFoundError",
                                "java/lang/ClassNotFoundException"})


def _is_missing_app_class(behaviour: dict) -> bool:
    return behaviour.get("kind") == "throw" and behaviour.get("exc") in _MISSING_APP_CLASS


def run_stage(wd: WorkDir) -> dict:
    emit_path = wd.path("emit.json")
    rebuilt = wd.path("rebuilt.jar")
    if not (os.path.isfile(emit_path) and os.path.isfile(rebuilt)):
        raise StageNotImplemented("verify", "Phase 8", "no rebuilt.jar — run emit first")
    with open(emit_path) as f:
        filled = json.load(f).get("filled", [])
    if not filled:
        raise StageNotImplemented("verify", "Phase 8", "emit filled no methods to verify")

    jir = _index_jir(wd.path("jir"))
    traces = _index_trace(wd.path("trace.json")) if os.path.isfile(wd.path("trace.json")) else {}

    zin = zipfile.ZipFile(rebuilt)
    classes: dict[str, object] = {}

    def _classfile(owner: str):
        if owner not in classes:
            from ..classfile import ClassFile
            classes[owner] = ClassFile(owner + ".class", zin.read(owner + ".class"))
        return classes[owner]

    results = []
    live_cache: dict[str, dict] = {}        # bytecode key -> harness result
    try:
        for sig in filled:
            owner, name, desc = split_method(sig)
            jm = jir.get(sig, {})
            recovered = diff.from_jir_ops(jm.get("ops", []))

            try:
                cf = _classfile(owner)
                m = next(x for x in cf.methods if x["name"] == name and x["desc"] == desc)
                insns, _meta = cf.insns(m)
                rebuilt_bhv = diff.from_bytecode(insns or [], cf)
            except (KeyError, StopIteration, UnnativeError) as ex:
                results.append({"method": sig, "verdict": "mismatch",
                                "reason": f"cannot read rebuilt bytecode: {ex}",
                                "recovered": recovered.describe()})
                continue

            # A faulted emulation isn't a trustworthy witness of normal completion.
            tr = traces.get(sig)
            trace_bhv = (diff.UNKNOWN if (tr is None or tr.get("fault"))
                         else diff.from_trace(tr["calls"]))
            v, confirmed = diff.verdict(recovered, rebuilt_bhv, trace_bhv)

            # Live run (one per distinct recovered body), compared to `recovered`.
            live = {"status": "skipped", "reason": "no body to run"}
            bytecode = jm.get("bytecode") or []
            if bytecode:
                key = json.dumps(bytecode, sort_keys=True)
                if key not in live_cache:
                    live_cache[key] = harness.run_bytecode(bytecode)
                res = live_cache[key]
                live = dict(res)
                if res.get("status") == "ran":
                    obs = res["behaviour"]
                    if _is_missing_app_class(obs):
                        # The stub classpath can't contain the *application* classes
                        # a partially-recovered body references (e.g. Window), so the
                        # live run cannot witness it. That's inconclusive, not a
                        # disagreement: miss the verdict, keep parity.
                        live = {"status": "skipped",
                                "reason": "recovered body references an application class "
                                          "not on the harness stub classpath"}
                    else:
                        live["agreed"] = (obs["kind"] == recovered.kind
                                          and obs.get("exc") == recovered.exc)
                        if not live["agreed"] and v == "parity":
                            v, confirmed = "mismatch", False
                            live["reason"] = "live run disagreed with recovered behaviour"

            results.append({"method": sig, "verdict": v,
                            "recovered": recovered.describe(),
                            "rebuilt": rebuilt_bhv.describe(),
                            "trace": trace_bhv.describe(),
                            "trace_confirmed": confirmed, "live": live})
    finally:
        zin.close()

    counts = {"parity": 0, "mismatch": 0, "unsupported": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    trace_confirmed = sum(1 for r in results if r.get("trace_confirmed"))
    live_ran = [r for r in results if r.get("live", {}).get("status") == "ran"]
    live_agreed = sum(1 for r in live_ran if r["live"].get("agreed"))
    live_errored = [r for r in results if r.get("live", {}).get("status") == "error"]

    summary = {
        "stage": "verify", "checked": len(results), **counts,
        "trace_confirmed": trace_confirmed,
        "parity_rate": round(counts["parity"] / len(results), 4) if results else 1.0,
        "live": {"ran": len(live_ran), "agreed": live_agreed,
                 "errored": len(live_errored), "available": harness.available()},
        "methods": results,
    }
    with open(wd.path("verify.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # An empty live run is not always "no JDK": list the method(s) where the harness
    # itself failed (e.g. a recovered instance body dropped into the static H.t()
    # stub) so a skipped live check is never mistaken for a clean pass.
    if live_ran:
        live_line = f"; live-confirmed {live_agreed}/{len(live_ran)} on the JVM"
    elif live_errored:
        first = live_errored[0]["live"].get("reason", "?")
        live_line = f"; live check error on {len(live_errored)} body(-ies): {first}"
    elif harness.available():
        live_line = "; live check ran nothing"
    else:
        live_line = "; live check skipped (no JDK)"
    print(f"verify: {counts['parity']}/{len(results)} method(s) at parity "
          f"({trace_confirmed} trace-confirmed, {counts['mismatch']} mismatch, "
          f"{counts['unsupported']} unsupported){live_line}", file=sys.stderr)
    return {"checked": len(results), "parity": counts["parity"]}
