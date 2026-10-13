"""Triage orchestration: census + fingerprints -> verdict -> report.json + summary.

The verdict is deliberately conservative. A specific protector is named only when
a distinctive structural signal supports it; a jar that is clearly native-translated
but shows nothing protector-specific is reported as ``unknown`` rather than guessed
into a bucket, and a jar with no native translation at all is reported as ``none``.
The protector *version* is always null for now, with a note: version fingerprinting
needs the Phase 0 corpus (plan.md §1, §4).
"""

from __future__ import annotations

import json
import os

from .. import __version__
from ..workdir import WorkDir
from . import jnic, jvmtp
from .census import Census, take_census
from .fingerprint import HIGH, LOW, MEDIUM, Fingerprint

# Score at/above which a single protector's structural signals are strong enough
# to name it (rather than report "unknown"). Tuned to the signal weights in
# jnic.py / jvmtp.py: native-methods + loader alone (2.0) is not enough; it takes a
# protector-distinctive signal on top.
_NAME_THRESHOLD = 3.0

PROTECTOR_NONE = "none"
PROTECTOR_UNKNOWN = "unknown"

_VERSION_NOTE = ("protector version not determined; version fingerprinting needs "
                 "the Phase 0 corpus (plan.md §1, §4)")


def _signal_present(fp: Fingerprint, name: str) -> bool:
    return any(s.name == name and s.present for s in fp.signals)


def classify(census: Census, jnic_fp: Fingerprint, jvmtp_fp: Fingerprint) -> dict:
    native_present = census.counts["native_methods"] > 0
    loader_present = census.counts["loader_classes"] > 0

    if not native_present and not loader_present:
        return {
            "protector": PROTECTOR_NONE,
            "confidence": HIGH,
            "version": None,
            "version_note": _VERSION_NOTE,
            "rationale": "no native methods and no native-loader call sites found; "
                         "this jar does not look native-translated.",
        }

    # JVMTP's per-triple packaging is distinctive enough to name on sight.
    if _signal_present(jvmtp_fp, "per_triple_fanout"):
        return {
            "protector": "jvmtp",
            "confidence": jvmtp_fp.confidence,
            "version": None,
            "version_note": _VERSION_NOTE,
            "rationale": "per-triple native packaging (fat jar + per-target "
                         "artifacts) is characteristic of JVMTP.",
        }

    best = max((jnic_fp, jvmtp_fp), key=lambda fp: fp.score)
    if best.score >= _NAME_THRESHOLD:
        return {
            "protector": best.protector,
            "confidence": best.confidence,
            "version": None,
            "version_note": _VERSION_NOTE,
            "rationale": f"structural signals favour {best.protector} "
                         f"(score {best.score:.1f} vs "
                         f"{(jvmtp_fp if best is jnic_fp else jnic_fp).score:.1f}).",
        }

    if native_present:
        return {
            "protector": PROTECTOR_UNKNOWN,
            "confidence": LOW,
            "version": None,
            "version_note": _VERSION_NOTE,
            "rationale": "native methods are present but no protector-distinctive "
                         "signal was found; see the per-protector fingerprints "
                         "and re-run once the Phase 0 corpus sharpens the "
                         "constants.",
        }
    return {
        "protector": PROTECTOR_NONE,
        "confidence": MEDIUM,
        "version": None,
        "version_note": _VERSION_NOTE,
        "rationale": "a native-loader call site exists but no native methods were "
                     "found; the loader may be unrelated to method translation.",
    }


def triage(source: str) -> dict:
    """Analyse a jar/dir and return the full triage report as a dict."""
    census = take_census(source)
    jnic_fp = jnic.fingerprint(census)
    jvmtp_fp = jvmtp.fingerprint(census)
    verdict = classify(census, jnic_fp, jvmtp_fp)
    return {
        "tool": "unnative",
        "tool_version": __version__,
        "stage": "triage",
        "source": os.path.abspath(source),
        "verdict": verdict,
        "fingerprints": {"jnic": jnic_fp.to_json(), "jvmtp": jvmtp_fp.to_json()},
        "census": census.to_json(),
    }


def human_summary(report: dict) -> str:
    v = report["verdict"]
    c = report["census"]["counts"]
    lines = [
        f"unnative triage — {report['source']}",
        "",
        f"  verdict     : {v['protector']}  (confidence: {v['confidence']})",
        f"  version     : {v['version'] if v['version'] else 'unknown'}",
        f"  rationale   : {v['rationale']}",
        "",
        f"  classes     : {c['classes_parsed']} parsed"
        + (f", {c['classes_failed']} failed" if c['classes_failed'] else ""),
        f"  native meth : {c['native_methods']} in {c['classes_with_native']} class(es)",
        f"  loaders     : {c['loader_classes']} class(es) with native-load calls",
        f"  resources   : {c['resources']} "
        f"({c['native_libs']} native lib, {c['embedded_jars']} jar, "
        f"{c['opaque_blobs']} blob)",
    ]

    native = report["census"]["classes"]
    nm = [(cl["name"], m) for cl in native for m in cl["native_methods"]]
    if nm:
        lines.append("")
        lines.append("  native methods:")
        for owner, m in nm[:15]:
            kind = "static " if m["static"] else ""
            lines.append(f"    - {kind}{owner}.{m['name']}{m['desc']}")
        if len(nm) > 15:
            lines.append(f"    … and {len(nm) - 15} more")

    for proto in ("jnic", "jvmtp"):
        fp = report["fingerprints"][proto]
        if fp["signals"]:
            present = ", ".join(s["name"] for s in fp["signals"])
            lines.append("")
            lines.append(f"  {proto} signals: {present}  (score {fp['score']})")

    return "\n".join(lines)


def run_stage(wd: WorkDir) -> dict:
    """Pipeline-stage entry point: triage the sample, write report.json + report.txt."""
    report = triage(wd.source)
    with open(wd.path("report.json"), "w") as f:
        json.dump(report, f, indent=2)
    summary = human_summary(report)
    with open(wd.path("report.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    v = report["verdict"]
    return {"protector": v["protector"], "confidence": v["confidence"],
            "native_methods": report["census"]["counts"]["native_methods"]}
