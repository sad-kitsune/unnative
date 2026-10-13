"""JNIC fingerprint (jnic.dev).

What plan.md §1.1 lets us key on *structurally*, before the Phase 0 corpus pins
the exact constants:

  * method bodies replaced by `native` declarations, bound to stubs at load time
    (so: native methods present, plus a `<clinit>`/loader that binds a library);
  * a single bundled native library, shipped as a jar resource (its path and the
    loader class name are "unknown / to confirm in Phase 0", so we do NOT hardcode
    a magic string — we score the *shape*: one embedded native lib or opaque blob,
    not the per-triple fan-out that distinguishes JVMTP).

Each place a confirmed constant would sharpen this is marked `TODO(Phase 0)`.
Until then JNIC detection is capped at MEDIUM: the structural signals genuinely
cannot, on their own, tell JNIC from any other single-library JNI packer.
"""

from __future__ import annotations

from .census import Census
from .fingerprint import HIGH, LOW, MEDIUM, Fingerprint, Signal

# TODO(Phase 0): once the corpus is built, fill these with the confirmed JNIC
# loader class name(s) and native-resource path(s) and add a high-confidence
# exact-match signal. The strings below are candidates to *look for*, not
# confirmed facts — keep them out of the scoring until verified.
_CANDIDATE_LOADER_HINTS = ("jnic", "Jnic", "JNIC")


def fingerprint(census: Census) -> Fingerprint:
    signals: list[Signal] = []

    natives = census.native_methods
    signals.append(Signal(
        name="native_methods_present",
        present=bool(natives),
        weight=1.0,
        detail=f"{len(natives)} native method(s) across "
               f"{census.counts['classes_with_native']} class(es)",
        evidence=[m.ref for m in natives[:20]],
        confidence_if_present=LOW,
    ))

    load_calls = [lc for c in census.classes for lc in c.load_calls]
    signals.append(Signal(
        name="native_loader_present",
        present=bool(load_calls),
        weight=1.0,
        detail=f"{len(load_calls)} System/Runtime load call site(s)",
        evidence=[f"{lc.where} -> {lc.target}({lc.lib or '?'})" for lc in load_calls[:10]],
        confidence_if_present=LOW,
    ))

    # JNIC ships ONE library. A single native lib / opaque blob with no per-triple
    # fan-out is JNIC-shaped (JVMTP would carry several, or per-triple jars).
    single_lib = (census.counts["native_libs"] + census.counts["opaque_blobs"]) >= 1
    few_triples = census.counts["embedded_jars"] == 0
    signals.append(Signal(
        name="single_bundled_library",
        present=single_lib and few_triples,
        # Weak on purpose: a single packed blob does NOT distinguish JNIC from a
        # JVMTP build with resources.compress, so it must not, on its own, push the
        # verdict over the naming threshold. Confirm with the Phase 0 corpus.
        weight=0.5,
        detail="a bundled native lib/blob and no per-triple jar fan-out",
        evidence=[r.name for r in census.resources
                  if r.kind in ("nativelib", "blob")][:10],
        confidence_if_present=LOW,
    ))

    # Name hint: a class or resource name that literally mentions JNIC. Weak, and
    # trivially stripped, so LOW — but when present it is worth surfacing.
    hinted = sorted({
        c.name for c in census.classes
        if any(h in c.name for h in _CANDIDATE_LOADER_HINTS)
    } | {
        r.name for r in census.resources
        if any(h in r.name for h in _CANDIDATE_LOADER_HINTS)
    })
    signals.append(Signal(
        name="jnic_name_hint",
        present=bool(hinted),
        weight=0.5,
        detail="a class/resource name mentions JNIC (heuristic, easily stripped)",
        evidence=hinted[:10],
        confidence_if_present=LOW,
    ))

    notes = [
        "JNIC-specific constants (loader class, resource path, RegisterNatives vs "
        "Java_* symbols) are unconfirmed; see plan.md §1.1 'Unknown' column and "
        "Phase 0. Detection is structural and capped at MEDIUM until the corpus "
        "lands.",
    ]
    fp = Fingerprint.from_signals("jnic", signals, notes)
    # Cap: structural-only evidence must not exceed MEDIUM for JNIC.
    if fp.confidence == HIGH:
        fp.confidence = MEDIUM
    return fp
