"""JVMTP fingerprint (jvmtp.dev).

JVMTP has a distinctive *packaging* shape (plan.md §1.2) that JNIC does not, and
that structural shape is strong enough to name on sight:

  * a fat jar PLUS one jar (or native lib) per target triple, picked at startup by
    an injected `NativeLoader`;
  * optionally `resources.compress`: all libs packed into one compressed blob
    streamed out at startup;
  * `<init>`/`<clinit>` bodies split into synthetic methods with machine-generated
    (`hash`/`alphabetic`/`random`) names.

The per-triple fan-out is the high-confidence signal: seeing several target-triple
artifacts in one jar is characteristic of JVMTP's multi-target packaging. The
loader class name and blob format are "unknown" per §1.2, so the `NativeLoader`
name match stays heuristic (MEDIUM) and is marked `TODO(Phase 0)`.
"""

from __future__ import annotations

import re

from .census import Census
from .fingerprint import HIGH, LOW, MEDIUM, Fingerprint, Signal

# Target-triple tokens JVMTP's per-target artifacts tend to carry. Matching two or
# more distinct triples in resource names is the characteristic multi-target shape.
_OS_TOKENS = ("linux", "windows", "win", "macos", "osx", "darwin", "mac")
_ARCH_TOKENS = ("x86_64", "x86-64", "amd64", "aarch64", "arm64", "x64")

# TODO(Phase 0): confirm the injected loader's real class name and tighten this.
_LOADER_NAME_RE = re.compile(r"(?:^|[/$])[A-Za-z0-9_]*NativeLoader")

# Machine-generated synthetic split-method names: the plan calls out hash /
# alphabetic / random naming. Heuristic matchers, deliberately conservative.
_HASHY_RE = re.compile(r"^[a-z0-9]{8,}$|^[0-9a-f]{8,}$")
_ALPHA_RE = re.compile(r"^[a-zA-Z]{8,}$")


def _triples_in(names: list[str]) -> list[str]:
    found = []
    for n in names:
        low = n.lower()
        os_hit = next((t for t in _OS_TOKENS if t in low), None)
        arch_hit = next((t for t in _ARCH_TOKENS if t in low), None)
        if os_hit and arch_hit:
            found.append(n)
    return found


def fingerprint(census: Census) -> Fingerprint:
    signals: list[Signal] = []

    res_names = [r.name for r in census.resources]

    # (1) Per-triple fan-out: multiple embedded jars, or lib/blob resources that
    # name two or more distinct target triples.
    triple_named = _triples_in(res_names)
    lib_like = [r for r in census.resources if r.kind in ("nativelib", "jar", "blob")]
    fanout = census.counts["embedded_jars"] >= 2 or len(triple_named) >= 2 or (
        census.counts["native_libs"] >= 2)
    signals.append(Signal(
        name="per_triple_fanout",
        present=fanout,
        weight=2.0,
        detail="multiple per-target artifacts (fat jar + per-triple jars/libs)",
        evidence=(triple_named[:10] or [r.name for r in lib_like[:10]]),
        confidence_if_present=HIGH,
    ))

    # (2) Injected NativeLoader class (heuristic name match until corpus-confirmed).
    loader_hits = sorted({c.name for c in census.classes
                          if _LOADER_NAME_RE.search(c.name)})
    signals.append(Signal(
        name="native_loader_class",
        present=bool(loader_hits),
        weight=1.0,
        detail="a class named like an injected NativeLoader (heuristic)",
        evidence=loader_hits[:10],
        confidence_if_present=MEDIUM,
    ))

    # (3) resources.compress: a single large compressed blob carrying the libs.
    big_blobs = [r for r in census.resources
                 if r.kind in ("blob", "other") and r.size > 256 * 1024]
    signals.append(Signal(
        name="compressed_resource_blob",
        present=bool(big_blobs),
        weight=0.5,
        detail="a large opaque resource blob (possible resources.compress payload)",
        evidence=[f"{r.name} ({r.size} B)" for r in big_blobs[:10]],
        confidence_if_present=LOW,
    ))

    # (4) Synthetic <init>/<clinit> split methods with machine-generated names.
    split = []
    for c in census.classes:
        for sig in c.synthetic_methods:
            nm = sig.split(":", 1)[0]
            if _HASHY_RE.match(nm) or _ALPHA_RE.match(nm):
                split.append(f"{c.name}.{sig}")
    signals.append(Signal(
        name="synthetic_split_methods",
        present=len(split) >= 3,
        weight=0.75,
        detail="several synthetic, machine-named methods (split <init>/<clinit>)",
        evidence=split[:10],
        confidence_if_present=LOW,
    ))

    notes = [
        "JVMTP packaging (per-triple fan-out, NativeLoader, resources.compress) is "
        "the distinctive signal; the loader class name and blob format are "
        "unconfirmed (plan.md §1.2) — NativeLoader matching is heuristic until the "
        "Phase 0 corpus confirms it.",
    ]
    return Fingerprint.from_signals("jvmtp", signals, notes)
