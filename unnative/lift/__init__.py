"""Phase 6 — lift native idioms to a JVM-level IR (JIR).

The behaviour trace (Phase 5) gives, per method, the ordered stream of JNI calls
the translated native code makes. Both JNIC and JVMTP lower bytecode roughly 1:1
onto JNI, so that stream *is* the method's interaction skeleton: a `GetMethodID`
followed by a `Call*Method` is an `invoke*`, `AllocObject`+`<init>` is `new`, and
so on (plan.md §6). This stage recognises those idioms and emits JIR.

This first slice is a **linear idiom lifter**: it walks the recorded call stream
in order and maps each consuming JNI call to a JIR op, resolving call/field targets
by following the handles that `FindClass`/`Get*ID` produced. It does not yet rebuild
control flow (`cfg.py`) or stack order (`stackify.py`) — those are the next slices —
and it cannot see pure computation that makes no JNI call. Both limits are reported
per method rather than hidden, so coverage is honest.
"""

from __future__ import annotations

from .stage import run_stage

__all__ = ["run_stage"]
