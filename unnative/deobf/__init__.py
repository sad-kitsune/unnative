"""Phase 4 — per-site deobfuscation, focused on MBA and hidden constants.

JVMTP's `obfuscation.constants.mba` hides int/long literals behind expressions that
are constant per site. This package recovers them: `mba.py` is the evaluation engine
(expression IR + exact wrapping arithmetic + a fold that detects opaque constants by
probing), `x86expr.py` lifts straight-line native code into that IR, and `stage.py`
walks each bound function to recover its hidden constants.

The string layer is intentionally *not* here: the sibling tool `unseed` already
recovers the caller-keyed Java string encryption these jars use. The stage says so in
its output and recommends `unseed patch`.
"""

from __future__ import annotations

from .stage import run_stage

__all__ = ["run_stage"]
