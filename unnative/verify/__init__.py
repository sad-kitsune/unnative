"""Phase 8 — differential verification of the rebuilt jar.

Checks every method the emit stage filled against the behaviour the lift stage
recovered, three independent ways: the JIR (the expectation), the bytecode re-decoded
from `rebuilt.jar` (catches emit bugs), and the emulated original's JNI trace (ground
truth). When a JDK is present, each recovered body is also run on a real JVM. The
honest limit: the original native method can't be run in place — initialising its
class triggers the protector's loader — so the emulated trace is its only independent
witness. See `differential.py`.
"""

from __future__ import annotations

from .stage import run_stage

__all__ = ["run_stage"]
