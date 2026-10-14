"""Phase 7 — emit bytecode for lifted methods and rebuild the jar.

The stackified JIR (Phase 6) gives, for the methods it could lower, a list of
operand-stack bytecode ops. This stage encodes those into a real Code attribute
with a pure-Python class-file writer (`writer.py`), fills the `native` stub, and
re-packs the jar, copying everything it does not touch. Methods that did not
stackify are left as native stubs — partial recovery, not all-or-nothing.
"""

from __future__ import annotations

from .stage import run_stage

__all__ = ["run_stage"]
