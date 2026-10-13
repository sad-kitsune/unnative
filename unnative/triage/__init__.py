"""Phase 1 — triage.

Fingerprint the protector (JNIC / JVMTP / unknown) and take a method census:
which methods are `native`, which classes carry a native-loader `<clinit>`, and
what the jar's resources look like. Produces `report.json` plus a human summary.

Public API:
    triage(source) -> dict           analyse a jar/dir, return the report
    run_stage(workdir) -> dict       the pipeline-stage entry point (writes files)
"""

from .report import run_stage, triage

__all__ = ["triage", "run_stage"]
