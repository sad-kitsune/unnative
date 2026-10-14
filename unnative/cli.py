"""`unnative` command-line driver.

Subcommands mirror the pipeline stages of plan.md §2:

    unnative run     <jar>      run every stage, skipping any whose outputs are
                                up to date, stopping cleanly at the first stage
                                that is not implemented yet
    unnative triage  <jar>      Phase 1 — protector fingerprint + method census
    unnative unpack  <jar>      Phase 2 — extract the native libraries
    unnative bind    <jar>      Phase 3 — native method <-> function-address map
    unnative deobf   <jar>      Phase 4 — per-site string/MBA/call/flow recovery
    unnative trace   <jar>      Phase 5 — dynamic JNI-call trace (emu / patched JDK)
    unnative lift    <jar>      Phase 6 — lift native idioms to JVM-level IR
    unnative emit    <jar>      Phase 7 — emit bytecode, rebuild the jar
    unnative verify  <jar>      Phase 8 — differential verification

triage, unpack, bind, trace and lift are implemented; deobf, emit and verify are
honest stubs that report the phase that will build them (see plan.md §3).
"""

from __future__ import annotations

import argparse
import os
import sys

from .bind import run_stage as bind_stage
from .emit import run_stage as emit_stage
from .errors import UnnativeError
from .lift import run_stage as lift_stage
from .trace import run_stage as trace_stage
from .verify import run_stage as verify_stage
from .triage import run_stage as triage_stage
from .unpack import run_stage as unpack_stage
from .workdir import SOURCE, Pipeline, Stage, StageNotImplemented, WorkDir


def _stub(name: str, phase: str, note: str):
    """A not-yet-implemented stage: raises StageNotImplemented so the driver can
    stop cleanly and tell the user which plan.md phase owns the work."""
    def fn(_wd: WorkDir):
        raise StageNotImplemented(name, phase, note)
    return fn


def build_pipeline() -> Pipeline:
    """The stage graph of plan.md §2. Inputs/outputs are declared even for stubbed
    stages so the make-like cache is correct the moment a stage is implemented."""
    # Execution order runs M1 (unpack -> bind -> trace) before the M2 deobf/lift
    # stages; deobf and trace are parallel siblings in the plan's architecture, so
    # ordering trace first simply means `run` reaches the behaviour trace before it
    # stops at the first unbuilt stage.
    stages = [
        Stage("triage", "Phase 1", triage_stage,
              inputs=[SOURCE], outputs=["report.json"],
              summary="protector fingerprint + method census"),
        Stage("unpack", "Phase 2", unpack_stage,
              inputs=[SOURCE], outputs=["libs"],
              summary="extract native libraries per target triple"),
        Stage("bind", "Phase 3", bind_stage,
              inputs=["libs"], outputs=["bindings.json"],
              summary="map each native method to one function address"),
        Stage("trace", "Phase 5", trace_stage,
              inputs=["bindings.json", "libs"], outputs=["trace.json"],
              summary="dynamic JNI-call trace (behaviour per method)"),
        # lift consumes the behaviour trace directly: the first slice is a linear
        # idiom lifter that does not yet depend on the per-site deobf pass, so it
        # runs here, before the still-stubbed deobf, and `run` reaches it.
        Stage("lift", "Phase 6", lift_stage,
              inputs=["trace.json", "bindings.json", "libs"], outputs=["jir"],
              summary="lift native idioms to JVM-level IR"),
        # emit and verify consume lift/emit outputs directly (no dependence on the
        # per-site deobf pass), so they run here before the still-stubbed deobf and
        # `run` reaches them.
        Stage("emit", "Phase 7", emit_stage,
              inputs=["jir"], outputs=["rebuilt.jar"],
              summary="emit bytecode, rebuild the jar"),
        Stage("verify", "Phase 8", verify_stage,
              inputs=["rebuilt.jar", "emit.json", "jir", "trace.json"],
              outputs=["verify.json"],
              summary="differential verification"),
        Stage("deobf", "Phase 4",
              _stub("deobf", "Phase 4", "needs per-site concrete evaluation (Phase 5a)"),
              inputs=["bindings.json"], outputs=["deobf.json"],
              summary="recover strings / MBA constants / call targets / flatten"),
    ]
    return Pipeline(stages)


_STATUS_TAG = {"ran": "[ran ]", "cached": "[skip]", "stopped": "[stop]", "skipped": "[----]"}


def _print_results(results, wd: WorkDir):
    print(f"\nwork dir: {wd.dir}", file=sys.stderr)
    for r in results:
        tag = _STATUS_TAG.get(r.status, f"[{r.status}]")
        line = f"  {tag} {r.name:<8} {r.phase}"
        if r.status == "stopped":
            line += " — not implemented yet" + (f": {r.note}" if r.note else "")
        print(line, file=sys.stderr)


def _apply_provided_libs(wd: WorkDir, args) -> bool:
    """Attach user-supplied libraries to the work dir; return whether a forced
    re-run is needed so `unpack` picks them up."""
    libs = getattr(args, "lib", None)
    if not libs:
        return False
    for p in libs:
        if not os.path.isfile(p):
            raise UnnativeError(f"--lib: no such file: {p}")
    wd.provided_libs = [os.path.abspath(p) for p in libs]
    return True


def _cmd_run(args) -> int:
    wd = WorkDir(args.jar, root=args.work_dir)
    forced = _apply_provided_libs(wd, args)
    pipe = build_pipeline()
    results = pipe.run(wd, force=args.force or forced)
    _print_results(results, wd)
    return 0


def _cmd_stage(args) -> int:
    wd = WorkDir(args.jar, root=args.work_dir)
    forced = _apply_provided_libs(wd, args)
    pipe = build_pipeline()
    if args.stage not in pipe.by_name:
        raise UnnativeError(f"unknown stage {args.stage!r}")
    results = pipe.run(wd, only=args.stage, force=args.force or forced)
    _print_results(results, wd)
    # A stubbed stage reports "stopped"; make that a non-zero, informative exit.
    for r in results:
        if r.name == args.stage and r.status == "stopped":
            return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they are accepted *after* the
    # subcommand, e.g. `unnative bind sample.jar --lib linux-x86_64.so`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("jar", help="protected .jar (or a directory of .class files)")
    common.add_argument("--work-dir", default=None,
                        help="root for per-sample work directories (default ./work)")
    common.add_argument("--force", action="store_true",
                        help="re-run stages even if their outputs are up to date")
    common.add_argument("--lib", action="append", default=None, metavar="PATH",
                        help="use an externally-extracted native library instead of "
                             "unpacking it from the jar (repeatable; e.g. one per "
                             "os/arch). Implies re-running unpack.")

    p = argparse.ArgumentParser(
        prog="unnative",
        description="Recover readable Java from JNIC/JVMTP native-translated jars.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", parents=[common],
                         help="run the pipeline as far as it goes")
    run.set_defaults(func=_cmd_run)

    # One subcommand per stage; they share an implementation that runs just that
    # stage. Help text is pulled from the pipeline definition so it can't drift.
    for stage in build_pipeline().stages:
        sp = sub.add_parser(stage.name, parents=[common],
                            help=f"{stage.phase}: {stage.summary}")
        sp.set_defaults(func=_cmd_stage, stage=stage.name)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UnnativeError as ex:
        print(f"unnative: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
