"""The work directory and the make-like stage cache.

Every stage reads and writes plain files under `work/<sample>/`, so a run can be
re-done, cached and inspected one stage at a time (plan.md §2). Staleness is keyed
on *content*, not mtime: each stage drops a `<stage>.stamp` recording a hash of the
inputs it consumed, and is considered up to date when its outputs all exist and
that hash still matches. Change the jar, or an upstream artifact, and exactly the
stages downstream of the change re-run.

This module is the generic machinery (`WorkDir`, `Stage`, `Pipeline`). The concrete
list of stages is wired up in `cli.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from .errors import UnnativeError

# The sentinel input key meaning "the sample jar itself" (the first stage's input).
SOURCE = "source"

STAMP_SUFFIX = ".stamp"


class StageNotImplemented(UnnativeError):
    """A stage that is scaffolded but whose phase has not been built yet.

    Carries the plan.md phase number so the driver can tell the user which
    milestone implements it, instead of failing opaquely.
    """

    def __init__(self, stage: str, phase: str, note: str = ""):
        self.stage = stage
        self.phase = phase
        self.note = note
        msg = f"stage {stage!r} is not implemented yet (plan.md {phase})"
        if note:
            msg += f": {note}"
        super().__init__(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _short(digest: str, n: int = 12) -> str:
    return digest[:n]


class WorkDir:
    """`work/<sample>/` for one input jar, plus artifact-path and stamp helpers.

    The sample id is `<basename>-<short source hash>` so it is both readable and
    unique to the jar's contents: analysing a changed jar lands in a fresh dir
    rather than silently reusing stale artifacts.
    """

    def __init__(self, source: str, root: str | None = None):
        self.source = os.path.abspath(source)
        if not os.path.exists(self.source):
            raise UnnativeError(f"{self.source}: no such file")
        self.root = os.path.abspath(root) if root else os.path.join(os.getcwd(), "work")
        self.source_hash = sha256_file(self.source)
        base = os.path.splitext(os.path.basename(self.source))[0]
        self.sample = f"{base}-{_short(self.source_hash)}"
        self.dir = os.path.join(self.root, self.sample)

    # -- paths --
    def ensure(self) -> "WorkDir":
        os.makedirs(self.dir, exist_ok=True)
        self._write_source_record()
        return self

    def path(self, *parts: str) -> str:
        """Absolute path to an artifact inside this work dir."""
        return os.path.join(self.dir, *parts)

    def _write_source_record(self):
        rec = {"source": self.source, "sha256": self.source_hash, "sample": self.sample}
        with open(self.path("source.json"), "w") as f:
            json.dump(rec, f, indent=2)

    # -- stamps / staleness --
    def _stamp_path(self, stage: str) -> str:
        return self.path(stage + STAMP_SUFFIX)

    def read_stamp(self, stage: str) -> dict | None:
        p = self._stamp_path(stage)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def write_stamp(self, stage: str, inputs_hash: str, meta: dict | None = None):
        rec = {"stage": stage, "inputs_hash": inputs_hash, "time": time.time()}
        if meta:
            rec["meta"] = meta
        with open(self._stamp_path(stage), "w") as f:
            json.dump(rec, f, indent=2)

    def inputs_hash(self, input_keys: list[str]) -> str:
        """A content hash over a stage's declared inputs.

        `SOURCE` resolves to the jar's hash; any other key is an artifact path
        inside this work dir. A missing artifact hashes as the literal token
        ``missing:<key>`` so a stamp can never falsely match when an upstream
        output has been deleted.
        """
        h = hashlib.sha256()
        for key in input_keys:
            h.update(key.encode())
            h.update(b"\x00")
            if key == SOURCE:
                h.update(self.source_hash.encode())
            else:
                ap = self.path(key)
                if os.path.isfile(ap):
                    h.update(sha256_file(ap).encode())
                elif os.path.isdir(ap):
                    h.update(self._dir_hash(ap).encode())
                else:
                    h.update(f"missing:{key}".encode())
            h.update(b"\x00")
        return h.hexdigest()

    @staticmethod
    def _dir_hash(root: str) -> str:
        h = hashlib.sha256()
        for dirpath, _dirs, files in sorted(os.walk(root)):
            for name in sorted(files):
                fp = os.path.join(dirpath, name)
                rel = os.path.relpath(fp, root)
                h.update(rel.encode())
                h.update(b"\x00")
                h.update(sha256_file(fp).encode())
        return h.hexdigest()

    def outputs_exist(self, outputs: list[str]) -> bool:
        return all(os.path.exists(self.path(o)) for o in outputs)


@dataclass
class Stage:
    """One pipeline stage: a name, the plan phase that owns it, the function that
    runs it, and the artifact keys it reads/writes (for the content-keyed cache).

    `fn(workdir) -> dict | None` does the work and returns optional stamp metadata.
    A stage whose `fn` raises `StageNotImplemented` is treated as a clean stop, not
    a crash.
    """

    name: str
    phase: str
    fn: Callable[[WorkDir], dict | None]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    summary: str = ""

    def is_up_to_date(self, wd: WorkDir) -> bool:
        if not self.outputs or not wd.outputs_exist(self.outputs):
            return False
        stamp = wd.read_stamp(self.name)
        if not stamp:
            return False
        return stamp.get("inputs_hash") == wd.inputs_hash(self.inputs)


@dataclass
class StageResult:
    name: str
    status: str                 # "ran" | "cached" | "skipped" | "stopped"
    phase: str = ""
    note: str = ""


class Pipeline:
    """Runs an ordered list of stages with make-like skipping."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self.by_name = {s.name: s for s in stages}

    def run(self, wd: WorkDir, *, only: str | None = None, force: bool = False,
            stop_on_unimplemented: bool = True) -> list[StageResult]:
        wd.ensure()
        results: list[StageResult] = []
        for stage in self.stages:
            if only and stage.name != only:
                # When targeting one stage, still honour that its upstream outputs
                # must exist; we simply don't execute the others here.
                continue
            if not force and stage.is_up_to_date(wd):
                results.append(StageResult(stage.name, "cached", stage.phase))
                continue
            try:
                meta = stage.fn(wd)
            except StageNotImplemented as ex:
                results.append(StageResult(stage.name, "stopped", ex.phase, ex.note))
                if stop_on_unimplemented:
                    break
                continue
            wd.write_stamp(stage.name, wd.inputs_hash(stage.inputs), meta)
            results.append(StageResult(stage.name, "ran", stage.phase))
            if only:
                break
        return results
