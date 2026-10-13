"""Tests for the make-like driver: work dir, stamps, and incremental skipping."""

from __future__ import annotations

import os
import unittest

from _fixtures import NATIVE_WITH_LOADER, JavaFixtureCase

from unnative.workdir import SOURCE, Pipeline, Stage, StageNotImplemented, WorkDir


class WorkDirTests(JavaFixtureCase):
    def test_sample_id_and_source_record(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        self.assertTrue(wd.sample.startswith("sample-"))
        self.assertTrue(os.path.isfile(wd.path("source.json")))

    def test_inputs_hash_tracks_source(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        h1 = wd.inputs_hash([SOURCE])
        # Rewriting the jar with different bytes changes the hash.
        with open(jar, "ab") as f:
            f.write(b"\x00trailing")
        wd2 = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        self.assertNotEqual(h1, wd2.inputs_hash([SOURCE]))

    def test_missing_input_does_not_match(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        # An artifact that does not exist must hash distinctly from one that does.
        h_missing = wd.inputs_hash(["bindings.json"])
        with open(wd.path("bindings.json"), "w") as f:
            f.write("{}")
        self.assertNotEqual(h_missing, wd.inputs_hash(["bindings.json"]))


class PipelineTests(JavaFixtureCase):
    def _mini_pipeline(self):
        def real_stage(wd: WorkDir):
            with open(wd.path("out.txt"), "w") as f:
                f.write("done")
            return {"ok": True}

        def stub_stage(wd: WorkDir):
            raise StageNotImplemented("second", "Phase 2", "not built")

        return Pipeline([
            Stage("first", "Phase 1", real_stage, inputs=[SOURCE], outputs=["out.txt"]),
            Stage("second", "Phase 2", stub_stage, inputs=["out.txt"], outputs=["x.json"]),
        ])

    def test_runs_then_caches_then_stops(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work"))
        pipe = self._mini_pipeline()

        first = pipe.run(wd)
        self.assertEqual([(r.name, r.status) for r in first],
                         [("first", "ran"), ("second", "stopped")])
        self.assertTrue(os.path.isfile(wd.path("out.txt")))

        # Second run: 'first' is up to date, so it is skipped; 'second' still stops.
        second = pipe.run(wd)
        self.assertEqual([(r.name, r.status) for r in second],
                         [("first", "cached"), ("second", "stopped")])

    def test_force_reruns(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work"))
        pipe = self._mini_pipeline()
        pipe.run(wd)
        forced = pipe.run(wd, force=True)
        self.assertEqual(forced[0].status, "ran")

    def test_only_runs_one_stage(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work"))
        pipe = self._mini_pipeline()
        res = pipe.run(wd, only="first")
        self.assertEqual([(r.name, r.status) for r in res], [("first", "ran")])


if __name__ == "__main__":
    unittest.main()
