"""Tests for Phase 1 triage: census, fingerprints, and the verdict."""

from __future__ import annotations

import os
import unittest

from _fixtures import NATIVE_WITH_LOADER, PLAIN, JavaFixtureCase

from unnative.triage import triage
from unnative.triage.census import take_census


class CensusTests(JavaFixtureCase):
    def test_native_method_and_loader(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        census = take_census(jar)
        self.assertEqual(census.counts["native_methods"], 1)
        nm = census.native_methods[0]
        self.assertEqual((nm.name, nm.desc, nm.static), ("add", "(II)I", True))
        # mul is a normal method and must NOT be counted native.
        self.assertEqual(census.counts["classes_with_native"], 1)
        # The loadLibrary argument is recovered from the <clinit>.
        load = [lc for c in census.classes for lc in c.load_calls]
        self.assertEqual(len(load), 1)
        self.assertEqual(load[0].target, "System.loadLibrary")
        self.assertEqual(load[0].lib, "demo_jni")

    def test_plain_jar_has_no_natives(self):
        jar = self.build_jar(PLAIN)
        census = take_census(jar)
        self.assertEqual(census.counts["native_methods"], 0)
        self.assertEqual(census.counts["loader_classes"], 0)

    def test_high_entropy_resource_is_blob(self):
        # A large, extension-less, high-entropy resource (like a protector's packed
        # native library) is flagged as an opaque blob; the same random bytes under
        # a media/text extension are not (already-compressed formats are expected).
        jar = self.build_jar(PLAIN, extra={
            "payload": os.urandom(8192),            # -> blob
            "art.png": os.urandom(8192),            # known media ext -> not a blob
            "readme": b"A" * 8192,                  # low entropy -> not a blob
        })
        census = take_census(jar)
        kinds = {r.name: r.kind for r in census.resources}
        self.assertEqual(kinds["payload"], "blob")
        self.assertNotEqual(kinds["art.png"], "blob")
        self.assertNotEqual(kinds["readme"], "blob")
        self.assertGreaterEqual(census.counts["opaque_blobs"], 1)


class VerdictTests(JavaFixtureCase):
    def test_plain_jar_is_none(self):
        jar = self.build_jar(PLAIN)
        report = triage(jar)
        self.assertEqual(report["verdict"]["protector"], "none")
        self.assertEqual(report["verdict"]["confidence"], "high")
        self.assertIsNone(report["verdict"]["version"])

    def test_native_without_distinctive_signal_is_unknown(self):
        # A hand-made native + loadLibrary jar is genuinely NOT identifiable as a
        # specific protector; triage must say so rather than guess "jnic".
        jar = self.build_jar(NATIVE_WITH_LOADER)
        report = triage(jar)
        self.assertEqual(report["verdict"]["protector"], "unknown")
        self.assertEqual(report["verdict"]["confidence"], "low")

    def test_per_triple_fanout_is_jvmtp(self):
        # Inject fake per-triple native libraries (real ELF/PE/Mach-O magic, under
        # triple-named paths). This is a synthetic *shape* test — it exercises the
        # resource classifier + jvmtp fingerprint, not a real JVMTP sample.
        elf = b"\x7fELF" + b"\x00" * 60
        pe = b"MZ" + b"\x00" * 62
        macho = b"\xcf\xfa\xed\xfe" + b"\x00" * 60
        extra = {
            "natives/linux-x86_64/libdemo.so": elf,
            "natives/windows-x86_64/demo.dll": pe,
            "natives/macos-aarch64/libdemo.dylib": macho,
        }
        jar = self.build_jar(NATIVE_WITH_LOADER, extra=extra)
        report = triage(jar)
        self.assertEqual(report["verdict"]["protector"], "jvmtp")
        self.assertEqual(report["verdict"]["confidence"], "high")
        # The native libs are classified and counted.
        self.assertGreaterEqual(report["census"]["counts"]["native_libs"], 3)
        jvmtp = report["fingerprints"]["jvmtp"]
        self.assertTrue(any(s["name"] == "per_triple_fanout" for s in jvmtp["signals"]))

    def test_version_is_never_guessed(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        report = triage(jar)
        self.assertIsNone(report["verdict"]["version"])
        self.assertIn("corpus", report["verdict"]["version_note"])


if __name__ == "__main__":
    unittest.main()
