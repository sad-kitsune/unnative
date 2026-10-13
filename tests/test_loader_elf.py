"""Tests for the ELF loader, against a real shared object built with the C compiler."""

from __future__ import annotations

import unittest

from _fixtures import LOADER_C, JavaFixtureCase, compile_shared

from unnative.errors import UnnativeError
from unnative.native.loader_elf import ElfImage


class ElfLoaderTests(JavaFixtureCase):
    def _image(self) -> ElfImage:
        so = compile_shared(LOADER_C, self.tmp, "libdemo.so")
        return ElfImage.from_path(so)

    def test_basic_shape(self):
        e = self._image()
        self.assertEqual(e.arch, "x86_64")
        self.assertTrue(e.segments)
        self.assertIsNotNone(e.jni_onload())
        names = [s.name for s in e.java_exports()]
        self.assertEqual(names, ["Java_Demo_add", "Java_Demo_mul"])

    def test_symbol_lookup_and_vaddr(self):
        e = self._image()
        add = e.symbol("Java_Demo_add")
        self.assertIsNotNone(add)
        self.assertTrue(add.defined and add.is_func)
        # Its vaddr translates to a real file offset inside a PT_LOAD segment.
        self.assertIsNotNone(e.vaddr_to_off(add.value))

    def test_relocations_present(self):
        # A PIC shared object has RELATIVE relocations for its GOT/data pointers.
        e = self._image()
        self.assertTrue(e.relocations)

    def test_dynamic_symbol_path(self):
        # Force the stripped-library code path (no section headers) and confirm the
        # dynamic-segment reader + GNU-hash symbol count recover the same exports.
        e = self._image()
        e._sh = []
        syms = e._symbols_from_dynamic()
        self.assertIsNotNone(syms)
        names = {s.name for s in syms if s.defined}
        self.assertIn("Java_Demo_add", names)
        self.assertIn("Java_Demo_mul", names)
        self.assertIn("JNI_OnLoad", names)

    def test_rejects_non_elf(self):
        with self.assertRaises(UnnativeError):
            ElfImage(b"MZ not really a PE but definitely not ELF")


if __name__ == "__main__":
    unittest.main()
