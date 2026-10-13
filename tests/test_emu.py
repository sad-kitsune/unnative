"""End-to-end emulator tests. Skipped unless Unicorn and a C compiler are present."""

from __future__ import annotations

import os
import unittest

from _fixtures import (
    MEMCPY_C,
    REG_C,
    TRACE_C,
    JavaFixtureCase,
    compile_shared,
    have_unicorn,
)

_COMPUTE_JAR = {"Demo.java": "public class Demo { public static native int compute(int x); }"}


@unittest.skipUnless(have_unicorn(), "unicorn not installed")
class EmuTests(JavaFixtureCase):
    def _setup(self):
        so = compile_shared(REG_C, self.tmp, "libreg.so", jni=True)
        jar = self.build_jar({
            "Demo.java": """
public class Demo {
    public static native int add(int a, int b);
    public static native int mul(int a, int b);
}
""",
        }, name="demo.jar")
        from unnative.native.emu import UnicornEmu
        return UnicornEmu.for_jar_and_lib(jar, so)

    def test_jni_onload_captures_bindings(self):
        emu = self._setup()
        rv = emu.call_jni_onload()
        self.assertEqual(rv & 0xFFFFFFFF, 0x00010006)       # JNI_VERSION_1_6
        got = {(b.owner, b.name, b.signature, b.fn_ptr) for b in emu.env.bindings}
        self.assertEqual(got, {
            ("Demo", "add", "(II)I", 0x1111),
            ("Demo", "mul", "(II)I", 0x2222),
        })
        fns = [ev.fn for ev in emu.env.trace]
        for expected in ("GetEnv", "FindClass", "RegisterNatives"):
            self.assertIn(expected, fns)

    def test_call_exported_function(self):
        # The classic path: call an exported Java_* directly and read RAX.
        emu = self._setup()
        add = emu.elf.symbol("Java_Demo_add")
        self.assertIsNotNone(add)
        result = emu.call(emu.guest_addr(add.value),
                          [emu.layout.env_struct, 0, 20, 22])
        self.assertEqual(result & 0xFFFFFFFF, 42)


@unittest.skipUnless(have_unicorn(), "unicorn not installed")
class TraceTests(JavaFixtureCase):
    def _trace_emu(self):
        so = compile_shared(TRACE_C, self.tmp, "libtrace.so", jni=True)
        jar = self.build_jar(_COMPUTE_JAR, name="demo.jar")
        from unnative.native.emu import UnicornEmu
        emu = UnicornEmu.for_jar_and_lib(jar, so)
        vaddr = emu.elf.symbol("Java_Demo_compute").value
        return emu, vaddr

    def test_trace_records_jni_stream(self):
        emu, vaddr = self._trace_emu()
        inv = emu.trace_method(vaddr, "Demo", "compute", "(I)I", [41])
        self.assertIsNone(inv.fault)
        self.assertEqual(inv.return_value & 0xFFFFFFFF, 42)
        self.assertEqual([e.fn for e in inv.trace],
                         ["GetVersion", "FindClass", "NewStringUTF", "ExceptionCheck"])

    def test_snapshot_restore_makes_calls_independent(self):
        emu, vaddr = self._trace_emu()
        a = emu.trace_method(vaddr, "Demo", "compute", "(I)I", [1])
        b = emu.trace_method(vaddr, "Demo", "compute", "(I)I", [100])
        self.assertEqual(a.return_value & 0xFFFFFFFF, 2)
        self.assertEqual(b.return_value & 0xFFFFFFFF, 101)
        self.assertEqual(len(a.trace), len(b.trace))     # state rolled back each time


@unittest.skipUnless(have_unicorn(), "unicorn not installed")
class ExternShimTests(JavaFixtureCase):
    def test_memcpy_via_plt_is_serviced(self):
        # Without the extern shim this method's `call memcpy@plt` jumps through a
        # NULL GOT entry and faults; with it, memcpy runs and the method returns.
        so = compile_shared(MEMCPY_C, self.tmp, "libcp.so", jni=True,
                            flags=("-fno-builtin",))
        jar = self.build_jar(
            {"Demo.java": "public class Demo { public static native int cp(int x); }"},
            name="demo.jar")
        from unnative.native.emu import UnicornEmu
        emu = UnicornEmu.for_jar_and_lib(jar, so)
        self.assertIn("memcpy", emu.unresolved_imports)
        vaddr = emu.elf.symbol("Java_Demo_cp").value
        inv = emu.trace_method(vaddr, "Demo", "cp", "(I)I", [38])
        self.assertIsNone(inv.fault)
        self.assertEqual(inv.return_value & 0xFFFFFFFF, 42)   # dst[3]==4, +38
        self.assertIn("memcpy", emu.extern_calls)


if __name__ == "__main__":
    unittest.main()
