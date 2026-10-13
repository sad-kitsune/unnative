"""Tests for the Phase 3 binding map: both resolution paths, decoys, unresolved."""

from __future__ import annotations

import unittest

from _fixtures import (
    DECOY_C,
    LOADER_C,
    JavaFixtureCase,
    compile_shared,
    have_unicorn,
)

from unnative.native.binder import SOURCE_REGISTER, SOURCE_SYMBOL, bind
from unnative.native.jvm_model import JvmModel
from unnative.native.loader_elf import ElfImage

_DEMO_JAR_SRC = {
    "Demo.java": """
public class Demo {
    public static native int add(int a, int b);
    public static native int mul(int a, int b);
}
""",
}


class StaticPathTests(JavaFixtureCase):
    def test_java_symbol_resolution(self):
        so = compile_shared(LOADER_C, self.tmp, "libdemo.so")
        jar = self.build_jar(_DEMO_JAR_SRC, name="demo.jar")
        result = bind(ElfImage.from_path(so), JvmModel.from_source(jar))
        self.assertEqual(result.to_json()["counts"]["resolved"], 2)
        self.assertEqual(result.unresolved, [])
        for b in result.bindings:
            self.assertEqual(b.sources, [SOURCE_SYMBOL])
            self.assertTrue(b.symbol.startswith("Java_Demo_"))

    def test_unresolved_when_absent(self):
        # Model has a native method the library does not implement.
        so = compile_shared(LOADER_C, self.tmp, "libdemo.so")
        jar = self.build_jar({
            "Demo.java": """
public class Demo {
    public static native int add(int a, int b);
    public static native int notThere(int a);
}
""",
        }, name="demo.jar")
        result = bind(ElfImage.from_path(so), JvmModel.from_source(jar))
        resolved = {b.name for b in result.bindings}
        self.assertIn("add", resolved)
        self.assertEqual([u["name"] for u in result.unresolved], ["notThere"])


@unittest.skipUnless(have_unicorn(), "unicorn not installed")
class DecoyTests(JavaFixtureCase):
    def test_decoy_and_register_natives(self):
        so = compile_shared(DECOY_C, self.tmp, "libdecoy.so", jni=True)
        jar = self.build_jar(_DEMO_JAR_SRC, name="demo.jar")
        model = JvmModel.from_source(jar)
        from unnative.native.emu import UnicornEmu
        emu = UnicornEmu(ElfImage.from_path(so), model)
        emu.call_jni_onload()
        rn = [(b.owner, b.name, b.signature, b.fn_ptr - emu.load_bias)
              for b in emu.env.bindings]
        result = bind(emu.elf, model, register_natives=rn)

        by_name = {b.name: b for b in result.bindings}
        # `add` is exported AND re-bound -> decoy, with both sources.
        self.assertTrue(by_name["add"].is_decoy)
        self.assertIn(SOURCE_SYMBOL, by_name["add"].sources)
        self.assertIn(SOURCE_REGISTER, by_name["add"].sources)
        # `mul` is register-only.
        self.assertEqual(by_name["mul"].sources, [SOURCE_REGISTER])
        self.assertEqual(result.to_json()["counts"]["decoys"], 1)


if __name__ == "__main__":
    unittest.main()
