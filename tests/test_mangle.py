"""Tests for JNI name mangling, cross-checked against javac -h."""

from __future__ import annotations

import glob
import os
import re
import subprocess
import unittest

from _fixtures import JavaFixtureCase, compile_java, require

from unnative.classfile import ACC_NATIVE, ClassSource, load_all
from unnative.native.mangle import candidate_symbols, long_symbol, short_symbol


class DirectMangleTests(unittest.TestCase):
    def test_spec_cases(self):
        self.assertEqual(short_symbol("Demo", "add"), "Java_Demo_add")
        self.assertEqual(short_symbol("com/x/Foo", "run"), "Java_com_x_Foo_run")
        self.assertEqual(short_symbol("Demo", "weird_name"), "Java_Demo_weird_1name")
        self.assertEqual(long_symbol("Demo", "f", "(I)I"), "Java_Demo_f__I")
        self.assertEqual(long_symbol("Demo", "f", "(II)I"), "Java_Demo_f__II")
        self.assertEqual(long_symbol("Demo", "arr", "([ILjava/lang/String;)J"),
                         "Java_Demo_arr___3ILjava_lang_String_2")


NATIVES_SRC = {
    "Demo.java": """
public class Demo {
    public static native int add(int a, int b);
    public native String greet(String who);
    public native int f(int a);
    public native int f(int a, int b);
    public native int weird_name(int a);
    public native long arr(int[] a, java.lang.String s);
}
""",
    "Foo.java": """
package com.x;
public class Foo { public native void run(Object o); }
""",
}


class JavacOracleTests(JavaFixtureCase):
    """Compile native methods, ask `javac -h` for the canonical symbol names, and
    confirm our mangler is both sound (every emitted name is one we produce) and
    complete (one symbol per native method)."""

    def test_matches_javac_h(self):
        require("javac")
        classes = compile_java(NATIVES_SRC, self.tmp)
        hdr = os.path.join(self.tmp, "hdr")
        os.makedirs(hdr, exist_ok=True)
        # Recompile with -h to emit the headers. compile_java writes each source flat
        # under src/<key>, so reference those exact paths.
        srcs = [os.path.join(self.tmp, "src", key) for key in NATIVES_SRC]
        proc = subprocess.run(["javac", "-h", hdr, "-d", classes, *srcs],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise unittest.SkipTest(f"javac -h failed: {proc.stderr.strip()}")

        emitted = set()
        for h in glob.glob(os.path.join(hdr, "*.h")):
            with open(h) as f:
                emitted |= set(re.findall(r"\bJava_[A-Za-z0-9_]+", f.read()))
        self.assertTrue(emitted)

        # Enumerate native methods from the compiled classes and build our candidates.
        with ClassSource(classes) as src:
            cfs, _ = load_all(src, quiet=True)
        native = []
        candidates = set()
        for cf in cfs.values():
            for m in cf.methods:
                if m["access"] & ACC_NATIVE:
                    native.append((cf.this, m["name"], m["desc"]))
                    candidates |= set(candidate_symbols(cf.this, m["name"], m["desc"]))

        # Soundness: every symbol javac emitted is one our mangler can produce.
        self.assertTrue(emitted <= candidates,
                        f"javac emitted symbols we don't produce: {emitted - candidates}")
        # Completeness: javac emits exactly one symbol per native method.
        self.assertEqual(len(emitted), len(native))


if __name__ == "__main__":
    unittest.main()
