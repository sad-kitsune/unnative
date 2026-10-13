"""Test fixtures built on the fly with `javac` + `zipfile`.

Using the real compiler gives us genuine class files (correct constant pools,
StackMapTables, modified-UTF-8) instead of hand-rolled bytes, and `zipfile` lets a
test inject arbitrary extra resources (fake per-triple native libs) to exercise the
resource classifier and fingerprints. Tests that need the toolchain skip cleanly
when it is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Make `import unnative` work when running the tests from a checkout without an
# install step.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

UNSEED_REF = os.path.join(os.path.dirname(REPO), "unseed", "unseed.py")


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise unittest.SkipTest(f"{tool} not on PATH")
    return path


def compile_java(sources: dict[str, str], tmp: str) -> str:
    """Compile `{filename: source}` into `<tmp>/classes`, return that directory."""
    require("javac")
    srcdir = os.path.join(tmp, "src")
    outdir = os.path.join(tmp, "classes")
    os.makedirs(srcdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, src in sources.items():
        p = os.path.join(srcdir, name)
        with open(p, "w") as f:
            f.write(src)
        paths.append(p)
    proc = subprocess.run(["javac", "-d", outdir, *paths],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise unittest.SkipTest(f"javac failed: {proc.stderr.strip()}")
    return outdir


def make_jar(classes_dir: str, jar_path: str, extra: dict[str, bytes] | None = None):
    """Zip every .class under `classes_dir` into `jar_path`, plus any `extra`
    resources (arcname -> bytes)."""
    with zipfile.ZipFile(jar_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\r\n\r\n")
        for root, _dirs, files in os.walk(classes_dir):
            for name in files:
                if not name.endswith(".class"):
                    continue
                full = os.path.join(root, name)
                arc = os.path.relpath(full, classes_dir)
                zf.write(full, arc)
        for arc, data in (extra or {}).items():
            zf.writestr(arc, data)
    return jar_path


# --- canned Java sources -------------------------------------------------------

# A class with a native method bound by a loadLibrary <clinit> — the shape triage's
# census must read (native method + loader call + the library name).
NATIVE_WITH_LOADER = {
    "Demo.java": """
public class Demo {
    static { System.loadLibrary("demo_jni"); }
    public static native int add(int a, int b);
    public int mul(int a, int b) { return a * b; }
    public static void main(String[] a) { System.out.println(new Demo().mul(6, 7)); }
}
""",
}

# A plain class: no native methods, no loader. Triage must call this "none".
PLAIN = {
    "Plain.java": """
public class Plain {
    public static int addmul(int a, int b) { return a * b + a - b; }
    public static void main(String[] a) { System.out.println(addmul(2, 3)); }
}
""",
}


# A class hierarchy, for method/field resolution across super + interface.
INHERIT = {
    "Shape.java": """
public abstract class Shape {
    protected int sides;
    public int getSides() { return sides; }
    public abstract double area();
}
""",
    "Square.java": """
public class Square extends Shape {
    double s;
    public Square(double s) { this.s = s; this.sides = 4; }
    public double area() { return s * s; }
    public native double nativeArea();
}
""",
}


# --- C fixtures (shared objects) ----------------------------------------------

# Self-contained (no jni.h): a JNI_OnLoad plus two exported Java_* stubs and an
# internal helper. For the ELF loader tests.
LOADER_C = """
#include <stdint.h>
typedef void JavaVM; typedef void JNIEnv;
__attribute__((visibility("default"))) int JNI_OnLoad(JavaVM* vm, void* r){ (void)vm;(void)r; return 0x00010006; }
__attribute__((visibility("default"))) int Java_Demo_add(JNIEnv* e, void* c, int a, int b){ (void)e;(void)c; return a+b; }
__attribute__((visibility("default"))) int Java_Demo_mul(JNIEnv* e, void* c, int a, int b){ (void)e;(void)c; return a*b; }
static int helper(int x){ return x+1; }
int use(int x){ return helper(x); }
"""

# Needs jni.h: a JNI_OnLoad that calls GetEnv/GetVersion/FindClass/RegisterNatives
# (with sentinel pointers, so the binding capture is easy to assert), plus a real
# exported Java_Demo_add. For the emulator tests.
REG_C = """
#include <jni.h>
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = 0;
    (*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6);
    jint v = (*env)->GetVersion(env);
    jclass cls = (*env)->FindClass(env, "Demo");
    static const JNINativeMethod m[2] = {
        {"add", "(II)I", (void*)0x1111},
        {"mul", "(II)I", (void*)0x2222},
    };
    (*env)->RegisterNatives(env, cls, m, 2);
    (void)v;
    return JNI_VERSION_1_6;
}
JNIEXPORT jint JNICALL Java_Demo_add(JNIEnv* e, jclass c, jint a, jint b){ (void)e;(void)c; return a+b; }
"""

# Needs jni.h: `add` is BOTH an exported Java_* symbol and RegisterNatives-bound (to
# a real function), so the binder should flag it a decoy; `mul` is register-only.
DECOY_C = """
#include <jni.h>
static jint impl_add(JNIEnv* e, jclass c, jint a, jint b){ (void)e;(void)c; return a+b; }
static jint impl_mul(JNIEnv* e, jclass c, jint a, jint b){ (void)e;(void)c; return a*b; }
JNIEXPORT jint JNICALL Java_Demo_add(JNIEnv* e, jclass c, jint a, jint b){ return impl_add(e,c,a,b); }
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = 0;
    (*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6);
    jclass cls = (*env)->FindClass(env, "Demo");
    JNINativeMethod m[2] = {
        {"add", "(II)I", (void*)impl_add},
        {"mul", "(II)I", (void*)impl_mul},
    };
    (*env)->RegisterNatives(env, cls, m, 2);
    return JNI_VERSION_1_6;
}
"""

# Needs jni.h: a method that calls memcpy via the PLT (-fno-builtin prevents gcc
# inlining it), exercising the emulator's extern/C-runtime shim.
MEMCPY_C = """
#include <jni.h>
#include <string.h>
JNIEXPORT jint JNICALL Java_Demo_cp(JNIEnv* e, jclass c, jint x) {
    volatile unsigned long n = 8;
    unsigned char src[8] = {1,2,3,4,5,6,7,8};
    unsigned char dst[8] = {0};
    memcpy(dst, src, (size_t)n);
    return dst[3] + x;      /* 4 + x */
}
"""

# Needs jni.h: a method body that makes a known sequence of JNI calls, for tracing.
TRACE_C = """
#include <jni.h>
JNIEXPORT jint JNICALL Java_Demo_compute(JNIEnv* env, jclass cls, jint x) {
    jint v = (*env)->GetVersion(env);
    jclass sc = (*env)->FindClass(env, "java/lang/String");
    jstring s = (*env)->NewStringUTF(env, "hello");
    (void)sc; (void)s;
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    return x + v - v + 1;
}
"""


def jdk_include() -> tuple[str, str] | None:
    cands = []
    jh = os.environ.get("JAVA_HOME")
    if jh:
        cands.append(os.path.join(jh, "include"))
    cands.append("/usr/lib/jvm/default/include")
    jc = shutil.which("javac")
    if jc:
        base = os.path.dirname(os.path.dirname(os.path.realpath(jc)))
        cands.append(os.path.join(base, "include"))
    for inc in cands:
        if os.path.isfile(os.path.join(inc, "jni.h")):
            return inc, os.path.join(inc, "linux")
    return None


def compile_shared(c_source: str, tmp: str, name: str = "lib.so", *,
                   jni: bool = False, flags=()) -> str:
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise unittest.SkipTest("no C compiler (gcc/clang) on PATH")
    src = os.path.join(tmp, name + ".c")
    with open(src, "w") as f:
        f.write(c_source)
    out = os.path.join(tmp, name)
    cmd = [cc, "-shared", "-fPIC", "-O2", *flags]
    if jni:
        inc = jdk_include()
        if inc is None:
            raise unittest.SkipTest("jni.h not found")
        cmd += ["-I", inc[0], "-I", inc[1]]
    cmd += ["-o", out, src]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise unittest.SkipTest(f"{os.path.basename(cc)} failed: {proc.stderr.strip()}")
    return out


def have_unicorn() -> bool:
    try:
        import unicorn  # noqa: F401
        return True
    except ImportError:
        return False


def tmpdir() -> str:
    return tempfile.mkdtemp(prefix="unnative-test-")


class JavaFixtureCase(unittest.TestCase):
    """Base class that provides a per-test temp dir and auto-cleans it."""

    def setUp(self):
        self.tmp = tmpdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def build_jar(self, sources: dict[str, str], name: str = "sample.jar",
                  extra: dict[str, bytes] | None = None) -> str:
        classes = compile_java(sources, self.tmp)
        return make_jar(classes, os.path.join(self.tmp, name), extra=extra)
