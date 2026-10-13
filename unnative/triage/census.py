"""Method census — the facts triage can establish from the class files alone.

No protector knowledge here: this walks the jar's own bytecode and records what is
objectively true (which methods are `native`, which `<clinit>`s reach a native
loader, what the resources look like). The fingerprints in `jnic.py`/`jvmtp.py`
read this census; keeping the two apart means the census stays trustworthy even
when the fingerprints are still guesswork pending the Phase 0 corpus.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field

from ..classfile import (
    ACC_NATIVE,
    ACC_STATIC,
    ACC_SYNTHETIC,
    INVOKE_OPS,
    ClassFile,
    ClassSource,
    load_all,
)
from ..errors import UnnativeError

# Direct JVM entry points that bind a native library. A protector's own loader
# class ultimately calls one of these; seeing it (directly or one hop from a
# `<clinit>`) is the clearest static signal that native code gets bound at load.
_LOAD_TARGETS = {
    ("java/lang/System", "loadLibrary", "(Ljava/lang/String;)V"),
    ("java/lang/System", "load", "(Ljava/lang/String;)V"),
    ("java/lang/Runtime", "loadLibrary", "(Ljava/lang/String;)V"),
    ("java/lang/Runtime", "load", "(Ljava/lang/String;)V"),
}

_CLASS_EXT = ".class"


@dataclass
class NativeMethod:
    owner: str          # internal name, e.g. com/x/Foo
    name: str
    desc: str
    static: bool

    @property
    def ref(self) -> str:
        return f"{self.owner}.{self.name}:{self.desc}"


@dataclass
class LoadCall:
    where: str          # "owner.method:desc" containing the call
    target: str         # "System.loadLibrary" etc.
    lib: str | None     # the constant string argument, if statically visible


@dataclass
class ClassInfo:
    name: str                               # internal name
    major: int
    native_methods: list[NativeMethod] = field(default_factory=list)
    load_calls: list[LoadCall] = field(default_factory=list)
    clinit_calls: list[str] = field(default_factory=list)   # "owner.name:desc"
    synthetic_methods: list[str] = field(default_factory=list)  # "name:desc"
    decode_errors: list[str] = field(default_factory=list)


@dataclass
class Resource:
    name: str
    size: int
    kind: str           # "class" | "jar" | "nativelib" | "blob" | "manifest" | "other"


@dataclass
class Census:
    classes: list[ClassInfo]
    resources: list[Resource]
    parse_failures: list[tuple[str, str]]
    counts: dict

    def to_json(self) -> dict:
        return {
            "counts": self.counts,
            "classes": [asdict(c) for c in self.classes],
            "resources": [asdict(r) for r in self.resources],
            "parse_failures": [{"entry": e, "reason": why} for e, why in self.parse_failures],
        }

    # -- convenience views used by the fingerprints --
    @property
    def native_methods(self) -> list[NativeMethod]:
        return [m for c in self.classes for m in c.native_methods]

    @property
    def loader_classes(self) -> list[ClassInfo]:
        """Classes that either call a native loader directly, or whose `<clinit>`
        calls into another class (a candidate injected loader)."""
        return [c for c in self.classes if c.load_calls or c.clinit_calls]


_NATIVE_LIB_SUFFIXES = (".so", ".dll", ".dylib", ".jnilib")

# Extensions whose high entropy is expected (already-compressed media / fonts), so
# they are not mistaken for an opaque packed payload.
_KNOWN_NONBLOB_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".ogg", ".mp3", ".wav", ".mp4", ".zip", ".gz", ".xz", ".ttf", ".otf", ".woff",
    ".json", ".txt", ".xml", ".properties", ".cfg", ".html", ".css", ".js", ".md",
    ".yml", ".yaml", ".lang", ".mcmeta", ".sf", ".rsa", ".dsa",
}
_BLOB_MIN_SIZE = 4096
_BLOB_ENTROPY = 7.90            # ~8.0 = encrypted or well-compressed
_ENTROPY_SAMPLE = 1 << 18


def _shannon_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = Counter(b)
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_opaque_blob(name: str, data: bytes) -> bool:
    """A large, high-entropy resource with no media/text extension — an encrypted
    or compressed payload (e.g. a protector's packed native library)."""
    if len(data) < _BLOB_MIN_SIZE:
        return False
    if os.path.splitext(name)[1].lower() in _KNOWN_NONBLOB_EXT:
        return False
    return _shannon_entropy(data[:_ENTROPY_SAMPLE]) >= _BLOB_ENTROPY


def _classify_resource(name: str, data: bytes) -> str:
    low = name.lower()
    if low.endswith(_CLASS_EXT):
        return "class"
    if low == "meta-inf/manifest.mf":
        return "manifest"
    if low.endswith(".jar"):
        return "jar"
    if any(low.endswith(s) for s in _NATIVE_LIB_SUFFIXES):
        return "nativelib"
    # Native objects by magic, regardless of extension (protectors rename them):
    #   \x7fELF (ELF), MZ (PE), 0xcafebabe/0xfeedface/0xcffaedfe (Mach-O),
    #   0xbebafeca (Mach-O fat, big-endian).
    head = data[:4]
    if head[:4] == b"\x7fELF" or head[:2] == b"MZ" or head in (
        b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    ):
        # 0xcafebabe is ALSO a Java class magic; only call it a native lib if the
        # entry is not a .class. (Mach-O fat shares the same first bytes.)
        if head == b"\xca\xfe\xba\xbe":
            return "blob"           # ambiguous; a non-.class cafebabe is suspicious
        return "nativelib"
    if _looks_like_opaque_blob(name, data):
        return "blob"              # encrypted/compressed payload (possible packed lib)
    return "other"


def _scan_method_calls(cf: ClassFile, m: dict, owner: str):
    """-> (load_calls, invoked_refs, decode_error_or_None).

    Walks one method's bytecode once, collecting native-load call sites (with the
    string literal argument when it is a plain `ldc`, recorded best-effort) and the
    set of methods it invokes (so a `<clinit>`'s call graph can be inspected).
    """
    load_calls: list[LoadCall] = []
    invoked: list[str] = []
    try:
        insns, _meta = cf.insns(m)
    except UnnativeError as ex:
        return load_calls, invoked, str(ex)
    if insns is None:
        return load_calls, invoked, None

    last_str: str | None = None
    where = f"{owner}.{m['name']}:{m['desc']}"
    for ins in insns:
        if ins.name in ("ldc", "ldc_w", "ldc2_w"):
            try:
                # const_value returns ('str', s), ('int', n), ... but a 4-tuple
                # ('mhandle', owner, name, desc) for a MethodHandle constant, so
                # key off the kind and only read the value when it is a string.
                cv = cf.const_value(ins.operands[0])
                last_str = cv[1] if cv[0] == "str" else None
            except UnnativeError:
                last_str = None
            continue
        if ins.name in INVOKE_OPS or ins.name == "invokedynamic":
            if ins.name == "invokedynamic":
                last_str = None
                continue
            try:
                o, n, d, _k = cf.ref(ins.operands[0])
            except UnnativeError:
                last_str = None
                continue
            invoked.append(f"{o}.{n}:{d}")
            if (o, n, d) in _LOAD_TARGETS:
                load_calls.append(LoadCall(where=where,
                                           target=f"{o.rsplit('/', 1)[-1]}.{n}",
                                           lib=last_str))
            last_str = None
            continue
        # Any other opcode between an ldc and the invoke means the string is not a
        # clean immediate argument; don't attribute it.
        if ins.name not in ("dup", "nop"):
            last_str = None
    return load_calls, invoked, None


def _class_info(cf: ClassFile) -> ClassInfo:
    ci = ClassInfo(name=cf.this, major=cf.major)
    for m in cf.methods:
        acc = m["access"]
        if acc & ACC_NATIVE:
            ci.native_methods.append(
                NativeMethod(owner=cf.this, name=m["name"], desc=m["desc"],
                             static=bool(acc & ACC_STATIC)))
        if acc & ACC_SYNTHETIC:
            ci.synthetic_methods.append(f"{m['name']}:{m['desc']}")
        # A native method has no Code; skip the scan for it.
        if acc & ACC_NATIVE:
            continue
        load_calls, invoked, err = _scan_method_calls(cf, m, cf.this)
        ci.load_calls.extend(load_calls)
        if err:
            ci.decode_errors.append(f"{m['name']}:{m['desc']}: {err}")
        if m["name"] == "<clinit>":
            ci.clinit_calls = invoked
    return ci


def take_census(source: str) -> Census:
    """Parse a jar/dir and build the full census."""
    with ClassSource(source) as src:
        cfs, fails = load_all(src, quiet=True)

        classes = [_class_info(cf) for cf in cfs.values()]

        resources: list[Resource] = []
        for name in src.names():
            if name.endswith("/"):
                continue
            try:
                data = src.read_name(name)
            except (UnnativeError, OSError, KeyError):
                resources.append(Resource(name=name, size=-1, kind="other"))
                continue
            resources.append(Resource(name=name, size=len(data),
                                       kind=_classify_resource(name, data)))

    n_native = sum(len(c.native_methods) for c in classes)
    counts = {
        "classes_parsed": len(classes),
        "classes_failed": len(fails),
        "native_methods": n_native,
        "classes_with_native": sum(1 for c in classes if c.native_methods),
        "loader_classes": sum(1 for c in classes if c.load_calls),
        "resources": len(resources),
        "embedded_jars": sum(1 for r in resources if r.kind == "jar"),
        "native_libs": sum(1 for r in resources if r.kind == "nativelib"),
        "opaque_blobs": sum(1 for r in resources if r.kind == "blob"),
    }
    return Census(classes=classes, resources=resources, parse_failures=fails, counts=counts)
