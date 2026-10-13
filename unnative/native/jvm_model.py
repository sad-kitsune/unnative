"""The Java-side model the fake JNIEnv resolves against.

Built from the jar's *own* class files (plan.md §5a: "model Java objects as opaque
handles backed by jvm_model.py, built from the jar's own classes"). When the native
code calls `FindClass("com/x/Foo")`, `GetMethodID(cls, "bar", "(I)I")` or
`GetFieldID(...)`, the JNIEnv stubs answer from this model and hand back opaque ids
that the emulator can later map back to a concrete `(owner, name, desc)` — which is
exactly the JNI-call stream Phase 6 lifts from.

This module is static: it knows classes, methods and fields and how JNI resolves
them (walking superclasses and interfaces). Live object *instances* are opaque
handles owned by the handle table in `jnienv.py`; they are not modelled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..classfile import (
    ACC_NATIVE,
    ACC_STATIC,
    ClassFile,
    ClassSource,
    load_all,
)


@dataclass(frozen=True)
class MemberRef:
    """A resolved method or field: where it was *declared* (after the inheritance
    walk), its name/descriptor, and whether it is static."""
    owner: str          # declaring class (internal name)
    name: str
    desc: str
    static: bool
    is_method: bool

    def key(self) -> tuple[str, str, str]:
        return (self.owner, self.name, self.desc)

    def __str__(self) -> str:
        kind = "static " if self.static else ""
        sep = "" if self.is_method else " "
        return f"{kind}{self.owner}.{self.name}{sep}{self.desc}"


@dataclass
class ClassModel:
    internal: str                       # e.g. com/x/Foo
    super_name: str | None
    interfaces: list[str]
    access: int
    methods: list[dict]                 # {access, name, desc}
    fields: list[dict]
    cf: ClassFile | None = field(default=None, repr=False)

    def own_method(self, name: str, desc: str) -> dict | None:
        for m in self.methods:
            if m["name"] == name and m["desc"] == desc:
                return m
        return None

    def own_field(self, name: str, desc: str) -> dict | None:
        for f in self.fields:
            if f["name"] == name and f["desc"] == desc:
                return f
        return None


class JvmModel:
    """A resolver over the jar's classes. JDK classes (java/lang/*, …) are not
    present; resolution against an absent class returns None, which the JNIEnv
    stubs surface honestly rather than inventing a handle."""

    def __init__(self, classes: dict[str, ClassModel]):
        self.classes = classes

    # -- construction --
    @classmethod
    def from_classfiles(cls, cfs: dict[str, ClassFile]) -> "JvmModel":
        classes: dict[str, ClassModel] = {}
        for cf in cfs.values():
            classes[cf.this] = ClassModel(
                internal=cf.this,
                super_name=cf.super,
                interfaces=list(cf.interfaces),
                access=cf.access,
                methods=[{"access": m["access"], "name": m["name"], "desc": m["desc"]}
                         for m in cf.methods],
                fields=[{"access": fld["access"], "name": fld["name"], "desc": fld["desc"]}
                        for fld in cf.fields],
                cf=cf,
            )
        return cls(classes)

    @classmethod
    def from_source(cls, path: str) -> "JvmModel":
        with ClassSource(path) as src:
            cfs, _fails = load_all(src, quiet=True)
        return cls.from_classfiles(cfs)

    # -- lookup --
    def find_class(self, internal: str) -> ClassModel | None:
        return self.classes.get(internal)

    def _mro(self, start: str):
        """Yield `start` then its superclasses then interfaces (breadth-ish), over
        classes we actually have. JNI method/field resolution searches this chain."""
        seen = set()
        queue = [start]
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            cm = self.classes.get(name)
            yield name, cm
            if cm:
                if cm.super_name:
                    queue.append(cm.super_name)
                queue.extend(cm.interfaces)

    def resolve_method(self, owner: str, name: str, desc: str) -> MemberRef | None:
        for cls_name, cm in self._mro(owner):
            if cm is None:
                continue
            m = cm.own_method(name, desc)
            if m is not None:
                return MemberRef(owner=cls_name, name=name, desc=desc,
                                 static=bool(m["access"] & ACC_STATIC), is_method=True)
        return None

    def resolve_field(self, owner: str, name: str, desc: str) -> MemberRef | None:
        for cls_name, cm in self._mro(owner):
            if cm is None:
                continue
            f = cm.own_field(name, desc)
            if f is not None:
                return MemberRef(owner=cls_name, name=name, desc=desc,
                                 static=bool(f["access"] & ACC_STATIC), is_method=False)
        return None

    def native_methods(self) -> list[MemberRef]:
        out = []
        for cm in self.classes.values():
            for m in cm.methods:
                if m["access"] & ACC_NATIVE:
                    out.append(MemberRef(owner=cm.internal, name=m["name"],
                                         desc=m["desc"],
                                         static=bool(m["access"] & ACC_STATIC),
                                         is_method=True))
        return out
