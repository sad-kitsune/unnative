"""The fake JNIEnv / JavaVM the emulated library talks to (plan.md §5a).

Three parts, all independent of Unicorn:

  * `Layout` — where the JNIEnv, JavaVM, their function tables and the per-slot
    trampoline sentinels live in the emulated address space, and the byte images
    to map there. Each table slot points at a unique unmapped sentinel address;
    `emu.py` hooks a fetch of that address and calls back into `dispatch`.
  * `HandleTable` — opaque ids handed to native code for jclass/jmethodID/
    jfieldID/jobject/jstring, each mapping back to a concrete descriptor so a JNI
    call can be recorded as `(owner, name, desc)` — the stream Phase 6 lifts.
  * `JNIEnv` — the servicer. A handler per JNI function consults `JvmModel` and the
    handle table and records a `TraceEvent`; unimplemented functions fall to a
    default that records the call and returns a benign value. `RegisterNatives` is
    the load-bearing one: it is how the binding map (Phase 3) is captured.

Handlers take a `JniCtx` (arguments already ABI-decoded, with memory read/write),
so they are testable with a fake context and carry no calling-convention knowledge.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import jni_api
from .jvm_model import JvmModel

PTR = jni_api.POINTER_SIZE

JNI_OK = 0
JNI_FALSE = 0
JNI_TRUE = 1
JNI_VERSION_DEFAULT = 0x00010008        # report JNI 1.8 from GetVersion

# Global slot index spaces for the two interfaces, kept disjoint so one trampoline
# region serves both (vm slots start at +256, well past JNIEnv's 234).
_VM_BASE = 256


class JniCtx(Protocol):
    """What a handler sees. `args` excludes the JNIEnv*/JavaVM* receiver, so for
    GetMethodID(cls, name, sig) args == [cls_handle, name_ptr, sig_ptr]."""
    args: list[int]

    def cstr(self, ptr: int) -> str: ...
    def read(self, ptr: int, n: int) -> bytes: ...
    def read_ptr(self, ptr: int) -> int: ...
    def write(self, ptr: int, data: bytes) -> None: ...


@dataclass
class Layout:
    env_struct: int = 0x1_0000_0000      # JNIEnv  -> holds env_table pointer
    env_table: int = 0x1_0000_1000       # JNINativeInterface_ : 234 pointers
    vm_struct: int = 0x1_0000_3000       # JavaVM  -> holds vm_table pointer
    vm_table: int = 0x1_0000_3100        # JNIInvokeInterface_ : 8 pointers
    tramp_base: int = 0x1_0001_0000      # per-slot sentinel addresses live here
    tramp_step: int = 16

    def slot_address(self, global_index: int) -> int:
        return self.tramp_base + global_index * self.tramp_step

    def tramp_span(self) -> tuple[int, int]:
        hi = self.slot_address(_VM_BASE + len(jni_api.JAVAVM_SLOTS)) + self.tramp_step
        return self.tramp_base, hi

    def _table_bytes(self, slots, base_index: int) -> bytes:
        out = bytearray()
        for i, (_idx, name, _ret) in enumerate(slots):
            # Reserved slots (name is None) are NULL in a real JVM.
            addr = 0 if name is None else self.slot_address(base_index + i)
            out += struct.pack("<Q", addr)
        return bytes(out)

    def env_table_bytes(self) -> bytes:
        return self._table_bytes(jni_api.JNIENV_SLOTS, 0)

    def vm_table_bytes(self) -> bytes:
        return self._table_bytes(jni_api.JAVAVM_SLOTS, _VM_BASE)

    def env_struct_bytes(self) -> bytes:
        return struct.pack("<Q", self.env_table)

    def vm_struct_bytes(self) -> bytes:
        return struct.pack("<Q", self.vm_table)

    def resolve(self, address: int) -> tuple[str, str] | None:
        """sentinel address -> (interface, function_name), or None if not a slot."""
        if (address - self.tramp_base) % self.tramp_step:
            return None
        gi = (address - self.tramp_base) // self.tramp_step
        if gi < 0:
            return None
        if gi < len(jni_api.JNIENV_SLOTS):
            name = jni_api.JNIENV_SLOTS[gi][1]
            return ("JNIEnv", name) if name else None
        j = gi - _VM_BASE
        if 0 <= j < len(jni_api.JAVAVM_SLOTS):
            name = jni_api.JAVAVM_SLOTS[j][1]
            return ("JavaVM", name) if name else None
        return None


@dataclass
class Handle:
    id: int
    kind: str           # "class" | "method" | "field" | "object" | "string" | ...
    descriptor: object  # internal name / MemberRef / str value / ...


class HandleTable:
    """Opaque ids for Java-side objects. Ids are tagged high so they are never
    confused with NULL or a real mapped address."""

    BASE = 0x8000_0000_0000_0001

    def __init__(self):
        self._by_id: dict[int, Handle] = {}
        self._next = self.BASE

    def new(self, kind: str, descriptor: object) -> int:
        hid = self._next
        self._next += 1
        self._by_id[hid] = Handle(hid, kind, descriptor)
        return hid

    def get(self, hid: int) -> Handle | None:
        return self._by_id.get(hid)

    def descriptor(self, hid: int):
        h = self._by_id.get(hid)
        return h.descriptor if h else None


@dataclass
class TraceEvent:
    interface: str
    fn: str
    detail: str
    result: object = None


@dataclass
class Binding:
    owner: str          # internal class name (or "?" if the class handle is unknown)
    name: str
    signature: str
    fn_ptr: int

    def to_json(self) -> dict:
        return {"owner": self.owner, "name": self.name, "signature": self.signature,
                "fn_ptr": self.fn_ptr}


# A handler returns the integer/pointer value for RAX, or None for void. (No current
# handler needs a non-zero float return; emu routes by the slot's return category.)
Handler = Callable[["JNIEnv", JniCtx], "int | None"]

_HANDLERS: dict[str, Handler] = {}

# JNI return types that are object/reference-like (a jobject/jclass/jstring/jarray/
# jmethodID/jfieldID/…). An unimplemented stub returning one of these must hand back
# a non-null handle, not 0: translated code frequently uses the result as a receiver
# or even a call target, and a NULL there faults the whole method. Raw C-pointer
# returns (jchar*, jbyte* from GetStringChars/Get*ArrayElements) are deliberately
# excluded — a fake pointer there would fault on dereference instead.
_OBJECT_RETURNS = frozenset({
    "jobject", "jclass", "jstring", "jthrowable", "jweak",
    "jmethodID", "jfieldID",
    "jobjectArray", "jbooleanArray", "jbyteArray", "jcharArray", "jshortArray",
    "jintArray", "jlongArray", "jfloatArray", "jdoubleArray",
})
_RET_BY_NAME = {name: ret for _i, name, ret in jni_api.JNIENV_SLOTS if name}
_RET_BY_NAME.update({name: ret for _i, name, ret in jni_api.JAVAVM_SLOTS if name})


def _handles(*names: str):
    def register(fn: Handler):
        for n in names:
            _HANDLERS[n] = fn
        return fn
    return register


class JNIEnv:
    def __init__(self, model: JvmModel, layout: Layout | None = None,
                 handles: HandleTable | None = None):
        self.model = model
        self.layout = layout or Layout()
        self.handles = handles or HandleTable()
        self.trace: list[TraceEvent] = []
        self.bindings: list[Binding] = []
        self.unimplemented: set[str] = set()

    # -- dispatch --
    def dispatch(self, interface: str, name: str, ctx: JniCtx) -> int | None:
        handler = _HANDLERS.get(name)
        if handler is None:
            return self._default(interface, name, ctx)
        return handler(self, ctx)

    def _record(self, interface: str, fn: str, detail: str, result: object = None):
        self.trace.append(TraceEvent(interface, fn, detail, result))

    def _default(self, interface: str, name: str, ctx: JniCtx) -> int | None:
        self.unimplemented.add(name)
        # Object/reference returns get a fresh non-null handle so callers can keep
        # going (and never call through a NULL); everything else gets 0.
        if _RET_BY_NAME.get(name) in _OBJECT_RETURNS:
            hid = self.handles.new("object", f"<{name}>")
            self._record(interface, name, f"(unimplemented) args={ctx.args}", hid)
            return hid
        self._record(interface, name, f"(unimplemented) args={ctx.args}")
        return 0

    def _class_name_of(self, handle: int) -> str:
        d = self.handles.descriptor(handle)
        return d if isinstance(d, str) else "?"

    # -- JavaVM --
    @_handles("GetEnv")
    def _get_env(self, ctx: JniCtx):
        penv, version = ctx.args[0], ctx.args[1]
        ctx.write(penv, struct.pack("<Q", self.layout.env_struct))
        self._record("JavaVM", "GetEnv", f"version={version:#x} -> env", JNI_OK)
        return JNI_OK

    # -- JNIEnv: versions / classes --
    @_handles("GetVersion")
    def _get_version(self, ctx: JniCtx):
        self._record("JNIEnv", "GetVersion", "", JNI_VERSION_DEFAULT)
        return JNI_VERSION_DEFAULT

    @_handles("FindClass")
    def _find_class(self, ctx: JniCtx):
        name = ctx.cstr(ctx.args[0])
        cm = self.model.find_class(name)
        hid = self.handles.new("class", name)
        self._record("JNIEnv", "FindClass",
                     f"{name!r} ({'known' if cm else 'absent'})", hid)
        return hid

    @_handles("GetObjectClass")
    def _get_object_class(self, ctx: JniCtx):
        obj = ctx.args[0]
        desc = self.handles.descriptor(obj)
        name = desc if isinstance(desc, str) else "?"
        hid = self.handles.new("class", name)
        self._record("JNIEnv", "GetObjectClass", f"obj#{obj:#x} -> {name}", hid)
        return hid

    @_handles("GetSuperclass")
    def _get_superclass(self, ctx: JniCtx):
        name = self._class_name_of(ctx.args[0])
        cm = self.model.find_class(name)
        sup = cm.super_name if cm else None
        hid = self.handles.new("class", sup) if sup else 0
        self._record("JNIEnv", "GetSuperclass", f"{name} -> {sup}", hid)
        return hid

    # -- JNIEnv: members (the core of the recovered call stream) --
    def _resolve_member(self, ctx: JniCtx, static: bool, is_method: bool, fn: str):
        owner = self._class_name_of(ctx.args[0])
        name = ctx.cstr(ctx.args[1])
        sig = ctx.cstr(ctx.args[2])
        ref = (self.model.resolve_method(owner, name, sig) if is_method
               else self.model.resolve_field(owner, name, sig))
        kind = "method" if is_method else "field"
        hid = self.handles.new(kind, ref if ref else (owner, name, sig))
        self._record("JNIEnv", fn,
                     f"{owner}.{name}{sig} ({'resolved' if ref else 'unresolved'})", hid)
        return hid

    @_handles("GetMethodID")
    def _get_method_id(self, ctx: JniCtx):
        return self._resolve_member(ctx, static=False, is_method=True, fn="GetMethodID")

    @_handles("GetStaticMethodID")
    def _get_static_method_id(self, ctx: JniCtx):
        return self._resolve_member(ctx, static=True, is_method=True,
                                    fn="GetStaticMethodID")

    @_handles("GetFieldID")
    def _get_field_id(self, ctx: JniCtx):
        return self._resolve_member(ctx, static=False, is_method=False, fn="GetFieldID")

    @_handles("GetStaticFieldID")
    def _get_static_field_id(self, ctx: JniCtx):
        return self._resolve_member(ctx, static=True, is_method=False,
                                    fn="GetStaticFieldID")

    # -- JNIEnv: strings --
    @_handles("NewStringUTF")
    def _new_string_utf(self, ctx: JniCtx):
        s = ctx.cstr(ctx.args[0])
        hid = self.handles.new("string", s)
        self._record("JNIEnv", "NewStringUTF", f"{s!r}", hid)
        return hid

    @_handles("GetStringUTFChars")
    def _get_string_utf_chars(self, ctx: JniCtx):
        desc = self.handles.descriptor(ctx.args[0])
        self._record("JNIEnv", "GetStringUTFChars",
                     f"string#{ctx.args[0]:#x} = {desc!r}")
        # A real emulator writes the bytes into guest memory and returns a pointer;
        # the scaffolding records the value and returns NULL for now.
        return 0

    @_handles("ReleaseStringUTFChars", "ExceptionClear", "ExceptionDescribe")
    def _void_noop(self, ctx: JniCtx):
        return None

    # -- JNIEnv: exceptions --
    @_handles("ExceptionCheck")
    def _exception_check(self, ctx: JniCtx):
        self._record("JNIEnv", "ExceptionCheck", "", JNI_FALSE)
        return JNI_FALSE

    @_handles("ExceptionOccurred")
    def _exception_occurred(self, ctx: JniCtx):
        self._record("JNIEnv", "ExceptionOccurred", "", 0)
        return 0

    # -- JNIEnv: RegisterNatives (the binding map, Phase 3) --
    @_handles("RegisterNatives")
    def _register_natives(self, ctx: JniCtx):
        clazz, methods_ptr, n = ctx.args[0], ctx.args[1], ctx.args[2]
        owner = self._class_name_of(clazz)
        for k in range(n):
            base = methods_ptr + k * (3 * PTR)      # JNINativeMethod{name,sig,fnPtr}
            name = ctx.cstr(ctx.read_ptr(base))
            sig = ctx.cstr(ctx.read_ptr(base + PTR))
            fn_ptr = ctx.read_ptr(base + 2 * PTR)
            self.bindings.append(Binding(owner, name, sig, fn_ptr))
        self._record("JNIEnv", "RegisterNatives", f"{owner}: {n} method(s)", JNI_OK)
        return JNI_OK
