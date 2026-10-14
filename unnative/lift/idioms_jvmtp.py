"""JVMTP/JNIC idiom recognition: a recorded JNI-call stream -> JIR ops.

The lifter walks a method's JNI calls in order, keeping a map from each live JNI
handle to what it denotes (a class, a resolved member, a string, an object). A
handle is born when `FindClass`/`Get*ID`/`NewStringUTF`/`AllocObject`/`NewObject`
returns it, and it is *consumed* when a `Call*Method`/`Get*Field`/`Throw`/… takes it
as an argument. Resolving the consumer's handle arguments against the map recovers
the `(owner, name, desc)` behind each call — which is the whole game for lifting,
because the protectors lower one bytecode instruction to one such idiom.

The idiom table is the one in plan.md §6. It is named `jvmtp` because that is the
protector the first real sample looks like, but nothing here is JVMTP-specific yet:
it is plain JNI shape recognition, shared with JNIC until the two need to diverge.
"""

from __future__ import annotations

import re

from . import jir

# JNI type infixes in the Call*/Get*/Set* family names.
_T = r"(?:Void|Object|Boolean|Byte|Char|Short|Int|Long|Float|Double)"
_CALL_STATIC = re.compile(rf"^CallStatic{_T}Method[AV]?$")
_CALL_NONVIRT = re.compile(rf"^CallNonvirtual{_T}Method[AV]?$")
_CALL_VIRTUAL = re.compile(rf"^Call{_T}Method[AV]?$")
_GET_STATIC_FIELD = re.compile(rf"^GetStatic{_T}Field$")
_SET_STATIC_FIELD = re.compile(rf"^SetStatic{_T}Field$")
_GET_FIELD = re.compile(rf"^Get{_T}Field$")
_SET_FIELD = re.compile(rf"^Set{_T}Field$")

# Primitive array constructors: New<Type>Array -> newarray <type>.
_PRIM = "(?P<t>Boolean|Byte|Char|Short|Int|Long|Float|Double)"
_NEW_PRIM_ARRAY = re.compile(rf"^New{_PRIM}Array$")

# JNI array-region / element-buffer and string-buffer calls. These are *recognised*
# idioms but not 1:1 with a single bytecode op: the protector lowers a run of array
# or string element accesses to a bulk native-buffer copy, so reconstructing the
# original loads/stores needs the native method body, not the call stream alone.
# They are deferred (counted honestly), not treated as unknown.
_DEFERRED = frozenset({
    "NewString", "GetStringLength", "GetStringChars", "ReleaseStringChars",
    "GetStringUTFLength", "GetStringUTFChars", "ReleaseStringUTFChars",
    "GetStringRegion", "GetStringUTFRegion", "GetStringCritical", "ReleaseStringCritical",
    "GetPrimitiveArrayCritical", "ReleasePrimitiveArrayCritical",
    "GetBooleanArrayElements", "GetByteArrayElements", "GetCharArrayElements",
    "GetShortArrayElements", "GetIntArrayElements", "GetLongArrayElements",
    "GetFloatArrayElements", "GetDoubleArrayElements",
    "ReleaseBooleanArrayElements", "ReleaseByteArrayElements", "ReleaseCharArrayElements",
    "ReleaseShortArrayElements", "ReleaseIntArrayElements", "ReleaseLongArrayElements",
    "ReleaseFloatArrayElements", "ReleaseDoubleArrayElements",
    "GetBooleanArrayRegion", "GetByteArrayRegion", "GetCharArrayRegion",
    "GetShortArrayRegion", "GetIntArrayRegion", "GetLongArrayRegion",
    "GetFloatArrayRegion", "GetDoubleArrayRegion",
    "SetBooleanArrayRegion", "SetByteArrayRegion", "SetCharArrayRegion",
    "SetShortArrayRegion", "SetIntArrayRegion", "SetLongArrayRegion",
    "SetFloatArrayRegion", "SetDoubleArrayRegion",
})

# Calls that produce a handle but emit no op themselves (they feed the consumers).
# Handle aliasing: a *Ref wrapper hands back a new handle for the same referent.
_ALIAS = frozenset({"NewGlobalRef", "NewLocalRef", "NewWeakGlobalRef"})

# Calls that carry no bytecode meaning for the linear lifter: exception polling,
# reference/frame lifecycle, binding. Skipped (counted as "ignored"), not unlifted.
_IGNORE = frozenset({
    "GetVersion", "GetJavaVM", "GetEnv",
    "ExceptionCheck", "ExceptionOccurred", "ExceptionClear", "ExceptionDescribe",
    "DeleteGlobalRef", "DeleteLocalRef", "DeleteWeakGlobalRef",
    "PushLocalFrame", "PopLocalFrame", "EnsureLocalCapacity",
    "IsSameObject", "IsInstanceOf", "IsAssignableFrom",
    "RegisterNatives", "UnregisterNatives",
})


class _Handles:
    """Handle -> referent, with alias chains followed on lookup."""

    def __init__(self):
        self._d: dict[int, dict] = {}

    def set(self, hid, ref: dict):
        if hid:
            self._d[hid] = ref

    def get(self, hid) -> dict | None:
        return self._d.get(hid)

    def class_of(self, hid) -> str | None:
        ref = self._d.get(hid)
        if not ref:
            return None
        if ref["kind"] == "class":
            return ref.get("class")
        if ref["kind"] == "object":
            return ref.get("class")
        return None


def _invoke_kind(fn: str) -> str | None:
    if _CALL_STATIC.match(fn):
        return "static"
    if _CALL_NONVIRT.match(fn):
        return "special"
    if _CALL_VIRTUAL.match(fn):
        return "virtual"
    return None


def _field_op(fn: str) -> str | None:
    if _GET_STATIC_FIELD.match(fn):
        return "getstatic"
    if _SET_STATIC_FIELD.match(fn):
        return "putstatic"
    if _GET_FIELD.match(fn):
        return "getfield"
    if _SET_FIELD.match(fn):
        return "putfield"
    return None


def _member(handles: _Handles, hid) -> dict | None:
    ref = handles.get(hid)
    if ref and ref["kind"] in ("method", "field"):
        return ref
    return None


def _has_params(desc: str) -> bool:
    try:
        return desc[desc.index("(") + 1] != ")"
    except (ValueError, IndexError):
        return True                      # unknown shape: assume operands, stay safe


def _ref_return(desc: str) -> bool:
    r = desc[desc.index(")") + 1:] if ")" in desc else ""
    return r[:1] in ("L", "[")


def lift_method(method: str, jni_calls: list[dict], fault: str | None = None) -> jir.JirMethod:
    owner, name, desc = jir.split_method(method)
    m = jir.JirMethod(owner, name, desc, jni_calls=len(jni_calls), fault=fault)
    handles = _Handles()

    # Value ids for Java-level reference values threaded between ops. A handle gets
    # a vid only when it names such a value (a new'd object, a ref-returning call);
    # class/method/field/string handles are JNI plumbing, not stack operands.
    vids: dict[int, str] = {}

    def vid(h) -> str:
        if h not in vids:
            vids[h] = f"v{len(vids)}"
        return vids[h]

    def use_of(h) -> str | None:
        return vids.get(h)                # None => untracked operand (opaque)

    def emit(op: dict, defv: str | None = None, uses=(), opaque: bool = True):
        m.ops.append(op)
        m.dataflow.append({"def": defv, "uses": list(uses), "opaque": opaque})

    if not jni_calls:
        # No JNI traffic at all: the body is pure computation (arithmetic, local
        # data flow) the JNI-stream lifter cannot see. Honest, and a pointer to the
        # future pcode/CFG path (plan.md §6 cfg.py/stackify.py).
        m.note = "no JNI calls — pure computation, needs native decompilation"
        return m

    for c in jni_calls:
        fn = c["fn"]
        args = c.get("args") or []
        info = c.get("info")
        ret = c.get("ret")

        # -- producers: record what the new handle denotes, emit nothing --
        if fn in ("FindClass", "GetObjectClass", "GetSuperclass") and info:
            handles.set(ret, {"kind": "class", "class": info.get("class")})
            m.ignored += 1
            continue
        if fn in ("GetMethodID", "GetStaticMethodID") and info:
            handles.set(ret, {"kind": "method", "owner": info["owner"],
                              "name": info["name"], "desc": info["desc"],
                              "static": info.get("static", False)})
            m.ignored += 1
            continue
        if fn in ("GetFieldID", "GetStaticFieldID") and info:
            handles.set(ret, {"kind": "field", "owner": info["owner"],
                              "name": info["name"], "desc": info["desc"],
                              "static": info.get("static", False)})
            m.ignored += 1
            continue
        if fn == "NewStringUTF" and info:
            handles.set(ret, {"kind": "string", "value": info.get("string")})
            m.ignored += 1
            continue
        if fn in _ALIAS:
            src = handles.get(args[0]) if args else None
            if src:
                handles.set(ret, src)
            m.ignored += 1
            continue

        # -- consumers: emit a JIR op --
        kind = _invoke_kind(fn)
        if kind:
            # static: Call(clazz, mid, …)      -> mid at args[1]
            # virtual: Call(obj, mid, …)        -> mid at args[1]
            # special: CallNonvirtual(obj, clazz, mid, …) -> mid at args[2]
            mid_ix = 2 if kind == "special" else 1
            ref = _member(handles, args[mid_ix]) if len(args) > mid_ix else None
            if ref:
                recv = use_of(args[0]) if args else None
                if (kind == "special" and ref["name"] == "<init>" and recv is not None):
                    # Constructor on a tracked object: threadable when it has no
                    # (unrecovered) arguments; the construct-merge in stackify folds
                    # this into the preceding `new`.
                    emit(jir.op_invoke(kind, ref["owner"], ref["name"], ref["desc"]),
                         uses=[recv], opaque=_has_params(ref["desc"]))
                else:
                    # The call's result, if a reference, becomes a tracked value; its
                    # operands are not resolved yet, so the op itself stays opaque.
                    defv = vid(ret) if ret is not None and _ref_return(ref["desc"]) else None
                    emit(jir.op_invoke(kind, ref["owner"], ref["name"], ref["desc"]),
                         defv=defv, opaque=True)
            else:
                emit(jir.op_unlifted(fn, "unresolved methodID handle"))
                m.unlifted += 1
            continue

        fop = _field_op(fn)
        if fop:
            ref = _member(handles, args[1]) if len(args) > 1 else None
            if ref:
                # A field's desc is its type; a ref-typed getter yields a value.
                is_ref_get = fop.startswith("get") and ref["desc"][:1] in ("L", "[")
                defv = vid(ret) if ret is not None and is_ref_get else None
                emit(jir.op_field(fop, ref["owner"], ref["name"], ref["desc"]),
                     defv=defv, opaque=True)
            else:
                emit(jir.op_unlifted(fn, "unresolved fieldID handle"))
                m.unlifted += 1
            continue

        if fn == "AllocObject":
            cls = handles.class_of(args[0]) if args else None
            handles.set(ret, {"kind": "object", "class": cls})
            # `new X` takes no stack operands and yields a fresh (uninitialised) ref.
            emit(jir.op_new(cls or "?"), defv=vid(ret), uses=(), opaque=False)
            continue

        if re.match(r"^NewObject[AV]?$", fn):
            cls = handles.class_of(args[0]) if args else None
            handles.set(ret, {"kind": "object", "class": cls})
            obj = vid(ret)
            emit(jir.op_new(cls or "?"), defv=obj, uses=(), opaque=False)
            ref = _member(handles, args[1]) if len(args) > 1 else None
            if ref:
                # Constructor with reference/primitive args is not threadable yet.
                emit(jir.op_invoke("special", ref["owner"], "<init>", ref["desc"]),
                     uses=[obj], opaque=_has_params(ref["desc"]))
            else:
                emit(jir.op_unlifted(fn, "unresolved <init> methodID"))
                m.unlifted += 1
            continue

        if fn == "GetArrayLength":
            emit(jir.op_simple("arraylength"))
            continue

        pa = _NEW_PRIM_ARRAY.match(fn)
        if pa:
            handles.set(ret, {"kind": "object", "class": None})
            emit(jir.op_newarray(pa.group("t").lower()), defv=vid(ret))
            continue

        if re.match(r"^NewObjectArray$", fn):
            # NewObjectArray(len, elementClass, initialElement) -> anewarray <elem>
            cls = handles.class_of(args[1]) if len(args) > 1 else None
            handles.set(ret, {"kind": "object", "class": None})
            emit(jir.op_anewarray(cls or "?"), defv=vid(ret))
            continue

        if fn == "GetObjectArrayElement":
            emit(jir.op_simple("aaload"))
            continue
        if fn == "SetObjectArrayElement":
            emit(jir.op_simple("aastore"))
            continue

        if fn == "MonitorEnter":
            emit(jir.op_simple("monitorenter"))
            continue
        if fn == "MonitorExit":
            emit(jir.op_simple("monitorexit"))
            continue

        if fn == "Throw":
            thrown = handles.class_of(args[0]) if args else None
            u = use_of(args[0]) if args else None
            emit(jir.op_athrow(thrown), uses=[u] if u else (), opaque=u is None)
            continue

        if fn == "ThrowNew":
            cls = handles.class_of(args[0]) if args else None
            # ThrowNew builds the throwable with a (String) ctor whose message string
            # we have not recovered, so this whole shape stays opaque for now.
            emit(jir.op_new(cls or "?"))
            emit(jir.op_invoke("special", cls or "?", "<init>", "(Ljava/lang/String;)V"))
            emit(jir.op_athrow(cls))
            continue

        if fn in _DEFERRED:
            emit(jir.op_deferred(fn, "bulk array/string buffer — needs "
                                     "native-body analysis (cfg)"))
            m.deferred += 1
            continue

        if fn in _IGNORE:
            m.ignored += 1
            continue

        # Anything else: keep an honest marker so coverage reflects reality.
        emit(jir.op_unlifted(fn, "no idiom yet"))
        m.unlifted += 1

    return m
