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
from ..classfile import parse_desc

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
    "GetStringLength", "GetStringChars", "ReleaseStringChars",
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


_REF_RETURN = {"L", "["}

# Int-category descriptors (ZBCSI): JVM stack category 1 ints, representable with
# iconst/bipush/sipush/ldc-int. J (long) needs ldc2_w, F/D ride in XMM (untracked).
_INT_CAT = frozenset({"Z", "B", "C", "S", "I"})


def lift_method(method: str, jni_calls: list[dict], fault: str | None = None,
                locals: list[dict] | None = None,
                call_sites: dict[int, dict[int, int]] | None = None) -> jir.JirMethod:
    """Lift one method's JNI stream to JIR.

    `call_sites` optionally maps the *unbiased vaddr* a JNI call was made from (the
    trace's per-event `retaddr`) to {args_index: constant}: arg-index 2 is RCX, 3
    is R8, 4 is R9 in the emulator's fixed-width arg array. When a primitive
    operand equals the statically-proven constant at its call site, it threads as a
    value-constant (`value_consts`) instead of staying opaque — the literal is
    proven, never guessed.
    """
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

    # Statically-proven primitive constants: keyed per (call site, operand) so each
    # proven literal is its own single-use value (stackify's single-use invariant).
    consts: dict = {}

    def const_use(c: dict, args_idx: int, value: int, ptype: str) -> str | None:
        """vid of a *proven* constant operand, or None.

        Proof is the static disassembly of the call site (`call_sites`), not the
        raw traced argument: `call_sites[retaddr][args_idx] == value` means the
        argument register held a literal at the call. Anything else stays opaque.
        """
        if not call_sites:
            return None
        site = call_sites.get(c.get("retaddr", 0))
        if site is None or site.get(args_idx) != value:
            return None
        key = ("const", c.get("retaddr", 0), args_idx)
        cv = consts.get(key)
        if cv is None:
            cv = vid(key)
            consts[key] = cv
            m.value_consts[cv] = {"value": value, "type": ptype}
        return cv

    def emit(op: dict, defv: str | None = None, uses=(), opaque: bool = True):
        m.ops.append(op)
        m.dataflow.append({"def": defv, "uses": list(uses), "opaque": opaque})

    # Seed the method's arguments as locals: a JNI operand equal to one of these
    # distinctive values (recorded by the emulator) is a pure argument load.
    for loc in (locals or []):
        m.value_locals[vid(loc["value"])] = {"slot": loc["slot"], "type": loc["type"]}

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
        if fn in ("NewStringUTF", "NewString") and info is not None:
            # The protector lowers `ldc "str"` to building the string natively. The
            # emulator recovered the literal, so this is a threadable `ldc` value.
            s = info.get("string")
            handles.set(ret, {"kind": "string", "value": s})
            emit(jir.op_ldc(s, "String"), defv=vid(ret), uses=(), opaque=s is None)
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
                    # Thread the operands: the actual method arguments follow the
                    # methodID handle, and the receiver (virtual/special only) is
                    # args[0]. The op is threadable iff the receiver and every argument
                    # are tracked values. A non-void result becomes a typed value
                    # (stackify picks the load/return opcode from the descriptor). The
                    # trace records a fixed-width arg array, so take exactly as many
                    # argument slots as the descriptor declares (one per param).
                    try:
                        params = parse_desc(ref["desc"])[0]
                    except Exception:
                        params = []
                    actual = args[mid_ix + 1: mid_ix + 1 + len(params)]
                    uses = [] if kind == "static" else [recv]
                    for i, a in enumerate(actual):
                        u = use_of(a)
                        if u is None and i < len(params) and params[i] in _INT_CAT:
                            u = const_use(c, mid_ix + 1 + i, a, params[i])
                        uses.append(u)
                    ok = all(u is not None for u in uses)
                    is_void = ref["desc"].endswith(")V")
                    defv = vid(ret) if ret is not None and not is_void else None
                    emit(jir.op_invoke(kind, ref["owner"], ref["name"], ref["desc"]),
                         defv=defv, uses=[u for u in uses if u is not None], opaque=not ok)
            else:
                emit(jir.op_unlifted(fn, "unresolved methodID handle"))
                m.unlifted += 1
            continue

        fop = _field_op(fn)
        if fop:
            ref = _member(handles, args[1]) if len(args) > 1 else None
            if ref:
                # A field's desc is its type; the value threads whatever the type,
                # and stackify emits the matching load/store/return opcode.
                fa = (ref["owner"], ref["name"], ref["desc"])
                if fop == "getstatic":                       # no receiver operand
                    defv = vid(ret) if ret is not None else None
                    emit(jir.op_field(fop, *fa), defv=defv, uses=(), opaque=False)
                elif fop == "getfield":                      # Get*Field(obj, fieldID)
                    recv = use_of(args[0]) if args else None
                    defv = vid(ret) if ret is not None else None
                    emit(jir.op_field(fop, *fa), defv=defv,
                         uses=[recv] if recv is not None else (),
                         opaque=recv is None)
                elif fop == "putstatic":                     # SetStatic*Field(cls, fid, val)
                    val = use_of(args[2]) if len(args) > 2 else None
                    if val is None and len(args) > 2 and ref["desc"] in _INT_CAT:
                        val = const_use(c, 2, args[2], ref["desc"])
                    emit(jir.op_field(fop, *fa), uses=[val] if val is not None else (),
                         opaque=val is None)
                else:                                        # putfield: Set*Field(obj, fid, val)
                    recv = use_of(args[0]) if args else None
                    val = use_of(args[2]) if len(args) > 2 else None
                    if val is None and len(args) > 2 and ref["desc"] in _INT_CAT:
                        val = const_use(c, 2, args[2], ref["desc"])
                    emit(jir.op_field(fop, *fa),
                         uses=[u for u in (recv, val) if u is not None],
                         opaque=recv is None or val is None)
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
