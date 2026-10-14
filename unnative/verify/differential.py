"""The behaviour model and the three independent derivations Phase 8 compares.

A method's *observable behaviour*, at the granularity we can recover today, is
"throws exception class X" or "returns" (and "unknown" when the shape isn't modelled
yet). Phase 8 derives that behaviour three independent ways and checks they agree:

- `from_jir_ops` — what the lift stage *recovered* (the JIR). This is the expectation.
- `from_bytecode` — what the *rebuilt jar* actually encodes, re-decoded from its bytes.
  Independent of the JIR, so a disagreement pins an emit-stage bug (a wrong
  constant-pool ref, a dropped op, a wrong opcode).
- `from_trace` — ground truth from the Phase 5 *emulation* of the original native
  method, by following the thrown object handle back to the class it came from. The
  original can't be run on a real JVM (its class initialiser triggers the protector's
  loader), so the emulated trace is the only independent witness of what it did.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Behaviour:
    kind: str                       # "throw" | "return" | "unknown"
    exc: str | None = None          # thrown class (internal name), for kind=="throw"

    def describe(self) -> str:
        if self.kind == "throw":
            return f"throw {self.exc}" if self.exc else "throw (class indeterminate)"
        return self.kind


UNKNOWN = Behaviour("unknown")


def from_jir_ops(ops: list[dict]) -> Behaviour:
    """The behaviour the JIR models (the recovered expectation)."""
    if ops and ops[-1].get("op") == "athrow" and ops[-1].get("class"):
        return Behaviour("throw", ops[-1]["class"])
    return UNKNOWN


def from_bytecode(insns: list, cf) -> Behaviour:
    """Behaviour re-derived from the emitted bytecode, resolving its own pool refs.

    Recognises the straight-line construct-and-throw shape the emitter produces:
    `new X; dup; invokespecial X.<init>…; athrow`. Anything else is `unknown` rather
    than a guess — the classifier grows with the emitter."""
    names = [i.name for i in insns]
    if names == ["new", "dup", "invokespecial", "athrow"]:
        cls = cf.class_name(insns[0].operands[0])
        owner, name, _desc, _kind = cf.ref(insns[2].operands[0])
        if owner == cls and name == "<init>":
            return Behaviour("throw", cls)
    return UNKNOWN


def from_trace(jni_calls: list[dict]) -> Behaviour:
    """Ground truth from the emulated original: did it throw, and of what class?

    Follows the thrown handle back through the recorded JNI stream rather than
    re-running the idiom lifter, so it is a genuinely independent witness. Returns
    `throw` with `exc=None` when it threw but the class can't be pinned down."""
    throws = [c for c in jni_calls if c.get("fn") in ("Throw", "ThrowNew")]
    if not throws:
        return UNKNOWN
    by_ret = {c["ret"]: c for c in jni_calls if c.get("ret") is not None}
    t = throws[-1]
    args = t.get("args") or []
    first = args[0] if args else None
    if t["fn"] == "ThrowNew":
        return Behaviour("throw", _resolve_class(by_ret, first))   # arg0 is the class
    return Behaviour("throw", _class_of_object(by_ret, first))     # Throw(objectref)


def _resolve_class(by_ret: dict, handle, depth: int = 0) -> str | None:
    """Follow a jclass handle back to the class name `FindClass` gave it."""
    if handle is None or depth > 8:
        return None
    c = by_ret.get(handle)
    if c is None:
        return None
    if c["fn"] == "FindClass":
        return (c.get("info") or {}).get("class")
    if c["fn"] in ("NewGlobalRef", "NewLocalRef"):      # these alias their arg[0]
        inner = c.get("args") or []
        return _resolve_class(by_ret, inner[0] if inner else None, depth + 1)
    return None


def _class_of_object(by_ret: dict, obj_handle) -> str | None:
    """The class of an object handle produced by AllocObject/NewObject."""
    c = by_ret.get(obj_handle)
    if c and c["fn"] in ("AllocObject", "NewObject"):
        inner = c.get("args") or []
        return _resolve_class(by_ret, inner[0] if inner else None)
    return None


def verdict(recovered: Behaviour, rebuilt: Behaviour, trace: Behaviour) -> tuple[str, bool]:
    """-> (verdict, trace_confirmed).

    `parity` means the rebuilt bytecode realises the recovered behaviour and the
    emulated original does not contradict it; `trace_confirmed` says the emulation
    positively agreed (same thrown class). `mismatch` means emit diverged from the
    recovered IR, or the ground-truth trace disagrees with it. `unsupported` means
    the recovered shape isn't modelled yet."""
    if recovered.kind == "unknown":
        return "unsupported", False
    if rebuilt != recovered:
        return "mismatch", False
    if trace.kind == "throw" and recovered.kind == "throw":
        if trace.exc is None:
            return "parity", False                      # threw, class not pinned
        return ("parity", True) if trace.exc == recovered.exc else ("mismatch", False)
    return "parity", False                              # trace couldn't confirm
