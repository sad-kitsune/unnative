"""The JVM-level IR (JIR) emitted by Phase 6.

JIR is a list of normalised operations that sit close to bytecode: `new`, `invoke*`
with a resolved `(owner, name, desc)`, field access, `athrow`, `ldc`. It is
deliberately *not* stack-accurate — operand stack bookkeeping (the `dup` before an
`invokespecial <init>`, argument order) is reconstructed later by `stackify.py` and
the emitter. An op the lifter could not recognise is kept as an explicit `unlifted`
marker so a method's JIR always accounts for every JNI call it was built from.

Ops are plain JSON-ready dicts built through the factory functions below; a
`JirMethod` bundles a method's ops with the coverage statistics the pipeline tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# invoke kinds mirror the JVM invoke* family; "virtual" covers invokevirtual and,
# until the lifter distinguishes them (owner is an interface), invokeinterface.
INVOKE_KINDS = ("static", "virtual", "special", "interface")
FIELD_OPS = ("getstatic", "putstatic", "getfield", "putfield")


def op_new(owner: str) -> dict:
    return {"op": "new", "owner": owner}


def op_invoke(kind: str, owner: str, name: str, desc: str) -> dict:
    return {"op": "invoke", "kind": kind, "owner": owner, "name": name, "desc": desc}


def op_field(kind: str, owner: str, name: str, desc: str) -> dict:
    return {"op": "field", "kind": kind, "owner": owner, "name": name, "desc": desc}


def op_athrow(thrown: str | None = None) -> dict:
    d = {"op": "athrow"}
    if thrown:
        d["class"] = thrown
    return d


def op_ldc(value, ctype: str = "String") -> dict:
    return {"op": "ldc", "type": ctype, "value": value}


def op_simple(name: str) -> dict:
    """A JIR op that carries no operands of its own (arraylength, aaload, athrow-less
    stack ops, monitorenter/exit). The stackifier supplies its inputs later."""
    return {"op": name}


def op_newarray(elem_type: str) -> dict:
    return {"op": "newarray", "type": elem_type}


def op_anewarray(owner: str) -> dict:
    return {"op": "anewarray", "owner": owner}


def op_deferred(fn: str, reason: str) -> dict:
    """A JNI call that is a *recognised* idiom but one the linear stream lifter
    can't resolve on its own — typically a bulk array-region or string-buffer copy
    that only makes sense against the native method body (plan.md §6 cfg.py path).
    Distinguished from `unlifted` so coverage separates "needs more machinery" from
    "unknown call"."""
    return {"op": "deferred", "fn": fn, "reason": reason}


def op_unlifted(fn: str, reason: str) -> dict:
    """A JNI call the lifter does not yet map to an op. Recorded, not dropped, so a
    method's JIR covers every call and coverage numbers stay truthful."""
    return {"op": "unlifted", "fn": fn, "reason": reason}


def split_method(method: str) -> tuple[str, str, str]:
    """Split `owner/internal/Name.method(desc)ret` into (owner, name, desc)."""
    paren = method.index("(")
    desc = method[paren:]
    owner, name = method[:paren].rsplit(".", 1)
    return owner, name, desc


@dataclass
class JirMethod:
    owner: str
    name: str
    desc: str
    ops: list[dict] = field(default_factory=list)
    # Index-aligned with `ops`: each entry is {"def": vid|None, "uses": [vid,...],
    # "opaque": bool}. `vid` is a small per-method value id ("v0", "v1", …) naming a
    # Java-level reference value threaded between ops. `opaque` marks an op whose
    # operands are not all tracked references (a computed primitive, an untracked
    # ref) — the signal that the method is not yet stackifiable. Conservatively True.
    dataflow: list[dict] = field(default_factory=list)
    # vid -> {"slot": int, "type": descriptor} for values that are method arguments
    # (incl. `this`): these materialise as local loads (aload_N/iload_N/…), can be
    # loaded more than once, and are seeded from the trace's recorded argument slots.
    value_locals: dict = field(default_factory=dict)
    # vid -> {"value": int, "type": descriptor} for arguments the static proof
    # showed are compile-time constants (a `mov $imm` immediates at the JNI call
    # site, confirmed by the traced value). Materialise as ldc (iconst/bipush/…),
    # re-materialisable like locals — a constant has no side effects to repeat.
    value_consts: dict = field(default_factory=dict)
    # counts over the JNI calls this method was lifted from
    jni_calls: int = 0
    ignored: int = 0            # structural calls (ExceptionCheck, ref mgmt, …)
    deferred: int = 0           # recognised idiom, needs the native-body path
    unlifted: int = 0           # calls with no idiom yet
    note: str | None = None     # e.g. "no JNI calls — pure computation"
    fault: str | None = None    # carried from the trace, if emulation stopped early

    _NONOP = frozenset({"deferred", "unlifted"})

    @property
    def lifted(self) -> int:
        return sum(1 for o in self.ops if o["op"] not in self._NONOP)

    @property
    def coverage(self) -> float:
        """Fraction of *liftable* calls (everything but structural noise) that
        became a real op. 1.0 when the method is all structure or empty."""
        liftable = self.jni_calls - self.ignored
        if liftable <= 0:
            return 1.0
        return round(self.lifted / liftable, 4)

    def to_json(self) -> dict:
        return {
            "owner": self.owner,
            "name": self.name,
            "desc": self.desc,
            "method": f"{self.owner}.{self.name}{self.desc}",
            "ops": self.ops,
            "dataflow": self.dataflow,
            "locals": self.value_locals,
            "consts": self.value_consts,
            "stats": {
                "jni_calls": self.jni_calls,
                "lifted": self.lifted,
                "ignored": self.ignored,
                "deferred": self.deferred,
                "unlifted": self.unlifted,
                "coverage": self.coverage,
            },
            "note": self.note,
            "fault": self.fault,
        }
