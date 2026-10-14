"""Phase 6 — operand-stack reconstruction (plan.md §6 stackify.py).

The lifter gives a method's JIR as an ordered list of operations with a data-flow
track (which reference value each op produces and consumes). The JVM is a stack
machine, so turning that into real bytecode means scheduling each value onto the
operand stack in the right order and inserting the `dup`s a construct-and-use needs.

This first slice handles the **reference-threaded straight-line** shape: a method
whose every op is a `new`, a reference-only constructor call, or an `athrow`/field
store over values the lifter tracked — the `throw new X()` family and its kin. It
emits verifiable bytecode for those and refuses, with a reason, any method that
needs an operand it cannot see (a computed primitive, an untracked reference) or
that is not straight-line. Broadening it is future work; the refusal is honest, and
Phase 8 is where emitted bytecode is actually proven against the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ops that compute no value and are emitted for effect; they end a statement.
_EFFECT_OPS = frozenset({"athrow"})


@dataclass
class StackifyResult:
    ok: bool
    bytecode: list[dict] = field(default_factory=list)
    reason: str | None = None

    def to_json(self) -> dict:
        return {"ok": self.ok, "bytecode": self.bytecode, "reason": self.reason}


def _bc(mnemonic: str, **kw) -> dict:
    return {"bc": mnemonic, **kw}


def stackify(jm) -> StackifyResult:
    """Schedule a JirMethod's ops into bytecode, or explain why it can't be yet."""
    ops, df = jm.ops, jm.dataflow
    if jm.note:
        return StackifyResult(False, reason=jm.note)
    if not ops:
        return StackifyResult(False, reason="no lifted ops")
    if len(df) != len(ops):
        return StackifyResult(False, reason="missing data-flow track")
    for o, d in zip(ops, df):
        if o["op"] in ("deferred", "unlifted"):
            return StackifyResult(False, reason=f"{o['op']} op present ({o.get('fn', o['op'])})")
        if d.get("opaque"):
            return StackifyResult(False, reason=f"opaque operand at {o['op']}")

    # Map each value to the op that produces it.
    producer: dict[str, int] = {}
    for i, d in enumerate(df):
        if d.get("def") is not None:
            producer[d["def"]] = i

    # Recognise the construct idiom: a `new` whose value is consumed by an
    # immediately-following `invokespecial <init>`. Collapse the pair so the value
    # is produced once (new;dup;<init>) and then used like any other reference.
    construct_init: dict[int, int] = {}          # new-op index -> <init>-op index
    for i, o in enumerate(ops):
        if o["op"] != "new":
            continue
        v = df[i].get("def")
        j = i + 1
        if (v is not None and j < len(ops) and ops[j]["op"] == "invoke"
                and ops[j].get("kind") == "special" and ops[j].get("name") == "<init>"
                and df[j].get("uses") == [v]):
            construct_init[i] = j
    init_ops = set(construct_init.values())

    # Count uses, but not the <init> uses folded into a construct: those are served
    # by the `dup`, not by the value's single live copy.
    use_count: dict[str, int] = {}
    for i, d in enumerate(df):
        if i in init_ops:
            continue
        for u in d.get("uses", []):
            use_count[u] = use_count.get(u, 0) + 1

    emitted: list[dict] = []
    stack: list[str] = []

    def materialize(v: str) -> StackifyResult | None:
        """Push value `v` onto the operand stack, recursively evaluating it."""
        pi = producer.get(v)
        if pi is None:
            return StackifyResult(False, reason=f"value {v} has no producer (external operand)")
        if use_count.get(v, 0) != 1:
            return StackifyResult(False, reason=f"value {v} used {use_count.get(v, 0)}x "
                                               "(needs dup/locals — not in this slice)")
        o = ops[pi]
        if o["op"] == "new":
            emitted.append(_bc("new", owner=o["owner"]))
            if pi in construct_init:
                # new; dup; invokespecial <init>()V  -> one initialised ref remains
                init = ops[construct_init[pi]]
                emitted.append(_bc("dup"))
                emitted.append(_bc("invokespecial", owner=init["owner"],
                                   name="<init>", desc=init["desc"]))
            stack.append(v)
            return None
        return StackifyResult(False, reason=f"cannot materialize value from {o['op']}")

    # Emit statement by statement, in execution order, skipping ops folded into a
    # construct. Only effect ops (athrow) are roots in this slice.
    for i, o in enumerate(ops):
        if i in init_ops:
            continue                              # emitted as part of its `new`
        if i in construct_init:
            continue                              # a value only used later; defer to its use
        if o["op"] == "athrow":
            uses = df[i].get("uses", [])
            if len(uses) != 1:
                return StackifyResult(False, reason="athrow without a single tracked operand")
            err = materialize(uses[0])
            if err:
                return err
            emitted.append(_bc("athrow"))
            stack.pop()
        else:
            return StackifyResult(False, reason=f"unhandled root op {o['op']} (slice limit)")

    if stack:
        return StackifyResult(False, reason="operands left on stack")
    return StackifyResult(True, bytecode=emitted)
