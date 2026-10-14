"""Phase 6 — operand-stack reconstruction (plan.md §6 stackify.py).

The lifter gives a method's JIR as an ordered list of operations with a data-flow
track: which value each op produces (`def`) and consumes (`uses`), whether any
operand is untracked (`opaque`), and which values are method arguments (`locals` —
threadable as `aload_N`/`iload_N`). The JVM is a stack machine, so turning that into
bytecode means scheduling each value onto the operand stack in the right order.

This handles the **straight-line, single-use, tracked-value** family: every op is
non-opaque (all operands are tracked references, constants, method arguments, or
field reads), each produced value is used once (method-argument locals may be
re-loaded freely), and control flow is linear. It covers:

- `throw new X()` and its kin (construct-and-throw),
- `obj.m(args…)` / `System.out.println("…")` statements,
- argument- and field-threaded methods such as `void m(C c){ this.setBackground(c);
  this.repaint(); }`, `int getX(){ return this.x; }`, `void set(int a,int b){…}`.

Each statement is scheduled as a tree: emit a root (an effectful op — a void call, a
field store, `athrow`) after materialising its operands left-to-right; values are
inlined at their single use; a construct (`new` + no-arg `<init>`) folds to
`new;dup;invokespecial`. A void method gets a trailing `return`; a value-returning
one returns the single produced-but-unused value of the matching type.

Anything outside the slice — a computed primitive (no tracked source), a multi-use
value needing a `dup`/local store, a branch, an unidentifiable return value — is
refused with a reason. Phase 8 proves the emitted bytecode against the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..classfile import parse_desc

# A genuinely straight-line method is small. A long op list almost always means the
# trace unrolled a Java-level loop into one linear run, which this slice must not
# straight-line (it would be wrong, and can exceed the 64 KB per-method code limit).
_MAX_OPS = 64

_INVOKE_BC = {"static": "invokestatic", "virtual": "invokevirtual",
              "special": "invokespecial", "interface": "invokeinterface"}
_TERMINATORS = frozenset({"athrow", "return", "areturn", "ireturn", "lreturn",
                          "freturn", "dreturn"})


@dataclass
class StackifyResult:
    ok: bool
    bytecode: list[dict] = field(default_factory=list)
    reason: str | None = None

    def to_json(self) -> dict:
        return {"ok": self.ok, "bytecode": self.bytecode, "reason": self.reason}


class _Refuse(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _bc(mnemonic: str, **kw) -> dict:
    return {"bc": mnemonic, **kw}


def _category(desc: str) -> str:
    """Stack category of a type descriptor: how it loads/returns."""
    c = desc[:1]
    if c in ("L", "["):
        return "ref"
    if c in ("Z", "B", "C", "S", "I"):
        return "int"
    return {"J": "long", "F": "float", "D": "double"}.get(c, "?")


_LOAD = {"ref": "aload", "int": "iload", "long": "lload", "float": "fload", "double": "dload"}
_RETURN = {"ref": "areturn", "int": "ireturn", "long": "lreturn",
           "float": "freturn", "double": "dreturn"}
# ldc ctype -> descriptor it materialises as (what the value's users see).
_LDC_TYPE = {"String": "Ljava/lang/String;", "I": "I"}


def _needs_cast(have: str | None, want: str | None) -> bool:
    """A `checkcast` is needed when a reference value is passed where a *more
    specific* reference type is expected — e.g. `Object` from `List.get` into a
    `Color` parameter (the erased-generics cast javac emits)."""
    if not have or not want:
        return False
    if have[:1] not in ("L", "[") or want[:1] not in ("L", "["):
        return False
    return have != want and want != "Ljava/lang/Object;"


def _cast_target(desc: str) -> str:
    return desc[1:-1] if desc.startswith("L") else desc        # Lx/y; -> x/y


def stackify(jm) -> StackifyResult:
    """Schedule a JirMethod's ops into bytecode, or explain why it can't be yet."""
    ops, df = jm.ops, jm.dataflow
    locals_map = jm.value_locals
    if jm.note:
        return StackifyResult(False, reason=jm.note)
    if not ops:
        return StackifyResult(False, reason="no lifted ops")
    if len(df) != len(ops):
        return StackifyResult(False, reason="missing data-flow track")
    if len(ops) > _MAX_OPS:
        return StackifyResult(False, reason=f"{len(ops)} ops — likely an unrolled "
                                           "loop, not a straight-line method")
    for o, d in zip(ops, df):
        if o["op"] in ("deferred", "unlifted"):
            return StackifyResult(False, reason=f"{o['op']} op present ({o.get('fn', o['op'])})")
        if d.get("opaque"):
            return StackifyResult(False, reason=f"opaque operand at {o['op']}")

    producer = {d["def"]: i for i, d in enumerate(df) if d.get("def") is not None}

    # Construct idiom: a `new` consumed by an immediately-following no-arg
    # `invokespecial <init>`, folded to new;dup;<init>.
    construct_init: dict[int, int] = {}
    for i, o in enumerate(ops):
        if o["op"] != "new":
            continue
        v, j = df[i].get("def"), i + 1
        if (v is not None and j < len(ops) and ops[j]["op"] == "invoke"
                and ops[j].get("kind") == "special" and ops[j].get("name") == "<init>"
                and df[j].get("uses") == [v]):
            construct_init[i] = j
    init_ops = set(construct_init.values())

    use_count: dict[str, int] = {}
    for i, d in enumerate(df):
        if i in init_ops:
            continue
        for u in d.get("uses", []):
            use_count[u] = use_count.get(u, 0) + 1

    def value_type(v: str) -> str | None:
        if v in locals_map:
            return locals_map[v]["type"]
        if v in jm.value_consts:
            return jm.value_consts[v]["type"]
        pi = producer.get(v)
        if pi is None:
            return None
        o = ops[pi]
        if o["op"] == "new":
            return f"L{o['owner']};"
        if o["op"] == "ldc":
            return _LDC_TYPE.get(o.get("type", "String"))
        if o["op"] == "field":
            return o["desc"]
        if o["op"] == "invoke":
            return o["desc"].split(")", 1)[1]
        return None

    void = jm.desc.endswith(")V")
    throws = any(o["op"] == "athrow" for o in ops)

    # A value-returning, non-throwing method returns its one produced-but-unused value
    # of the matching category (what native code left in the return register).
    return_vid = None
    if not throws and not void:
        rettype = jm.desc.split(")", 1)[1]
        cands = [v for v, pi in producer.items()
                 if pi not in init_ops and use_count.get(v, 0) == 0
                 and value_type(v) and _category(value_type(v)) == _category(rettype)]
        if len(cands) != 1:
            return StackifyResult(False, reason="cannot identify the returned value")
        return_vid = cands[0]

    # Every produced value is used exactly once (the return value zero times); a
    # multi-use *construct* is spilled to an extra local (new;dup;<init>;astore N,
    # then aload N per use — javac compiles `Window w = new Window(); w.f(); w.g()`
    # exactly this way). Any other multi-use value keeps the honest refusal.
    spill_slots: dict[str, int] = {}
    next_local = max((loc["slot"] + (2 if loc["type"] in ("J", "D") else 1)
                      for loc in locals_map.values()), default=0)
    for v, pi in producer.items():
        if pi in init_ops:
            continue
        expected = 0 if v == return_vid else 1
        n = use_count.get(v, 0)
        if n != expected:
            if n > 1 and pi in construct_init:
                spill_slots[v] = next_local
                next_local += 1
            else:
                return StackifyResult(False, reason=f"value {v} used {n}x "
                                                   "(needs dup/locals — not in this slice)")

    emitted: list[dict] = []
    stack: list[str] = []

    def materialize(v: str) -> None:
        if v in locals_map:
            loc = locals_map[v]
            emitted.append(_bc(_LOAD[_category(loc["type"])], slot=loc["slot"]))
            stack.append(v)
            return
        if v in jm.value_consts:
            c = jm.value_consts[v]
            emitted.append(_bc("ldc", ctype="I", value=c["value"]))
            stack.append(v)
            return
        if v in spill_slots:
            emitted.append(_bc("aload", slot=spill_slots[v]))
            stack.append(v)
            return
        pi = producer.get(v)
        if pi is None:
            raise _Refuse(f"value {v} has no producer (external operand)")
        o = ops[pi]
        op = o["op"]
        if op == "new":
            emitted.append(_bc("new", owner=o["owner"]))
            if pi in construct_init:
                init = ops[construct_init[pi]]
                emitted.append(_bc("dup"))
                emitted.append(_bc("invokespecial", owner=init["owner"],
                                   name="<init>", desc=init["desc"]))
            stack.append(v)
        elif op == "ldc":
            emitted.append(_bc("ldc", ctype=o.get("type", "String"), value=o["value"]))
            stack.append(v)
        elif op == "field" and o["kind"] == "getstatic":
            emitted.append(_bc("getstatic", owner=o["owner"], name=o["name"], desc=o["desc"]))
            stack.append(v)
        elif op == "field" and o["kind"] == "getfield":
            _emit_consuming(pi, "getfield", [f"L{o['owner']};"],
                            owner=o["owner"], name=o["name"], desc=o["desc"])
            stack.append(v)
        elif op == "invoke":
            _emit_invoke(pi)
            stack.append(v)
        else:
            raise _Refuse(f"cannot materialize value from {op}")

    def _emit_consuming(i: int, mnemonic: str, expected: list, **operands) -> None:
        uses = df[i].get("uses", [])
        for u, want in zip(uses, expected):
            materialize(u)
            if _needs_cast(value_type(u), want):
                emitted.append(_bc("checkcast", owner=_cast_target(want)))
        for _ in uses:
            stack.pop()
        emitted.append(_bc(mnemonic, **operands))

    def _emit_invoke(i: int) -> None:
        o = ops[i]
        params = parse_desc(o["desc"])[0]
        expected = ([] if o["kind"] == "static" else [f"L{o['owner']};"]) + params
        _emit_consuming(i, _INVOKE_BC[o["kind"]], expected,
                        owner=o["owner"], name=o["name"], desc=o["desc"])

    try:
        for i, o in enumerate(ops):
            if i in init_ops or i in construct_init:
                # A construct is folded into its first use — except a *spilled*
                # one, which is materialised here once (new;dup;<init>;astore N)
                # and then re-loaded per use. Net stack effect zero.
                if i in construct_init and df[i].get("def") in spill_slots:
                    v = df[i]["def"]
                    init = ops[construct_init[i]]
                    emitted.append(_bc("new", owner=o["owner"]))
                    emitted.append(_bc("dup"))
                    emitted.append(_bc("invokespecial", owner=init["owner"],
                                       name="<init>", desc=init["desc"]))
                    emitted.append(_bc("astore", slot=spill_slots[v]))
                continue                              # folded, or a value used later
            if df[i].get("def") is not None:
                continue                              # a value: materialised at its use
            op = o["op"]
            if op == "athrow":
                uses = df[i].get("uses", [])
                if len(uses) != 1:
                    raise _Refuse("athrow without a single tracked operand")
                materialize(uses[0])
                stack.pop()
                emitted.append(_bc("athrow"))
            elif op == "invoke":                      # void call, a statement
                _emit_invoke(i)
            elif op == "field" and o["kind"] in ("putstatic", "putfield"):
                expected = ([] if o["kind"] == "putstatic" else [f"L{o['owner']};"]) + [o["desc"]]
                _emit_consuming(i, o["kind"], expected,
                                owner=o["owner"], name=o["name"], desc=o["desc"])
            else:
                raise _Refuse(f"unhandled root op {op} (slice limit)")

        if throws:
            pass                                      # athrow was the terminating root
        elif void:
            emitted.append(_bc("return"))
        else:
            materialize(return_vid)
            stack.pop()
            emitted.append(_bc(_RETURN[_category(jm.desc.split(')', 1)[1])]))
    except _Refuse as r:
        return StackifyResult(False, reason=r.reason)

    if stack:
        return StackifyResult(False, reason="operands left on stack")
    return StackifyResult(True, bytecode=emitted)
