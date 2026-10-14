"""Lift a straight-line x86-64 snippet to MBA expressions over registers.

The MBA engine (`mba.py`) folds expressions; this is the front-end that turns a
run of real instructions into them. It models the integer arithmetic/logic subset
JVMTP's constant obfuscation is built from (mov/add/sub/and/or/xor/shifts/lea/
imul/neg/not and the `xor r,r` zeroing idiom) as operations on a per-register
expression map. Registers start as free variables; an immediate is a constant.

It is deliberately **sound over complete**: anything it does not model — a memory
load, a call, a division, an unmodelled opcode — invalidates every register that
instruction writes (to a fresh free variable), using capstone's own
register-access info. So a register only ever carries an expression built entirely
from immediates and modelled ops; a recovered constant is therefore real, never a
misread of an instruction we glossed over.

capstone-only; nothing imports it unless capstone is present.
"""

from __future__ import annotations

from .mba import Bin, Const, Un, Var, fold, variables

try:
    from capstone import x86 as cs_x86
    _REG, _IMM, _MEM = cs_x86.X86_OP_REG, cs_x86.X86_OP_IMM, cs_x86.X86_OP_MEM
except ImportError:                                  # pragma: no cover - env-gated
    cs_x86 = None
    _REG = _IMM = _MEM = None

_MASK64 = 0xFFFFFFFFFFFFFFFF

# Sub-register -> (64-bit parent, width in bits). High-byte regs (ah/bh/…) are
# deliberately absent: reads/writes of them fall through to invalidation.
_LEGACY = {
    "rax": ("eax", "ax", "al"), "rbx": ("ebx", "bx", "bl"),
    "rcx": ("ecx", "cx", "cl"), "rdx": ("edx", "dx", "dl"),
    "rsi": ("esi", "si", "sil"), "rdi": ("edi", "di", "dil"),
    "rbp": ("ebp", "bp", "bpl"), "rsp": ("esp", "sp", "spl"),
}
_GP: dict[str, tuple[str, int]] = {}
for _p, (_e, _w, _b) in _LEGACY.items():
    _GP[_p] = (_p, 64)
    _GP[_e], _GP[_w], _GP[_b] = (_p, 32), (_p, 16), (_p, 8)
for _n in range(8, 16):
    _p = f"r{_n}"
    _GP[_p] = (_p, 64)
    _GP[_p + "d"], _GP[_p + "w"], _GP[_p + "b"] = (_p, 32), (_p, 16), (_p, 8)

_BINOP = {"add": "+", "sub": "-", "and": "&", "or": "|", "xor": "^", "imul": "*"}
_SHIFT = {"shl": "<<", "sal": "<<", "shr": "shr", "sar": "sar"}


class Lifter:
    """Per-register symbolic state across a straight-line instruction run."""

    def __init__(self):
        self.regs: dict[str, object] = {}
        self._fresh = 0

    def _new_var(self) -> Var:
        self._fresh += 1
        return Var(f"u{self._fresh}")

    def _read(self, name: str | None):
        if name not in _GP:
            return None                               # untracked reg -> unknown
        parent, width = _GP[name]
        base = self.regs.get(parent) or Var(parent)
        return base if width == 64 else Bin("&", base, Const((1 << width) - 1))

    def _write(self, name: str, val):
        parent, width = _GP[name]
        if width == 64:
            self.regs[parent] = val
        elif width == 32:
            self.regs[parent] = Bin("&", val, Const(0xFFFFFFFF))   # zero-extends upper
        else:
            old = self.regs.get(parent) or Var(parent)
            keep = Bin("&", old, Const(~((1 << width) - 1) & _MASK64))
            self.regs[parent] = Bin("|", keep, Bin("&", val, Const((1 << width) - 1)))

    def _opval(self, md, op):
        if op.type == _REG:
            return self._read(md.reg_name(op.reg))
        if op.type == _IMM:
            return Const(op.imm)
        return None                                   # memory / other -> unknown

    def _invalidate_written(self, md, ins):
        try:
            _read, written = ins.regs_access()
        except Exception:                             # pragma: no cover - capstone quirk
            written = [op.reg for op in ins.operands if op.type == _REG]
        for rid in written:
            name = md.reg_name(rid)
            if name in _GP:
                self.regs[_GP[name][0]] = self._new_var()   # whole parent is now unknown

    def step(self, md, ins):
        m, ops = ins.mnemonic, ins.operands
        dst = ops[0] if ops else None

        if m == "nop" or m.startswith("nop"):
            return
        if m in ("mov", "movabs") and dst is not None and dst.type == _REG and len(ops) == 2:
            v = self._opval(md, ops[1])
            return self._set_or_kill(md, dst, v)
        if m in _BINOP and dst is not None and dst.type == _REG and len(ops) >= 2:
            name = md.reg_name(dst.reg)
            if m in ("xor", "sub") and ops[1].type == _REG and md.reg_name(ops[1].reg) == name:
                return self._write(name, Const(0))               # r ^= r / r -= r
            a = self._read(name)
            if m == "imul" and len(ops) == 3:                    # imul dst, src, imm
                a, b = self._opval(md, ops[1]), self._opval(md, ops[2])
            else:
                b = self._opval(md, ops[1])
            return self._set_or_kill(md, dst, None if a is None or b is None
                                     else Bin(_BINOP[m], a, b))
        if m in _SHIFT and dst is not None and dst.type == _REG and len(ops) == 2:
            a, b = self._read(md.reg_name(dst.reg)), self._opval(md, ops[1])
            return self._set_or_kill(md, dst, None if a is None or b is None
                                     else Bin(_SHIFT[m], a, b))
        if m in ("neg", "not") and dst is not None and dst.type == _REG and len(ops) == 1:
            a = self._read(md.reg_name(dst.reg))
            return self._set_or_kill(md, dst, None if a is None else Un("neg" if m == "neg" else "not", a))
        if m in ("inc", "dec") and dst is not None and dst.type == _REG and len(ops) == 1:
            a = self._read(md.reg_name(dst.reg))
            return self._set_or_kill(md, dst, None if a is None
                                     else Bin("+", a, Const(1 if m == "inc" else -1)))
        if m == "lea" and dst is not None and dst.type == _REG and len(ops) == 2 and ops[1].type == _MEM:
            return self._set_or_kill(md, dst, self._addr(md, ops[1].mem))

        self._invalidate_written(md, ins)            # unmodelled: stay sound

    def _set_or_kill(self, md, dst, val):
        name = md.reg_name(dst.reg)
        if name not in _GP:
            return
        self._write(name, val if val is not None else self._new_var())

    def _addr(self, md, mem):
        expr = Const(mem.disp)
        if mem.base != 0:
            b = self._read(md.reg_name(mem.base))
            expr = Bin("+", b if b is not None else self._new_var(), expr)
        if mem.index != 0:
            idx = self._read(md.reg_name(mem.index))
            expr = Bin("+", expr, Bin("*", idx if idx is not None else self._new_var(),
                                      Const(mem.scale)))
        return expr


_CALLER_SAVED = ("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11")


def call_site_constants(md, insns: list) -> dict[int, dict[int, int]]:
    """-> {retaddr: {args_index: value}} for every `call` in a straight-line run.

    A value is reported only when the Lifter *proves* the register holds an
    immediate at the call — a plain `mov reg, imm` (or a foldable constant
    expression) that no memory load, call, or unmodelled op has invalidated.
    Caller-saved registers are reset after each call, so a value never survives
    a *real* call boundary. args_index is the JNI call's arg-vector position the
    register feeds: RCX=2, R8=3, R9=4 (the emulator's fixed-width array).
    """
    out: dict[int, dict[int, int]] = {}
    lifter = Lifter()
    for ins in insns:
        if ins.mnemonic.startswith("call"):
            site = {}
            for args_ix, reg in ((2, "rcx"), (3, "r8"), (4, "r9")):
                expr = lifter.regs.get(reg)
                if expr is None:
                    continue
                c = fold(expr, 64) if not variables(expr) else None
                if c is not None:
                    site[args_ix] = c
            for reg in _CALLER_SAVED:              # the callee may clobber these
                lifter.regs[reg] = lifter._new_var()
            if site:
                out[ins.address + ins.size] = site
        else:
            lifter.step(md, ins)
    return out


def constants_in(md, insns, width: int = 64) -> dict[str, int]:
    """Lift a straight-line run and return {register: constant} for every register
    that holds an *opaque* constant: an expression with free variables (so it reads
    as runtime-dependent) that nonetheless folds to one value. That is exactly the
    MBA signature; a plain `mov r, imm` literal has no variables and is not reported."""
    lifter = Lifter()
    for ins in insns:
        lifter.step(md, ins)
    out = {}
    for reg, expr in lifter.regs.items():
        if variables(expr):                          # looks input-dependent...
            c = fold(expr, width)
            if c is not None:                        # ...but is actually constant
                out[reg] = c
    return out
