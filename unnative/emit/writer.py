"""A surgical class-file writer (plan.md Phase 7).

The plan assumed Kotlin + ASM for this stage, for ASM's StackMapTable generation.
The first emit target makes that machinery unnecessary: the methods `stackify`
can lower today are *straight-line* (the `throw new X()` family), and a method
with no branches needs no StackMapTable at all — so a small pure-Python writer is
enough, and keeps the whole pipeline in one language with no extra toolchain.

The writer is deliberately *surgical*. The vendored parser (`classfile.py`) keeps
the original bytes and the constant-pool boundary, so we rebuild a class by copying
every byte we do not touch and re-emitting only the methods we fill:

    magic+version │ cp_count' │ <original CP bytes> │ <appended CP> │
    <access…fields, verbatim> │ <methods, re-emitted> │ <class attrs, verbatim>

Existing constant-pool indices are never renumbered — new entries are appended —
so every attribute we copy as raw bytes (StackMapTable, annotations, the whole
fields section, …) keeps pointing at the right pool entry. An unmodified class
therefore re-serialises byte-for-byte identically; that identity is the writer's
main correctness test.

Only the bytecode `stackify` actually emits is encodable (`new`, `dup`,
`invokespecial`, `athrow`). Anything else raises, rather than guessing an encoding
that nothing exercises — the encoder grows, with a test, as stackify does.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..classfile import ACC_NATIVE, ACC_STATIC, ClassFile, enc_modified_utf8, parse_desc
from ..errors import UnnativeError

_U2_MAX = 0xFFFF


# -- constant-pool tags we append (JVMS Table 4.4-A) --
_TAG_UTF8 = 1
_TAG_INTEGER = 3
_TAG_CLASS = 7
_TAG_STRING = 8
_TAG_FIELDREF = 9
_TAG_METHODREF = 10
_TAG_INTERFACEMETHODREF = 11
_TAG_NAMEANDTYPE = 12


def _skip_attrs(d: bytes, p: int) -> int:
    cnt = struct.unpack_from(">H", d, p)[0]
    p += 2
    for _ in range(cnt):
        ln = struct.unpack_from(">I", d, p + 2)[0]
        p += 6 + ln
    return p


def _members(d: bytes, p: int) -> tuple[int, list[tuple[int, int]]]:
    """-> (offset past the section, [(start, end) of each member_info])."""
    cnt = struct.unpack_from(">H", d, p)[0]
    p += 2
    spans = []
    for _ in range(cnt):
        start = p
        p = _skip_attrs(d, p + 6)             # access+name+desc = 6, then attributes
        spans.append((start, p))
    return p, spans


def _arg_slots(desc: str) -> int:
    args, _ret = parse_desc(desc)
    return sum(2 if a in ("J", "D") else 1 for a in args)


def _ret_slots(desc: str) -> int:
    _args, ret = parse_desc(desc)
    if ret == "V":
        return 0
    return 2 if ret in ("J", "D") else 1


def _type_slots(desc: str) -> int:
    """Operand-stack slots a field of type `desc` occupies (2 for long/double)."""
    return 2 if desc in ("J", "D") else 1


# load mnemonic -> (general opcode, <op>_0 short-form base, stack slots pushed)
_LOAD_OP = {"iload": (0x15, 0x1A, 1), "lload": (0x16, 0x1E, 2), "fload": (0x17, 0x22, 1),
            "dload": (0x18, 0x26, 2), "aload": (0x19, 0x2A, 1)}
# return mnemonic -> (opcode, stack slots popped)
_RETURN_OP = {"ireturn": (0xAC, 1), "lreturn": (0xAD, 2), "freturn": (0xAE, 1),
              "dreturn": (0xAF, 2), "areturn": (0xB0, 1)}


@dataclass
class _MethodMod:
    code_attr: bytes          # the full Code attribute (name_index + length + body)
    clear_native: bool


class ClassWriter:
    """Rebuild one `ClassFile`, optionally replacing method bodies.

    Append-only on the constant pool, verbatim everywhere we do not edit; see the
    module docstring. Call `set_method_code` for each method to fill, then
    `to_bytes`.
    """

    def __init__(self, cf: ClassFile):
        self.cf = cf
        d = cf.data

        # Section boundaries, re-walked from the parser's constant-pool end so we can
        # copy the untouched spans byte-for-byte.
        n_ifaces = struct.unpack_from(">H", d, cf.cp_end + 6)[0]
        fields_start = cf.cp_end + 8 + 2 * n_ifaces
        methods_off, _ = _members(d, fields_start)
        self._methods_off = methods_off                      # offset of methods_count
        self._methods_count = struct.unpack_from(">H", d, methods_off)[0]
        class_attrs_off, self._method_spans = _members(d, methods_off)
        self._class_attrs_off = class_attrs_off

        # Reverse indices over the existing pool, for dedup when appending.
        self._next_index = cf.cp_count
        self._new_entries: list[bytes] = []
        self._utf8: dict[str, int] = {}
        self._class: dict[str, int] = {}
        self._nat: dict[tuple[str, str], int] = {}
        self._ref: dict[tuple[int, str, str, int], int] = {}
        self._index_existing()

        self._mods: dict[int, _MethodMod] = {}               # method index -> mod

    # -- constant pool ---------------------------------------------------------
    def _index_existing(self):
        cf = self.cf
        for i in range(1, cf.cp_count):
            e = cf.cp[i]
            if e is None:
                continue
            kind = e[0]
            if kind == "utf8":
                self._utf8[cf.utf8(i)] = i
            elif kind == "class":
                self._class[cf.utf8(e[1])] = i
            elif kind == "nameandtype":
                self._nat[cf.nat(i)] = i
            elif kind in ("methodref", "fieldref", "ifmethodref"):
                owner, name, desc, _k = cf.ref(i)
                self._ref[(e[0], owner, name, desc)] = i

    def _append(self, raw: bytes) -> int:
        if self._next_index > _U2_MAX:
            raise UnnativeError("emit: constant pool overflow (> 65535 entries)")
        idx = self._next_index
        self._new_entries.append(raw)
        self._next_index += 1
        return idx

    def add_utf8(self, s: str) -> int:
        if s in self._utf8:
            return self._utf8[s]
        enc = enc_modified_utf8(s)
        if len(enc) > _U2_MAX:
            raise UnnativeError(f"emit: utf8 constant too long ({len(enc)} bytes)")
        idx = self._append(struct.pack(">BH", _TAG_UTF8, len(enc)) + enc)
        self._utf8[s] = idx
        return idx

    def add_class(self, internal_name: str) -> int:
        if internal_name in self._class:
            return self._class[internal_name]
        name_i = self.add_utf8(internal_name)
        idx = self._append(struct.pack(">BH", _TAG_CLASS, name_i))
        self._class[internal_name] = idx
        return idx

    def add_nameandtype(self, name: str, desc: str) -> int:
        key = (name, desc)
        if key in self._nat:
            return self._nat[key]
        name_i, desc_i = self.add_utf8(name), self.add_utf8(desc)
        idx = self._append(struct.pack(">BHH", _TAG_NAMEANDTYPE, name_i, desc_i))
        self._nat[key] = idx
        return idx

    def _add_ref(self, tag: int, kind: str, owner: str, name: str, desc: str) -> int:
        key = (kind, owner, name, desc)
        if key in self._ref:
            return self._ref[key]
        class_i = self.add_class(owner)
        nat_i = self.add_nameandtype(name, desc)
        idx = self._append(struct.pack(">BHH", tag, class_i, nat_i))
        self._ref[key] = idx
        return idx

    def add_methodref(self, owner: str, name: str, desc: str) -> int:
        return self._add_ref(_TAG_METHODREF, "methodref", owner, name, desc)

    def add_fieldref(self, owner: str, name: str, desc: str) -> int:
        return self._add_ref(_TAG_FIELDREF, "fieldref", owner, name, desc)

    def add_interfacemethodref(self, owner: str, name: str, desc: str) -> int:
        return self._add_ref(_TAG_INTERFACEMETHODREF, "ifmethodref", owner, name, desc)

    def add_string(self, s: str) -> int:
        utf8_i = self.add_utf8(s)
        return self._append(struct.pack(">BH", _TAG_STRING, utf8_i))

    def add_int(self, value: int) -> int:
        return self._append(struct.pack(">Bi", _TAG_INTEGER, value))

    # -- bytecode encoding -----------------------------------------------------
    def _encode(self, bytecode: list[dict]) -> tuple[bytes, int]:
        """-> (code bytes, max_stack) for a straight-line op list.

        Only the ops `stackify` emits are handled; an unknown one raises so we never
        ship an untested encoding. max_stack is exact for branchless code (a linear
        high-water mark of the operand stack)."""
        out = bytearray()
        depth = max_stack = 0

        def push_pop(pop: int, push: int):
            nonlocal depth, max_stack
            if depth < pop:
                raise UnnativeError("emit: operand stack underflow")
            depth = depth - pop + push
            max_stack = max(max_stack, depth)

        for op in bytecode:
            bc = op["bc"]
            if bc == "dup":
                out.append(0x59)
                push_pop(0, 1)
            elif bc == "new":
                out += struct.pack(">BH", 0xBB, self.add_class(op["owner"]))
                push_pop(0, 1)
            elif bc == "athrow":
                out.append(0xBF)
                push_pop(1, 0)
            elif bc == "checkcast":
                out += struct.pack(">BH", 0xC0, self.add_class(op["owner"]))
                push_pop(1, 1)                            # narrows the ref's type
            elif bc == "return":
                out.append(0xB1)
            elif bc in _LOAD_OP:
                gen, short, slots = _LOAD_OP[bc]
                n = op["slot"]
                if n <= 3:
                    out.append(short + n)
                elif n < 256:
                    out += bytes([gen, n])
                else:
                    raise UnnativeError(f"emit: local slot {n} needs wide form (unsupported)")
                push_pop(0, slots)
            elif bc in _RETURN_OP:
                opcode, slots = _RETURN_OP[bc]
                out.append(opcode)
                push_pop(slots, 0)
            elif bc == "ldc":
                ctype = op.get("ctype", "String")
                if ctype == "String":
                    out += struct.pack(">BH", 0x13, self.add_string(op["value"]))  # ldc_w
                elif ctype == "I":
                    v = op["value"]
                    if -1 <= v <= 5:
                        out.append(0x03 + v)                 # iconst_m1..iconst_5
                    elif -128 <= v <= 127:
                        out += bytes([0x10, v & 0xFF])       # bipush
                    elif -32768 <= v <= 32767:
                        out += struct.pack(">Bh", 0x11, v)   # sipush
                    else:
                        out += struct.pack(">BH", 0x13, self.add_int(v))  # ldc_w int
                else:
                    raise UnnativeError(f"emit: unsupported ldc type {ctype!r}")
                push_pop(0, 1)
            elif bc == "astore":
                n = op["slot"]
                if n <= 3:
                    out.append(0x4B + n)                     # astore_0..astore_3
                elif n < 256:
                    out += bytes([0x3A, n])
                else:
                    raise UnnativeError(f"emit: local slot {n} needs wide form (unsupported)")
                push_pop(1, 0)
            elif bc == "getstatic":
                out += struct.pack(">BH", 0xB2, self.add_fieldref(op["owner"], op["name"], op["desc"]))
                push_pop(0, _type_slots(op["desc"]))
            elif bc == "getfield":
                out += struct.pack(">BH", 0xB4, self.add_fieldref(op["owner"], op["name"], op["desc"]))
                push_pop(1, _type_slots(op["desc"]))
            elif bc == "putstatic":
                out += struct.pack(">BH", 0xB3, self.add_fieldref(op["owner"], op["name"], op["desc"]))
                push_pop(_type_slots(op["desc"]), 0)
            elif bc == "putfield":
                out += struct.pack(">BH", 0xB5, self.add_fieldref(op["owner"], op["name"], op["desc"]))
                push_pop(1 + _type_slots(op["desc"]), 0)
            elif bc == "invokespecial":
                out += struct.pack(">BH", 0xB7, self.add_methodref(op["owner"], op["name"], op["desc"]))
                push_pop(1 + _arg_slots(op["desc"]), _ret_slots(op["desc"]))
            elif bc == "invokevirtual":
                out += struct.pack(">BH", 0xB6, self.add_methodref(op["owner"], op["name"], op["desc"]))
                push_pop(1 + _arg_slots(op["desc"]), _ret_slots(op["desc"]))
            elif bc == "invokestatic":
                out += struct.pack(">BH", 0xB8, self.add_methodref(op["owner"], op["name"], op["desc"]))
                push_pop(_arg_slots(op["desc"]), _ret_slots(op["desc"]))
            elif bc == "invokeinterface":
                ref = self.add_interfacemethodref(op["owner"], op["name"], op["desc"])
                count = 1 + _arg_slots(op["desc"])
                out += struct.pack(">BHBB", 0xB9, ref, count, 0)
                push_pop(count, _ret_slots(op["desc"]))
            else:
                raise UnnativeError(f"emit: unsupported bytecode op {bc!r}")
        return bytes(out), max_stack

    # -- method replacement ----------------------------------------------------
    def _find_method(self, name: str, desc: str) -> int:
        for i, m in enumerate(self.cf.methods):
            if m["name"] == name and m["desc"] == desc:
                return i
        raise UnnativeError(f"emit: no method {name}{desc} in {self.cf.this}")

    def set_method_code(self, name: str, desc: str, bytecode: list[dict],
                        *, clear_native: bool = True):
        """Attach a Code attribute (built from `bytecode`) to method name+desc.

        Resolves every constant-pool entry the body needs now, so `to_bytes` is pure
        assembly. Refuses a method that already has a Code attribute: this slice only
        fills empty `native` stubs, and overwriting real code would be a silent
        behaviour change."""
        idx = self._find_method(name, desc)
        m = self.cf.methods[idx]
        if "Code" in m["attrs"]:
            raise UnnativeError(f"emit: {name}{desc} already has a Code attribute")

        code, max_stack = self._encode(bytecode)
        is_static = bool(m["access"] & ACC_STATIC)
        arg_locals = (0 if is_static else 1) + _arg_slots(desc)
        # A spill slot (multi-use construct stored to an extra local) can exceed the
        # args: the verifier needs max_locals to cover every slot the code touches.
        used = max((op["slot"] + (2 if op.get("bc") in ("lload", "dload", "lstore", "dstore")
                                  else 1) for op in bytecode if "slot" in op), default=0)
        max_locals = max(arg_locals, used)
        body = (struct.pack(">HHI", max_stack, max_locals, len(code)) + code
                + struct.pack(">HH", 0, 0))          # empty exception table, no sub-attrs
        code_attr = struct.pack(">HI", self.add_utf8("Code"), len(body)) + body
        self._mods[idx] = _MethodMod(code_attr=code_attr, clear_native=clear_native)

    # -- serialisation ---------------------------------------------------------
    def _method_bytes(self, idx: int) -> bytes:
        d = self.cf.data
        start, end = self._method_spans[idx]
        mod = self._mods.get(idx)
        if mod is None:
            return d[start:end]                       # untouched: copy verbatim
        access, name_i, desc_i = struct.unpack_from(">HHH", d, start)
        if mod.clear_native:
            access &= ~ACC_NATIVE
        acount = struct.unpack_from(">H", d, start + 6)[0]
        existing_attrs = d[start + 8:end]             # raw bytes of the current attrs
        return (struct.pack(">HHHH", access, name_i, desc_i, acount + 1)
                + existing_attrs + mod.code_attr)

    def to_bytes(self) -> bytes:
        cf, d = self.cf, self.cf.data
        out = bytearray()
        out += d[0:8]                                             # magic + version
        out += struct.pack(">H", self._next_index)               # cp_count'
        out += d[10:cf.cp_end]                                    # original pool
        for e in self._new_entries:                              # appended pool
            out += e
        out += d[cf.cp_end:self._methods_off]                    # access … fields
        out += struct.pack(">H", self._methods_count)            # methods_count
        for i in range(self._methods_count):
            out += self._method_bytes(i)
        out += d[self._class_attrs_off:]                         # class attributes
        return bytes(out)
