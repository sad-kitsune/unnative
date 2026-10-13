"""Opcode table and instruction decoder.

Vendored from unseed §3; see PROVENANCE.md. Copied, not imported. The table is
cross-validated against `javap` by unseed's own test suite; the shift-opcode
ordering comment below records a subtlety that is easy to get silently wrong.

Operand formats:
  ''     none               'b'  int8            'B'    uint8
  'bb'   uint8, int8        's'  int16           'u2'   constant-pool index
  'o2'   int16 branch       'i4' int32 branch    'u2bb' invokeinterface/dynamic
  'u2b'  multianewarray     'switch' | 'wide'    (variable length)
"""

from __future__ import annotations

import re
import struct

from ..errors import UnnativeError

# struct layout of the operand bytes, for the fixed-length formats.
_FIXED_OPERANDS = {'': '>', 'b': '>b', 'B': '>B', 'bb': '>Bb', 's': '>h', 'u2': '>H',
                   'o2': '>h', 'i4': '>i', 'u2bb': '>HBB', 'u2b': '>HB'}

# Minimum total instruction length per format, opcode byte included. 'switch' and
# 'wide' are variable-length, so this records only the bytes their length
# calculation must be able to read.
FMT_MIN_LEN = {fmt: 1 + struct.calcsize(layout) for fmt, layout in _FIXED_OPERANDS.items()}
FMT_MIN_LEN.update({'switch': 5, 'wide': 2})

OPS: dict = {}                      # opcode byte -> (mnemonic, operand format)


def _op(code: int, name: str, fmt: str = '') -> None:
    assert code not in OPS, f'duplicate opcode {code:#x}'
    OPS[code] = (name, fmt)


def _ops(base: int, names, fmt: str = '', stride: int = 1) -> None:
    """Register consecutive opcodes `base`, `base+stride`, ... under `names`."""
    for i, name in enumerate(names):
        _op(base + i * stride, name, fmt)


_op(0x00, 'nop')
_op(0x01, 'aconst_null')
_ops(0x02, ['iconst_m1'] + [f'iconst_{v}' for v in range(6)])
_ops(0x09, ['lconst_0', 'lconst_1'])
_ops(0x0B, ['fconst_0', 'fconst_1', 'fconst_2'])
_ops(0x0E, ['dconst_0', 'dconst_1'])
_op(0x10, 'bipush', 'b')
_op(0x11, 'sipush', 's')
_op(0x12, 'ldc', 'B')
_op(0x13, 'ldc_w', 'u2')
_op(0x14, 'ldc2_w', 'u2')
_ops(0x15, [f'{t}load' for t in 'ilfda'], 'B')
for _t, _base in zip('ilfda', (0x1A, 0x1E, 0x22, 0x26, 0x2A)):
    _ops(_base, [f'{_t}load_{i}' for i in range(4)])
_ops(0x2E, [f'{t}aload' for t in ('i', 'l', 'f', 'd', 'a', 'b', 'c', 's')])
_ops(0x36, [f'{t}store' for t in 'ilfda'], 'B')
for _t, _base in zip('ilfda', (0x3B, 0x3F, 0x43, 0x47, 0x4B)):
    _ops(_base, [f'{_t}store_{i}' for i in range(4)])
_ops(0x4F, [f'{t}astore' for t in ('i', 'l', 'f', 'd', 'a', 'b', 'c', 's')])
_ops(0x57, ['pop', 'pop2', 'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2', 'swap'])
# add/sub/mul/div/rem/neg: the four types are interleaved, four opcodes apart.
for _off, _t in enumerate('ilfd'):
    _ops(0x60 + _off, [f'{_t}{o}' for o in ('add', 'sub', 'mul', 'div', 'rem', 'neg')],
         stride=4)
# Shifts are grouped by OPERATION with the int/long variants adjacent
# (ishl, lshl, ishr, lshr, iushr, lushr), NOT by type. Reading this range as
# "all four int shifts then all four long shifts" is an easy and silent slip:
# it misnames four of six opcodes, and because ishr/lshl differ in both signedness
# and operand category the interpreter then pops the wrong stack width.
# Verified against javap and against javac's own codegen for >> / >>> / <<.
_ops(0x78, ['ishl', 'lshl', 'ishr', 'lshr', 'iushr', 'lushr',
            'iand', 'land', 'ior', 'lor', 'ixor', 'lxor'])
_op(0x84, 'iinc', 'bb')
_ops(0x85, ['i2l', 'i2f', 'i2d', 'l2i', 'l2f', 'l2d', 'f2i', 'f2l',
            'f2d', 'd2i', 'd2l', 'd2f', 'i2b', 'i2c', 'i2s'])
_ops(0x94, ['lcmp', 'fcmpl', 'fcmpg', 'dcmpl', 'dcmpg'])
_ops(0x99, ['ifeq', 'ifne', 'iflt', 'ifge', 'ifgt', 'ifle',
            'if_icmpeq', 'if_icmpne', 'if_icmplt', 'if_icmpge', 'if_icmpgt', 'if_icmple',
            'if_acmpeq', 'if_acmpne', 'goto', 'jsr'], 'o2')
_op(0xA9, 'ret', 'B')
_op(0xAA, 'tableswitch', 'switch')
_op(0xAB, 'lookupswitch', 'switch')
_ops(0xAC, ['ireturn', 'lreturn', 'freturn', 'dreturn', 'areturn', 'return'])
_ops(0xB2, ['getstatic', 'putstatic', 'getfield', 'putfield',
            'invokevirtual', 'invokespecial', 'invokestatic'], 'u2')
_op(0xB9, 'invokeinterface', 'u2bb')
_op(0xBA, 'invokedynamic', 'u2bb')
_op(0xBB, 'new', 'u2')
_op(0xBC, 'newarray', 'B')
_op(0xBD, 'anewarray', 'u2')
_op(0xBE, 'arraylength')
_op(0xBF, 'athrow')
_op(0xC0, 'checkcast', 'u2')
_op(0xC1, 'instanceof', 'u2')
_op(0xC2, 'monitorenter')
_op(0xC3, 'monitorexit')
_op(0xC4, 'wide', 'wide')
_op(0xC5, 'multianewarray', 'u2b')
_op(0xC6, 'ifnull', 'o2')
_op(0xC7, 'ifnonnull', 'o2')
_op(0xC8, 'goto_w', 'i4')
_op(0xC9, 'jsr_w', 'i4')

BRANCH_OPS = {n for n, f in OPS.values() if f in ('o2', 'i4', 'switch')}

INVOKE_OPS = ('invokevirtual', 'invokespecial', 'invokestatic', 'invokeinterface')
GOTO_OPS = ('goto', 'goto_w', 'jsr', 'jsr_w')
TERMINATORS = frozenset(('return', 'ireturn', 'lreturn', 'freturn', 'dreturn',
                         'areturn', 'athrow'))


def names_matching(pattern: str):
    """Mnemonics whose name fully matches `pattern` — handy for building handler sets."""
    rx = re.compile(pattern)
    return [n for n, _f in OPS.values() if rx.fullmatch(n)]


class Insn:
    __slots__ = ('off', 'op', 'name', 'fmt', 'operands', 'length', 'target')

    def __init__(self, off, op, name, fmt, operands, length, target=None):
        self.off = off
        self.op = op
        self.name = name
        self.fmt = fmt
        self.operands = operands      # tuple of raw ints
        self.length = length
        self.target = target          # list of absolute branch targets, if any

    def __repr__(self):
        return f'{self.off}:{self.name}{self.operands if self.operands else ""}'


def _switch_len(code: bytes, start: int) -> tuple:
    """Return (total_length, [targets]) for tableswitch/lookupswitch at `start`."""
    pad = (4 - ((start + 1) % 4)) % 4
    p = start + 1 + pad
    default = struct.unpack_from('>i', code, p)[0]
    targets = [start + default]
    if code[start] == 0xAA:                       # tableswitch
        low, high = struct.unpack_from('>ii', code, p + 4)
        if high < low or high - low > 1 << 20:
            raise UnnativeError(f'implausible tableswitch range {low}..{high} at {start}')
        for k in range(high - low + 1):
            targets.append(start + struct.unpack_from('>i', code, p + 12 + 4 * k)[0])
        return p + 12 + 4 * (high - low + 1) - start, targets
    npairs = struct.unpack_from('>i', code, p + 4)[0]
    if npairs < 0 or npairs > 1 << 20:
        raise UnnativeError(f'implausible lookupswitch npairs {npairs} at {start}')
    for k in range(npairs):
        targets.append(start + struct.unpack_from('>i', code, p + 8 + 8 * k + 4)[0])
    return p + 8 + 8 * npairs - start, targets


def _read_operands(fmt: str, code: bytes, i: int) -> tuple:
    """-> (operands, length, branch_targets_or_None) for the instruction at `i`."""
    if fmt in _FIXED_OPERANDS:
        layout = _FIXED_OPERANDS[fmt]
        operands = struct.unpack_from(layout, code, i + 1) if len(layout) > 1 else ()
        length = 1 + struct.calcsize(layout)
        targets = [i + operands[0]] if fmt in ('o2', 'i4') else None
        return operands, length, targets
    if fmt == 'switch':
        length, targets = _switch_len(code, i)
        return (), length, targets
    if fmt == 'wide':
        op2 = code[i + 1]
        if op2 not in OPS:
            raise UnnativeError(f'wide: unknown opcode {op2:#02x} at {i}')
        return (op2,), (6 if op2 == 0x84 else 4), None     # wide iinc has an extra int16
    raise UnnativeError(f'unhandled operand format {fmt!r}')


def decode(code: bytes) -> list:
    """Linear decode of a Code attribute body.

    Raises UnnativeError on any unknown opcode, any unhandled operand format, and
    any instruction truncated by the end of the Code body. Truncation is checked
    before the operands are read, and the read itself is fenced, so a malformed
    classfile can never escape as a raw struct.error — that would break the
    contract every caller relies on, namely that UnnativeError is the only way
    this decoder fails.
    """
    insns = []
    i, n = 0, len(code)
    while i < n:
        op = code[i]
        if op not in OPS:
            raise UnnativeError(f'unknown opcode {op:#02x} at bci {i}')
        name, fmt = OPS[op]
        need = FMT_MIN_LEN.get(fmt)
        if need is None:
            raise UnnativeError(f'unhandled operand format {fmt!r} at bci {i}')
        if i + need > n:
            raise UnnativeError(f'{name} at bci {i} runs past end of code ({n})')
        try:
            operands, length, targets = _read_operands(fmt, code, i)
        except (struct.error, IndexError) as ex:
            raise UnnativeError(f'{name} at bci {i} runs past end of code ({n}) '
                                f'while reading its operands') from ex
        if i + length > n:
            raise UnnativeError(f'{name} at bci {i} runs past end of code ({n})')
        insns.append(Insn(i, op, name, fmt, operands, length, targets))
        i += length
    return insns
