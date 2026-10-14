"""Static control-flow recovery for a translated native method (plan.md §6).

The linear idiom lifter sees the JNI-call stream in execution order but not the
*shape* it ran through — the branches, the loops, which block each call sits in.
This module recovers that shape statically: it disassembles the native function
with capstone and walks its direct control flow into basic blocks and edges.

It is a recursive-descent recovery: starting at the function entry it follows
fall-through, direct `jmp`/`jcc` targets and returns, stopping at `ret`, at an
indirect jump it can't resolve statically (a jump table / flattening dispatcher —
recorded as an open exit, not guessed), and at a budget. The first real sample's
methods use plain direct control flow, so this covers them; a method that turns out
to be flattened shows up as a pile of indirect exits rather than a wrong CFG.

capstone is optional: when it is missing the lift stage keeps producing the linear
JIR and simply skips this enrichment. Nothing else in the package imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import capstone
    from capstone import x86 as cs_x86
    CAPSTONE_AVAILABLE = True
except ImportError:                                  # pragma: no cover - env-gated
    capstone = None
    cs_x86 = None
    CAPSTONE_AVAILABLE = False

# How far from the entry a direct branch target is still considered "inside this
# function" (vs. a tail call into another one), and a hard cap on instructions so a
# pathological decode can't run away. Both are deliberately generous and revisable.
_WINDOW = 0x10000
_BUDGET = 50_000


@dataclass
class Insn:
    addr: int
    size: int
    mnemonic: str
    term: str | None = None          # ret | jmp | jcc | ijmp | None (falls through)
    target: int | None = None        # direct branch target, if any
    is_call: bool = False


@dataclass
class Block:
    start: int
    end: int                          # address just past the block's last instruction
    n_insns: int
    terminator: str                   # ret | jmp | jcc | ijmp | fallthrough
    succs: list[int] = field(default_factory=list)
    calls: int = 0

    def to_json(self) -> dict:
        return {"start": self.start, "end": self.end, "insns": self.n_insns,
                "terminator": self.terminator, "succs": self.succs, "calls": self.calls}


@dataclass
class Cfg:
    entry: int
    blocks: dict[int, Block]
    n_insns: int = 0
    calls: int = 0
    indirect_exits: int = 0
    back_edges: int = 0
    truncated: bool = False

    @property
    def has_loops(self) -> bool:
        return self.back_edges > 0

    def to_json(self) -> dict:
        return {
            "entry": self.entry,
            "blocks": {hex(a): b.to_json() for a, b in sorted(self.blocks.items())},
            "stats": {
                "blocks": len(self.blocks),
                "insns": self.n_insns,
                "edges": sum(len(b.succs) for b in self.blocks.values()),
                "calls": self.calls,
                "indirect_exits": self.indirect_exits,
                "back_edges": self.back_edges,
                "has_loops": self.has_loops,
                "truncated": self.truncated,
            },
        }


def elf_reader(elf):
    """A `read(vaddr, n) -> bytes` over a loaded ElfImage, for build_cfg."""
    def read(vaddr: int, n: int) -> bytes:
        off = elf.vaddr_to_off(vaddr)
        if off is None:
            return b""
        return bytes(elf.data[off:off + n])
    return read


def bytes_reader(base: int, blob: bytes):
    """A `read` over an in-memory blob based at `base` vaddr (used by tests)."""
    def read(vaddr: int, n: int) -> bytes:
        start = vaddr - base
        if start < 0 or start >= len(blob):
            return b""
        return blob[start:start + n]
    return read


def _decode_one(md, reader, addr: int) -> Insn | None:
    chunk = reader(addr, 16)
    if not chunk:
        return None
    ins = next(md.disasm(chunk, addr), None)
    if ins is None:
        return None
    rec = Insn(addr=addr, size=ins.size, mnemonic=ins.mnemonic)
    groups = set(ins.groups)
    is_jump = cs_x86.X86_GRP_JUMP in groups
    is_call = cs_x86.X86_GRP_CALL in groups
    is_ret = cs_x86.X86_GRP_RET in groups
    rec.is_call = is_call
    if is_ret:
        rec.term = "ret"
    elif is_jump:
        op = ins.operands[0] if ins.operands else None
        direct = op is not None and op.type == cs_x86.X86_OP_IMM
        rec.target = op.imm if direct else None
        if ins.mnemonic == "jmp":
            rec.term = "jmp" if direct else "ijmp"
        else:
            rec.term = "jcc" if direct else "ijmp"
    return rec


def build_cfg(reader, entry: int, *, window: int = _WINDOW, budget: int = _BUDGET) -> Cfg:
    """Recover the CFG of the function at `entry`, reading bytes through `reader`."""
    if not CAPSTONE_AVAILABLE:
        raise RuntimeError("build_cfg needs capstone")
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True

    def in_window(t: int) -> bool:
        return entry - 0x200 <= t <= entry + window

    decoded: dict[int, Insn] = {}
    leaders: set[int] = {entry}
    worklist = [entry]
    truncated = False

    while worklist:
        addr = worklist.pop()
        while True:
            if addr in decoded or len(decoded) >= budget:
                truncated = truncated or len(decoded) >= budget
                break
            ins = _decode_one(md, reader, addr)
            if ins is None:
                truncated = True
                break
            decoded[addr] = ins
            nxt = addr + ins.size
            if ins.term == "ret" or ins.term == "ijmp":
                break
            if ins.term == "jmp":
                if ins.target is not None and in_window(ins.target):
                    leaders.add(ins.target)
                    worklist.append(ins.target)
                else:
                    ins.term = "ijmp"          # external/unresolvable target: open exit
                break
            if ins.term == "jcc":
                leaders.add(nxt)
                if ins.target is not None and in_window(ins.target):
                    leaders.add(ins.target)
                    worklist.append(ins.target)
                worklist.append(nxt)
                break
            # plain instruction (incl. call): fall through to the next one
            addr = nxt

    return _assemble_blocks(decoded, leaders, entry, truncated)


def _assemble_blocks(decoded: dict[int, Insn], leaders: set[int], entry: int,
                     truncated: bool) -> Cfg:
    addrs = sorted(decoded)
    cfg = Cfg(entry=entry, blocks={}, n_insns=len(decoded), truncated=truncated)

    cur: Block | None = None
    for i, a in enumerate(addrs):
        ins = decoded[a]
        if a in leaders or cur is None:
            cur = Block(start=a, end=a, n_insns=0, terminator="fallthrough")
            cfg.blocks[a] = cur
        cur.n_insns += 1
        cur.end = a + ins.size
        if ins.is_call:
            cur.calls += 1
            cfg.calls += 1
        nxt = a + ins.size
        next_is_leader = (i + 1 < len(addrs) and addrs[i + 1] in leaders)
        if ins.term in ("ret", "jmp", "jcc", "ijmp"):
            cur.terminator = ins.term
            if ins.term == "jmp" and ins.target is not None:
                cur.succs = [ins.target]
            elif ins.term == "jcc":
                cur.succs = [t for t in (ins.target, nxt) if t is not None]
            elif ins.term == "ijmp":
                cfg.indirect_exits += 1
                cur.succs = []
            else:                               # ret
                cur.succs = []
            cur = None
        elif next_is_leader:
            cur.terminator = "fallthrough"
            cur.succs = [nxt]
            cur = None

    # keep only edges whose target is a real block, and count back-edges (loops)
    for b in cfg.blocks.values():
        b.succs = [s for s in b.succs if s in cfg.blocks]
    cfg.back_edges = _count_back_edges(cfg)
    return cfg


def _count_back_edges(cfg: Cfg) -> int:
    """Back-edges via a colour DFS from the entry: an edge to a grey (on-stack)
    block is a loop back-edge."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {a: WHITE for a in cfg.blocks}
    back = 0
    if cfg.entry not in cfg.blocks:
        return 0
    stack = [(cfg.entry, iter(cfg.blocks[cfg.entry].succs))]
    color[cfg.entry] = GREY
    while stack:
        node, it = stack[-1]
        advanced = False
        for s in it:
            if color.get(s) == GREY:
                back += 1
            elif color.get(s) == WHITE:
                color[s] = GREY
                stack.append((s, iter(cfg.blocks[s].succs)))
                advanced = True
                break
        if not advanced:
            color[node] = BLACK
            stack.pop()
    return back
