"""ELF64 loader: the memory-map plan and the dynamic symbols of a native library.

linux-x86_64 is the priority analysis triple (plan.md Phase 2), and this covers
both x86-64 and aarch64 (both 64-bit little-endian). It gives the emulator two
things:

  * `segments` — the PT_LOAD list (vaddr, file bytes, zero-filled bss tail, rwx
    flags) to map into Unicorn;
  * `symbols` / `exports` — the dynamic symbol table, so Phase 3 can find
    `JNI_OnLoad` and the exported `Java_*` stubs.

Protected libraries are frequently stripped of section headers, so symbols are
read from `.dynsym` when section headers survive and otherwise from `PT_DYNAMIC`
(`DT_SYMTAB`/`DT_STRTAB`, sized via `DT_GNU_HASH` or `DT_HASH`). PE and Mach-O get
their own loaders later; this module raises on non-ELF or 32-bit/big-endian input
rather than guessing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..errors import UnnativeError

# e_machine
_EM = {0x3E: "x86_64", 0xB7: "aarch64"}

# p_type / p_flags
PT_LOAD = 1
PT_DYNAMIC = 2
PF_X, PF_W, PF_R = 1, 2, 4

# sh_type
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_DYNSYM = 11

# st_info
STT_FUNC = 2
STT_OBJECT = 1
STB_GLOBAL = 1
STB_WEAK = 2
SHN_UNDEF = 0

# d_tag
DT_NULL = 0
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_STRSZ = 10
DT_SYMENT = 11
DT_PLTRELSZ = 2
DT_JMPREL = 23
DT_GNU_HASH = 0x6FFFFEF5

# x86-64 relocation types we care about.
R_X86_64_64 = 1
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_IRELATIVE = 37

_SYM_SIZE = 24  # Elf64_Sym
_RELA_SIZE = 24  # Elf64_Rela


@dataclass
class Segment:
    vaddr: int
    memsz: int          # mapped size (>= filesz; tail is zero-filled bss)
    filesz: int
    offset: int         # file offset of the file-backed bytes
    read: bool
    write: bool
    execute: bool
    data: bytes         # exactly filesz bytes


@dataclass
class Symbol:
    name: str
    value: int          # vaddr (0 if undefined)
    size: int
    is_func: bool
    bind_global: bool
    defined: bool


@dataclass
class Reloc:
    offset: int         # vaddr to patch
    type: int           # R_X86_64_*
    sym: int            # dynsym index (0 if none)
    addend: int


def _flags(p_flags: int) -> tuple[bool, bool, bool]:
    return bool(p_flags & PF_R), bool(p_flags & PF_W), bool(p_flags & PF_X)


class ElfImage:
    def __init__(self, data: bytes):
        self.data = data
        if data[:4] != b"\x7fELF":
            raise UnnativeError(f"not an ELF file (magic {data[:4].hex()})")
        ei_class, ei_data = data[4], data[5]
        if ei_class != 2:
            raise UnnativeError("only ELF64 is supported (32-bit input)")
        if ei_data != 1:
            raise UnnativeError("only little-endian ELF is supported")
        try:
            self._parse()
        except struct.error as ex:
            raise UnnativeError(f"truncated or malformed ELF ({ex})") from ex

    # -- construction --
    @classmethod
    def from_path(cls, path: str) -> "ElfImage":
        with open(path, "rb") as f:
            return cls(f.read())

    def _parse(self):
        d = self.data
        (self.e_type, e_machine, _ver, self.e_entry, e_phoff, e_shoff, _flg,
         _ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = \
            struct.unpack_from("<HHIQQQIHHHHHH", d, 16)
        self.arch = _EM.get(e_machine, f"machine:{e_machine:#x}")

        self.segments: list[Segment] = []
        self._dynamic: list[tuple[int, int]] = []
        for k in range(e_phnum):
            base = e_phoff + k * e_phentsize
            p_type, p_flags, p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, _al = \
                struct.unpack_from("<IIQQQQQQ", d, base)
            if p_type == PT_LOAD:
                r, w, x = _flags(p_flags)
                self.segments.append(Segment(
                    vaddr=p_vaddr, memsz=p_memsz, filesz=p_filesz, offset=p_offset,
                    read=r, write=w, execute=x,
                    data=d[p_offset:p_offset + p_filesz]))
            elif p_type == PT_DYNAMIC:
                self._read_dynamic(p_offset, p_filesz)

        # Section headers (present in unstripped libs).
        self._sh = []
        if e_shoff and e_shnum:
            for k in range(e_shnum):
                base = e_shoff + k * e_shentsize
                sh = struct.unpack_from("<IIQQQQIIQQ", d, base)   # 10 fields
                self._sh.append(sh)
            self._shstrndx = e_shstrndx
        else:
            self._shstrndx = 0

        self.symbols: list[Symbol] = self._read_symbols()
        self.exports: dict[str, Symbol] = {
            s.name: s for s in self.symbols
            if s.defined and s.is_func and s.bind_global and s.name}
        self.relocations: list[Reloc] = self._read_relocations()

    def _read_dynamic(self, offset: int, size: int):
        d = self.data
        n = size // 16
        for k in range(n):
            tag, val = struct.unpack_from("<qQ", d, offset + k * 16)
            self._dynamic.append((tag, val))
            if tag == DT_NULL:
                break

    def _dyn(self, tag: int) -> int | None:
        for t, v in self._dynamic:
            if t == tag:
                return v
        return None

    # -- vaddr/file translation --
    def vaddr_to_off(self, vaddr: int) -> int | None:
        for seg in self.segments:
            if seg.offset is not None and seg.vaddr <= vaddr < seg.vaddr + seg.filesz:
                return seg.offset + (vaddr - seg.vaddr)
        return None

    def _cstr(self, off: int) -> str:
        end = self.data.find(b"\x00", off)
        return self.data[off:end].decode("latin-1") if end >= 0 else ""

    # -- symbols --
    def _read_symbols(self) -> list[Symbol]:
        syms = self._symbols_from_sections()
        if syms is not None:
            return syms
        syms = self._symbols_from_dynamic()
        return syms if syms is not None else []

    def _parse_symtab(self, sym_off: int, count: int, str_off: int,
                      str_is_file_off: bool) -> list[Symbol]:
        out = []
        d = self.data
        for i in range(count):
            base = sym_off + i * _SYM_SIZE
            st_name, st_info, _other, st_shndx, st_value, st_size = \
                struct.unpack_from("<IBBHQQ", d, base)
            name = self._cstr(str_off + st_name) if str_is_file_off else \
                self._cstr((self.vaddr_to_off(str_off) or 0) + st_name)
            out.append(Symbol(
                name=name, value=st_value, size=st_size,
                is_func=(st_info & 0xF) == STT_FUNC,
                bind_global=(st_info >> 4) in (STB_GLOBAL, STB_WEAK),
                defined=st_shndx != SHN_UNDEF))
        return out

    def _symbols_from_sections(self) -> list[Symbol] | None:
        for sh in self._sh:
            (_name, sh_type, _fl, _addr, sh_offset, sh_size, sh_link, _info,
             _align, sh_entsize) = sh
            if sh_type == SHT_DYNSYM and sh_entsize == _SYM_SIZE and sh_link < len(self._sh):
                str_off = self._sh[sh_link][4]       # linked STRTAB sh_offset
                count = sh_size // _SYM_SIZE
                return self._parse_symtab(sh_offset, count, str_off,
                                          str_is_file_off=True)
        return None

    def _symbols_from_dynamic(self) -> list[Symbol] | None:
        sym_v = self._dyn(DT_SYMTAB)
        str_v = self._dyn(DT_STRTAB)
        if sym_v is None or str_v is None:
            return None
        sym_off = self.vaddr_to_off(sym_v)
        if sym_off is None:
            return None
        count = self._count_dynsyms()
        if count is None:
            return None
        return self._parse_symtab(sym_off, count, str_v, str_is_file_off=False)

    def _count_dynsyms(self) -> int | None:
        """Symbol count for a stripped lib: prefer DT_GNU_HASH, fall back to DT_HASH."""
        gnu = self._dyn(DT_GNU_HASH)
        if gnu is not None:
            off = self.vaddr_to_off(gnu)
            if off is not None:
                return self._count_via_gnu_hash(off)
        h = self._dyn(DT_HASH)
        if h is not None:
            off = self.vaddr_to_off(h)
            if off is not None:
                # DT_HASH: nbucket, nchain, ...; nchain == number of symbols.
                _nbucket, nchain = struct.unpack_from("<II", self.data, off)
                return nchain
        return None

    def _count_via_gnu_hash(self, off: int) -> int | None:
        d = self.data
        nbuckets, symoffset, bloom_size, _bloom_shift = struct.unpack_from("<IIII", d, off)
        buckets_off = off + 16 + bloom_size * 8
        max_bucket = 0
        for i in range(nbuckets):
            b = struct.unpack_from("<I", d, buckets_off + i * 4)[0]
            if b > max_bucket:
                max_bucket = b
        if max_bucket < symoffset:
            return symoffset
        # Walk the chain from the highest bucket's symbol index to its end (the
        # chain word with its low bit set), giving the last symbol index + 1.
        chain_off = buckets_off + nbuckets * 4
        idx = max_bucket
        while True:
            word = struct.unpack_from("<I", d, chain_off + (idx - symoffset) * 4)[0]
            if word & 1:
                return idx + 1
            idx += 1
            if idx - symoffset > (1 << 24):          # runaway guard
                raise UnnativeError("GNU hash chain did not terminate")

    # -- relocations (x86-64 RELA; the emulator applies them at the load bias) --
    def _read_relocations(self) -> list[Reloc]:
        out: list[Reloc] = []
        for tbl, sz in ((DT_RELA, DT_RELASZ), (DT_JMPREL, DT_PLTRELSZ)):
            v = self._dyn(tbl)
            size = self._dyn(sz)
            if v is None or not size:
                continue
            off = self.vaddr_to_off(v)
            if off is None:
                continue
            for k in range(size // _RELA_SIZE):
                r_offset, r_info, r_addend = struct.unpack_from(
                    "<QQq", self.data, off + k * _RELA_SIZE)
                out.append(Reloc(offset=r_offset, type=r_info & 0xFFFFFFFF,
                                 sym=r_info >> 32, addend=r_addend))
        return out

    # -- queries the binder uses --
    def symbol(self, name: str) -> Symbol | None:
        return self.exports.get(name) or next(
            (s for s in self.symbols if s.name == name), None)

    def jni_onload(self) -> int | None:
        s = self.symbol("JNI_OnLoad")
        return s.value if s and s.defined else None

    def java_exports(self) -> list[Symbol]:
        return sorted((s for s in self.exports.values() if s.name.startswith("Java_")),
                      key=lambda s: s.name)
