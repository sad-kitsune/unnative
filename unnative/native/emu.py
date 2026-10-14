"""Unicorn CPU wrapper: run a native library against the fake JNIEnv.

This is the only module that needs Unicorn, and it imports it lazily so the rest of
the package stays standard-library-only. It maps the ELF's PT_LOAD segments at a
load bias, maps the JNIEnv/JavaVM structures and their function tables from
`jnienv.Layout`, and leaves the per-slot trampoline region *unmapped* so that when
the library does `(*env)->Fn(...)` the fetch of the sentinel address faults into a
hook, which decodes the SysV arguments, calls `JNIEnv.dispatch`, places the return
value, and resumes at the caller's return address.

Scope note (plan.md Phase 5a): x86-64 first — argument/return handling here is
System V AMD64. aarch64 (the other target arch) reuses the mapping and hook shape
with a different register file and is a follow-up; non-x86-64 input raises rather
than silently mis-decoding arguments.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..classfile import parse_desc
from ..errors import UnnativeError
from .jnienv import PTR, JNIEnv, Layout, TraceEvent
from .jvm_model import JvmModel
from .loader_elf import ElfImage

PAGE = 0x1000
LOAD_BIAS = 0x40_0000
STACK_BASE = 0x7000_0000
STACK_SIZE = 0x10_0000
TLS_BASE = 0x6000_0000
TLS_SIZE = 0x2_0000
JNI_REGION_SIZE = 0x4000
HALT_ADDR = 0x5000_0000             # where a top-level call "returns" to
EXTERN_BASE = 0x1_0002_0000         # sentinels for the library's external imports
EXTERN_STEP = 16


@dataclass
class Invocation:
    """The outcome of running one native method: its return value, the ordered JNI
    calls it made, and a fault message if emulation stopped early."""
    return_value: int | None
    trace: list[TraceEvent]
    fault: str | None = None

    def to_json(self) -> dict:
        return {
            "return_value": self.return_value,
            "fault": self.fault,
            "jni_calls": [self._event_json(e) for e in self.trace],
        }

    @staticmethod
    def _event_json(e) -> dict:
        # Keep the record compact: the structured fields the lifter reads (ret /
        # args / info) are emitted only when present, so ExceptionCheck-style events
        # stay three keys wide while Call*/Get*ID events carry what Phase 6 needs.
        d = {"interface": e.interface, "fn": e.fn, "detail": e.detail}
        if e.result is not None:
            d["ret"] = e.result
        if e.args is not None:
            d["args"] = e.args
        if e.info is not None:
            d["info"] = e.info
        return d


def _align_down(x: int) -> int:
    return x & ~(PAGE - 1)


def _align_up(x: int) -> int:
    return (x + PAGE - 1) & ~(PAGE - 1)


def _require_unicorn():
    try:
        import unicorn  # noqa: F401
        from unicorn import x86_const  # noqa: F401
    except ImportError as ex:
        raise UnnativeError(
            "the emulation back-end needs Unicorn — install it, e.g. "
            "`pip install unicorn` (or `pip install -e '.[emu]'`)") from ex
    return unicorn, x86_const


class _UcCtx:
    """Concrete JniCtx: decodes System V AMD64 integer args and reads/writes guest
    memory. `args` excludes the receiver (RDI), matching jnienv's contract."""

    def __init__(self, uc, x86, mem_read, mem_write):
        self._uc = uc
        self._mem_read = mem_read
        self._mem_write = mem_write
        rsi = uc.reg_read(x86.UC_X86_REG_RSI)
        rdx = uc.reg_read(x86.UC_X86_REG_RDX)
        rcx = uc.reg_read(x86.UC_X86_REG_RCX)
        r8 = uc.reg_read(x86.UC_X86_REG_R8)
        r9 = uc.reg_read(x86.UC_X86_REG_R9)
        rsp = uc.reg_read(x86.UC_X86_REG_RSP)
        # At the fetch fault RSP points at the pushed return address; further
        # integer args (7th+) would sit just above it.
        stack = [struct.unpack("<Q", mem_read(rsp + 8 + 8 * i, 8))[0] for i in range(3)]
        self.args = [rsi, rdx, rcx, r8, r9, *stack]

    def read(self, ptr: int, n: int) -> bytes:
        return bytes(self._mem_read(ptr, n))

    def read_ptr(self, ptr: int) -> int:
        return struct.unpack("<Q", self.read(ptr, PTR))[0]

    def write(self, ptr: int, data: bytes) -> None:
        self._mem_write(ptr, bytes(data))

    def cstr(self, ptr: int, limit: int = 4096) -> str:
        if ptr == 0:
            return ""
        out = bytearray()
        while len(out) < limit:
            chunk = bytes(self._mem_read(ptr + len(out), 16))
            nul = chunk.find(b"\x00")
            if nul >= 0:
                out += chunk[:nul]
                break
            out += chunk
        return out.decode("latin-1")


class UnicornEmu:
    def __init__(self, elf: ElfImage, model: JvmModel, layout: Layout | None = None):
        if elf.arch != "x86_64":
            raise UnnativeError(f"emu.py currently targets x86_64; got {elf.arch} "
                                "(aarch64 support is a follow-up)")
        self.unicorn, self.x86 = _require_unicorn()
        self.elf = elf
        self.model = model
        self.layout = layout or Layout()
        self.env = JNIEnv(model, self.layout)
        self.load_bias = LOAD_BIAS
        self._build()

    # -- setup --
    def _build(self):
        uc_mod, x86 = self.unicorn, self.x86
        self.uc = uc_mod.Uc(uc_mod.UC_ARCH_X86, uc_mod.UC_MODE_64)
        uc = self.uc

        # Library: one region spanning all PT_LOAD, segment bytes written in place.
        lo = _align_down(min(s.vaddr for s in self.elf.segments))
        hi = _align_up(max(s.vaddr + s.memsz for s in self.elf.segments))
        self.lib_base = self.load_bias + lo
        self.lib_size = hi - lo
        uc.mem_map(self.lib_base, self.lib_size,
                   uc_mod.UC_PROT_ALL)
        for seg in self.elf.segments:
            if seg.data:
                uc.mem_write(self.load_bias + seg.vaddr, seg.data)
        self._apply_relocations()

        # Stack.
        uc.mem_map(STACK_BASE, STACK_SIZE, uc_mod.UC_PROT_READ | uc_mod.UC_PROT_WRITE)
        self.stack_top = STACK_BASE + STACK_SIZE - 0x1000

        # TLS block + FS base, so stack-protector canary reads (%fs:0x28) and other
        # TLS accesses hit mapped memory instead of faulting at a tiny absolute
        # address. FS base points mid-block; the self-pointer convention glibc uses
        # (*(fs:0) == fs base) is honoured so TCB reads behave.
        uc.mem_map(TLS_BASE, TLS_SIZE, uc_mod.UC_PROT_READ | uc_mod.UC_PROT_WRITE)
        self.fs_base = TLS_BASE + TLS_SIZE // 2
        uc.mem_write(self.fs_base, struct.pack("<Q", self.fs_base))
        self._set_fs_base(self.fs_base)

        # JNIEnv / JavaVM structures + function tables (data).
        lay = self.layout
        self.jni_base = _align_down(lay.env_struct)
        uc.mem_map(self.jni_base, JNI_REGION_SIZE,
                   uc_mod.UC_PROT_READ | uc_mod.UC_PROT_WRITE)
        uc.mem_write(lay.env_struct, lay.env_struct_bytes())
        uc.mem_write(lay.env_table, lay.env_table_bytes())
        uc.mem_write(lay.vm_struct, lay.vm_struct_bytes())
        uc.mem_write(lay.vm_table, lay.vm_table_bytes())

        # Trampoline: each JNI slot points here. We map it executable and fill it
        # with `ret` (0xC3) so a `call` to a slot lands on a one-byte stub; a code
        # hook over this range services the JNI function, sets the return register,
        # and the `ret` returns to the caller.
        lo, hi = lay.tramp_span()
        tlo, tsz = _align_down(lo), _align_up(hi) - _align_down(lo)
        uc.mem_map(tlo, tsz, uc_mod.UC_PROT_READ | uc_mod.UC_PROT_EXEC)
        uc.mem_write(tlo, b"\xc3" * tsz)

        # Extern shim: GOT entries for the library's imports point here; a code hook
        # services each (memcpy, memset, …). ret-filled like the JNI trampoline.
        self.extern_calls: list[str] = []
        self.extern_unimplemented: set[str] = set()
        if self._extern_name:
            elo = _align_down(EXTERN_BASE)
            esz = _align_up(self._extern_next) - elo
            uc.mem_map(elo, esz, uc_mod.UC_PROT_READ | uc_mod.UC_PROT_EXEC)
            uc.mem_write(elo, b"\xc3" * esz)

        # Halt page: a top-level call returns here and emulation stops at `until`.
        uc.mem_map(HALT_ADDR, PAGE, uc_mod.UC_PROT_ALL)

        self._install_hook()

    def _set_fs_base(self, base: int):
        x86 = self.x86
        reg = getattr(x86, "UC_X86_REG_FS_BASE", None)
        if reg is not None:
            self.uc.reg_write(reg, base)
        else:                                       # fall back to the MSR
            self.uc.msr_write(0xC0000100, base)

    def _extern_sentinel(self, name: str) -> int:
        """Address a GOT entry for external symbol `name` points at, so a call
        through the PLT lands on a hooked stub the C-runtime shim services."""
        a = self._extern_addr.get(name)
        if a is None:
            a = self._extern_next
            self._extern_next += EXTERN_STEP
            self._extern_addr[name] = a
            self._extern_name[a] = name
        return a

    def _apply_relocations(self):
        """Patch the mapped image as the dynamic linker would. RELATIVE and
        symbol-based relocs against lib-defined symbols resolve to load-biased
        addresses; an **undefined external symbol** (libc etc.) is pointed at an
        extern-shim sentinel so a call through its PLT entry is serviced in Python
        instead of jumping to NULL."""
        from .loader_elf import (
            R_X86_64_64, R_X86_64_GLOB_DAT, R_X86_64_IRELATIVE, R_X86_64_JUMP_SLOT,
            R_X86_64_RELATIVE,
        )
        uc = self.uc
        self.unresolved_imports: list[str] = []
        self._extern_addr: dict[str, int] = {}
        self._extern_name: dict[int, str] = {}
        self._extern_next = EXTERN_BASE
        for r in self.elf.relocations:
            where = self.load_bias + r.offset
            if r.type in (R_X86_64_RELATIVE, R_X86_64_IRELATIVE):
                value = self.load_bias + r.addend
            elif r.type in (R_X86_64_64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT):
                sym = self.elf.symbols[r.sym] if r.sym < len(self.elf.symbols) else None
                if sym and sym.defined:
                    value = self.load_bias + sym.value + (
                        r.addend if r.type == R_X86_64_64 else 0)
                elif sym and sym.name:
                    self.unresolved_imports.append(sym.name)
                    value = self._extern_sentinel(sym.name)
                else:
                    value = 0
            else:
                continue                # other types: not needed for the JNI paths yet
            uc.mem_write(where, struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))

    def _install_hook(self):
        uc_mod, x86 = self.unicorn, self.x86
        lo, hi = self.layout.tramp_span()
        self.fault: tuple | None = None

        def on_tramp(uc, address, size, user):
            slot = self.layout.resolve(address)
            if slot is None:
                return                  # padding byte in the trampoline region
            interface, name = slot
            ctx = _UcCtx(uc, x86, uc.mem_read, uc.mem_write)
            try:
                result = self.env.dispatch(interface, name, ctx)
            except Exception as ex:                       # noqa: BLE001
                self.fault = ("stub", name, str(ex))
                uc.emu_stop()
                return
            self._place_return(name, result)
            # Let the `ret` at this address execute to return to the caller.

        def on_extern(uc, address, size, user):
            name = self._extern_name.get(address)
            if name is not None:
                self._extern_dispatch(name)
            # the `ret` at the sentinel returns to the PLT's caller

        def on_invalid(uc, access, address, size, value, user):
            self.fault = ("mem", access, address, uc.reg_read(x86.UC_X86_REG_RIP))
            return False                # let emu_start raise; call() reports it

        self.uc.hook_add(uc_mod.UC_HOOK_CODE, on_tramp, begin=lo, end=hi)
        if self._extern_name:
            self.uc.hook_add(uc_mod.UC_HOOK_CODE, on_extern,
                             begin=EXTERN_BASE, end=self._extern_next)
        self.uc.hook_add(uc_mod.UC_HOOK_MEM_UNMAPPED, on_invalid)

    def _read_cstr(self, ptr: int, limit: int = 4096) -> bytes:
        if not ptr:
            return b""
        out = bytearray()
        while len(out) < limit:
            chunk = bytes(self.uc.mem_read(ptr + len(out), 16))
            nul = chunk.find(b"\x00")
            if nul >= 0:
                return bytes(out + chunk[:nul])
            out += chunk
        return bytes(out)

    def _extern_dispatch(self, name: str):
        """A minimal C-runtime shim. Enough for the translated methods' real calls
        (memcpy is the one this corpus needs); unknown imports are no-ops returning
        0 and recorded, so they surface instead of faulting at a NULL PLT slot."""
        uc, x86 = self.uc, self.x86
        self.extern_calls.append(name)

        def reg(r):
            return uc.reg_read(getattr(x86, "UC_X86_REG_" + r))

        rdi, rsi, rdx = reg("RDI"), reg("RSI"), reg("RDX")
        ret = 0
        if name in ("memcpy", "memmove"):
            if rdx:
                uc.mem_write(rdi, bytes(uc.mem_read(rsi, rdx)))
            ret = rdi
        elif name == "memset":
            if rdx:
                uc.mem_write(rdi, bytes([rsi & 0xFF]) * rdx)
            ret = rdi
        elif name == "strlen":
            ret = len(self._read_cstr(rdi))
        elif name in ("strcmp", "strncmp"):
            a, b = self._read_cstr(rdi), self._read_cstr(rsi)
            ret = (a > b) - (a < b)
        elif name == "__gmon_start__":
            ret = 0
        else:
            self.extern_unimplemented.add(name)
        uc.reg_write(x86.UC_X86_REG_RAX, ret & 0xFFFFFFFFFFFFFFFF)

    def _place_return(self, name: str, result):
        import unnative.native.jni_api as api
        uc, x86 = self.uc, self.x86
        cat = api.return_category(_ret_type_of(name))
        if cat == "void":
            return
        value = int(result or 0)
        if cat == "float":
            # No current handler returns a non-zero float; place 0.0 in XMM0.
            uc.reg_write(x86.UC_X86_REG_XMM0, 0)
        else:
            uc.reg_write(x86.UC_X86_REG_RAX, value & 0xFFFFFFFFFFFFFFFF)

    # -- calling --
    def call(self, addr: int, args: list[int] | None = None, *,
             timeout_us: int = 0, count: int = 0) -> int:
        """Call a function at guest `addr` (a load-biased address), returning RAX.
        The first six integer args go in registers, the rest on the stack; the stack
        is 16-byte aligned per the System V ABI. Emulation stops at HALT, or early on
        `timeout_us`/`count` bounds (0 = unbounded) so one runaway function can't
        hang a batch."""
        uc, x86 = self.uc, self.x86
        args = list(args or [])
        regs = [x86.UC_X86_REG_RDI, x86.UC_X86_REG_RSI, x86.UC_X86_REG_RDX,
                x86.UC_X86_REG_RCX, x86.UC_X86_REG_R8, x86.UC_X86_REG_R9]
        for reg, val in zip(regs, args[:6]):
            uc.reg_write(reg, val & 0xFFFFFFFFFFFFFFFF)
        stack_args = args[6:]
        # Push stack args then the return address, keeping rsp ≡ 8 (mod 16) at entry
        # (i.e. an odd number of 8-byte pushes from a 16-aligned base).
        pushes = stack_args + [HALT_ADDR]
        if len(pushes) % 2 == 0:
            pushes = stack_args + [0, HALT_ADDR]      # pad to an odd count
        rsp = self.stack_top & ~0xF
        for val in reversed(pushes):
            rsp -= 8
            uc.mem_write(rsp, struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))
        uc.reg_write(x86.UC_X86_REG_RSP, rsp)
        self.fault = None
        try:
            uc.emu_start(addr, HALT_ADDR, timeout=timeout_us, count=count)
        except self.unicorn.UcError as ex:
            raise UnnativeError(self._fault_message(ex)) from ex
        return uc.reg_read(x86.UC_X86_REG_RAX)

    # -- snapshot / restore (per-snippet evaluation, Phase 4) --
    def _snapshot_regions(self):
        return [(STACK_BASE, STACK_SIZE), (TLS_BASE, TLS_SIZE),
                (self.jni_base, JNI_REGION_SIZE), (self.lib_base, self.lib_size)]

    def snapshot(self):
        """Capture CPU + guest memory so a call can be undone and re-run (the
        protector's per-site constants are evaluated by running a snippet, then
        rolling back). Python-side state (handles, trace) is intentionally *not*
        rolled back; callers slice the trace by length instead."""
        ctx = self.uc.context_save()
        mem = {addr: bytes(self.uc.mem_read(addr, size))
               for addr, size in self._snapshot_regions()}
        return ctx, mem

    def restore(self, snap):
        ctx, mem = snap
        self.uc.context_restore(ctx)
        for addr, data in mem.items():
            self.uc.mem_write(addr, data)

    # -- native-method invocation + tracing (M1: behaviour per method) --
    def new_class(self, owner_internal: str) -> int:
        return self.env.handles.new("class", owner_internal)

    def new_object(self, owner_internal: str) -> int:
        return self.env.handles.new("object", owner_internal)

    def invoke(self, vaddr: int, int_args: list[int], *, restore: bool = True,
               timeout_us: int = 0, count: int = 0) -> "Invocation":
        """Call the native function at lib `vaddr` with pre-built integer args,
        returning its result, the slice of JNI calls it made, and any fault. With
        `restore`, guest state is rolled back afterwards so calls are independent."""
        snap = self.snapshot() if restore else None
        start = len(self.env.trace)
        value: int | None = None
        fault: str | None = None
        try:
            value = self.call(self.guest_addr(vaddr), int_args,
                              timeout_us=timeout_us, count=count)
        except UnnativeError as ex:
            fault = str(ex)
        events = list(self.env.trace[start:])
        if snap is not None:
            self.restore(snap)
        return Invocation(return_value=value, trace=events, fault=fault)

    def trace_method(self, vaddr: int, owner: str, name: str, desc: str,
                     java_args: list[int] | None = None, *,
                     timeout_us: int = 0, count: int = 0) -> "Invocation":
        """Invoke a bound native method and record the JNI-call stream it makes.

        The receiver is a synthetic class handle (static) or object handle
        (instance); arguments default to zero/null per the descriptor's arity.
        Float/double arguments are not yet passed in XMM (a follow-up); methods that
        take them are traced with those slots zeroed."""
        ref = self.model.resolve_method(owner, name, desc)
        static = ref.static if ref else True
        nargs = len(parse_desc(desc)[0])
        if java_args is None:
            java_args = [0] * nargs
        receiver = self.new_class(owner) if static else self.new_object(owner)
        int_args = [self.layout.env_struct, receiver, *java_args]
        return self.invoke(vaddr, int_args, timeout_us=timeout_us, count=count)

    def _fault_message(self, ex) -> str:
        f = self.fault
        if f and f[0] == "stub":
            return f"JNI stub {f[1]} failed: {f[2]}"
        if f and f[0] == "mem":
            _kind, access, address, rip = f
            return (f"emulation faulted: memory access {access} at {address:#x} "
                    f"(rip {rip:#x}; rip-bias {rip - self.load_bias:#x})")
        return f"emulation failed: {ex}"

    def guest_addr(self, vaddr: int) -> int:
        return self.load_bias + vaddr

    def call_jni_onload(self) -> int | None:
        v = self.elf.jni_onload()
        if v is None:
            return None
        return self.call(self.guest_addr(v), [self.layout.vm_struct, 0])

    @classmethod
    def for_jar_and_lib(cls, jar_path: str, lib_path: str) -> "UnicornEmu":
        return cls(ElfImage.from_path(lib_path), JvmModel.from_source(jar_path))


def _ret_type_of(name: str) -> str:
    import unnative.native.jni_api as api
    for _i, n, ret in api.JNIENV_SLOTS:
        if n == name:
            return ret
    for _i, n, ret in api.JAVAVM_SLOTS:
        if n == name:
            return ret
    return "void"
