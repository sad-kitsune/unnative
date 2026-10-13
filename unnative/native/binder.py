"""The binding map (plan.md Phase 3): native method <-> implementing address.

Two paths, because a protector may use either (or both):

  * the classic path — an exported `Java_<mangled>` symbol the JVM's lazy linker
    resolves (recovered statically here, via `mangle`);
  * `RegisterNatives` — a table the library installs at load, recovered by the
    emulator running `JNI_OnLoad` (the caller passes those results in, so this
    module needs no Unicorn and stays unit-testable).

A method resolved by *both* is logged as a decoy (plan.md Phase 3): a real
`Java_*` symbol that is also re-bound by `RegisterNatives` is a classic
misdirection. A `native` method the model has but neither path resolves is left
unresolved — often a sign it is VM-backed (Phase 9) rather than a plain translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .jvm_model import JvmModel
from .loader_elf import ElfImage
from .mangle import candidate_symbols

SOURCE_SYMBOL = "java_symbol"
SOURCE_REGISTER = "register_natives"


@dataclass
class ResolvedBinding:
    owner: str
    name: str
    desc: str
    vaddr: int                      # address within the library (load-bias removed)
    sources: list[str] = field(default_factory=list)
    symbol: str | None = None       # the Java_* export name, when resolved that way

    @property
    def is_decoy(self) -> bool:
        return SOURCE_SYMBOL in self.sources and SOURCE_REGISTER in self.sources

    def to_json(self) -> dict:
        return {"owner": self.owner, "name": self.name, "desc": self.desc,
                "vaddr": self.vaddr, "sources": self.sources, "symbol": self.symbol,
                "decoy": self.is_decoy}


@dataclass
class BindResult:
    bindings: list[ResolvedBinding]
    unresolved: list[dict]
    arch: str

    @property
    def decoys(self) -> list[ResolvedBinding]:
        return [b for b in self.bindings if b.is_decoy]

    def to_json(self) -> dict:
        return {
            "arch": self.arch,
            "counts": {
                "native_methods": len(self.bindings) + len(self.unresolved),
                "resolved": len(self.bindings),
                "unresolved": len(self.unresolved),
                "via_symbol": sum(1 for b in self.bindings if SOURCE_SYMBOL in b.sources),
                "via_register_natives":
                    sum(1 for b in self.bindings if SOURCE_REGISTER in b.sources),
                "decoys": len(self.decoys),
            },
            "bindings": [b.to_json() for b in self.bindings],
            "unresolved": self.unresolved,
        }


def bind(elf: ElfImage, model: JvmModel,
         register_natives: list[tuple[str, str, str, int]] = ()) -> BindResult:
    """Resolve every `native` method in `model` against `elf`.

    `register_natives` is an optional list of `(owner, name, desc, vaddr)` recovered
    dynamically (vaddr already load-bias-relative), e.g. from `emu` running
    `JNI_OnLoad`.
    """
    by_key: dict[tuple[str, str, str], ResolvedBinding] = {}

    # Classic Java_* symbol path.
    for m in model.native_methods():
        key = (m.owner, m.name, m.desc)
        for sym in candidate_symbols(m.owner, m.name, m.desc):
            exp = elf.exports.get(sym)
            if exp is not None:
                by_key[key] = ResolvedBinding(
                    owner=m.owner, name=m.name, desc=m.desc, vaddr=exp.value,
                    sources=[SOURCE_SYMBOL], symbol=sym)
                break

    # RegisterNatives path (dynamic).
    for owner, name, desc, vaddr in register_natives:
        key = (owner, name, desc)
        existing = by_key.get(key)
        if existing is not None:
            if SOURCE_REGISTER not in existing.sources:
                existing.sources.append(SOURCE_REGISTER)
            # RegisterNatives wins on address: it is what actually runs.
            existing.vaddr = vaddr
        else:
            by_key[key] = ResolvedBinding(owner=owner, name=name, desc=desc,
                                          vaddr=vaddr, sources=[SOURCE_REGISTER])

    native_keys = {(m.owner, m.name, m.desc) for m in model.native_methods()}
    unresolved = [{"owner": o, "name": n, "desc": d}
                  for (o, n, d) in sorted(native_keys - set(by_key))]

    bindings = sorted(by_key.values(), key=lambda b: (b.owner, b.name, b.desc))
    return BindResult(bindings=bindings, unresolved=unresolved, arch=elf.arch)
