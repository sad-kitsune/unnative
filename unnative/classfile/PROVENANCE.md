# Provenance of the class-file core

**English** | [Русский](PROVENANCE.ru.md)

These modules are **vendored from [`unseed`](https://github.com/sad-kitsune/unseed)** (`unseed.py`): its class-file parser
and opcode table, copied rather than imported, so unnative has no dependency on
unseed's layout and the two can evolve independently.

`unseed` is MIT-licensed, Copyright (c) 2026 Sad Kitsune.

## What was vendored, and from where

| this module         | from `unseed.py` section                         |
|---------------------|--------------------------------------------------|
| `modified_utf8.py`  | §2 modified UTF-8 / UTF-16 codec                 |
| `opcodes.py`        | §3 opcode table and instruction decoder          |
| `classfile.py`      | §4 ClassFile, §5 ClassSource, and `parse_desc`   |

## What was deliberately **not** vendored

unseed's §6 concrete stack interpreter, §7 decryptor extraction, §8–14 (its
caller-seeded string scheme, patcher, invokedynamic census, CLI). Those are
specific to the one bytecode-level obfuscation unseed defeats. unnative's problem
is native code, so its own lifter/emitter live elsewhere in the tree.

## Changes made during vendoring

- `UnseedError` → `unnative.errors.UnnativeError`.
- The monolith was split into three focused modules; the public names
  (`decode`, `Insn`, `ClassFile`, `ClassSource`, `load_all`, `parse_desc`,
  the modified-UTF-8 functions) are re-exported from `unnative.classfile`.
- Correctness-critical comments (modified UTF-8 vs. `replace`, the shift-opcode
  ordering, the loud-failure decoder contract) were kept verbatim, because they
  record *why* the code is the way it is, not just what it does.

Keep these in sync by hand when unseed's core gets a correctness fix. A drift
check lives in `tests/test_classfile.py`.
