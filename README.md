# unnative

**English** | [Русский](README.ru.md)

Recover readable Java from jars whose method bodies have been replaced with native
code by [JNIC](https://jnic.dev) or [JVMTP](https://jvmtp.dev).

These protectors compile selected JVM methods to machine code behind a `native`
stub and ship the result as a bundled library that binds itself at load time. The
original bytecode is gone, so recovering what a method does means loading the native
library, watching the JNI calls it makes, and lifting that back toward bytecode.
`unnative` is a command-line pipeline for doing exactly that. Its class-file parser
is shared with the sibling project [unseed](https://github.com/sad-kitsune/unseed).

## What it does

`unnative` runs a jar through a series of stages. Each stage writes a plain file
into a per-jar work directory, so results can be inspected and re-run independently.

- **triage** — parse the classes, list the `native` methods and the loader, and make
  a best guess at the protector. Writes `report.json`.
- **unpack** — get the bundled native libraries onto disk. Libraries shipped as
  plain jar resources are extracted directly; protectors that pack them into an
  encrypted blob aren't unpacked yet, so you can also supply pre-extracted libraries
  with `--lib`.
- **bind** — map each `native` method to the function that implements it, both by
  resolving the exported `Java_*` symbols and by running the library's registration
  code under an emulator to capture its `RegisterNatives` tables. Writes
  `bindings.json`.
- **trace** — run each bound method under the emulator and record the sequence of
  JNI calls it makes and its return value. Writes `trace.json`.
- **lift** — recognise the JNI idioms in each trace (a `GetMethodID` + `Call*Method`
  is an `invoke*`, `AllocObject` + `<init>` is `new`, a field access, a `throw`, …)
  and emit a JVM-level IR. When capstone is installed, each method's IR is also
  enriched with the control-flow graph recovered statically from its native code —
  basic blocks, edges, loops, and any indirect jump left as an open exit. Writes a
  `jir/` directory, one file per method, with a coverage summary. For the simplest
  shapes — the reference-threaded straight-line methods such as `throw new X()` — it
  also schedules the IR onto the operand stack and emits candidate bytecode; it
  refuses, with a reason, any method that needs an operand it can't see (a computed
  primitive, an untracked reference) or that isn't straight-line. It does not yet
  recover the pure computation that makes no JNI call. Each gap is reported per
  method rather than hidden, and emitted bytecode is candidate output until Phase 8
  verifies it against the original.
- **emit** — rebuild the jar from the lifted IR. For each method the lift stage
  lowered to bytecode, fill the `native` stub: clear `ACC_NATIVE` and attach a real
  `Code` attribute, written by a small pure-Python class-file writer (no external
  toolchain — the straight-line methods lifted today need no StackMapTable). Methods
  that didn't lift are left as native stubs and the loader is kept, so the rebuilt
  jar still runs — partial recovery, not all-or-nothing. A signed jar has its
  signature stripped, since rewriting a class invalidates it. Writes `rebuilt.jar`
  and an `emit.json` summary.
- **verify** — differentially check the rebuilt methods. For each method `emit`
  filled, compare the behaviour the lift stage recovered against two independent
  witnesses: the bytecode re-decoded straight from `rebuilt.jar` (which catches any
  emit-stage corruption) and the emulated original's JNI trace (ground truth, reached
  by following the thrown object handle back to the class it came from). When a JDK
  is present, each recovered body is also run on a real JVM and its outcome compared.
  Writes `verify.json` with a per-method verdict. The honest limit: the original
  native method can't be run in place — initialising its class triggers the
  protector's loader — so the emulated trace is its only independent witness.

A later stage — recovering obfuscated strings and constants (Phase 4) — is not
implemented yet.

The emulator is a [Unicorn](https://www.unicorn-engine.org/) CPU with a synthetic
`JNIEnv`/`JavaVM` and a small C-runtime shim, so the native library executes in a
sandbox that can't reach the host.

## Requirements

- Python 3.11+ (the core uses only the standard library).
- A JDK (17+) for the test fixtures and the sample corpus.
- Unicorn, for the `bind`/`trace` emulation. The static `Java_*` binding path works
  without it.
- capstone, for the `lift` stage's control-flow recovery. Without it `lift` still
  emits the linear IR, minus the per-method CFG.

## Usage

No install is needed for the core:

```sh
python3 -m unnative triage path/to/app.jar
python3 -m unnative run     path/to/app.jar
python3 -m unnative --help
```

`run` executes the stages in order and stops at the first one that isn't built yet.
Output lands under `work/<jar>/` (`report.json`, `libs/`, `bindings.json`,
`trace.json`).

The emulation stages need Unicorn. Where `pip` can't install into the system Python
(for example a PEP 668 "externally managed" setup), use a virtualenv:

```sh
python3 -m venv .venv
.venv/bin/pip install unicorn
.venv/bin/python -m unnative run path/to/app.jar
```

If the native library is packed in a form `unpack` can't extract, pass the already
extracted library (flags go after the subcommand):

```sh
.venv/bin/python -m unnative run app.jar --lib linux-x86_64.so
```

Optional editable install:

```sh
pip install -e .          # installs the `unnative` console script
pip install -e '.[emu]'   # also installs Unicorn
```

## Tests

Standard-library `unittest`. Fixtures are built on the fly with `javac` and a C
compiler; the emulator tests skip themselves when Unicorn is absent.

```sh
python3 -m unittest discover -s tests
```

## Layout

```
unnative/   the package: CLI and driver, the class-file parser, triage, the native back-end
corpus/     small Java programs used as test material
tools/      helper scripts
tests/
```

## Authorisation

Analyse only software you are authorised to analyse, and respect the protectors'
licence terms when building test material with them.

## License

MIT (see [LICENSE](LICENSE)). The class-file parser under `unnative/classfile/` is
vendored from `unseed`, which is also MIT-licensed.
