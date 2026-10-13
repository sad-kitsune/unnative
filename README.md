# unnative

**English** | [Русский](README.ru.md)

Recover readable Java from jars whose method bodies have been replaced with native
code by [JNIC](https://jnic.dev) or [JVMTP](https://jvmtp.dev).

These protectors compile selected JVM methods to machine code behind a `native`
stub and ship the result as a bundled library that binds itself at load time. The
original bytecode is gone, so recovering what a method does means loading the native
library, watching the JNI calls it makes, and lifting that back toward bytecode.
`unnative` is a command-line pipeline for doing exactly that. Its class-file parser
is shared with the sibling project `unseed`.

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

Later stages — recovering obfuscated strings and constants, lifting the JNI-call
streams back to bytecode, rebuilding the jar, and checking it against the original —
are not implemented yet.

The emulator is a [Unicorn](https://www.unicorn-engine.org/) CPU with a synthetic
`JNIEnv`/`JavaVM` and a small C-runtime shim, so the native library executes in a
sandbox that can't reach the host.

## Requirements

- Python 3.11+ (the core uses only the standard library).
- A JDK (17+) for the test fixtures and the sample corpus.
- Unicorn, for the `bind`/`trace` emulation. The static `Java_*` binding path works
  without it.

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
