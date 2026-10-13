# Corpus

**English** | [Русский](README.ru.md)

Small Java programs used as test material. Each program exercises one area of the
JVM spec. We build each one unprotected (the baseline), and — where the protectors
are available — protected with JNIC and JVMTP across a matrix of options, then
measure the tool's output against the baseline.

- `src/` — the Java programs (checked in).
- `configs/` — per-protector option configs (see the matrix below).
- `build/`, `out/` — build outputs and recorded expected output (gitignored).
- `build.sh` — builds the baselines and records their output; also builds the
  protected variants when the protectors are configured.

## Building

```sh
corpus/build.sh                                                  # baseline only
JNIC_CLI=/path/to/jnic JVMTP_CLI=/path/to/jvmtp corpus/build.sh  # + protected
```

Baseline builds compile each `src/<Name>.java`, jar it with a `Main-Class`, run it,
and save stdout+stderr+exit to `out/expected/<Name>.txt`. That recorded output is
the ground truth a rebuilt jar is later compared against.

## Programs

| program | exercises |
|---|---|
| `ArithEdges` | int overflow, idiv/irem by zero, shift masking, long compare, NaN, d2i saturation |
| `Exceptions` | nested try/catch/finally, finally on every path, rethrow with cause |
| `Switches` | tableswitch, lookupswitch, String switch, enum switch |
| `Indy` | lambda metafactory (capturing + non-capturing), string-concat `invokedynamic` |
| `Invokes` | every `invoke*` kind, default methods, `super`, field + static access, `new` |

Worth adding: arrays (multi-dimensional, covariance), `synchronized`/monitors,
`<clinit>` ordering, generics + bridge methods, inner/anonymous classes, recursion,
varargs, reflection. Drop one `.java` per area into `src/` and `build.sh` picks it
up automatically.

## Option matrix (`configs/`)

For every program, build:

- **baseline** — no obfuscation (also the reference for control-flow recovery).
- **each option alone** — one knob at a time, so a failure pins the responsible
  feature:
  - JNIC: `stringObf`, `flowObf`, `useIntrinsics`.
  - JVMTP: `obfuscation.constants.mba`, string encryption, call concealment,
    opaque predicates, flattening, `obfuscation.jumps`, `resources.compress`,
    `compiler.batch-resolve`, `compiler.safe-env`, `antidebug.enable`.
- **everything on**.
- **JVMTP VM backend** — `vm.stack`, `vm.register`, `vm.dual`.

Record the protector version in each output filename, e.g.
`out/jvmtp-1.4.2/Switches.vm-dual.jar`, so a version bump shows up as a diff.

The concrete config file formats are filled in once the protector docs are in hand;
they aren't guessed here.

> **Authorisation.** Respect the protectors' licence terms when building protected
> corpora, and only analyse jars you are authorised to analyse.
