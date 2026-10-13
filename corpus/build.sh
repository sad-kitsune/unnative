#!/usr/bin/env bash
# Build the test corpus.
#
# Baseline (unprotected) builds run anywhere a JDK is present; they are the ground
# truth the tool's output is measured against. The protected builds need JNIC
# and/or JVMTP, which are licensed tools and are NOT bundled here; when they are
# configured (see corpus/README.md) this script drives them across the option
# matrix, otherwise it skips them with a notice rather than failing.
#
# Usage:
#   corpus/build.sh                              # baseline builds + record output
#   JNIC_CLI=... JVMTP_CLI=... corpus/build.sh   # + protected builds
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
BUILD="$HERE/build"
OUT="$HERE/out"

mkdir -p "$BUILD" "$OUT/baseline" "$OUT/expected"

command -v javac >/dev/null || { echo "error: javac not on PATH" >&2; exit 1; }

JDK_VER="$(javac -version 2>&1 | awk '{print $2}')"
echo "== baseline build (JDK $JDK_VER) =="

build_baseline() {
    local src="$1" name outdir jar
    name="$(basename "$src" .java)"
    outdir="$BUILD/baseline/$name"
    jar="$OUT/baseline/$name.jar"
    rm -rf "$outdir"; mkdir -p "$outdir"
    javac -d "$outdir" "$src"
    # Entry point = the file's main class (same as the stem, by convention here).
    ( cd "$outdir" && jar --create --file "$jar" --main-class "$name" . )
    # Record ground-truth stdout+stderr and exit code for Phase 8 differential verify.
    local exp="$OUT/expected/$name.txt"
    if java -jar "$jar" >"$exp" 2>&1; then
        echo "exit=0" >>"$exp"
    else
        echo "exit=$?" >>"$exp"
    fi
    echo "  built $name.jar  (expected output: $(wc -l <"$exp") lines)"
}

shopt -s nullglob
sources=("$SRC"/*.java)
if [ ${#sources[@]} -eq 0 ]; then
    echo "no sources in $SRC" >&2; exit 1
fi
for src in "${sources[@]}"; do
    build_baseline "$src"
done

# --- protected builds (need the licensed protectors) --------------------------
# Fill these in once JNIC/JVMTP are available. Keep the tool VERSION in the output
# filename so a version bump shows up as a corpus diff. The option matrix (baseline
# / each option alone / all-on; for JVMTP also vm.stack|register|dual) lives in
# corpus/configs — see corpus/README.md.
protect_jnic() {
    if [ -z "${JNIC_CLI:-}" ]; then
        echo "== JNIC: skipped (set JNIC_CLI to enable; see corpus/README.md) =="
        return 0
    fi
    echo "== JNIC: TODO wire '$JNIC_CLI' across corpus/configs (Phase 0) ==" >&2
    # for cfg in "$HERE"/configs/jnic/*.cfg; do ... "$JNIC_CLI" ... ; done
}

protect_jvmtp() {
    if [ -z "${JVMTP_CLI:-}" ]; then
        echo "== JVMTP: skipped (set JVMTP_CLI to enable; see corpus/README.md) =="
        return 0
    fi
    echo "== JVMTP: TODO wire '$JVMTP_CLI' across corpus/configs (Phase 0) ==" >&2
    # for cfg in "$HERE"/configs/jvmtp/*.toml; do ... "$JVMTP_CLI" ... ; done
}

protect_jnic
protect_jvmtp
echo "== done =="
