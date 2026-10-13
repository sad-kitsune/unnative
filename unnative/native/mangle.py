"""JNI name mangling — the classic (non-RegisterNatives) binding path.

When a native method is *not* bound by `RegisterNatives`, the JVM's lazy linker
resolves it by looking up an exported C symbol whose name is a mangling of the
method's class and name (and, for overloaded methods, its argument signature). We
run that mangling forwards — for each `native` method in the model we compute the
symbol(s) the linker would look for and match them against the library's exports —
which both resolves `(owner, name, desc) -> address` and tells us which methods use
this path versus `RegisterNatives`.

The scheme (JNI spec): `Java_` + mangle(class) + `_` + mangle(method), and for an
overloaded method also `__` + mangle(arg-descriptor). mangle() keeps ASCII
alphanumerics, maps `/` and `.` to `_`, and escapes `_`->`_1`, `;`->`_2`,
`[`->`_3`, any other character to `_0` + its 4-hex-digit UTF-16 code unit(s).
Validated against `javac -h` output (see tests/test_mangle.py).
"""

from __future__ import annotations

_SIMPLE = {"_": "_1", ";": "_2", "[": "_3", "/": "_", ".": "_"}


def mangle_segment(s: str) -> str:
    out = []
    for ch in s:
        if ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch in _SIMPLE:
            out.append(_SIMPLE[ch])
        else:
            # Non-alphanumeric, non-special: escape each UTF-16 code unit.
            for unit in s_to_utf16_units(ch):
                out.append(f"_0{unit:04x}")
    return "".join(out)


def s_to_utf16_units(s: str) -> list[int]:
    raw = s.encode("utf-16-be")
    return [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw), 2)]


def _arg_descriptor(desc: str) -> str:
    """The parenthesised argument part of a method descriptor, without the return."""
    try:
        return desc[desc.index("(") + 1:desc.index(")")]
    except ValueError:
        return ""


def short_symbol(owner_internal: str, name: str) -> str:
    """`Java_<class>_<method>` — the form the linker tries first."""
    return f"Java_{mangle_segment(owner_internal)}_{mangle_segment(name)}"


def long_symbol(owner_internal: str, name: str, desc: str) -> str:
    """`Java_<class>_<method>__<args>` — used for overloaded methods."""
    return f"{short_symbol(owner_internal, name)}__{mangle_segment(_arg_descriptor(desc))}"


def candidate_symbols(owner_internal: str, name: str, desc: str) -> list[str]:
    """Symbols to look for, in the order the JVM would: short form, then long form."""
    return [short_symbol(owner_internal, name), long_symbol(owner_internal, name, desc)]
