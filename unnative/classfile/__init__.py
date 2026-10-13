"""Class-file core: modified-UTF-8 codec, opcode table + decoder, ClassFile reader.

Vendored from unseed (see PROVENANCE.md). This package is the only place unnative
parses `.class` files; everything downstream (triage, lift, emit) builds on it.
"""

from .classfile import (
    ACC_ABSTRACT,
    ACC_FINAL,
    ACC_NATIVE,
    ACC_PRIVATE,
    ACC_PROTECTED,
    ACC_PUBLIC,
    ACC_STATIC,
    ACC_SYNCHRONIZED,
    ACC_SYNTHETIC,
    ClassFile,
    ClassSource,
    load_all,
    load_class,
    parse_desc,
)
from .modified_utf8 import (
    dec_modified_utf8,
    enc_modified_utf8,
    from_utf16_units,
    to_utf16_units,
)
from .opcodes import (
    BRANCH_OPS,
    GOTO_OPS,
    INVOKE_OPS,
    OPS,
    TERMINATORS,
    Insn,
    decode,
    names_matching,
)

__all__ = [
    # classfile
    "ClassFile", "ClassSource", "load_all", "load_class", "parse_desc",
    "ACC_PUBLIC", "ACC_PRIVATE", "ACC_PROTECTED", "ACC_STATIC", "ACC_FINAL",
    "ACC_SYNCHRONIZED", "ACC_NATIVE", "ACC_ABSTRACT", "ACC_SYNTHETIC",
    # modified_utf8
    "dec_modified_utf8", "enc_modified_utf8", "to_utf16_units", "from_utf16_units",
    # opcodes
    "OPS", "Insn", "decode", "names_matching",
    "BRANCH_OPS", "INVOKE_OPS", "GOTO_OPS", "TERMINATORS",
]
