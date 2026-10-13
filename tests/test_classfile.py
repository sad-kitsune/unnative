"""Tests for the vendored class-file core, plus a drift check against unseed."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

from _fixtures import NATIVE_WITH_LOADER, UNSEED_REF, JavaFixtureCase

from unnative.classfile import (
    OPS,
    ClassFile,
    ClassSource,
    dec_modified_utf8,
    decode,
    enc_modified_utf8,
    load_all,
    parse_desc,
)
from unnative.errors import UnnativeError


class ModifiedUtf8Tests(unittest.TestCase):
    def test_nul_is_two_bytes(self):
        self.assertEqual(enc_modified_utf8("\x00"), b"\xc0\x80")
        self.assertEqual(dec_modified_utf8(b"\xc0\x80"), "\x00")

    def test_raw_nul_rejected(self):
        with self.assertRaises(UnnativeError):
            dec_modified_utf8(b"\x00")

    def test_supplementary_roundtrip(self):
        # U+1F600, a supplementary character, must go via a CESU-8 surrogate pair.
        s = "a\U0001F600b"
        enc = enc_modified_utf8(s)
        self.assertEqual(dec_modified_utf8(enc), s)
        # 6 bytes for the pair (two 3-byte forms), 1 each for 'a' and 'b'.
        self.assertEqual(len(enc), 8)

    def test_lone_surrogate_rejected(self):
        with self.assertRaises(UnnativeError):
            # 0xED 0xA0 0x80 is a lone high surrogate in CESU-8 form.
            dec_modified_utf8(b"\xed\xa0\x80")


class OpcodeTableTests(unittest.TestCase):
    def test_shift_ordering(self):
        # The subtle one: shifts are grouped by operation, int/long adjacent.
        self.assertEqual(OPS[0x78][0], "ishl")
        self.assertEqual(OPS[0x79][0], "lshl")
        self.assertEqual(OPS[0x7A][0], "ishr")
        self.assertEqual(OPS[0x7C][0], "iushr")
        self.assertEqual(OPS[0x7E][0], "iand")

    def test_decode_simple_body(self):
        # iload_1; iload_2; imul; ireturn  ->  mul(int,int)
        body = bytes([0x1B, 0x1C, 0x68, 0xAC])
        insns = decode(body)
        self.assertEqual([i.name for i in insns], ["iload_1", "iload_2", "imul", "ireturn"])
        self.assertEqual([i.off for i in insns], [0, 1, 2, 3])

    def test_decode_branch_target(self):
        # goto +3 (to offset 3), then nop, nop, nop
        body = bytes([0xA7, 0x00, 0x03, 0x00, 0x00, 0x00])
        insns = decode(body)
        self.assertEqual(insns[0].name, "goto")
        self.assertEqual(insns[0].target, [3])

    def test_unknown_opcode_raises(self):
        with self.assertRaises(UnnativeError):
            decode(bytes([0xFF]))

    def test_truncation_raises(self):
        with self.assertRaises(UnnativeError):
            decode(bytes([0x13, 0x00]))   # ldc_w needs a 2-byte operand, only 1 given


class ParseDescTests(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(parse_desc("()V"), ([], "V"))
        self.assertEqual(parse_desc("(II)I"), (["I", "I"], "I"))
        self.assertEqual(
            parse_desc("(Ljava/lang/Object;JJ)V"),
            (["Ljava/lang/Object;", "J", "J"], "V"))
        self.assertEqual(parse_desc("([[ILjava/lang/String;)[B"),
                         (["[[I", "Ljava/lang/String;"], "[B"))

    def test_malformed_raises(self):
        with self.assertRaises(UnnativeError):
            parse_desc("II)I")


class ClassFileTests(JavaFixtureCase):
    def test_reads_native_method(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        src = ClassSource(jar)
        cfs, fails = load_all(src, quiet=True)
        self.assertEqual(fails, [])
        demo = next(cf for cf in cfs.values() if cf.this == "Demo")
        self.assertEqual(demo.native_methods, {("add", "(II)I")})

    def test_malformed_classfile_raises(self):
        with self.assertRaises(UnnativeError):
            ClassFile("bad.class", b"not a class file at all")


class _UnseedRef:
    """Lazily load unseed.py from the sibling checkout for the drift check."""

    _mod = None

    @classmethod
    def load(cls):
        if cls._mod is not None:
            return cls._mod
        if not os.path.exists(UNSEED_REF):
            raise unittest.SkipTest(f"unseed reference not found at {UNSEED_REF}")
        spec = importlib.util.spec_from_file_location("unseed_ref", UNSEED_REF)
        mod = importlib.util.module_from_spec(spec)
        # Register before exec: unseed.py defines @dataclass types, and on 3.14 the
        # decorator looks the module up in sys.modules by name while executing.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        cls._mod = mod
        return mod


class VendorDriftTests(unittest.TestCase):
    """The class-file core is vendored, not imported (plan.md §2.1). These guard
    against the copy silently drifting from unseed's source of truth."""

    def test_opcode_table_matches_unseed(self):
        ref = _UnseedRef.load()
        self.assertEqual(OPS, ref.OPS)

    def test_utf8_codec_matches_unseed(self):
        ref = _UnseedRef.load()
        for s in ["", "\x00", "abc", "a\U0001F600b", "héllo", "ࠀ"]:
            self.assertEqual(enc_modified_utf8(s), ref.enc_modified_utf8(s))
            self.assertEqual(dec_modified_utf8(enc_modified_utf8(s)),
                             ref.dec_modified_utf8(ref.enc_modified_utf8(s)))

    def test_decode_matches_unseed(self):
        ref = _UnseedRef.load()
        body = bytes([0x1B, 0x1C, 0x68, 0xAC, 0xA7, 0x00, 0x03, 0x00, 0x00, 0x00])
        ours = [(i.off, i.name, i.operands) for i in decode(body)]
        theirs = [(i.off, i.name, i.operands) for i in ref.decode(body)]
        self.assertEqual(ours, theirs)


if __name__ == "__main__":
    unittest.main()
