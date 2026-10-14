"""Tests for the x86 -> MBA-expression lifter (unnative/deobf/x86expr).

Hand-encoded instruction bytes (capstone decodes them, and the tests assert the
mnemonics, so a wrong encoding fails loudly) are lifted and folded. The headline
case is a real opaque constant: a short run whose result looks register-dependent
but is always 42.
"""

from __future__ import annotations

import unittest

try:
    import capstone
    HAVE_CAPSTONE = True
except ImportError:
    HAVE_CAPSTONE = False

if HAVE_CAPSTONE:
    from unnative.deobf import x86expr


@unittest.skipUnless(HAVE_CAPSTONE, "capstone not installed")
class X86ExprTests(unittest.TestCase):
    def _md(self):
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        return md

    def _disasm(self, code: bytes):
        md = self._md()
        insns = list(md.disasm(code, 0x1000))
        return md, insns

    def test_plain_literal_is_not_reported(self):
        # mov eax, 42 ; ret  -> a bare literal, no free variable, not an MBA constant.
        md, insns = self._disasm(bytes([0xB8, 0x2A, 0, 0, 0, 0xC3]))
        self.assertEqual([i.mnemonic for i in insns], ["mov", "ret"])
        self.assertEqual(x86expr.constants_in(md, insns, 32), {})

    def test_xor_self_zeroing_is_not_reported(self):
        # xor eax, eax ; ret  -> constant 0 but trivial (no variable), not reported.
        md, insns = self._disasm(bytes([0x31, 0xC0, 0xC3]))
        self.assertEqual([i.mnemonic for i in insns], ["xor", "ret"])
        self.assertEqual(x86expr.constants_in(md, insns, 32), {})

    def test_opaque_constant_32bit(self):
        # mov eax,ecx ; xor eax,ecx ; add eax,42 ; ret  ->  (ecx ^ ecx) + 42 == 42.
        code = bytes([0x89, 0xC8, 0x31, 0xC8, 0x83, 0xC0, 0x2A, 0xC3])
        md, insns = self._disasm(code)
        self.assertEqual([i.mnemonic for i in insns], ["mov", "xor", "add", "ret"])
        self.assertEqual(x86expr.constants_in(md, insns, 32), {"rax": 42})

    def test_opaque_constant_64bit(self):
        # mov rax,rcx ; xor rax,rcx ; add rax,42 ; ret  (REX.W) -> 42 at 64-bit.
        code = bytes([0x48, 0x89, 0xC8, 0x48, 0x31, 0xC8, 0x48, 0x83, 0xC0, 0x2A, 0xC3])
        md, insns = self._disasm(code)
        self.assertEqual([i.mnemonic for i in insns], ["mov", "xor", "add", "ret"])
        self.assertEqual(x86expr.constants_in(md, insns, 64), {"rax": 42})

    def test_genuinely_variable_result_is_not_reported(self):
        # mov eax,ecx ; add eax,edx ; ret  ->  ecx + edx, not constant.
        code = bytes([0x89, 0xC8, 0x01, 0xD0, 0xC3])
        md, insns = self._disasm(code)
        self.assertEqual([i.mnemonic for i in insns], ["mov", "add", "ret"])
        self.assertEqual(x86expr.constants_in(md, insns, 32), {})

    def test_unmodelled_load_invalidates_register(self):
        # mov eax,ecx ; xor eax,ecx ; mov eax,[rdi] ; ret
        # the load clobbers eax with unknown memory, so no constant survives.
        code = bytes([0x89, 0xC8, 0x31, 0xC8, 0x8B, 0x07, 0xC3])
        md, insns = self._disasm(code)
        self.assertEqual(insns[2].mnemonic, "mov")       # mov eax, dword [rdi]
        self.assertEqual(x86expr.constants_in(md, insns, 32), {})

    def _blob_call_sites(self, code: bytes):
        md = self._md()
        insns = list(md.disasm(code, 0x1000))
        return insns, x86expr.call_site_constants(md, insns)

    def test_call_site_rcx_r8_immediates(self):
        # mov ecx, 300 ; mov r8d, 300 ; call rax
        code = bytes([0xB9, 0x2C, 0x01, 0x00, 0x00, 0x41, 0xB8, 0x2C, 0x01, 0x00, 0x00,
                      0xFF, 0xD0])
        insns, sites = self._blob_call_sites(code)
        self.assertEqual([i.mnemonic for i in insns], ["mov", "mov", "call"])
        retaddr = insns[-1].address + insns[-1].size
        self.assertEqual(sites, {retaddr: {2: 300, 3: 300}})

    def test_call_site_register_copy_of_unknown_is_not_a_constant(self):
        # mov rcx, rsi ; call rax  -> rcx came from an unwritten register, unknown.
        code = bytes([0x48, 0x89, 0xF1, 0xFF, 0xD0])
        insns, sites = self._blob_call_sites(code)
        self.assertEqual(sites, {})

    def test_call_site_register_copy_of_known_propagates(self):
        # mov esi, 7 ; mov rcx, rsi ; call rax
        # the move is an *expression* copy: rcx provably holds 7 too.
        code = bytes([0xBE, 0x07, 0x00, 0x00, 0x00, 0x48, 0x89, 0xF1, 0xFF, 0xD0])
        insns, sites = self._blob_call_sites(code)
        self.assertEqual(list(sites.values())[0], {2: 7})

    def test_call_site_call_clobbers_caller_saved(self):
        # mov ecx, 300 ; call rax ; call rbx  -> the second call cannot inherit rcx
        code = bytes([0xB9, 0x2C, 0x01, 0x00, 0x00, 0xFF, 0xD0, 0xFF, 0xD3])
        _insns, sites = self._blob_call_sites(code)
        # exactly one call site is proven (the first); the second has no constants
        self.assertEqual(len(sites), 1)
        self.assertEqual(list(sites.values())[0], {2: 300})


if __name__ == "__main__":
    unittest.main()
