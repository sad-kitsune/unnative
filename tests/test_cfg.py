"""Tests for static CFG recovery (unnative/lift/cfg).

capstone is an optional dependency, so these skip when it is absent (as under the
plain system Python); they run under the project's venv. Fixtures are hand-encoded
x86-64 byte sequences so the expected block/edge/loop structure is known exactly.
"""

from __future__ import annotations

import unittest

from unnative.lift import cfg


@unittest.skipUnless(cfg.CAPSTONE_AVAILABLE, "capstone not installed")
class CfgTests(unittest.TestCase):
    def _cfg(self, base, hexbytes):
        blob = bytes.fromhex(hexbytes.replace(" ", ""))
        return cfg.build_cfg(cfg.bytes_reader(base, blob), base)

    def test_conditional_diamond(self):
        # test rdi,rdi ; je +1 ; nop ; ret  -> 3 blocks, 3 edges, no loop
        c = self._cfg(0x1000, "4885ff 7401 90 c3")
        s = c.to_json()["stats"]
        self.assertEqual(s["blocks"], 3)
        self.assertEqual(s["edges"], 3)
        self.assertFalse(s["has_loops"])
        self.assertFalse(s["truncated"])
        # the entry block ends in a two-way conditional branch
        self.assertEqual(c.blocks[0x1000].terminator, "jcc")
        self.assertEqual(sorted(c.blocks[0x1000].succs), [0x1005, 0x1006])
        self.assertEqual(c.blocks[0x1006].terminator, "ret")

    def test_self_loop_is_a_back_edge(self):
        # nop ; jmp -3  -> one block that jumps to itself
        c = self._cfg(0x2000, "90 ebfd")
        s = c.to_json()["stats"]
        self.assertEqual(s["blocks"], 1)
        self.assertTrue(s["has_loops"])
        self.assertEqual(s["back_edges"], 1)
        self.assertEqual(c.blocks[0x2000].succs, [0x2000])

    def test_indirect_jump_is_open_exit(self):
        # jmp rax  -> unresolved target, recorded as an indirect exit, not guessed
        c = self._cfg(0x3000, "ffe0")
        s = c.to_json()["stats"]
        self.assertEqual(s["indirect_exits"], 1)
        self.assertEqual(c.blocks[0x3000].terminator, "ijmp")
        self.assertEqual(c.blocks[0x3000].succs, [])

    def test_calls_do_not_split_blocks(self):
        # call rel32 (to self+5, i.e. next insn) ; ret  -> one block, one call
        c = self._cfg(0x4000, "e800000000 c3")
        s = c.to_json()["stats"]
        self.assertEqual(s["blocks"], 1)
        self.assertEqual(s["calls"], 1)
        self.assertEqual(c.blocks[0x4000].terminator, "ret")


if __name__ == "__main__":
    unittest.main()
