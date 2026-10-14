"""Tests for the Phase 4 deobf stage (unnative/deobf/stage).

The engine and the lifter are tested in test_mba / test_x86expr; here the focus is
the stage glue: recovering a constant end-to-end through CFG -> lift -> fold on a
hand-encoded function, and stopping cleanly when its inputs or capstone are absent.
"""

from __future__ import annotations

import os
import unittest

from _fixtures import NATIVE_WITH_LOADER, JavaFixtureCase

from unnative.deobf import stage as deobf
from unnative.lift import cfg as cfgmod
from unnative.workdir import StageNotImplemented, WorkDir

# mov eax,ecx ; xor eax,ecx ; add eax,42 ; ret   ->  (ecx ^ ecx) + 42 == 42
_MBA_FUNC = bytes([0x89, 0xC8, 0x31, 0xC8, 0x83, 0xC0, 0x2A, 0xC3])


@unittest.skipUnless(cfgmod.CAPSTONE_AVAILABLE, "capstone not installed")
class RecoverConstantsTests(unittest.TestCase):
    def test_recovers_opaque_constant_via_cfg(self):
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        base = 0x400000
        reader = cfgmod.bytes_reader(base, _MBA_FUNC)
        found = deobf._recover_constants(md, reader, base)
        self.assertIn(42, found)
        self.assertEqual(found[42], [hex(base)])      # materialised in the entry block


class DeobfStageStopTests(JavaFixtureCase):
    def test_stops_without_bindings(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        wd = WorkDir(jar, root=os.path.join(self.tmp, "work")).ensure()
        if not cfgmod.CAPSTONE_AVAILABLE:
            self.skipTest("capstone gates this on the no-bindings branch")
        with self.assertRaises(StageNotImplemented) as cm:
            deobf.run_stage(wd)                        # no bindings.json / libs yet
        self.assertEqual(cm.exception.phase, "Phase 4")


if __name__ == "__main__":
    unittest.main()
