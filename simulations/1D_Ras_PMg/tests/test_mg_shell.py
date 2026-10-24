"""mg_shell on the real start structure: Mg2+ in traj_0.restart.pdb is octahedral (Ser17 OG, Thr35 OG1,
one GDP Pbeta oxygen, one Pi oxygen, two waters). Needs the system files: set RAS_SYSTEM_DIR to the
folder with step3_input.psf and traj_0.restart.pdb (skipped otherwise)."""
import os
import numpy as np
import pytest

import mg_shell

SYS = os.environ.get("RAS_SYSTEM_DIR")
pytestmark = pytest.mark.skipif(not SYS or not os.path.exists(os.path.join(SYS or "", "step3_input.psf")),
                                reason="set RAS_SYSTEM_DIR to the Ras system folder")


@pytest.fixture(scope="module")
def system():
    from openmm.app import CharmmPsfFile, PDBFile
    import openmm.unit as unit
    psf = CharmmPsfFile(os.path.join(SYS, "step3_input.psf"))
    x = np.asarray(PDBFile(os.path.join(SYS, "traj_0.restart.pdb")).getPositions().value_in_unit(unit.angstrom))
    return psf, x


def test_selection_counts(system):
    psf, _ = system
    sets = mg_shell.shell_sets(psf)
    assert sets["counts"] == {"water": 30664, "Pi": 4, "GDP_Pbeta": 3, "GDP_Palpha": 2, "GDP_other": 6, "protein": 744}


def test_start_structure_is_octahedral(system):
    psf, x = system
    row = mg_shell.shell_row(x, mg_shell.shell_sets(psf))
    s, n_w, n_pi, n_pb, n_pa, n_go, n_prot, d_ser, d_thr, d_asp = row
    assert s == pytest.approx(3.32, abs=0.01)
    assert (n_w, n_pi, n_pb, n_pa, n_go, n_prot) == (2, 1, 1, 0, 0, 2)
    assert d_ser < 2.5 and d_thr < 2.5 and d_asp > 3.0
    # the minimum image changes nothing for a compact frame in a large box
    assert mg_shell.shell_row(x, mg_shell.shell_sets(psf), box=(100.0, 100.0, 100.0)) == row
    line = mg_shell.format_row(500, row)
    assert line.count(",") == mg_shell.HEADER.count(",")
