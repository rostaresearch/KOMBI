"""Mg2+ first-shell bookkeeping, shared by explore_sim_PMg.py and pbc_equil_prod.py.

One row per logged frame:
  s_A                      Mg-P distance (from positions)
  n_<kind>_O               oxygens within CUTOFF of Mg2+, by kind:
                           water, Pi (H2PO4- O1-O4), GDP_Pbeta (nonbridging O1B/O2B/O3B),
                           GDP_Palpha (nonbridging O1A/O2A), GDP_other (O3A bridge, ribose, base),
                           protein (and anything else)
  d_Ser17OG_A, d_Thr35OG1_A, d_Asp57OD_A
                           Mg2+ distance to Ras (segment PROA) Ser17 OG, Thr35 OG1 and the nearer
                           of Asp57 OD1/OD2
These are what the coordination statistics in the KOMBI SI (section SI.5) need: a third or fourth
water, a second Pbeta oxygen, alpha-beta coordination, Thr35 release and Asp57 entry.
Distances use the minimum image when a box is given.
"""
import numpy as np

MG, P = 7799, 7840
CUTOFF = 3.0      # A; first-shell cutoff (Mg-O ~2.1 A, second shell ~4 A)
KINDS = ("water", "Pi", "GDP_Pbeta", "GDP_Palpha", "GDP_other", "protein")
HEADER = "step,s_A," + ",".join(f"n_{k}_O" for k in KINDS) + ",d_Ser17OG_A,d_Thr35OG1_A,d_Asp57OD_A"


def shell_sets(psf):
    """oxygen indices by kind and the reference atoms, from an OpenMM CharmmPsfFile
    (which names water residues HOH, not TIP3, and the phosphate H2PO4)"""
    idx, kind, ser17, thr35, asp57 = [], [], None, None, []
    for i, a in enumerate(psf.atom_list):
        res = a.residue
        ras = a.system == "PROA"
        if ras and int(res.idx) == 17 and res.resname == "SER" and a.name == "OG": ser17 = i
        if ras and int(res.idx) == 35 and res.resname == "THR" and a.name == "OG1": thr35 = i
        if ras and int(res.idx) == 57 and res.resname == "ASP" and a.name in ("OD1", "OD2"): asp57.append(i)
        if not a.name.startswith("O"):
            continue
        if res.resname in ("HOH", "TIP3"): k = 0
        elif res.resname == "H2PO4": k = 1
        elif res.resname == "GDP": k = 2 if a.name in ("O1B", "O2B", "O3B") else 3 if a.name in ("O1A", "O2A") else 4
        else: k = 5
        idx.append(i); kind.append(k)
    idx, kind = np.array(idx), np.array(kind)
    n = np.bincount(kind, minlength=len(KINDS))
    assert psf.atom_list[MG].name == "MG" and psf.atom_list[P].residue.resname == "H2PO4", "Mg/P indices"
    assert n[0] > 1000 and n[1] == 4 and n[2] == 3 and n[3] == 2 and ser17 is not None and thr35 is not None and len(asp57) == 2, \
        f"shell atom selection: oxygens by kind {dict(zip(KINDS, n.tolist()))}, Ser17 {ser17}, Thr35 {thr35}, Asp57 {asp57}"
    pos = {int(j): q for q, j in enumerate(idx)}
    return {"O_idx": idx, "O_kind": kind, "counts": dict(zip(KINDS, n.tolist())),
            "q_ser17": pos[ser17], "q_thr35": pos[thr35], "q_asp57": [pos[j] for j in asp57]}


def shell_row(x, sets, box=None):
    """x: positions in A, shape (N, 3); box: (Lx, Ly, Lz) in A for a rectangular periodic box, or None.
    returns (s, n_water, n_Pi, n_Pbeta, n_Palpha, n_GDP_other, n_protein, d_Ser17, d_Thr35, d_Asp57)"""
    d = x[sets["O_idx"]] - x[MG]
    dp = x[P] - x[MG]
    if box is not None:
        box = np.asarray(box, dtype=float)
        d = d - box * np.round(d / box)
        dp = dp - box * np.round(dp / box)
    dist = np.linalg.norm(d, axis=1)
    cnt = np.bincount(sets["O_kind"][dist < CUTOFF], minlength=len(KINDS))
    return (float(np.linalg.norm(dp)), *[int(c) for c in cnt],
            float(dist[sets["q_ser17"]]), float(dist[sets["q_thr35"]]), float(dist[sets["q_asp57"]].min()))


def format_row(step, row):
    return "%d,%.4f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n" % ((step,) + tuple(row))
