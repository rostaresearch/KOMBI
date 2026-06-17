# CHARMM toppar — fetch separately

The CHARMM topology/parameter files needed to run this example are not
redistributed here. Download `toppar_c36_jul24.tgz` (or newer) from
MacKerell's site: <https://mackerell.umaryland.edu/charmm_ff.shtml>.

The driver `unbias_NaCl_sim.py` expects `toppar_water_ions.str` here.
The simulations use a **modified Na–Cl Lennard-Jones combining rule**
(−6.0 kcal/mol at 2.5 Å) — see `unbias_NaCl_sim.py:58` for details.
The corresponding `step3_input.{psf,pdb,crd}` from CHARMM-GUI also
needs to live alongside this folder.
