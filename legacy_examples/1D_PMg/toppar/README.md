# CHARMM toppar — fetch separately

The CHARMM topology/parameter files needed to run this example are not
redistributed here. Download them from one of:

- MacKerell lab: <https://mackerell.umaryland.edu/charmm_ff.shtml> (`toppar_c36_jul24.tgz` or newer)
- CHARMM-GUI: re-export this system from <https://charmm-gui.org/> and the `toppar/` directory comes with it

Drop the standard CHARMM36 stream files into this folder. The driver
`explore_sim_PMg.py` expects at minimum:

- `top_all36_prot.rtf`, `par_all36m_prot.prm`
- `toppar_water_ions.str`
- whatever lipid/RNA streams the original system needed (see `step3_input.psf`)
