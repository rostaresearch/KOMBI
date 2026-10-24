# KOMBI

Simulation code for *Kinetics-Optimized Enhanced Sampling Using Mean First Passage Times* by Tiejun Wei, Balint Dudas and Edina Rosta.

KOMBI uses DHAM models with directed transition counts to optimize a Gaussian bias by minimizing a mean first-passage-time objective.

## Contents

| Folder | Contents |
|---|---|
| `simulations/1D_langevin_sim` | One-dimensional model potential and simulation drivers |
| `simulations/2D_langevin_sim` | Two-dimensional model potential and simulation drivers |
| `simulations/1D_NaCl` | Ion-pair simulation drivers; production PME configuration |
| `simulations/1D_Ras_PMg` | Ras phosphate-release driver and its regression tests |

Use Python 3.12 and install the dependencies with `python -m pip install -r requirements.txt`. Run each driver from its own simulation directory so that its local `config.py` and `util.py` are used. The production environments used OpenMM 8.1 for the analytical and NaCl systems and 8.2 for Ras.


