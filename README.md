# KOMBI — Kinetics-Optimised Markov-state Bias Iteration

Python package implementing the MFPT-driven adaptive-bias scheme
described in the manuscript. Given a target endpoint on a CV, KOMBI
builds a discrete Markov model of the current trajectory ensemble,
optimises a sum-of-Gaussians bias against the analytic mean first
passage time (closed-form on the rate matrix), runs new biased
dynamics under that bias, and iterates. DHAM is used to recover an
unbiased free-energy surface from the cumulative biased data.

The numerical core is decoupled from the propagator: it works on any
discretised CV with a column-stochastic transition matrix, so the same
optimiser drives synthetic 1D/2D model potentials and OpenMM
all-atom systems behind the same API.

## Layout

```
kombi/
├── src/kombi/             — installable Python package
│   ├── msm.py             — MSM utilities (eigvec, MFPT, bias_K, bias_M)
│   ├── dham.py            — DHAM unbiasing with ESS-per-state diagnostic
│   ├── bias.py            — Gaussian bias + L-BFGS-B optimiser
│   ├── potentials.py      — analytic 1D double-well and 2D Müller-Brown
│   ├── orchestrator.py    — run_kombi_1d loop
│   ├── config.py          — RunConfig dataclass + YAML loader
│   ├── io.py              — Paths() helper (auto-creates every subdir)
│   ├── analysis.py        — plotting + DHAM-based unbiased FES recovery
│   └── cli.py             — `python -m kombi run --config …`
├── tests/                 — pytest suite (15 tests, all green)
├── examples/
│   ├── 1d_model_potential/  — fully working smoke test (≈ 1 min)
│   ├── 2d_model_potential/  — placeholder (Müller-Brown port pending)
│   ├── nacl_ion_pair/       — placeholder (port of legacy_examples/1D_NaCl_TW pending)
│   └── ras_gap_pmg/         — placeholder (port of legacy_examples/1D_PMg pending)
├── legacy_examples/       — original manuscript-run scripts, verbatim
│   ├── 1D_NaCl_TW/        — NaCl association/dissociation, OpenMM+CUDA
│   ├── 1D_PMg/            — Ras–GAP P–Mg coordination, OpenMM+CUDA
│   ├── 2D_Dialanine/      — dialanine φ/ψ, OpenMM
│   ├── 1D_langevin_sim/   — 1D Langevin model potential, custom integrator
│   ├── 2D_langevin_sim/   — 2D Langevin model potential, custom integrator
│   └── 2D_langevin_sim_benchmark_BD/  — 2D Brownian-dynamics benchmark
├── manuscript_scripts/    — figure-generation scripts and figure-backing data
│   └── paper_publication/ — figure_gen_1D.py, figure_gen_2D.py, analysis.py,
│                            reach-time CSVs, rendered PNG/MAT artifacts
├── alt_optimisers/        — alternative bias-optimisation strategies
│   └── CE_Opt/            — Cross-Entropy method variant (vs. L-BFGS-B)
├── pyproject.toml         — PEP 621, pip-installable
├── MECHANISM_AUDIT.md     — protocol for the v2 paper's Ω_J / Ω_ρ / D_edge / R_unsup audit
│                            applied to every KOMBI-optimised system in this repo
└── CITATION.cff
```

## Status

| Component                                | Status |
|------------------------------------------|--------|
| `src/kombi/` numerical core              | refactored, deduplicated, tested |
| `tests/` (15 unit tests)                 | green on Python 3.10/3.11/3.12 |
| 1D model-potential example               | end-to-end verified (MFPT 163 → 106 ps over 3 iters) |
| 2D / NaCl / Ras-GAP examples on the package | **not yet** ported — placeholders |
| `legacy_examples/`                        | TJ's original manuscript-run drivers, copied verbatim |

The legacy scripts run on the older standalone-`MSM.py` codepath
(see <https://github.com/DerienFe/RL_MFPT>) and are kept here for
reproducibility of the published numbers. Porting each onto
`src/kombi/` is staged work; the algorithms are the same, only the
OpenMM driver loop differs per system.

## Install

```
pip install -e ./kombi
```

The `tests/` suite needs only `numpy`, `scipy`, `matplotlib`, `pyyaml`.
The legacy and future OpenMM examples additionally need `openmm`,
`mdtraj`, and a working CUDA build of OpenMM.

## Smoke test

```
python -m kombi run --config kombi/examples/1d_model_potential/config.yaml
```

Output (≤ 1 min on a CPU laptop):

```
[iter 0]  MFPT(start→target) = 163.4 ps
[iter 1]  MFPT(start→target) = 106.7 ps
[iter 2]  MFPT(start→target) = 106.9 ps
```

Plots and per-iteration biases land in `runs/model_1d/`.

## CHARMM toppar

The `legacy_examples/1D_NaCl_TW/` and `legacy_examples/1D_PMg/`
folders need the standard CHARMM36 toppar set, which is not bundled
here (MacKerell licence). Download from
<https://mackerell.umaryland.edu/charmm_ff.shtml> and place the
`toppar/` contents in the example folder. The per-example
`toppar/README.md` lists exactly which streams are needed.

## Reproducing the manuscript

The clean package gives the same MFPT optimisation as the legacy
scripts on the systems where the port has been completed (currently:
1D model potential only). For 2D / NaCl / Ras-GAP, please run the
corresponding `legacy_examples/` driver until the port lands.
