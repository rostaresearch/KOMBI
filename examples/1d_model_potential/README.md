# 1D model potential

Tilted double-well sin potential on `[0, 2π]` — the smoke-test system from
Fig. 1 of the KOMBI manuscript.

## Run

From the repo root:

```bash
python -m kombi run --config examples/1d_model_potential/config.yaml
# or, equivalently,
python examples/1d_model_potential/run.py
```

## Outputs

Everything lands in `runs/model_1d/` (auto-created):

```
runs/model_1d/
├── bias_history/
│   └── bias_iter_000.npy ... bias_iter_004.npy
├── figures/
│   ├── final_fes.png
│   └── mfpt_convergence.png
├── logs/
└── trajectories/   (empty for the toy system)
```

`final_fes.png` should show the recovered unbiased FES collapsing toward the
known reference (the tilted double-well), with the final bias overlaid;
`mfpt_convergence.png` should show the biased MFPT(start → target) dropping
by a couple of orders of magnitude over the 5 iterations.

Expected wall time: under a minute on any modern laptop.
