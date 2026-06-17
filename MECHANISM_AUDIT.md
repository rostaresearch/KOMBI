# Mechanism-conservation audit for KOMBI-optimised biases

This protocol applies the four mechanism-conservation diagnostics from
the MSM Roundtable 2026 paper (`rosta_jctc_draft_v2.tex`, lines 650–825)
to each KOMBI-optimised system in this repository. The goal is to
report, for every example KOMBI accelerates, how strongly the
MFPT-optimal bias distorts the unbiased reactive mechanism and how much
of the biased flux is routed through under-sampled edges.

Run once after a KOMBI run converges; the output is a small per-system
table and a companion bias-profile figure.

## Why audit

KOMBI minimises $\log\tau_{AB}(\mathbf{K}_b)$ on the rate matrix. On
multi-channel landscapes this objective can yield large speedups while
re-routing the reactive current onto a disjoint mechanism. The v2 paper
shows that on a 9-well landscape the MFPT-optimal bias gives a
$2.4\times 10^4$ acceleration with $\Omega_J=0.67$ — i.e. ⅓ of the
unbiased current is lost. On a real-protein MSM (Ras+GDP+Pi, see
`MSM_Roundtable_2026/ras_msm_demo/`) the corresponding numbers are
$9\times$ speedup with $\Omega_J\approx 0.51$ and 47 % unsupported flux.

The four diagnostics are:

| symbol | definition | what it tells you |
|---|---|---|
| $\Omega_J$ | $\sum_{ij}\min(\widehat J^{(0)}_{ij},\widehat J^{(b)}_{ij})$ | fraction of unbiased reactive current preserved |
| $\Omega_\rho$ | $\sum_i\min(\rho^{(0)}_i,\rho^{(b)}_i),\ \rho_i\propto\pi_i q_i(1-q_i)$ | overlap of the transition-path density on states |
| $D_{\rm edge}$ | $\tfrac12\sum_{ij}\widehat J^{(0)}_{ij}\,|b_j-b_i|$ | current-weighted edge-rate distortion |
| $R_{\rm unsup}$ | $\sum_{ij}\widehat J^{(b)}_{ij}\,\bigl(1-\tfrac{C_{ij}}{C_{ij}+C_0}\bigr)$ | fraction of biased flux routed through low-count edges |

All four are computed from a reversible MSM, a chosen $A,B$ pair, and a
bias vector $\mathbf{b}$ on the discretised states.

## What you need per system

| input | source |
|---|---|
| $\mathbf{K}^{(0)}$ | unbiased reference rate matrix on the chosen state discretisation |
| $C_{ij}$ | unbiased transition counts on the same discretisation (for $R_{\rm unsup}$) |
| $\mathbf{b}^*$ | KOMBI's converged bias (e.g. `runs/<system>/bias_history/bias_iter_last.npy`) |
| $A,B$ | source / target state index sets |

If you only have the analytic potential (1D / 2D model systems),
$\mathbf{K}^{(0)}$ and the "counts" $C_{ij}$ are both produced directly
from the discretised drift: see Step 1 below for the closed-form path.

## Per-system parameter table

| system | folder | unbiased data | discretisation | suggested $A,B$ | lag (frames) |
|---|---|---|---|---|---|
| 1D model potential | `examples/1d_model_potential/` | analytic $V(x)$ in `potentials.py` | 50 bins on $[0,2\pi]$ | $A=\{0,1\}$, $B=\{48,49\}$ | 1 |
| 2D Müller–Brown | `legacy_examples/2D_langevin_sim/` | `plain_multi_well_2D_fes.py` analytic FES | $30\times30$ grid | well-A index, well-B index | 1 |
| 1D Langevin double-well | `legacy_examples/1D_langevin_sim/` | `langevin_sim_unbiased.py` trajectory | 50 bins on the sampled range | bottom-2 / top-2 bins | 1 |
| 2D Langevin grid | `legacy_examples/2D_langevin_sim/` | `langevin_sim_unbiased.py` trajectory | $30\times30$ on $(x,y)$ box | corner-A / corner-B | 1 |
| 2D Brownian-dynamics benchmark | `legacy_examples/2D_langevin_sim_benchmark_BD/` | `langevin_sim_unbiased.py` BD trajectory | $30\times30$ | as 2D Langevin | 1 |
| NaCl ion pair | `legacy_examples/1D_NaCl_TW/` | `unbias_NaCl_sim.py` + `Unbiased_Profile.npy` | 30 distance bins on $[2,9]$ Å | contact $<3$ Å / dissociated $>7$ Å | 50 |
| Ras–GAP P–Mg | `legacy_examples/1D_PMg/` | `explore_sim_PMg.py` initial unbiased segment | 30 P–Mg distance bins | bound $<2.8$ Å / released $>5$ Å | 100 |
| 2D dialanine | `legacy_examples/2D_Dialanine/` | `unbiased_sim.py` $\phi,\psi$ trajectory | $36\times36$ $\phi,\psi$ bins | C7eq basin / C7ax basin | 200 |

(All "lag (frames)" numbers refer to the dcdfreq written by each
driver, not the integration step. Adjust based on the implied
implied-timescale-test on each system.)

## Protocol

The pipeline mirrors `MSM_Roundtable_2026/ras_msm_demo/build_ras_msm.py`,
which is the canonical reference implementation. Re-use its helpers
wherever possible; only Step 1 differs system-by-system.

### Step 1 — Build $\mathbf{K}^{(0)}$ and $C_{ij}$

**Analytic systems (1D / 2D model, Müller–Brown).** Discretise the
potential to a grid, build the symmetric nearest-neighbour rate matrix
from a Smoluchowski-style hopping rule
$K^{(0)}_{ij}\propto\exp\!\bigl(-(V_j-V_i)/(2k_BT)\bigr)$, and treat
$C_{ij}=N_{\rm syn}\cdot \pi_i K^{(0)}_{ij}\Delta t$ as a synthetic
count matrix at a fictitious sample size $N_{\rm syn}$ (set $N_{\rm syn}$
to the actual number of frames in any biased trajectory you care to
compare against, otherwise $R_{\rm unsup}$ has no scale).
`src/kombi/msm.py` already implements the 1D nearest-neighbour fast
path used by KOMBI itself.

**Empirical-trajectory systems (NaCl, P–Mg, dialanine, Langevin
benchmarks).** Featurise each unbiased trajectory onto the chosen
discretisation, then compute the count matrix at the chosen lag:

```python
from kombi.msm import count_matrix, reversible_T, generator_from_T

C   = count_matrix(dtrajs, n_states, lag=LAG)   # discrete-traj counts
Csym = C + C.T
T, pi = reversible_T(Csym, n_iter=200)
K0    = generator_from_T(T, lag_time=1.0)       # in units of the lag
```

If `kombi.msm` does not yet expose `count_matrix` / `reversible_T` /
`generator_from_T` as standalone helpers, copy them from
`MSM_Roundtable_2026/ras_msm_demo/build_ras_msm.py` (≈ 30 lines).

### Step 2 — Pull KOMBI's converged $\mathbf{b}^*$

KOMBI writes one Gaussian-bias snapshot per iteration to
`runs/<system>/bias_history/bias_iter_NNN.npy`. The last one is the
converged bias. Project it onto the same discretisation used in
Step 1:

```python
from kombi.bias import GaussianBias

gb = GaussianBias.load("runs/<system>/bias_history/bias_iter_last.npy")
b_star = gb.evaluate_on_grid(centres)            # 1D: centres of the bins
                                                  # 2D: flattened grid centres
b_star -= b_star[A_index]                         # gauge: zero on A
```

For 2D systems unfold the $n_x\times n_y$ grid to a length-$n_xn_y$
vector so it can be used with the 1D-state-indexed helpers.

### Step 3 — Tilt, committor, fluxes, four metrics

```python
from build_ras_msm import (tilt_K, commit_pi, reactive_current,
                            tube_density, edge_count_scale, metrics,
                            mfpt_AB)

Kb     = tilt_K(K0, b_star)
q0     = commit_pi(K0, A, B, pi)
qb     = commit_pi(Kb, A, B, pi)              # pi unchanged under symm. tilt
tau0   = mfpt_AB(K0, A, B)
tau_b  = mfpt_AB(Kb, A, B)
C0     = edge_count_scale(C)                  # median nonzero off-diag count

Omega_J, Omega_rho, D_edge, R_unsup = metrics(
    K0, Kb, pi, pi, q0, qb, b_star, C, C0
)
print(f"speedup = {tau0/tau_b:.1f}x, Omega_J = {Omega_J:.3f}, "
      f"Omega_rho = {Omega_rho:.3f}, D_edge = {D_edge:.3f}, "
      f"R_unsup = {R_unsup:.3f}")
```

`tilt_K` uses the detailed-balance-preserving form
$K^{(b)}_{ij}=K^{(0)}_{ij}\exp\!\bigl((b_j-b_i)/2\bigr)$ for $i\neq j$,
with the diagonal adjusted so row sums vanish. The stationary $\pi$ is
identical for $\mathbf{K}^{(0)}$ and $\mathbf{K}^{(b)}$ under this tilt.

### Step 4 — Bootstrap

Use the two-level scheme from the Ras demo: resample replicates with
replacement (outer), then row-Dirichlet on the resulting count matrix
(inner). The outer level matters whenever you have a small number of
independent trajectories, as in every empirical example here.

```python
from build_ras_msm import bootstrap_metrics

boot = bootstrap_metrics(
    dtrajs, n_states=N_STATES, lag=LAG, A=A, B=B,
    b_star=b_star, n_boot=500, rng=np.random.default_rng(42)
)
valid = boot[~np.isnan(boot).any(1)]
med = np.median(valid, axis=0)
lo  = np.quantile(valid, 0.025, axis=0)
hi  = np.quantile(valid, 0.975, axis=0)
```

For the analytic systems, replace the trajectory resample with a
parametric bootstrap: draw each row of $C$ from a Dirichlet centred on
$\pi_i K^{(0)}_{ij}$ at the chosen $N_{\rm syn}$.

### Step 5 — Report

Emit one LaTeX `tabular` per system in the same shape as the Ras-MSM
table (`MSM_Roundtable_2026/tab_ras_msm.tex`). Optionally produce a
2-panel figure: (a) $\mathbf{b}^*$ on the discretisation with the $A,B$
basins shaded, and (b) bootstrap distributions of the four metrics.

A driver that loops over every system and produces a single combined
table is straightforward; the per-system Step-1 wrapper is the only
piece that varies.

## Suggested combined table

```latex
\begin{table}[!h]
\centering
\caption{Mechanism-conservation diagnostics for the KOMBI-optimised
bias $\mathbf{b}^*$ across all systems in this repository. Medians and
95\% CIs from 500 two-level bootstrap MSMs.}
\label{tab:kombi-mechanism-audit}
\begin{tabular}{lcccccc}
\hline
system & dim & $\tau_{AB}/\tau_{AB}^*$ &
 $\Omega_J$ & $\Omega_\rho$ & $D_{\rm edge}$ & $R_{\rm unsup}$ \\
\hline
1D model potential        & 1 & ... & ... & ... & ... & ... \\
1D Langevin double-well   & 1 & ... & ... & ... & ... & ... \\
2D Müller--Brown          & 2 & ... & ... & ... & ... & ... \\
2D Langevin grid          & 2 & ... & ... & ... & ... & ... \\
2D BD benchmark           & 2 & ... & ... & ... & ... & ... \\
NaCl ion pair             & 1 & ... & ... & ... & ... & ... \\
Ras--GAP P--Mg            & 1 & ... & ... & ... & ... & ... \\
2D dialanine              & 2 & ... & ... & ... & ... & ... \\
\hline
\end{tabular}
\end{table}
```

A reader can then see at a glance which KOMBI examples preserve mechanism
($\Omega_J\gtrsim 0.7$, $R_{\rm unsup}\lesssim 0.2$) vs. which trade
mechanism for raw acceleration ($\Omega_J\lesssim 0.5$).

## Practical knobs and pitfalls

- **$C_0$ calibration.** Use `edge_count_scale(C)` (median nonzero
  off-diagonal count), not the median row-sum. The row-sum choice is
  too strict for MSMs whose off-diagonal counts are an order of
  magnitude smaller than the diagonal — every transition then looks
  "unsupported".

- **Bias gauge.** $\mathbf{b}^*$ is defined up to an additive constant
  (the tilt is invariant). Always subtract $\mathbf{b}^*[A]$ before
  reporting $D_{\rm edge}$, otherwise the absolute scale of $b^*$ on
  $A$ leaks into the distortion.

- **Bias bounds.** KOMBI's converged $\mathbf{b}^*$ may exceed the
  $U_{\max}$ budget the v2 paper uses for cross-system comparison
  ($U_{\max}=6\,k_BT$). For a clean comparison, evaluate the metrics
  along $\alpha\mathbf{b}^*$ for $\alpha\in\{0.25,0.5,0.75,1\}$ and
  report the curve, not just the endpoint.

- **Implied timescales.** Pick the lag $\tau$ from the
  implied-timescale plateau of the unbiased MSM, not arbitrarily. The
  default `lag=5 frames` in the Ras demo was checked this way for the
  Ras system; do the analogous check per system here.

- **Reproducibility seed.** All bootstrap draws should pass an explicit
  `np.random.default_rng(seed)` so the table can be regenerated bit-
  for-bit from a CI run.

## Reference

- Paper: `MSM_Roundtable_2026/rosta_jctc_draft_v2.tex`, sections
  "Mechanism conservation" (around line 630), "Implications for
  protein simulations" (line 801), unsupported-flux equation (line 825).

- Reference implementation: `MSM_Roundtable_2026/ras_msm_demo/`
  – `build_ras_msm.py` exports `tilt_K`, `commit_pi`,
    `reactive_current`, `tube_density`, `edge_count_scale`, `metrics`,
    `bootstrap_metrics`, `optimise_b`, `mfpt_AB`.

- Ras+GDP+Pi worked example: `MSM_Roundtable_2026/tab_ras_msm.tex`,
  `MSM_Roundtable_2026/fig_ras_msm_demo.pdf`.
