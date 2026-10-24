"""Consistency checks of KOMBI Ras runs.

1. Per-step bias record: max |V_openmm - V_python| and the size of the applied bias.
2. Bias switch at propagation boundaries: at each boundary only the bias switches, so the jump in
   the per-step second difference of s, regressed on the jump in -V'(s)/mu of the intended bias
   (applied form, once), has slope lambda = the number of times the bias acts. A run that applies
   the bias once gives ~1; the placebo at mid-propagation gives ~0.
3. First passages (first_passage.json) and censored runs.

usage: python audit_runs.py runs/nonpbc_run*  [--out audit.json]
"""
import argparse, glob, json, os
import numpy as np

DT = 0.002                                   # ps
M_MG, M_P = 24.3050, 30.9738
MU = M_MG * M_P / (M_MG + M_P)
N_G = 10


def acc_bias(s_nm, params):
    """-V'(s)/mu in nm/ps^2; params [a(10) kcal/mol, b(10) A, c(10) A], applied form, once"""
    a, b, c = params[:N_G] * 4.184, params[N_G:2 * N_G] / 10, params[2 * N_G:] / 10
    z = s_nm[:, None] - b[None, :]
    dv = np.sum(a * np.exp(-z ** 2 / (2 * c ** 2)) * (-z / c ** 2), axis=1)
    return -dv / MU


def jump(w, t, J, p_old, p_new):
    yc = lambda c: (w[c + 1] - 2 * w[c] + w[c - 1]) / DT ** 2
    old = np.arange(t - J, t); new = np.arange(t, t + J)
    dy = np.mean([yc(c) for c in new]) - np.mean([yc(c) for c in old])
    dx = np.mean(acc_bias(w[new], p_new)) - np.mean(acc_bias(w[old], p_old))
    return dy, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="audit_runs.json")
    args = ap.parse_args()
    runs = sorted(set(d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(os.path.join(d, "cv"))))
    rec = {J: {"switch": [], "placebo": []} for J in (1, 2, 3, 5)}
    per_run = {}
    for d in runs:
        files = sorted(glob.glob(os.path.join(d, "cv", "prop_[0-8]*.npz")))
        files = [f for f in files if not f.endswith("prop_9999.npz")]   # --check-only output
        if not files:
            continue
        Z = [np.load(f) for f in files]
        dv = max(float(np.abs(z["V_openmm_kcal"] - z["V_python_kcal"]).max()) for z in Z)
        vmax = max(float(z["V_openmm_kcal"].max()) for z in Z)
        fp = json.load(open(os.path.join(d, "first_passage.json"))) if os.path.exists(os.path.join(d, "first_passage.json")) else None
        # optimiser health: a model MFPT that is not finite and positive marks a broken decision;
        # n_nonfinite_mfpt counts failed evaluations
        broken = nonfin = 0
        for f in glob.glob(os.path.join(d, "decisions", "prop_*.npz")):
            z = np.load(f, allow_pickle=True)
            if "optimised_mfpt_ns" in z.files:
                m = float(z["optimised_mfpt_ns"])
                broken += int(not (np.isfinite(m) and m > 0))
                nonfin += int(z["n_nonfinite_mfpt"]) if "n_nonfinite_mfpt" in z.files else 0
        per_run[os.path.basename(d)] = {"n_props": len(Z), "max_abs_dV_kcal": dv, "max_applied_bias_kcal": vmax,
                                        "max_s_A": max(float(z["s_A"].max()) for z in Z),
                                        "broken_decisions": broken, "nonfinite_mfpt_evaluations": nonfin,
                                        "first_passage": fp}
        name = os.path.basename(d)
        for k in range(len(Z) - 1):
            s0, s1 = Z[k]["s_A"] / 10, Z[k + 1]["s_A"] / 10
            w = np.concatenate([s0[-12:], s1[:12]])
            m = len(s0) // 2
            wp = s0[m - 12:m + 12]
            for J in rec:
                rec[J]["switch"].append((name, *jump(w, 11, J, Z[k]["params"], Z[k + 1]["params"])))
                rec[J]["placebo"].append((name, *jump(wp, 11, J, Z[k]["params"], Z[k + 1]["params"])))
    res = {"runs": per_run, "n_boundaries": len(rec[1]["switch"])}
    rng = np.random.default_rng(11)
    for J in rec:
        res[f"J{J}"] = {}
        for lab in ("switch", "placebo"):
            R = rec[J][lab]
            if not R:
                continue
            names = np.array([r for r, _, _ in R]); dy = np.array([a for _, a, _ in R]); dx = np.array([b for _, _, b in R])
            slope = float(np.sum(dx * dy) / np.sum(dx * dx))
            ur = np.unique(names); idx = {r: np.flatnonzero(names == r) for r in ur}
            bs = []
            for _ in range(2000):
                sel = np.concatenate([idx[r] for r in rng.choice(ur, len(ur))])
                bs.append(np.sum(dx[sel] * dy[sel]) / np.sum(dx[sel] * dx[sel]))
            res[f"J{J}"][lab] = {"lambda": slope, "boot_sd": float(np.std(bs, ddof=1)), "n": int(len(dx))}
    fps = [v["first_passage"] for v in per_run.values() if v["first_passage"]]
    t = [f["first_passage_ps"] for f in fps if f["reached"]]
    res["first_passage_summary"] = {"n_finished": len(fps), "n_reached": len(t),
                                    "mean_ps": float(np.mean(t)) if t else None,
                                    "sd_ps": float(np.std(t, ddof=1)) if len(t) > 1 else None,
                                    "censored_simulated_ps": [f["simulated_ps"] for f in fps if not f["reached"]]}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    worst = max((v["max_abs_dV_kcal"] for v in per_run.values()), default=0.0)
    print(f"max |V_openmm - V_python| over all runs: {worst:.2e} kcal/mol")


if __name__ == "__main__":
    main()
