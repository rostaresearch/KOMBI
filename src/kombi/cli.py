"""Command-line entry point: ``python -m kombi run --config <path>``."""

from __future__ import annotations

import argparse
import sys

from kombi.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kombi",
        description="Kinetics-Optimized Markovian Bias Integration",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a KOMBI iteration loop from a config file")
    run_p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    run_p.add_argument("--seed", type=int, default=None, help="Override RunConfig.seed")
    run_p.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose iteration output"
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.verbose:
            cfg.verbose = True
        # late import so `python -m kombi --help` doesn't pay the cost
        from kombi.orchestrator import run_kombi

        if cfg.system == "model_1d":
            # 1D analytic potential needs a reference K matrix; build one
            # from the canonical double-well used in the manuscript.
            import numpy as np

            from kombi.msm import MSM
            from kombi.potentials import double_well_1d

            x = cfg.cv_grid
            y = double_well_1d(x)
            kbT = cfg.kbT
            n = x.size
            K0 = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in (i - 1, i + 1):
                    if 0 <= j < n:
                        u_ij = y[j] - y[i]
                        K0[j, i] = np.exp(u_ij / (2.0 * kbT))
                K0[i, i] = -np.sum(K0[:, i])
            result = run_kombi(cfg, K0=K0, cv_grid=x)
            print(f"Final MFPT: {result.final_mfpt:.3f} ps")
            print(f"Output dir: {result.paths.root}")
            return 0
        else:
            from kombi.orchestrator import run_kombi as _run

            try:
                _run(cfg)
                return 0
            except NotImplementedError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
