"""Entry point so ``python -m kombi ...`` works."""

from kombi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
