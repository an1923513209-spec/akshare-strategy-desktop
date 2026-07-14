"""Smoke-test the local A-share strategy research environment."""

from __future__ import annotations

import importlib
import sys


MODULES = [
    "akshare",
    "backtesting",
    "vectorbt",
    "optuna",
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")

    print("OK")


if __name__ == "__main__":
    main()
