#!/usr/bin/env python3
"""Run the BBDM analytical and synthetic experiments on CPU."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "figure2": "clouds",
    "figure3": "analytic",
    "figure5": "analytic",
    "figure6": "dimension",
    "figure7": "clouds",
    "table-d6": "analytic",
}
SYNTHETIC = {"figure2", "figure6", "figure7"}
MODEL_OPTIONS = ("components", "dimension", "steps", "seed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["all", *EXPERIMENTS])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller simulation/search budgets to check the installation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU workers or numerical-library threads (default: up to 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="Parent directory for generated results (default: results/).",
    )
    model = parser.add_argument_group("model settings (figure2, figure6 or figure7)")
    model.add_argument(
        "--components",
        type=int,
        help="Number of Gaussians; for figure6, select one component count.",
    )
    model.add_argument(
        "--dimension",
        type=int,
        help="Data dimension; for figure6, select one dimension.",
    )
    model.add_argument("--steps", type=int, help="Reverse steps (default: 20).")
    model.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: 1234 for figure2/7, 0 for figure6).",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error(
            "The pinned dependencies require Python 3.11 or newer; Python 3.12 was tested."
        )
    if args.workers < 1:
        parser.error("--workers must be positive.")
    overrides = {
        name: getattr(args, name)
        for name in MODEL_OPTIONS
        if getattr(args, name) is not None
    }
    if overrides and args.experiment not in SYNTHETIC:
        parser.error(
            "Model settings require figure2, figure6 or figure7. "
            "Choose one of these experiments instead of all to customize it."
        )
    for name, minimum in (
        ("components", 1),
        ("dimension", 1),
        ("steps", 2),
        ("seed", 0),
    ):
        if name in overrides and overrides[name] < minimum:
            parser.error(f"--{name} must be at least {minimum}.")
    if args.experiment == "figure6" and overrides.get("components") == 1:
        parser.error("Figure 6 needs at least two Gaussians for pairwise separation.")

    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode = "quick" if args.quick else "full"
    print(f"CPU experiments: {mode} mode. Results: {output}", flush=True)
    if args.quick:
        print(
            "Quick mode uses reduced budgets; omit --quick for full simulation budgets.",
            flush=True,
        )
    if overrides:
        print(
            "Requested model settings: "
            + ", ".join(f"{name}={value}" for name, value in overrides.items()),
            flush=True,
        )

    # Child scripts may deliberately increase one numerical thread pool or use
    # several single-threaded processes. Start with one thread to avoid nesting.
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env["MPLBACKEND"] = "Agg"
    summary = {
        "mode": mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "workers": args.workers,
        "model_overrides": overrides,
        "experiments": [],
    }
    for name in selected:
        module = ROOT / "experiments" / EXPERIMENTS[name] / "run.py"
        directory = output / name
        command = [
            sys.executable,
            str(module),
            "--experiment",
            name,
            "--output",
            str(directory),
            "--workers",
            str(args.workers),
        ]
        if args.quick:
            command.append("--quick")
        for option_name, value in overrides.items():
            command.extend([f"--{option_name}", str(value)])
        print(f"\nRunning {name}...", flush=True)
        started = time.perf_counter()
        subprocess.run(command, env=env, check=True)
        elapsed = time.perf_counter() - started
        summary["experiments"].append(
            {"experiment": name, "directory": directory.name, "seconds": elapsed}
        )
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Finished {name} in {elapsed:.1f} seconds.", flush=True)
    print(f"\nDone. Open the PNG plots and CSV/JSON results in {output}.")


if __name__ == "__main__":
    main()
