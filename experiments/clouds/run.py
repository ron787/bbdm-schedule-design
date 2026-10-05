#!/usr/bin/env python3
"""Reproduce Figure 2 or Figure 7 using CPU-only NumPy/SciPy calculations."""

import argparse
import csv
import json
import os
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["figure2", "figure7"], required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for CSV, JSON, and PNG results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU BLAS/OpenMP threads, not GPU workers.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=32,
        help="Number of Gaussian components (paper: 32).",
    )
    parser.add_argument(
        "--dimension", type=int, default=512, help="Signal dimension (paper: 512)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Reverse transitions, at least 2 (paper: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Nonnegative random seed (paper: 1234).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.components < 1:
        parser.error("--components must be a positive integer")
    if args.dimension < 1:
        parser.error("--dimension must be a positive integer")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, schedules, experiment, output, custom_settings=None):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = (
        ["#6a6a6a", "#440154", "#21918c", "#5ec962", "#fde725"]
        if experiment == "figure2"
        else ["#6a6a6a", "#fde725"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
    for schedule, color in zip(schedules, colors):
        for kind, marker, alpha in [("frozen", "o", 0.25), ("exact", "x", 0.6)]:
            points = np.array(
                [
                    [r["mse_gt"], r["sliced_w2"]]
                    for r in rows
                    if r["schedule_id"] == schedule["id"] and r["sampler_kind"] == kind
                ]
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                marker=marker,
                s=13,
                color=color,
                alpha=alpha,
                linewidths=0.6,
            )
            if kind == "exact":
                parameters = ", ".join(f"{v:.2f}" for v in schedule["parameters"])
                label = schedule["label"]
                if schedule["lambda_mse"] is not None:
                    label = f"$\\lambda={schedule['lambda_mse']:g}$"
                ax.scatter(
                    *points.mean(axis=0),
                    marker="X",
                    s=70,
                    color=color,
                    edgecolor="black",
                    linewidth=0.55,
                    label=f"{label} ({parameters})",
                )
    if experiment == "figure2":
        for sid, marker, color, label in [
            ("posterior_sampler", "^", "#0072b2", "Posterior sampler"),
            ("mmse_estimator", "D", "#111111", "MMSE estimator"),
        ]:
            points = np.array(
                [[r["mse_gt"], r["sliced_w2"]] for r in rows if r["schedule_id"] == sid]
            )
            ax.scatter(
                points[:, 0], points[:, 1], marker=marker, s=12, color=color, alpha=0.5
            )
            ax.scatter(
                *points.mean(axis=0),
                marker=marker,
                s=60,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=label,
            )
    ax.set(xlabel="MSE to true $x_0$", ylabel="Sliced $W_2$")
    if custom_settings is not None:
        components, dimension, steps, seed = custom_settings
        ax.set_title(
            f"Custom run: R={components}, d={dimension}, S={steps}, seed={seed}",
            fontsize=10,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    position = (
        {}
        if experiment == "figure2"
        else {"loc": "center right", "bbox_to_anchor": (0.99, 0.48)}
    )
    if custom_settings is not None:
        position = {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0)}
    legend = ax.legend(
        fontsize=7,
        framealpha=0.9,
        title="Schedule: $(\\alpha,\\beta,c,\\gamma)$",
        title_fontsize=8,
        **position,
    )
    ax.add_artist(legend)
    handles = [
        Line2D(
            [],
            [],
            color="#555555",
            marker=marker,
            linestyle="None",
            markersize=size,
            label=label,
        )
        for marker, size, label in [
            ("o", 4, "Frozen per measurement"),
            ("x", 5, "Exact per measurement"),
            ("X", 7, "Exact mean"),
        ]
    ]
    position = (
        {"loc": "lower left", "bbox_to_anchor": (0.01, 0.015)}
        if experiment == "figure2"
        else {"loc": "center left", "bbox_to_anchor": (0.01, 0.48)}
    )
    if custom_settings is not None:
        position = {"loc": "lower left", "bbox_to_anchor": (1.01, 0.0)}
    ax.legend(
        handles=handles,
        fontsize=7,
        framealpha=0.92,
        title="Markers",
        title_fontsize=7,
        **position,
    )
    save_options = (
        {"bbox_inches": "tight", "bbox_extra_artists": [legend]}
        if custom_settings is not None
        else {}
    )
    fig.savefig(output / f"{experiment}.png", dpi=180, **save_options)
    plt.close(fig)


def main():
    args = parse_args()
    # Configure CPU threads before importing libraries that initialize BLAS.
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[variable] = str(args.workers)
    import numpy as np
    import scipy
    import model

    started = time.perf_counter()
    measurements, samples, projections = 100, 256, 256
    steps, restarts, endpoint_eps = args.steps, 16, 1e-6
    args.output.mkdir(parents=True, exist_ok=True)
    problem = model.make_problem(
        args.experiment, args.seed, components=args.components, dimension=args.dimension
    )
    custom_settings = (args.components, args.dimension, steps, args.seed)
    if custom_settings == (32, 512, 20, 1234):
        custom_settings = None
    schedules = []

    def add_schedule(sid, label, parameters, weight=None):
        w2, mse = model.objectives(problem, parameters, steps)
        schedules.append(
            {
                "id": sid,
                "label": label,
                "parameters": list(map(float, parameters)),
                "lambda_mse": weight,
                "surrogate_w2_squared": w2,
                "surrogate_mse": mse,
            }
        )

    if args.experiment == "figure2":
        add_schedule("default", "default", model.DEFAULT)
        for index, weight in enumerate([0.0, 0.5, 0.75, 1.0]):
            parameters = model.optimize(
                problem, steps, weight, args.seed + 101 * (index + 1), restarts
            )
            add_schedule(
                "blend_lam_" + f"{weight:.2f}".replace(".", "p"),
                f"lambda={weight:g}",
                parameters,
                weight,
            )
    else:
        add_schedule("MSE-oriented", "MSE-oriented", model.MSE_EDGE)
        add_schedule("W2-oriented", "W2-oriented", model.W2_EDGE)
    caches = {
        s["id"]: model.reverse_cache(problem, s["parameters"], steps, endpoint_eps)
        for s in schedules
    }
    directions = model.projection_directions(
        problem.dimension, projections, args.seed + 404
    )
    rows = []
    stream = model.contexts(problem, measurements, samples, steps, args.seed + 303)
    for index, context in enumerate(stream):

        def record(sid, kind, generated):
            rows.append(
                {
                    "y_index": index,
                    "sampler_kind": kind,
                    "schedule_id": sid,
                    "true_component": context.true_component,
                    "mse_gt": float(
                        np.mean(
                            np.sum((generated - context.x_true[None, :]) ** 2, axis=1)
                        )
                    ),
                    "sliced_w2": model.sliced_w2(
                        generated, context.posterior_reference, directions
                    ),
                }
            )

        record("posterior_sampler", "posterior", context.posterior_samples)
        record(
            "mmse_estimator",
            "mmse",
            np.repeat(context.posterior_mean[None, :], samples, axis=0),
        )
        for schedule in schedules:
            for kind in ["exact", "frozen"]:
                generated = model.sample_reverse(caches[schedule["id"]], context, kind)
                record(schedule["id"], kind, generated)
        if (index + 1) % 10 == 0 or index + 1 == measurements:
            print(
                f"{args.experiment}: {index + 1}/{measurements} measurements",
                flush=True,
            )

    summaries = []
    groups = sorted(set((r["schedule_id"], r["sampler_kind"]) for r in rows))
    for sid, kind in groups:
        points = np.array(
            [
                [r["mse_gt"], r["sliced_w2"]]
                for r in rows
                if r["schedule_id"] == sid and r["sampler_kind"] == kind
            ]
        )
        summaries.append(
            {
                "schedule_id": sid,
                "sampler_kind": kind,
                "measurements": len(points),
                "mean_mse_gt": float(points[:, 0].mean()),
                "mean_sliced_w2": float(points[:, 1].mean()),
                "se_mse_gt": float(points[:, 0].std(ddof=1) / np.sqrt(len(points))),
                "se_sliced_w2": float(points[:, 1].std(ddof=1) / np.sqrt(len(points))),
            }
        )
    save_csv(args.output / "per_measurement.csv", rows)
    save_csv(args.output / "summary.csv", summaries)
    settings = {
        "experiment": args.experiment,
        "matches_paper_settings": custom_settings is None,
        "seed": args.seed,
        "components": problem.components,
        "dimension": problem.dimension,
        "steps": steps,
        "measurements": measurements,
        "samples_per_measurement": samples,
        "projections": projections,
        "sigma_y": problem.sigma_y,
        "covariance_min": float(problem.covariance.min()),
        "covariance_max": float(problem.covariance.max()),
        "mixture_weights": "uniform",
        "means": "iid Uniform[-1,1]",
        "normalization": "none",
        "endpoint_eps": endpoint_eps,
        "measurement_seed": args.seed + 303,
        "projection_seed": args.seed + 404,
        "optimizer_restart_entries": restarts if args.experiment == "figure2" else None,
        "optimizer_unique_starts": 15 if args.experiment == "figure2" else None,
        "optimizer_maxiter": 300 if args.experiment == "figure2" else None,
        "workers": args.workers,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "computation_seconds": time.perf_counter() - started,
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "schedules.json").write_text(
        json.dumps(schedules, indent=2) + "\n", encoding="utf-8"
    )
    plot(rows, schedules, args.experiment, args.output, custom_settings)
    print(
        f"Saved results to {args.output.resolve()} ({settings['computation_seconds']:.1f} seconds of computation)."
    )


if __name__ == "__main__":
    main()
