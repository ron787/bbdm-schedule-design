"""CPU-only analytical experiments from the BBDM schedule-design paper.

Run from any working directory, for example:
    python experiments/analytic/run.py --experiment figure5 --output results

Only NumPy, SciPy, and Matplotlib are required. No model, dataset, checkpoint,
compiler, GPU, or paper PDF is needed. Parameter order is alpha, beta, c, gamma.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import brentq, differential_evolution
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCHEDULES = {
    "default": (1.0, 1.0, 0.5, 1.0),
    "MSE": (1.0, 2.0, 2.0, 0.2),
    "W2": (2.0, 1.0, 0.2, 2.0),
}
STEP_BUDGETS = (20, 50, 200, 1000)
INTERMEDIATE = (1.0, 2.0, 0.2, 2.0)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def save_csv(path, data, columns):
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="")


def rho_grid(steps, parameters):
    """Interior bridge precision plus its separately defined zero endpoint.

    Accepts one parameter tuple or an array with shape (number_of_schedules, 4).
    Stable log/expm1 formulas avoid cancellation close to m=0 or m=1.
    """
    theta = np.asarray(parameters, dtype=float)
    single = theta.ndim == 1
    theta = np.atleast_2d(theta)
    alpha, beta, c, gamma = theta.T[:, :, None]
    tau = np.arange(1, steps, dtype=float)[None, :] / steps
    log_one_minus_m = beta * np.log1p(-(tau**alpha))
    m = -np.expm1(log_one_minus_m)
    interior = np.exp(
        -np.log(c)
        - gamma * np.log(4.0)
        - gamma * np.log(m)
        + (2.0 - gamma) * log_one_minus_m
    )
    result = np.column_stack([interior, np.zeros(len(theta))])
    return result[0] if single else result


def bbdm_variance(steps, parameters, precision):
    """Exact selected-label variance, Eq. (148) in the audited final paper."""
    rho = rho_grid(steps, parameters)
    precision = np.atleast_1d(precision)
    return np.sum(
        (rho[:-1] - rho[1:])[:, None] / (precision[None, :] + rho[:-1, None]) ** 2,
        axis=0,
    )


def ddim_variance(steps, precision):
    beta = np.linspace(1e-4, 2e-2, 1000)
    train = np.r_[1.0, np.cumprod(1.0 - beta)]
    # numpy.rint uses round-to-nearest with ties to even.
    indices = np.rint(np.arange(steps + 1) * 1000 / steps).astype(int)
    abar = train[indices]
    a = np.sqrt((1.0 - abar[:-1]) / (1.0 - abar[1:]))
    b = np.sqrt(abar[:-1]) - a * np.sqrt(abar[1:])
    den = precision[None, :] + (abar[1:] / (1.0 - abar[1:]))[:, None]
    gain = a[:, None] + (b * np.sqrt(abar[1:]) / (1.0 - abar[1:]))[:, None] / den
    return np.prod(gain, axis=0) ** 2


def crossings(steps, difference):
    changes = np.flatnonzero(difference[1:] * difference[:-1] < 0)
    return [
        float(steps[i] - difference[i] / (difference[i + 1] - difference[i]))
        for i in changes
    ]


def figure3(output):
    precision = 1.0 / np.geomspace(0.5, 2.0, 512) + 4.0
    steps = np.arange(2, 1001)
    names = ["DDIM", *SCHEDULES]
    losses = {name: np.empty((len(steps), 2)) for name in names}
    for i, S in enumerate(steps):
        variances = {"DDIM": ddim_variance(S, precision)}
        variances.update(
            {
                name: bbdm_variance(S, theta, precision)
                for name, theta in SCHEDULES.items()
            }
        )
        for name, variance in variances.items():
            losses[name][i] = [
                np.mean((np.sqrt(variance) - 1.0 / np.sqrt(precision)) ** 2),
                np.mean(variance + 1.0 / precision),
            ]
    colors = {
        "DDIM": "black",
        "default": "tab:blue",
        "MSE": "tab:green",
        "W2": "tab:red",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in names:
        for j, ax in enumerate(axes):
            ax.semilogx(steps, losses[name][:, j], label=name, color=colors[name])
    for ax in axes:
        ax.set_xlabel("Number of reverse steps S")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$J^{W_2}/d$")
    axes[1].set_ylabel(r"$J^{\mathrm{MSE}}/d$")
    fig.tight_layout()
    fig.savefig(output / "figure3.png", dpi=180)
    plt.close(fig)
    save_csv(
        output / "figure3.csv",
        np.column_stack([steps, *losses.values()]),
        ["S"]
        + [f"{name}_{metric}" for name in names for metric in ("W2_squared", "MSE")],
    )
    summary = {
        f"{name}_vs_DDIM": {
            metric: crossings(steps, losses[name][:, j] - losses["DDIM"][:, j])
            for j, metric in enumerate(("W2_squared", "MSE"))
        }
        for name in SCHEDULES
    }
    save_json(output / "figure3_crossings.json", summary)
    return {
        "mode": "full",
        "step_range": [2, 1000],
        "dimension": 512,
        "sigma_y": 0.5,
        "prior_variances": "geomspace(0.5,2,512)",
        "schedules": SCHEDULES,
        "ddim_training_grid_size": 1000,
        "ddim_beta_endpoints": [1e-4, 2e-2],
        "rounding": "nearest, ties to even",
        "ddim_mean_convention": "paper's small-terminal-alpha approximation",
    }


def grouped_lower_bounds(rho, precision, blocks):
    """Lower bound on positive variance sums for every schedule/precision pair.

    Within each contiguous interval block, use its largest denominator.
    This only decreases each term, so a bound above the incumbent permits
    safe rejection. All candidates are considered; no grid points are skipped.
    """
    n = rho.shape[1] - 1
    ends = np.arange(blocks + 1) * n // blocks
    bound = np.zeros((len(rho), len(precision)))
    for left, right in zip(ends[:-1], ends[1:]):
        if left != right:
            bound += (rho[:, left] - rho[:, right])[:, None] / (
                precision[None, :] + rho[:, left, None]
            ) ** 2
    return bound


def paired_variances(rho, precision, rows, columns, blocks=None):
    """Evaluate selected pairs in small batches to bound peak memory."""
    result = np.empty(len(rows))
    n = rho.shape[1] - 1
    ends = np.arange(blocks + 1) * n // blocks if blocks else np.arange(n + 1)
    for start in range(0, len(rows), 256):
        rr = rows[start : start + 256]
        cc = columns[start : start + 256]
        left = rho[rr[:, None], ends[:-1]]
        right = rho[rr[:, None], ends[1:]]
        result[start : start + 256] = np.sum(
            (left - right) / (precision[cc, None] + left) ** 2, axis=1
        )
    return result


def regime_grid(job):
    """Minimize the exact objective over the entire declared finite grid."""
    S, quick = job
    started = time.perf_counter()
    counts = (5, 5, 5, 5) if quick else (21, 21, 15, 15)
    axes = [
        np.linspace(lo, hi, count)
        for lo, hi, count in zip((1, 1, 0.2, 0.2), (2, 2, 2, 2), counts)
    ]
    theta = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    precision = np.geomspace(1e-3, 1e4, 101 if quick else 701)
    edges = np.asarray([SCHEDULES["MSE"], SCHEDULES["W2"], INTERMEDIATE])
    edge_variances = np.array([bbdm_variance(S, p, precision) for p in edges])
    best_edge = np.argmin(edge_variances, axis=0)
    best = np.min(edge_variances, axis=0)
    winner = edges[best_edge].copy()
    full_sums = 0
    # A batch has at most 128*701 entries (about 0.7 MB per work array).
    for start in range(0, len(theta), 128):
        candidates = theta[start : start + 128]
        rho = rho_grid(S, candidates)
        lower = grouped_lower_bounds(rho, precision, min(16, S - 1))
        rows, cols = np.nonzero(lower <= best[None, :] * (1.0 + 1e-12))
        if S > 64 and len(rows):
            tighter = paired_variances(rho, precision, rows, cols, blocks=64)
            keep = tighter <= best[cols] * (1.0 + 1e-12)
            rows, cols = rows[keep], cols[keep]
        values = paired_variances(rho, precision, rows, cols)
        full_sums += len(values)
        exact = np.full(lower.shape, np.inf)
        exact[rows, cols] = values
        local_rows = np.argmin(exact, axis=0)
        local_best = exact[local_rows, np.arange(len(precision))]
        improve = local_best < best
        best[improve] = local_best[improve]
        winner[improve] = candidates[local_rows[improve]]
    changes = np.flatnonzero(np.any(winner[1:] != winner[:-1], axis=1))
    transitions = []
    for i in changes:
        before, after = winner[i], winner[i + 1]
        root = brentq(
            lambda p: float(
                bbdm_variance(S, before, p)[0] - bbdm_variance(S, after, p)[0]
            ),
            precision[i],
            precision[i + 1],
            xtol=1e-14,
        )
        transitions.append(
            {"precision": root, "from": before.tolist(), "to": after.tolist()}
        )
    return {
        "S": S,
        "precision": precision,
        "winner": winner,
        "variance": best,
        "transitions": transitions,
        "grid_counts": counts,
        "candidate_pairs": len(theta) * len(precision),
        "full_sums": full_sums,
        "seconds": time.perf_counter() - started,
    }


def parallel_map(function, jobs, workers):
    if workers <= 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        return list(executor.map(function, jobs))


def figure5(output, quick, workers):
    results = parallel_map(regime_grid, [(S, quick) for S in STEP_BUDGETS], workers)
    colors = {
        (2.0, 1.0, 0.2, 2.0): "tab:blue",
        INTERMEDIATE: "tab:orange",
        (1.0, 2.0, 2.0, 0.2): "tab:green",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    summary = []
    for row, result in enumerate(results):
        S = result["S"]
        precision, winner = result["precision"], result["winner"]
        save_csv(
            output / f"figure5_S{S}.csv",
            np.column_stack(
                [np.full(len(precision), S), precision, winner, result["variance"]]
            ),
            ["S", "precision", "alpha", "beta", "c", "gamma", "min_variance"],
        )
        transitions = result["transitions"]
        endpoints = [1e-3] + [t["precision"] for t in transitions] + [1e4]
        regimes = [winner[0].tolist()] + [t["to"] for t in transitions]
        for ax in axes:
            for left, right, theta in zip(endpoints[:-1], endpoints[1:], regimes):
                ax.plot(
                    [left, right],
                    [row, row],
                    linewidth=12,
                    color=colors.get(tuple(theta), "tab:purple"),
                    solid_capstyle="butt",
                )
        summary.append(
            {
                k: v
                for k, v in result.items()
                if k not in ("precision", "winner", "variance")
            }
        )
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-3, 1e4)
    axes[1].set_xlim(0.275, 0.37)
    for ax in axes:
        ax.set_yticks(range(4), [f"S={S}" for S in STEP_BUDGETS])
        ax.invert_yaxis()
        ax.set_xlabel("Posterior precision")
        ax.grid(alpha=0.2)
    labels = [
        ("tab:blue", r"Low-precision $W_2$ edge $(2,1,0.2,2)$"),
        ("tab:orange", r"Intermediate edge $(1,2,0.2,2)$"),
        ("tab:green", r"High-precision MSE edge $(1,2,2,0.2)$"),
    ]
    fig.legend(
        handles=[Line2D([0], [0], color=c, linewidth=8, label=t) for c, t in labels],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        title=r"Parameter order: $(\alpha,\beta,c,\gamma)$",
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output / "figure5.png", dpi=180)
    plt.close(fig)
    save_json(output / "figure5_transitions.json", summary)
    return {
        "mode": "quick-reduced-grid" if quick else "full",
        "S": STEP_BUDGETS,
        "grid_counts": [5, 5, 5, 5] if quick else [21, 21, 15, 15],
        "precision_range": [1e-3, 1e4],
        "precision_count": 101 if quick else 701,
        "candidate_pairs": sum(r["candidate_pairs"] for r in results),
        "workers": min(workers, 4),
        "method": "Complete finite grid with rigorous positive-term lower-bound pruning",
        "limitation": "Discrete grid minimizers do not certify the continuous-box global optimum.",
    }


def optimize_qs(S):
    searches = []

    # Since rho=R/c, lambda*variance depends only on u=c*lambda.
    # Fixing c=1 does not restrict the attainable objective.
    def objective(x):
        alpha, beta, gamma, log_u = x
        u = np.exp(log_u)
        return float(-u * bbdm_variance(S, (alpha, beta, 1.0, gamma), u)[0])

    for seed in (20261005, 1729):
        result = differential_evolution(
            objective,
            [(1.0, 2.0), (1.0, 2.0), (0.2, 2.0), (-20.0, 20.0)],
            seed=seed,
            popsize=15,
            tol=1e-10,
            maxiter=500,
            polish=True,
            workers=1,
        )
        searches.append(
            {
                "seed": seed,
                "Q": float(-result.fun),
                "parameters": [
                    float(result.x[0]),
                    float(result.x[1]),
                    1.0,
                    float(result.x[2]),
                ],
                "precision": float(np.exp(result.x[3])),
                "converged": bool(result.success),
                "function_evaluations": int(result.nfev),
            }
        )
    Q = max(r["Q"] for r in searches)
    return {
        "S": S,
        "Q": Q,
        "sqrt_Q": float(np.sqrt(Q)),
        "minimum_relative_std_deficit_percent": float(100 * (1.0 - np.sqrt(Q))),
        "searches": searches,
    }


def table_d6(output, workers):
    results = parallel_map(optimize_qs, list(STEP_BUDGETS), workers)
    save_csv(
        output / "table_d6.csv",
        [
            [r["S"], r["Q"], r["sqrt_Q"], r["minimum_relative_std_deficit_percent"]]
            for r in results
        ],
        ["S", "Q", "sqrt_Q", "minimum_relative_std_deficit_percent"],
    )
    save_json(output / "table_d6_searches.json", results)
    return {
        "mode": "full",
        "S": STEP_BUDGETS,
        "seeds": [20261005, 1729],
        "workers": min(workers, 4),
        "optimizer": "scipy differential_evolution",
        "population_multiplier": 15,
        "tolerance": 1e-10,
        "maximum_iterations": 500,
        "bounds": {
            "alpha": [1, 2],
            "beta": [1, 2],
            "gamma": [0.2, 2],
            "log(c*precision)": [-20, 20],
        },
        "limitation": "The paper does not specify its optimizer. These seeded numerical searches reproduce its table but do not certify the continuous unbounded global supremum.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=("figure3", "figure5", "table-d6"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a clearly marked reduced grid for Figure 5 only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU processes for Figure 5/Table D.6 (default: 1).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if args.experiment == "figure3":
        config = figure3(args.output)
    elif args.experiment == "figure5":
        config = figure5(args.output, args.quick, args.workers)
    else:
        config = table_d6(args.output, args.workers)
    config.update(
        {
            "experiment": args.experiment,
            "started_at_utc": started_at,
            "elapsed_seconds": time.perf_counter() - started,
            "parameter_order": ["alpha", "beta", "c", "gamma"],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
    )
    save_json(args.output / f"{args.experiment.replace('-', '_')}_config.json", config)
    print(
        f"{args.experiment}: completed ({config['mode']}) in {config['elapsed_seconds']:.2f}s; results: {args.output}"
    )


if __name__ == "__main__":
    main()
