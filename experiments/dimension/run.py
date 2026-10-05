#!/usr/bin/env python3
"""Fresh CPU simulation of Figure 6's Gaussian-mixture experiment.

NumPy uses different random draws from the paper's original CUDA run, so
reproduction is statistical rather than bitwise.
"""

import os

for _key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
import argparse, csv, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.special import log_softmax, softmax
from scipy.stats import t

DIMS = (16, 32, 64, 128, 256, 512)
COMPONENTS = (2, 4, 8, 16)
METRICS = ("separation", "selected_label_error", "denoiser_gap", "final_gap")


def schedule(steps=20):
    """m_s=s/S, delta_s=2m_s(1-m_s), with explicit singular endpoints."""
    m = np.linspace(0, 1, steps + 1)
    delta = 2 * m * (1 - m)
    a = np.zeros(steps + 1)
    b = a.copy()
    c = a.copy()
    sigma = a.copy()
    a[1] = 1
    a[steps] = 1 - m[steps - 1]
    b[steps] = m[steps - 1]
    sigma[steps] = np.sqrt(delta[steps - 1])
    for s in range(2, steps):
        c[s] = delta[s - 1] * (1 - m[s]) / ((1 - m[s - 1]) * delta[s])
        a[s] = 1 - m[s - 1] - (1 - m[s]) * c[s]
        b[s] = m[s - 1] - m[s] * c[s]
        sigma[s] = np.sqrt(max(0, delta[s - 1] - c[s] ** 2 * delta[s]))
    return m, delta, a, b, c, sigma


def simulate_batch(means, cov, rng, paths=64, steps=20, sigma_y=2.0, trace=None):
    """One batch of independent priors; return one summary per prior.

    Shapes: means=(mixtures,R,d), states=(mixtures,paths,d). Shared diagonal
    covariance lets responsibility scores and posterior means use matrix
    products, without materializing a paths×components×dimensions tensor.
    """
    mixtures, R, d = means.shape
    sy2 = sigma_y**2
    truth = rng.integers(R, size=(mixtures, paths))
    selected_true = means[np.arange(mixtures)[:, None], truth]
    clean = selected_true + np.sqrt(cov) * rng.normal(size=(mixtures, paths, d))
    y = clean + sigma_y * rng.normal(size=(mixtures, paths, d))
    logits = (
        np.matmul(y, np.swapaxes(means / (cov + sy2), 1, 2))
        - 0.5 * np.sum(means**2 / (cov + sy2), axis=-1)[:, None, :]
    )
    loggy = log_softmax(logits, axis=-1)
    gy = np.exp(loggy)
    labels = np.minimum(
        (rng.random((mixtures, paths, 1)) > np.cumsum(gy, axis=-1)).sum(-1), R - 1
    )
    selected_means = means[np.arange(mixtures)[:, None], labels]
    post = 1 / (1 / cov + 1 / sy2)
    first_exact = post * (np.matmul(gy, means) / cov + y / sy2)
    first_frozen = post * (selected_means / cov + y / sy2)
    m, delta, a, b, c, sigma = schedule(steps)
    exact = y.copy()
    frozen = y.copy()
    stable = np.ones((mixtures, paths), dtype=bool)
    local = np.zeros((mixtures, paths))
    for s in range(steps, 0, -1):
        z = rng.normal(size=y.shape)  # shared Gaussian innovation, including s=1
        if s == steps:
            den_exact = first_exact
            den_at_frozen = first_exact
            den_frozen = first_frozen
        else:
            u = 1 - m[s]
            var = u * u * post + delta[s]
            shift = u * post / cov * means
            shift_scaled = np.swapaxes(shift / var, 1, 2)
            offset = -0.5 * np.sum(shift**2 / var, axis=-1)[:, None, :]
            baseline = (u * post / sy2 + m[s]) * y
            post_s = 1 / (1 / cov + 1 / sy2 + u * u / delta[s])

            def oracle(x):
                gamma = softmax(
                    loggy + np.matmul(x - baseline, shift_scaled) + offset, axis=-1
                )
                den = post_s * (
                    np.matmul(gamma, means) / cov
                    + y / sy2
                    + u / delta[s] * (x - m[s] * y)
                )
                return den, gamma

            den_exact, _ = oracle(exact)
            den_at_frozen, gamma_frozen = oracle(frozen)
            den_frozen = post_s * (
                selected_means / cov + y / sy2 + u / delta[s] * (frozen - m[s] * y)
            )
            stable &= np.argmax(gamma_frozen, axis=-1) == labels
        # Average every denoiser call, including the measurement endpoint x_S=y.
        local += np.mean((den_at_frozen - den_frozen) ** 2, axis=-1) / steps
        exact = a[s] * den_exact + b[s] * y + c[s] * exact + sigma[s] * z
        frozen = a[s] * den_frozen + b[s] * y + c[s] * frozen + sigma[s] * z
        if trace is not None:
            trace.append(
                (
                    exact.copy(),
                    frozen.copy(),
                    stable.copy(),
                    den_exact.copy(),
                    den_at_frozen.copy(),
                    den_frozen.copy(),
                )
            )
    pair_dist = np.sum(
        (means[:, :, None, :] - means[:, None, :, :]) ** 2 / (cov + sy2), axis=-1
    )
    pair_dist[:, np.arange(R), np.arange(R)] = np.inf
    return dict(
        separation=np.min(pair_dist, axis=(1, 2)) / d,
        selected_label_error=np.mean(~stable, axis=1),
        denoiser_gap=np.mean(local, axis=1),
        final_gap=np.mean((exact - frozen) ** 2, axis=(1, 2)),
        failures=np.sum(~stable, axis=1),
    )


def run_condition(task):
    R, d, config = task
    start = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], R, d]))
    cov = np.geomspace(0.5, 2, d)
    parts = []
    remaining = config["n_mixtures"]
    while remaining:
        count = min(config["mixture_batch_size"], remaining)
        means = rng.uniform(-1, 1, (count, R, d))
        parts.append(
            simulate_batch(
                means,
                cov,
                rng,
                paths=config["n_samples_per_mixture"],
                steps=config["S"],
            )
        )
        remaining -= count
    samples = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = config["n_mixtures"]
    N = n * config["n_samples_per_mixture"]
    mult = t.ppf(0.975, n - 1)
    result = dict(
        R=R,
        d=d,
        n_mixtures=n,
        n_samples_per_mixture=config["n_samples_per_mixture"],
        n_paths=N,
        failures=int(samples["failures"].sum()),
    )
    for metric in METRICS:
        v = samples[metric]
        mean = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(n))
        result[metric] = mean
        result[metric + "_cluster_se"] = se
        low = max(0, mean - mult * se)
        high = mean + mult * se
        if metric == "selected_label_error":
            high = min(1, high)
        result[metric + "_ci95_low"] = low
        result[metric + "_ci95_high"] = high
    result["zero_error_conditional_upper95"] = (
        float(1 - 0.05 ** (1 / N)) if result["failures"] == 0 else None
    )
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = config["Rs"]
    dims = config["dims"]
    steps = config["S"]
    titles = (
        "Separation statistic",
        "Selected-label error",
        "Local denoiser discrepancy",
        "Final reconstruction discrepancy",
    )
    ylabels = (
        "Minimum measurement separation / d",
        "Probability of interior label mismatch",
        f"Mean squared denoiser gap / d, all {steps} calls",
        "Final squared reconstruction gap / d",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, title, ylabel in zip(axes.flat, METRICS, titles, ylabels):
        for index, R in enumerate(components):
            marker = ("o", "s", "^", "D")[index % 4]
            values = sorted([r for r in rows if r["R"] == R], key=lambda r: r["d"])
            xx = np.array([r["d"] for r in values])
            yy = np.array([r[metric] for r in values])
            positive = yy > 0
            line = ax.plot(
                xx[positive], yy[positive], marker=marker, ms=4, label=f"R={R}"
            )[0]
            lower = np.array([r[metric + "_ci95_low"] for r in values])
            upper = np.array([r[metric + "_ci95_high"] for r in values])
            if metric != "separation":
                lower = np.where(lower > 0, lower, np.nan)
            if len(xx) == 1:
                ax.vlines(xx, lower, upper, color=line.get_color(), alpha=0.5)
            else:
                ax.fill_between(xx, lower, upper, color=line.get_color(), alpha=0.12)
            if metric == "selected_label_error":
                for row in values:
                    if row["failures"] == 0:
                        ax.scatter(
                            row["d"],
                            row["zero_error_conditional_upper95"],
                            marker="v",
                            facecolors="none",
                            edgecolors=line.get_color(),
                        )
        if metric == "separation":
            theory = [
                (2 / 3) * np.mean(1 / (np.geomspace(0.5, 2, d) + 4)) for d in dims
            ]
            ax.plot(
                dims,
                theory,
                "k--",
                marker="_" if len(dims) == 1 else None,
                lw=1,
                label="pair mean theory",
            )
        else:
            ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims, labels=[str(d) for d in dims])
        ax.set_xlabel("dimension d")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    settings = (
        "paper settings" if config["matches_paper_settings"] else "custom settings"
    )
    heading = f"Figure 6 — fresh CPU simulation ({settings}): {config['n_mixtures']:,} priors × {config['n_samples_per_mixture']} paths"
    heading += (
        f"\nR={','.join(map(str, components))}; d={','.join(map(str, dims))}; "
        f"S={steps}; seed={config['seed']}"
    )
    heading += "\nBands/bars: approximate cluster 95% CI; open triangles: zero-error conditional upper bounds"
    fig.suptitle(heading, fontsize=11)
    fig.savefig(output / "figure6.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["figure6"], default="figure6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Use one component count R >= 2 instead of the paper sweep.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Use one dimension d >= 1 instead of the paper sweep.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of reverse steps S >= 2 (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Nonnegative random seed (default: 0).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name, minimum in (
        ("components", 2),
        ("dimension", 1),
        ("steps", 2),
        ("seed", 0),
    ):
        value = getattr(args, name)
        if value is not None and value < minimum:
            parser.error(f"--{name} must be at least {minimum}")
    args.output.mkdir(parents=True, exist_ok=True)
    components = [args.components] if args.components is not None else list(COMPONENTS)
    dims = [args.dimension] if args.dimension is not None else list(DIMS)
    steps = args.steps if args.steps is not None else 20
    seed = args.seed if args.seed is not None else 0
    matches_paper_settings = (
        components == list(COMPONENTS)
        and dims == list(DIMS)
        and steps == 20
        and seed == 0
    )
    config = dict(
        seed=seed,
        n_mixtures=4096,
        n_samples_per_mixture=64,
        mixture_batch_size=16,
        S=steps,
        sigma_y=2.0,
        cov_min=0.5,
        cov_max=2.0,
        Rs=components,
        dims=dims,
        alpha=1.0,
        beta=1.0,
        schedule_c=0.5,
        schedule_gamma=1.0,
        label_mode="sample",
        dtype="float64",
        rng="NumPy default_rng(SeedSequence([seed,R,d])); batch order is part of the realization",
        numpy_version=np.__version__,
        local_gap=f"Squared Euclidean gap/d, averaged over all {steps} denoiser calls",
        stability=f"Argmax matches selected label at all {steps - 1} interior states",
        uncertainty="Approximate Student-t 95% intervals across independent priors; rare-event tails can be poorly resolved. Zero-error upper bounds condition on the sampled prior ensemble.",
        matches_paper_settings=matches_paper_settings,
        workers=args.workers,
    )
    start = time.perf_counter()
    rows = []
    tasks = [(R, d, config) for R in components for d in dims]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(
            [pool.submit(run_condition, task) for task in tasks]
        ):
            row = future.result()
            rows.append(row)
            print(
                f"R={row['R']:2d} d={row['d']:3d}: error={row['selected_label_error']:.6g}",
                flush=True,
            )
    rows.sort(key=lambda r: (r["R"], r["d"]))
    result = dict(
        mode="fresh CPU simulation",
        config=config,
        elapsed_seconds=time.perf_counter() - start,
        results=rows,
    )
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    write_csv(args.output / "results.csv", rows)
    plot(rows, args.output, config)
    condition_label = "condition" if len(rows) == 1 else "conditions"
    print(
        f"Completed {len(rows)} {condition_label} in {result['elapsed_seconds']:.1f}s; results written to {args.output}"
    )


if __name__ == "__main__":
    main()
