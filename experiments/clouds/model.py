"""Diagonal shared-covariance MoG and BBDM equations for Figures 2 and 7.

Only NumPy and SciPy are used. Arithmetic order and random-number draws retain
compatibility with the paper experiments. Parameter order is alpha, beta, c, gamma.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize

BOUNDS = ((1.0, 2.0), (1.0, 2.0), (0.2, 2.0), (0.2, 2.0))
DEFAULT = np.array([1.0, 1.0, 0.5, 1.0])
MSE_EDGE = np.array([1.0, 2.0, 2.0, 0.2])
W2_EDGE = np.array([2.0, 1.0, 0.2, 2.0])


@dataclass
class Problem:
    means: np.ndarray
    covariance: np.ndarray
    sigma_y: float
    weights: np.ndarray
    prior_precision: np.ndarray
    measurement_precision: np.ndarray
    posterior_precision: np.ndarray
    posterior_variance: np.ndarray
    posterior_mean_offsets: np.ndarray
    posterior_y_coefficient: np.ndarray

    @property
    def dimension(self):
        return self.means.shape[1]

    @property
    def components(self):
        return self.means.shape[0]


def make_problem(experiment, seed=1234, *, components=32, dimension=512):
    """Construct either noise/covariance setting, with paper-sized defaults."""
    for name, value in [("components", components), ("dimension", dimension)]:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    weights = np.full(components, 1.0 / components, dtype=np.float64)
    means = rng.uniform(-1.0, 1.0, size=(components, dimension)).astype(np.float64)
    covariance = (
        np.geomspace(0.5, 2.0, dimension).astype(np.float64)
        if experiment == "figure2"
        else np.full(dimension, 8.0)
    )
    sigma_y = 0.1 if experiment == "figure2" else 3.0
    prior_precision = 1.0 / covariance
    measurement_precision = 1.0 / (covariance + sigma_y**2)
    posterior_precision = prior_precision + (1.0 / sigma_y**2)
    posterior_variance = 1.0 / posterior_precision
    offsets = posterior_variance[None, :] * prior_precision[None, :] * means
    y_coefficient = posterior_variance / (sigma_y**2)
    return Problem(
        means,
        covariance,
        sigma_y,
        weights,
        prior_precision,
        measurement_precision,
        posterior_precision,
        posterior_variance,
        offsets,
        y_coefficient,
    )


def softmax(logits):
    maximum = np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(logits - maximum)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def measurement_posterior(problem, y):
    difference = y[None, :] - problem.means
    quadratic = np.sum(
        difference * difference * problem.measurement_precision[None, :], axis=-1
    )
    weights = softmax((np.log(problem.weights) - 0.5 * quadratic)[None, :])[0]
    means = (
        problem.posterior_mean_offsets
        + y[None, :] * problem.posterior_y_coefficient[None, :]
    )
    return weights, means


@dataclass
class Context:
    x_true: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    posterior_mean: np.ndarray
    posterior_reference: np.ndarray
    posterior_samples: np.ndarray
    noises: np.ndarray
    frozen_labels: np.ndarray
    true_component: int


def contexts(problem, measurements, samples, steps, seed):
    """Yield one measurement at a time, preserving the paper's exact RNG order.

    Only the clean signals and observations are generated together. Posterior
    samples and reverse innovations are generated just before each measurement
    is evaluated; a full run no longer stores about 2.2 GB of innovations.
    """
    rng = np.random.default_rng(seed)
    component = rng.choice(problem.components, size=measurements, p=problem.weights)
    clean = (
        problem.means[component]
        + rng.standard_normal((measurements, problem.dimension))
        * np.sqrt(problem.covariance)[None, :]
    )
    observations = clean + problem.sigma_y * rng.standard_normal(
        (measurements, problem.dimension)
    )
    posterior_sd = np.sqrt(problem.posterior_variance)
    for index in range(measurements):
        weights, means = measurement_posterior(problem, observations[index])
        posterior_mean = weights @ means
        labels = rng.choice(problem.components, size=samples, p=weights)
        reference = (
            means[labels]
            + rng.standard_normal((samples, problem.dimension)) * posterior_sd[None, :]
        )
        labels = rng.choice(problem.components, size=samples, p=weights)
        posterior_samples = (
            means[labels]
            + rng.standard_normal((samples, problem.dimension)) * posterior_sd[None, :]
        )
        noises = rng.standard_normal((steps, samples, problem.dimension))
        frozen_labels = rng.choice(problem.components, size=samples, p=weights)
        yield Context(
            clean[index],
            observations[index],
            weights,
            means,
            posterior_mean,
            reference,
            posterior_samples,
            noises,
            frozen_labels,
            int(component[index]),
        )


def schedule(parameters, steps):
    alpha, beta, c, gamma = [float(value) for value in parameters]
    tau = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    m = np.empty(steps + 1, dtype=np.float64)
    m[0], m[-1] = 0.0, 1.0
    m[1:-1] = 1.0 - np.power(1.0 - np.power(tau[1:-1], alpha), beta)
    arch = np.clip(4.0 * m * (1.0 - m), 0.0, None)
    delta = c * np.power(np.clip(arch, 1e-16, None), gamma)
    delta[0], delta[-1] = 0.0, 0.0
    return m, delta


def selected_variance(problem, parameters, steps):
    """Exact selected-label variance, using the paper's reciprocal-scale sum."""
    m, delta = schedule(parameters, steps)
    denominator = np.clip((1.0 - m[:-1]) ** 2, 1e-16, None)
    conditional_delta = delta[1:] - delta[:-1] * ((1.0 - m[1:]) ** 2) / denominator
    if (
        np.any(np.diff(m) <= 0.0)
        or np.any(delta < -1e-10)
        or np.any(conditional_delta < -1e-10)
    ):
        return None
    rho = np.zeros(steps + 1, dtype=np.float64)
    rho[0] = np.inf
    if steps > 1:
        if np.any(delta[1:steps] <= 0.0) or np.any(1.0 - m[1:steps] <= 0.0):
            return None
        rho[1:steps] = (1.0 - m[1:steps]) ** 2 / delta[1:steps]
        if not np.all(np.isfinite(rho[1:steps])) or np.any(rho[1:steps] <= 0.0):
            return None
        if np.any(np.diff(rho[1:]) > 1e-8):
            return None
    variance = np.zeros_like(problem.posterior_precision, dtype=np.float64)
    for index in range(2, steps + 1):
        left, right = rho[index - 1], rho[index]
        gap = left - right
        if gap < -1e-8:
            return None
        if gap > 0.0:
            variance += gap / np.square(problem.posterior_precision + left)
    return np.clip(variance, 0.0, None)


def objectives(problem, parameters, steps):
    variance = selected_variance(problem, parameters, steps)
    if variance is None:
        return 1e18, 1e18
    target = 1.0 / problem.posterior_precision
    w2 = float(
        np.sum(np.square(np.sqrt(np.clip(variance, 0.0, None)) - np.sqrt(target)))
    )
    mse = float(np.sum(variance + target))
    return w2, mse


def clip_parameters(parameters):
    return np.array(
        [
            np.clip(parameters[index], lower, upper)
            for index, (lower, upper) in enumerate(BOUNDS)
        ],
        dtype=np.float64,
    )


def optimize(problem, steps, weight, seed, restarts=16):
    """Minimize the final paper's unnormalized blend with the original starts.

    The historical driver supplied the default as both automatic and explicit
    start. It filled the list to 16 entries and then deduplicated it, yielding
    one default and 14 random starts. Preserve this to reproduce its result.
    """

    def objective(parameters):
        w2, mse = objectives(problem, parameters, steps)
        return float((1.0 - float(weight)) * w2 + float(weight) * mse)

    rng = np.random.default_rng(seed)
    starts = [clip_parameters(DEFAULT.copy()), clip_parameters(DEFAULT.copy())]
    while len(starts) < max(1, restarts):
        starts.append(
            np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS], dtype=np.float64)
        )
    unique = []
    for start in starts:
        if not any(np.allclose(start, earlier) for earlier in unique):
            unique.append(start)
    best, best_value = clip_parameters(DEFAULT.copy()), float("inf")
    for start in unique:
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={"maxiter": 300},
        )
        candidate = clip_parameters(np.asarray(result.x, dtype=np.float64))
        value = float(objective(candidate))
        if value < best_value:
            best, best_value = candidate, value
    return best


@dataclass
class ReverseCache:
    m: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    sigma: np.ndarray
    responsibility_precision: list
    component_terms: list
    x_terms: list
    y_terms: list


def reverse_cache(problem, parameters, steps, endpoint_eps=1e-6):
    """Simulation coefficients, retaining the original numerical endpoint clip."""
    m, delta = schedule(parameters, steps)
    delta = np.clip(delta.copy(), endpoint_eps, None)
    m[-1] = min(m[-1], 1.0 - endpoint_eps)
    a = np.zeros(steps + 1, dtype=np.float64)
    b, c, sigma = a.copy(), a.copy(), a.copy()
    for s in range(1, steps + 1):
        ratio = ((1.0 - m[s]) ** 2) / max((1.0 - m[s - 1]) ** 2, 1e-16)
        conditional_delta = max(delta[s] - delta[s - 1] * ratio, 1e-16)
        variance = max(conditional_delta * delta[s - 1] / max(delta[s], 1e-16), 1e-16)
        sigma[s] = math.sqrt(variance)
        inside = (delta[s - 1] - variance) / max(delta[s], 1e-16)
        c[s] = math.sqrt(max(inside, 0.0))
        a[s] = (1.0 - m[s - 1]) - (1.0 - m[s]) * c[s]
        b[s] = m[s - 1] - m[s] * c[s]
    responsibility_precision = [np.zeros(problem.dimension)]
    component_terms = [np.zeros_like(problem.means)]
    x_terms, y_terms = [np.zeros(problem.dimension)], [np.zeros(problem.dimension)]
    for s in range(1, steps + 1):
        one_minus = 1.0 - m[s]
        responsibility_precision.append(
            1.0 / (one_minus**2 * problem.posterior_variance + delta[s])
        )
        rho = one_minus**2 / delta[s]
        variance = 1.0 / (problem.prior_precision + (1.0 / problem.sigma_y**2) + rho)
        kappa = one_minus / delta[s]
        x_terms.append(kappa * variance)
        y_terms.append((1.0 / problem.sigma_y**2 - m[s] * kappa) * variance)
        component_terms.append(
            variance[None, :] * problem.prior_precision[None, :] * problem.means
        )
    return ReverseCache(
        m, a, b, c, sigma, responsibility_precision, component_terms, x_terms, y_terms
    )


def oracle_denoiser(cache, step, x, context):
    """Exact MoG posterior mean, evaluated without a sample×component×d tensor."""
    ms = cache.m[step]
    means = (1.0 - ms) * context.means + ms * context.y[None, :]
    precision = cache.responsibility_precision[step]
    anchor = means[0]
    offsets = means - anchor[None, :]
    centered = x - anchor[None, :]
    logits = (
        np.log(np.clip(context.weights, 1e-300, None))[None, :]
        + (centered * precision[None, :]) @ offsets.T
        - 0.5 * np.sum(offsets * offsets * precision[None, :], axis=1)[None, :]
    )
    responsibilities = softmax(logits)
    return (
        x * cache.x_terms[step][None, :]
        + context.y[None, :] * cache.y_terms[step][None, :]
        + responsibilities @ cache.component_terms[step]
    )


def sample_reverse(cache, context, kind):
    x = np.repeat(context.y[None, :], len(context.frozen_labels), axis=0)
    for step in range(len(cache.m) - 1, 0, -1):
        if kind == "exact":
            denoised = oracle_denoiser(cache, step, x, context)
        elif kind == "frozen":
            denoised = (
                x * cache.x_terms[step][None, :]
                + (context.y * cache.y_terms[step])[None, :]
                + cache.component_terms[step][context.frozen_labels]
            )
        else:
            raise ValueError("kind must be exact or frozen")
        x = (
            cache.a[step] * denoised
            + cache.b[step] * context.y[None, :]
            + cache.c[step] * x
            + cache.sigma[step] * context.noises[step - 1]
        )
    return x


def projection_directions(dimension, count, seed):
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((count, dimension))
    norms = np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-16, None)
    return (directions / norms).astype(np.float64)


def sliced_w2(samples, reference, directions):
    first, second = samples @ directions.T, reference @ directions.T
    first.sort(axis=0)
    second.sort(axis=0)
    costs = np.mean((first - second) ** 2, axis=0)
    return float(math.sqrt(max(float(np.mean(costs)), 0.0)))
