# BBDM analytical and synthetic experiments

CPU code for **Mixture-of-Gaussians-Guided Schedule Design for Brownian Bridge
Diffusion Models**, by Ron Levy and Michael Elad, accepted at TMLR (2026).

[Paper on OpenReview](https://openreview.net/forum?id=NHccmvrBPO)

This repository covers the analytical calculations and synthetic Gaussian-mixture
experiments in Figures 1, 2, 3, 5, 6 and 7, and the Appendix D.6 table.
It uses Python, NumPy, SciPy and Matplotlib. All calculations run on CPU.

The MNIST fitted-prior experiments in Figures 8–9 and Section 8.4, and the
trained-model, FFHQ and DPS experiments, are outside this release. This is code
for the analytical and synthetic part of the paper, not its complete image pipeline.

## Install

Use Python 3.11 or newer; Python 3.12.14 was tested. Clone this repository,
or download and extract its ZIP archive:

```sh
git clone https://github.com/ron787/bbdm-schedule-design.git
cd bbdm-schedule-design
```

From the repository directory, install the three dependencies:

```sh
python -m pip install -r requirements.txt
```

The three dependency versions are pinned to the tested environment. A fresh
virtual environment is useful if you already have a separate research environment.

## Run

First check the installation with smaller simulation and search budgets:

```sh
python run.py all --quick --output quick_results
```

Run every experiment at the full settings:

```sh
python run.py all
```

Or select one experiment:

| Command | Paper result |
|---|---|
| `python run.py figure1` | Schedule shapes |
| `python run.py figure2` | High-precision MSE / sliced-Wasserstein tradeoff |
| `python run.py figure3` | Analytical BBDM / DDIM comparison |
| `python run.py figure5` | Full schedule-grid optimizer regimes |
| `python run.py figure6` | Dimension-dependent exact / frozen MoG comparison |
| `python run.py figure7` | Low-precision reversal |
| `python run.py table-d6` | Finite-step variance-ratio table |

Add `--workers 1` to use one CPU worker, or `--workers 4` to allow four.
Experiments run sequentially. The setting controls processes or numerical
threads as appropriate; it does not request a GPU. All outputs go under
`results/<experiment>/`, or under the parent supplied with `--output`.
Rerunning an experiment replaces its generated files in that output directory.

On the validation machine, the complete quick run took about 19 seconds.
Full Figure 6 took about 6.6 minutes with three CPU processes; Figures 2 and 7
took about 27 and 12 seconds, respectively. The full Figure 5 grid took about
13 seconds on one process. Runtime on other CPUs will differ.

The figure experiments write PNG plots, numerical CSV/JSON results and the
settings used; Appendix D.6 writes its table as CSV/JSON. Quick outputs are
explicitly identified as reduced-budget runs; omit
`--quick` for the full experiment. The deterministic schedule curves remain
cheap enough to calculate fully in either mode.

## Change the model settings

Figures **2, 6 and 7** accept four optional settings. Leave them out to use
the paper's configuration.

| Option | Meaning | Figure 2 / 7 default | Figure 6 default |
|---|---|---|---|
| `--components R` | Number of Gaussians | 32 | Sweep over 2, 4, 8, 16 |
| `--dimension d` | Data dimension | 512 | Sweep over 16, 32, 64, 128, 256, 512 |
| `--steps S` | Number of reverse steps | 20 | 20 |
| `--seed N` | Random seed | 1234 | 0 |

For example, run the tradeoff experiment with a smaller mixture and dimension:

```sh
python run.py figure2 --components 16 --dimension 128 --steps 10 --seed 42 --output custom_results
```

For Figure 6, a supplied component count or dimension selects a single value
on that axis; an omitted option keeps the paper's sweep. For example:

```sh
python run.py figure6 --components 8 --output dimension_sweep
```

Dimensions and component counts must be positive; Figure 6 requires at least
two components because it measures pairwise separation. Steps must be at least
2 and seeds nonnegative. Larger settings can increase runtime and memory use.

These options apply to one selected synthetic experiment, so they cannot be
combined with `all`, the analytical experiments, or `--saved`. The analytical
commands retain their prescribed curves and parameter sweeps. Observation noise,
covariance setup, sampling budgets and optimizer settings remain fixed in each
experiment; use `--quick` for a smaller simulation budget.

Custom plots identify the changed model settings, and result metadata records
the effective configuration and `matches_paper_settings`. This flag is true
only for the full default run. A custom or quick run explores the method and
does not reproduce the paper's exact configuration. Use a separate `--output`
directory when you want to retain several runs.

## What reproduction means here

- **Figures 1 and 3:** deterministic calculations from the paper's equations.
- **Figure 2:** the unnormalized objective `(1-lambda)*J_W + lambda*J_M`, with
  seed 1234, the paper's `n_restarts=16` setting (15 distinct starting points
  after deduplication), 100 observations, 256 samples per observation
  and 256 projection directions. The paper's random-number protocol is preserved.
- **Figure 5:** the stated grid of 99,225 schedule candidates, 701 precisions,
  and budgets 20, 50, 200 and 1,000. Exact positive-term bounds skip sums that
  cannot improve the current grid optimum. This is a grid search, not a proof
  of a continuous global optimum.
- **Figure 6:** a fresh NumPy CPU simulation with the recorded scientific
  settings: 4,096 mixtures, 64 paths per mixture, batch size 16, seed 0 and
  sampled posterior labels. Its random draws differ from the historical CUDA
  run, so agreement is statistical. The local squared discrepancy averages all
  20 denoiser calls; label stability checks the 19 interior states. Gaps are
  divided by dimension.
- **Figure 7:** the paper's low-precision experiment with the two edge schedules.
- **Appendix D.6:** numerical maximization of the finite-step variance ratio.
  Matching the table is numerical evidence, not a new global-optimality proof.

The saved Figure 6 results are included as a JSON reference. Its 96 metric
values were verified against the paper. To plot those saved values instead of
running a simulation:

```sh
python run.py figure6 --saved
```

This command is labeled as saved-data plotting and writes to `results/figure6_saved/`.
Ordinary experiment commands always perform fresh calculations; they do not
substitute the saved results.

Parameter tuples throughout the code use `(alpha, beta, c, gamma)` order.
Cloud MSE is summed over dimensions; Figure 6 reports squared gaps per dimension.
The empirical sliced-Wasserstein metric in Figures 2 and 7 differs from the
analytical squared-Wasserstein objective used to optimize schedules.

## Code layout

- `run.py`: the common command-line entry point.
- `experiments/analytic/`: schedule equations, DDIM comparison, grid search and table.
- `experiments/clouds/`: diagonal MoG tradeoff and reversal experiments.
- `experiments/dimension/`: the dimension study and its saved historical reference.
- `VALIDATION.md`: checks performed on this CPU release and measured runtimes.

The scripts can also be run directly; use `--help` for their options. Each group
keeps its equations close to the experiment so that the code can be read without
a separate framework.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{levy2026mogbbdm,
  title   = {{Mixture-of-Gaussians-Guided Schedule Design for Brownian Bridge Diffusion Models}},
  author  = {Levy, Ron and Elad, Michael},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Accepted for publication},
  url     = {https://openreview.net/forum?id=NHccmvrBPO}
}
```

The same citation is available in [CITATION.cff](CITATION.cff).

## License

The code and accompanying repository files are distributed under the
[MIT License](LICENSE). The paper is available separately through OpenReview.
