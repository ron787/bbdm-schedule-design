# BBDM schedule design

Code for **Mixture-of-Gaussians-Guided Schedule Design for Brownian Bridge
Diffusion Models**, by Ron Levy and Michael Elad, accepted at TMLR (2026).
[Paper](https://openreview.net/forum?id=NHccmvrBPO).

Includes the analytical and synthetic MoG experiments below. MNIST and
trained image-model experiments are outside this repository.

## Install

Requires Python 3.11 or newer; tested with Python 3.12. With Conda installed:

```sh
git clone https://github.com/ron787/bbdm-schedule-design.git
cd bbdm-schedule-design
conda create -n bbdm_cpu python=3.12 pip -y
conda activate bbdm_cpu
python -m pip install -r requirements.txt
```

## Run

```sh
python run.py all       # All experiments with the paper's settings
python run.py figure2   # One experiment
```

| Experiment | Result |
|---|---|
| `figure2` | MSE / sliced-Wasserstein tradeoff |
| `figure3` | Analytical BBDM / DDIM comparison |
| `figure5` | Optimal schedules across precision and step budgets |
| `figure6` | Dimension-dependent exact / frozen MoG comparison |
| `figure7` | Low-precision reversal |
| `table-d6` | Finite-step variance-ratio table |

Each experiment saves its files in its own folder. For example,
`results/figure2/` contains the Figure 2 plot (PNG), numerical results (CSV)
and run settings (JSON).

To use two CPU workers and save Figure 2 under `my_run/figure2/`:

```sh
python run.py figure2 --workers 2 --output my_run
```

`--workers N` allows up to N CPU workers/threads (default: up to four).
Use a positive integer, such as 1, 2 or 4, within your available CPU core count.
Larger values can be slower or exceed system resources. Figure 3 runs serially.
Choose a different `--output` folder to preserve earlier runs; reusing one
overwrites its result files.

## Optional settings

For an individual `figure2`, `figure6` or `figure7` run:

| Option | Figure 2 / 7 default | Figure 6 default |
|---|---|---|
| `--components R` | 32 Gaussians | 2, 4, 8, 16 |
| `--dimension d` | 512 | 16, 32, 64, 128, 256, 512 |
| `--steps S` | 20 | 20 |
| `--seed N` | 1234 | 0 |

```sh
python run.py figure2 --components 16 --dimension 128 --steps 10 --seed 42
```

For Figure 6, setting R or d selects one value on that axis; omitting it keeps
the sweep. Other model settings, such as observation noise and component
covariances, keep the paper's values. The plots and saved settings indicate
when these synthetic experiments use parameters different from the paper.

All calculations run fresh on CPU.
Figure 6 uses the paper's scientific settings with NumPy random draws, so
agreement with the original CUDA run is statistical rather than bitwise.

[Citation](CITATION.cff) · [MIT license](LICENSE)
