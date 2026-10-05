# BBDM schedule design

CPU code for **Mixture-of-Gaussians-Guided Schedule Design for Brownian Bridge
Diffusion Models**, by Ron Levy and Michael Elad, accepted at TMLR (2026).
[Paper](https://openreview.net/forum?id=NHccmvrBPO).

Includes the analytical and synthetic MoG experiments below. MNIST and
trained image-model experiments are outside this repository.

## Install

Requires Python 3.11 or newer.

```sh
git clone https://github.com/ron787/bbdm-schedule-design.git
cd bbdm-schedule-design
python -m pip install -r requirements.txt
```

## Run

```sh
python run.py all --quick   # Smaller budgets to check the installation
python run.py all           # Full paper settings
python run.py figure2       # One experiment
```

| Experiment | Result |
|---|---|
| `figure2` | MSE / sliced-Wasserstein tradeoff |
| `figure3` | Analytical BBDM / DDIM comparison |
| `figure5` | Optimal schedules across precision and step budgets |
| `figure6` | Dimension-dependent exact / frozen MoG comparison |
| `figure7` | Low-precision reversal |
| `table-d6` | Finite-step variance-ratio table |

Outputs go to `results/<experiment>/`: plots, numerical results and settings.
Use `--output PATH` to keep separate runs; rerunning replaces previous outputs.
`--workers N` controls CPU parallelism.

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
the sweep. Other scientific settings remain fixed. Custom and quick runs are
identified in their outputs.

All calculations run fresh on CPU. Figure 2 uses the unnormalized objective.
Figure 6 uses the paper's scientific settings with NumPy random draws, so
agreement with the original CUDA run is statistical rather than bitwise.

[Citation](CITATION.cff) · [MIT license](LICENSE)
