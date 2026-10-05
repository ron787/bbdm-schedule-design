# Synthetic MoG clouds: Figures 2 and 7

These scripts generate all observations from the mathematical model using NumPy and SciPy on the CPU. Matplotlib draws the result. There is no model training, pretrained checkpoint, GPU framework, data download, or stored plot-coordinate input.

From the repository root:

```sh
python experiments/clouds/run.py --experiment figure2 --output results/figure2 --workers 3
python experiments/clouds/run.py --experiment figure7 --output results/figure7 --workers 3
```

Add `--quick` for an explicitly marked smoke run with **8 measurements, 64 reconstruction samples, and 32 projections**. Quick results are not the paper figures. Full defaults are **100 measurements, 256 samples, and 256 projections**. `--workers` controls CPU numerical-library threads.

Four scientific settings can be changed in either mode:

| Option | Paper default | Allowed values |
|---|---:|---|
| `--components` | 32 | Positive integer |
| `--dimension` | 512 | Positive integer |
| `--steps` | 20 | Integer at least 2 |
| `--seed` | 1234 | Nonnegative integer |

For example, a small custom run is:

```sh
python experiments/clouds/run.py --experiment figure2 --output results/custom-figure2 --components 3 --dimension 7 --steps 5 --seed 0 --quick
```

Other scientific choices, including the noise levels, covariance family, sample counts, and optimization settings, remain fixed by the experiment and quick/full mode. Custom plots show R, d, S, and seed in the title. `settings.json` records all effective values and `matches_paper_settings`, which is true only for a full run with the paper's R, d, S, and seed. Changing only CPU workers does not affect that flag.

## Output

Each command writes only to its `--output` directory:

- `figure2.png` or `figure7.png`: newly generated experiment plot.
- `per_measurement.csv`: every exact/frozen measurement result and posterior/MMSE reference.
- `summary.csv`: means and standard errors across measurements.
- `settings.json`: effective dimensions, component/step/sample counts, seeds, whether the settings match the paper, numerical options, package versions, and runtime.
- `schedules.json`: exact optimized or fixed parameters and analytical objectives.

MSE is summed over all coordinates before averaging reconstruction samples. Sliced W2 uses equal-sized independent posterior reference samples, sorts their one-dimensional projections, and takes the square root of the mean squared difference. Figure 7's CSV includes posterior/MMSE references as extra diagnostics; the plot shows only its two edge schedules.

## Paper configuration

The prior has uniform component weights and independently uniform component means on [-1,1]. Figure 2 uses diagonal covariance `geomspace(0.5,2,512)` and measurement-noise standard deviation 0.1. Its blend is **unnormalized**, with weights 0, 0.5, 0.75, and 1. The default schedule is also evaluated.

The bounded parameter order is `(alpha,beta,c,gamma)`, with alpha,beta in [1,2] and c,gamma in [0.2,2]. Optimization uses the paper's 16 restart-list entries, including the duplicate default start, yielding 15 distinct initializations. Each uses SciPy L-BFGS-B with maximum 300 iterations.

Figure 7 uses covariance `8 I`, measurement-noise standard deviation 3, and fixed schedules `(1,2,2,0.2)` and `(2,1,0.2,2)`. Both experiments use seed 1234 for component means, seed 1537 for measurement contexts, and seed 1638 for projection directions. Frozen labels are independently sampled for each reconstruction and held fixed through its reverse trajectory. Exact/frozen chains share innovations.

The simulation uses endpoint clipping at 1e-6, as in the paper experiments; the analytical objective uses the exact endpoint formula. Small optimizer changes between SciPy versions can select different, nearly equivalent parameter tuples, so `settings.json` records the installed versions.

## Implementation and validation

`model.py` contains the diagonal Gaussian posterior, schedule objective, optimizer, streaming random-data generator, oracle/frozen reverse chains, and sliced-W2 metric. `run.py` supplies the paper settings, records outputs, and plots the fresh observations.

The implementation preserves the paper's numerical calculation and random-number order while processing one measurement at a time. This avoids storing about 2.2 GB of reverse innovations at the paper defaults. At those defaults, a measurement's innovation array is about 21 MB; at most two are temporarily alive during the next measurement's generation.

Both experiments were validated with full runs. All **1,200 Figure 2 metric rows and all 600 Figure 7 rows** (including the 200 additional posterior/MMSE reference rows) matched the reference CPU calculation exactly. All plotted observations also matched the paper within numerical/PDF-coordinate precision. Full computation took approximately **27 seconds for Figure 2 and 12 seconds for Figure 7** with three CPU threads on the validation machine; other CPUs will differ.
