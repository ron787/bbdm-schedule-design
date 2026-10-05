# Figure 6: frozen-label concentration

This CPU implementation uses NumPy, SciPy, and Matplotlib. It generates fresh Gaussian-mixture data and compares exact and frozen-label BBDM chains. No neural model is trained or loaded.

```sh
python experiments/dimension/run.py --experiment figure6 --output results/figure6 --workers 3
```

The default run uses the paper's experiment protocol: all 24 combinations of `R = 2,4,8,16` and `d = 16,32,64,128,256,512`, with **4,096 independently generated mixtures and 64 measurements per mixture**. That is 262,144 paired paths per condition. Batches contain 16 mixtures. A reduced smoke run uses 32 mixtures and 16 measurements for each selected condition:

```sh
python experiments/dimension/run.py --experiment figure6 --output results/figure6_quick --quick
```

Four scientific parameters are available. `--components R` (at least 2) selects one component count, and `--dimension d` (at least 1) selects one dimension; omitting either keeps its paper sweep. `--steps S` (at least 2) changes the reverse-step count, and `--seed` accepts any nonnegative integer. Their defaults are 20 steps and seed 0. For example:

```sh
python experiments/dimension/run.py --experiment figure6 --output results/figure6_custom --quick --components 3 --dimension 8 --steps 5 --seed 1
```

The remaining settings are shared diagonal covariance `geomspace(0.5,2,d)`, denoising `H=I`, measurement-noise standard deviation 2, uniform mixture weights, and the schedule `m_s=s/S`, `delta_s=2m_s(1-m_s)`. Frozen labels are sampled from the measurement posterior. The two chains share every reverse Gaussian innovation.

The local squared denoiser gap is evaluated on the frozen path and averaged over **all S calls**, including the measurement endpoint. Label stability tests **the S−1 interior states**. The final squared gap compares the terminal reconstructions. Both squared gaps are divided by dimension. Thus the default uses 20 denoiser calls and 19 stability checks.

The implementation uses float64 and a deterministic NumPy RNG per `(seed,R,d)` condition. Changing the worker count does not change random draws. NumPy and the original CUDA RNG use different draws, so a fresh run should reproduce the statistical findings rather than the historical numbers bit for bit.

Outputs are `results.json`, `results.csv`, and `figure6.png`. The JSON records the complete configuration, runtime, and estimates. `matches_paper_settings` is true only for the full paper grid and sample counts with 20 steps and seed 0; restricted grids, quick runs, and changed scientific settings are marked as custom. Confidence intervals use independent mixture summaries as clusters; estimates of very rare events can remain uncertain. Zero empirical probabilities remain zero in the data. An open triangle on the log plot marks a conditional 95% upper bound, not a positive error estimate.

The saved paper values can also be rendered without simulation:

```sh
python experiments/dimension/run.py --experiment figure6 --output results/figure6_saved --plot-reference
```

This reads the preserved `references/historical_run.json` and writes `reference_results.json`, `reference_results.csv`, and `figure6_reference.png`. Its output is labeled as saved historical data. This mode rejects `--quick` and all four scientific flags, even if an explicitly supplied value equals a default. Default execution always performs a fresh simulation.

The matrix implementation was checked against direct component-wise Gaussian posterior formulas at every one of 20 reverse steps for representative low- and high-dimensional cases. Chain states and denoisers agreed to approximately `1.6e-14`; sampled-label events and final RNG states were identical.

A complete fresh run with the default counts finished in 397.6 seconds using three CPU workers in the validation environment. Median relative differences from the historical results were 0.103% for separation, 0.558% for label error, 0.549% for the local gap, and 0.699% for the final gap. Rare tails vary more: at dimension 512 the four label-failure counts were 5, 3, 17, and 40, compared with 3, 5, 21, and 35 historically. The largest local-gap relative difference was 125.5%, from approximately 4.55e-7 to 1.03e-6. These are independent Monte Carlo realizations, not an exact replay.
