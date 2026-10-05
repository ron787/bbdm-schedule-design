# Validation of the CPU release

Checked on 5 October 2026 on Linux, Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0 and Matplotlib 3.10.8. All executed calculations used the CPU.
The reference is the 74-page camera-ready manuscript linked in the README.

## Full calculations

The experiments were run in full and compared with reference calculations.
The comparison data were used only to check outputs;
they were not inputs to the new simulations.

| Experiment | Check | Measured computation time |
|---|---|---:|
| Figure 1 | All six schedule curves identical to the reference calculation | 0.2 s |
| Figure 2 | All 1,200 measurement rows exactly equal to the reference CPU calculation, including optimized schedules | 26.9 s |
| Figure 3 | All eight curves for S=2,...,1000 identical | 5.9 s |
| Figure 5 | All 278,226,900 candidate pairs considered; identical winning schedules; maximum variance difference 6.22e-15 | 12.6 s |
| Figure 6 | All 24 historical scientific settings run with fresh NumPy draws; 6,291,456 paired trajectories | 397.6 s |
| Figure 7 | All 400 plotted rows and 200 additional Bayesian-reference rows exactly equal to the reference CPU calculation | 12.4 s |
| Appendix D.6 | All four numerical table values identical to the reference calculation and matching the paper's printed precision | 2.2 s |

The Figure 5 timing uses one process. Figures 2 and 7 use three numerical
threads, and Figure 6 uses three processes. These timings are measurements on the validation machine, not runtime
guarantees for other CPUs. Outputs record their own elapsed time and settings.

Figure 2 uses the final paper's unnormalized objective. The figures generated
by this repository emphasize the numerical results; their typography and layout
are not intended as pixel-for-pixel copies of the manuscript.

## Figure 6

The NumPy implementation was checked against direct component-wise Gaussian
formulas along all 20 steps of both coupled chains. Controlled cases used
R=2,d=8; R=16,d=64; and R=16,d=512. Maximum state/denoiser difference was
1.58e-14; label-stability events and NumPy generator states matched exactly.

Its scientific configuration matches the saved paper results: 4,096 independent
mixtures, 64 paths per mixture, batch size 16, seed 0, sampled posterior labels,
the default schedule, 20 steps, and every combination of four component counts
and six dimensions. The NumPy random stream is explicitly different from the
historical CUDA stream. The saved historical file is used for comparison and
the separately labeled `--saved` command.

The full fresh CPU run completed all 6,291,456 paired trajectories. Its median
absolute relative differences from the saved paper values, across 24 settings,
were 0.10% for separation, 0.56% for label error, 0.55% for local denoiser gap
and 0.70% for final sample gap. Maximum relative differences were 0.82%, 66.67%,
125.47% and 66.52%, respectively; the large relative differences occur in the
rare-event tail. At d=512 the fresh label-failure counts were 5, 3, 17 and 40,
compared with the historical counts 3, 5, 21 and 35 for R=2,4,8,16. This is a
statistical replication of the trends, not an exact replay of the CUDA draws.

`validation.json` records the checked configuration, full Figure 6 estimates,
controlled numerical checks and comparison summaries. It is verification
evidence only; the experiment scripts do not read it.

All 96 values in that historical JSON were checked against the manuscript's
vector plot. The maximum relative discrepancy is 4.87e-6, consistent with
PDF-coordinate rounding. This historical-data check is distinct from running
a fresh simulation.

## Interface checks

The common entry point completed all seven experiments in quick mode from a
different working directory, using two CPU workers, in about 19 seconds total.
The explicit saved-data Figure 6 command was also checked. Scripts were parsed
for valid Python and their imports audited: the only third-party runtime
packages are NumPy, SciPy and Matplotlib. No compiler invocation or GPU backend
is part of the release.

Reduced-budget quick results are explicitly identified in their settings;
the cloud quick plots also carry a quick-run title. The cheap deterministic
calculations are still performed fully. All generated files are written under
the requested output directory.

The checked numerical results support the stated experimental claims. A finite
schedule grid and numerical optimization do not establish new global-optimality
proofs, and Monte Carlo rare-event tails have sampling uncertainty.

## Optional model settings

The four controls (`--components`, `--dimension`, `--steps`, `--seed`) were
checked through the common entry point and the individual synthetic scripts.
All seven default quick experiments were rerun. Their 12 numerical CSV files
match the results obtained before adding these controls exactly, excluding elapsed time; both cloud experiments'
schedule files also match exactly. These checks preserve the full-run validation
above without repeating the expensive full Figure 6 simulation.

Custom common-entry-point checks used R=3,d=7,S=5 for Figure 2;
R=4,d=16,S=8 for Figure 7; and R=3,d=8,S=5 for Figure 6, each with seed 23.
All settings reached the experiment, all numerical CSV values were finite,
and the saved configurations correctly identified the runs as custom.

Both cloud scripts also passed the minimum case R=1,d=1,S=2; exact and
frozen-label results coincide for a single component. Figure 6 passed
R=2,d=1,S=2. Its custom S=5 and S=2 trajectories were checked against direct
component-wise formulas, agreeing within 8.9e-16 with identical label events
and RNG states. Selecting a single component count retained the dimension
sweep and exactly matched all six corresponding prior quick results.

Invalid values and unsupported combinations were rejected before simulation.
Saved Figure 6 plotting still works and rejects model overrides. Default and
explicit-default cloud settings produce identical results. Custom and
single-point plots were visually inspected. Compact evidence is recorded under
`optional_model_settings` in `validation.json`.
