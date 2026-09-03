# STAMP: Small-Signal Toolbox for Analysis of Modern Power Systems

STAMP is a MATLAB toolbox for automatic EMT modelling, time-domain simulation,
and small-signal stability analysis of hybrid AC/DC power systems. It was
developed at CITCEA-UPC, Universitat Politècnica de Catalunya.

This working repository also contains a VeraGrid port and validation workflow
for the STAMP `WSCC_SG_GFOR_GFOL` case. The comparison reproduces the same
topology, operating point, 88 differential states, initialization, Jacobian,
and eigenvalue spectrum in both tools.

## STAMP installation

Clone or download the repository and add the `STAMP` directory and its
subdirectories to the MATLAB path. MATLAB R2023b or newer is recommended.

The SSA and time-domain features require:

- MATLAB;
- Simulink;
- Simscape Electrical / SimPowerSystems;
- Signal Processing Toolbox; and
- DSP System Toolbox.

Run the WSCC example from MATLAB with:

```matlab
cd STAMP
SSA
```

The example sets `fanals = 2`, so STAMP runs a new MATPOWER power flow before
initializing the dynamic models and constructing the state-space system. The
stored `PF` worksheet is not the authoritative operating point for this run.

## STAMP–VeraGrid WSCC comparison

The Python bridge is in `veragrid_stamp/`; VeraGrid is an external source
dependency. Configure its source tree and grid-data root before running:

```bash
export VERAGRID_ROOT=/path/to/VeraGrid
export PYTHONPATH="$VERAGRID_ROOT/src:$PWD"
export NUMBA_CACHE_DIR=/tmp/veragrid-numba
export MPLCONFIGDIR=/tmp/matplotlib
```

The validated case contains:

- a 17-state STAMP synchronous-generator, AC4A exciter, and IEEEG1 governor;
- a 21-state grid-forming converter at bus 4;
- a 20-state grid-following converter at bus 6;
- 12 dynamic series-line current states;
- 12 dynamic bus-capacitor voltage states; and
- 6 dynamic RL-load current states.

The full STAMP-equivalent network is selected with:

```python
build_stamp_wscc_grid(
    dynamic_lines=True,
    full_dynamic_network=True,
)
```

### Parameter sources

The authoritative case data remain in the STAMP workbooks:

- `STAMP/01_data/cases/WSCC_SG_GFOR_GFOL.xlsx`
- `STAMP/01_data/cases/WSCC_SG_GFOR_GFOL_data_sg.xlsx`
- `STAMP/01_data/cases/WSCC_SG_GFOR_GFOL_data_vsc.xlsm`

`veragrid_stamp/parameters.py` is a typed transcription of the system,
generator, converter, exciter, and governor parameters. Derived controller
gains and per-unit base conversions are kept explicit there.

The dynamic line parameters come from the main workbook's `AC-NET` sheet.
For each bus, the dynamic shunt susceptance is assembled from the incident
half-line charging terms:

```text
B_bus,i = sum(B_ij / 2)
```

The bus capacitor currently uses a synthetic zero-power VeraGrid `Load` as an
adapter to the existing P/Q device interface. Electrically it is an aggregated
line-charging shunt capacitor. The physical loads use a separate dynamic
parallel-RL model.

### Power flow and initialization

STAMP runs MATPOWER through `PF_results.m`. VeraGrid runs `PowerFlowDriver`.
Each implementation uses its solved power-flow result to initialize its
dynamic model.

The VeraGrid comparison starts from `problem.get_x0()` and then initializes
the explicit line-current, load-current, and bus-voltage states consistently
with the solved network. Before SSA, the full DAE residual is checked.

Current maximum initial residuals are approximately:

```text
differential: 2.6e-9
algebraic:    9.7e-17
```

### Small-signal comparison

Run the complete 88-state comparison with:

```bash
python3 scripts/compare_stamp_veragrid_full_dynamic_network.py
```

Current result:

```text
STAMP unstable modes:     0
VeraGrid unstable modes:  0
STAMP rightmost mode:    -0.197765415617
VeraGrid rightmost mode: -0.197765415658
maximum Jacobian error:   6.54e-5
Jacobian RMS error:       1.16e-6
```

Generate the eigenvalue comparison figure with:

```bash
python3 scripts/plot_stamp_veragrid_eigenvalues.py
```

STAMP eigenvalues are plotted as circles and VeraGrid eigenvalues as crosses.
The output is written to
`STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_eigenvalue_comparison.png`.

The network-state elimination experiments are also retained. The 58-, 70-,
and 76-state formulations can contain unstable modes even though the complete
88-state network is stable. These are changes in network formulation rather
than unexplained VeraGrid-only modes.

For implementation details, equations, model mappings, and validation notes,
see [README_VERAGRID_STAMP.md](README_VERAGRID_STAMP.md).

## Reproducible Multivac benchmark

STAMP and VeraGrid can be benchmarked sequentially on the same Multivac
compute node and one CPU core:

```bash
cd ~/STAMP_Public
sbatch scripts/benchmark_multivac_same_node.sh
```

The benchmark uses 20 warmed repetitions and records raw samples under
`STAMP/02_results/comparison/`. Queue time and program startup are excluded
from the repeated kernels.

Measured single-core medians for the 88-state case were approximately:

| Measurement | STAMP | VeraGrid |
|---|---:|---:|
| Dense eigenvalue solution, 88-by-88 | 3.45 ms | 2.32 ms |
| Native SSA routine | 29.27 ms | 21.56 ms |

The dense 88-by-88 eigenvalue calculation is the strict numerical-kernel
comparison. Native SSA routines perform different post-processing and must be
reported separately.

VeraGrid sparse SSA with `k=10` took about 25.39 ms for this small case, so its
overhead exceeded its benefit. The separate scaling experiment in
`scripts/benchmark_veragrid_ssa_scaling.py` uses block-diagonal replicas of the
validated WSCC descriptor system to demonstrate solver scaling. Those systems
are synthetic benchmark matrices, not additional physical grids with ported
dynamic models.

Generate the runtime plot with:

```bash
python3 scripts/plot_ssa_benchmarks.py
```

## Experimental EMT model

The three-phase EMT assembly under `veragrid_stamp/emt_case.py` is
experimental and is not part of the validated RMS equivalence. Its periodic
steady-state initialization remains under investigation; EMT modes should not
currently be presented as an RMS-versus-EMT equivalence result.

## License

STAMP is licensed under the [Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/)
(MPLv2).

Original author: CITCEA-UPC

Copyright (c) 2025 CITCEA-UPC

## Citation

If you use STAMP in your research, please cite:

> Arevalo-Soler, Josep, et al. “A Matlab-based Toolbox for Automatic EMT
> Modeling and Small-Signal Stability Analysis of Modern Power Systems.”
> arXiv preprint arXiv:2506.22201 (2025).

If you use STAMP's static power-flow features, please also cite:

> Zimmerman, Ray Daniel, Carlos Edmundo Murillo-Sánchez, and Robert John
> Thomas. “MATPOWER: Steady-state operations, planning, and analysis tools for
> power systems research and education.” IEEE Transactions on Power Systems
> 26.1 (2010): 12–19.

> Beerten, Jef, and Ronnie Belmans. “Development of an open source power flow
> software for high voltage direct current grids and hybrid AC/DC systems:
> MATACDC.” IET Generation, Transmission & Distribution 9.10 (2015): 966–974.

## Acknowledgement

STAMP development was partially funded by the REFORMING Project
(PID2021-127788OA-I00), supported by the Fondo Europeo de Desarrollo
Regional/Ministerio de Ciencia e Innovación–Agencia Estatal de Investigación.

## Contact

For inquiries, contact:

- dionysios.moutevelis@upc.edu
- marc.cheah@upc.edu
- josep.arevalo.soler@gmail.com

Visitors since 2026-03-09:

![Visits](https://visitor-badge.laobi.icu/badge?page_id=CITCEA-UPC.STAMP_Public)
