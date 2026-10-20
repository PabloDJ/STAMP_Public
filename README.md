# STAMP: Small-Signal Toolbox for Analysis of Modern Power Systems

STAMP is a MATLAB toolbox for automatic EMT modelling, time-domain simulation,
and small-signal stability analysis of hybrid AC/DC power systems. It was
developed at CITCEA-UPC, Universitat Politècnica de Catalunya.

This fork of [CITCEA-UPC/STAMP_Public](https://github.com/CITCEA-UPC/STAMP_Public)
adds a VeraGrid port and validation workflow for the STAMP
`WSCC_SG_GFOR_GFOL` case. The comparison reproduces the same
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

### Nonlinear converter comparison

Run the separate GFOR/GFOL nonlinear-equation comparison with the same
six-bus power flow and full 88-state dynamic network:

```bash
python3 scripts/compare_stamp_veragrid_nonlinear_converters.py
```

It checks the bus voltages and angles against STAMP, reconstructs the converter
local q-d operating points, checks the full DAE residual, and assigns all 88
VeraGrid eigenvalues to the STAMP reference. It saves the spectrum to
`STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_veragrid_full_dynamic_nonlinear_eigenvalues.csv`.
The nonlinear converter equations are defined in
`veragrid_stamp/nonlinear_converters.py`; the established linearized comparison
continues to use `veragrid_stamp/source_linear_converters.py`.

The comparison maps the nonlinear state coordinates to STAMP's state matrix and
reports the largest matrix-entry differences. At the current operating point,
the maximum matrix-entry difference is about `6.54e-5` and the maximum assigned
eigenvalue difference is about `2.38e-7 1/s`.

The Python bridge is in `veragrid_stamp/`; VeraGrid is an external source
dependency. The starting network is bundled at
`veragrid_stamp/data/IEEE_9_Christoph.gridcal`, so the VeraGrid checkout only
needs to provide the Python source. Configure its source tree before running:

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

Add `nonlinear_converters=True` to this call for the nonlinear GFOR/GFOL
comparison.

### How the dynamic and nonlinear models are copied

`veragrid_stamp/wscc_case.py` opens the bundled grid file as the starting six-bus
topology, sets the 100 MVA, 50 Hz STAMP base, replaces the stored load and
generator dispatch with the STAMP case values, and adds the
GFOR bus-4 and GFOL bus-6 generators as PV power-flow devices. It then attaches
custom VeraGrid RMS `Block` models with `set_rms_model`; these equations are
ported from STAMP, rather than being VeraGrid's stock dynamic templates.

With `dynamic_lines=True`, each of the six `Line` devices gets the
`veragrid_stamp/dynamic_line.py` series-RL model. Its two differential states
are synchronous-frame `iq` and `id`, driven by the from/to q-d voltage
difference, resistance `R`, reactance `X`, and `omega_base = 2*pi*50`. The
four algebraic equations return the from/to P and Q injections to VeraGrid's
network. When `full_dynamic_network=False`, each end also contributes its
algebraic `B/2` charging current. When it is true, those charging terms are
removed from the line equations and summed at each bus instead, avoiding
double counting.

With `full_dynamic_network=True`, `veragrid_stamp/dynamic_network.py` replaces
the three physical loads with dynamic parallel-RL models. Each load has two
inductor-current states (`ilq`, `ild`) and a conductance branch. Case P and Q
and the solved power-flow voltage set its equilibrium conductance
`G=P/|V|^2` and inverse inductance `1/L=omega_base*Q/|V|^2` in per unit; the algebraic P/Q
outputs use VeraGrid's injection sign convention. The same switch adds one
zero-power `Load` device per bus as an interface for a dynamic shunt capacitor.
Its susceptance is the sum of the incident line `B/2` values, and its two
states (`vc_q`, `vc_d`) are constrained to the bus q-d voltage by algebraic
equations. The line, load, and capacitor blocks contribute 12, 6, and 12
differential states, respectively. The full-network comparison explicitly
initializes them from the solved power flow and solves the remaining algebraic
variables before forming the 88-state DAE Jacobian.

The synchronous generator always uses the nonlinear STAMP rotor-circuit,
AC4A exciter, and IEEEG1 governor port in
`veragrid_stamp/nonlinear_generator.py` (17 states). By default the GFOR and
GFOL devices use the source-linear converter blocks in
`veragrid_stamp/source_linear_converters.py`. Set
`nonlinear_converters=True` to attach the full equations in
`veragrid_stamp/nonlinear_converters.py` instead (21 and 20 states). These
include the transformer and damped LCL filter, delayed current and voltage
measurements, current controllers, GFOR voltage and frequency droop, and GFOL
PLL and P/Q control. The nonlinear blocks initialize their q-d currents,
capacitor voltages, controller states, and references from VeraGrid's solved
terminal P/Q/voltage; bus RMS line-to-line voltage is converted to STAMP's
peak phase-voltage base inside the converter model.

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

## Three-phase EMT study

The EMT assembly is in `veragrid_stamp/emt_case.py`. It reuses VeraGrid's abc
bus, PI-line, and parallel R||L load templates. The SG wrapper retains the
validated STAMP six-winding/transformer/control equations, and the GFOR/GFOL
wrapper retains the named q-d controller and LCL states while exposing abc
terminal voltage and current.

By default the EMT case uses the linearized-deviation GFOR/GFOL wrapper in
`veragrid_stamp/emt_converters.py`. Run its initialization gate and Floquet
small-signal study with:

```bash
NUMBA_OPT=0 python3 scripts/run_veragrid_stamp_wscc_emt.py \
  --simulation-time 0.04 --assessment-time 0.04 --time-step 2e-5
```

EMT steady state is a periodic orbit, so the runner compares snapshots one
50-Hz cycle apart (with unwrapped angle states reduced modulo `2*pi`) and uses
`SmallSignalStabilityEmtDriver`, not the RMS state-matrix driver. This
linearized wrapper initializes with zero device-envelope derivatives, but its
unperturbed trajectory excites an EMT-only unstable pair. Its six-mode run
finds rightmost Floquet exponents of approximately
`25.622 +/- j100.961 rad/s`.

The full nonlinear converter equations can instead be connected to the abc
network with:

```bash
NUMBA_OPT=0 python3 scripts/run_veragrid_stamp_wscc_emt.py \
  --nonlinear-converters \
  --simulation-time 0.04 --assessment-time 0.04 --time-step 2e-5 --modes 6
```

This selects `veragrid_stamp/emt_nonlinear_converters.py`, which wraps the same
nonlinear GFOR/GFOL q-d equations used by the RMS comparison. The wrapper now
passes the synchronous generator speed into those equations, matching both the
RMS and linearized EMT frame references. Before that correction, the nonlinear
EMT run used a fixed `1.0 pu` frame speed and its dominant mode was near
`-0.2305 s^-1`, rather than the STAMP RMS value `-0.197765 s^-1`.

With the corrected nonlinear `.veragrid` grid and periodic state saved in
`veragrid_stamp/data/`, the full monodromy benchmark in the VeraGrid repository
(`benchmarks/stamp_wscc/run_emt.py`) closes the orbit within `9.8e-10`. Its
10 rightmost matched modes differ from STAMP RMS by at most `0.00204335 s^-1`;
the dominant EMT mode is `-0.197657 s^-1`. This is case-specific validation;
the extra EMT modes are not forced into a one-to-one RMS comparison.

Each Floquet run writes its exponents and RMS-comparison plot under `scripts/`.
The nonlinear outputs are `stamp_wscc_emt_floquet_exponents_nonlinear.csv` and
`stamp_wscc_emt_floquet_spectrum_nonlinear.png`. The corrected full-monodromy
nonlinear result is saved separately as
`scripts/stamp_wscc_nonlinear_full_emt_floquet_exponents.csv` and
`scripts/stamp_wscc_nonlinear_full_emt_spectrum.png`. The older 88-state
tangent-reduction plot uses the linearized converter wrapper; its clarified
filename is `scripts/stamp_wscc_linearized_tangent_spectrum.png`.

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
