# STAMP–VeraGrid WSCC small-signal bridge

This fork contains the reference STAMP model and a Python bridge for
rebuilding the `WSCC_SG_GFOR_GFOL` case in VeraGrid. The bridge lives in
`veragrid_stamp/`; VeraGrid itself remains an external dependency.

The case is the reduced six-bus WSCC/IEEE-9 topology used by STAMP: one AC4A +
IEEEG1 synchronous generator at bus 1, one grid-forming converter at bus 4,
one grid-following converter at bus 6, six dynamic pi lines, and three loads.

Run from this repository with the VeraGrid source tree on `PYTHONPATH`. The grid case is bundled
at `veragrid_stamp/data/IEEE_9_Christoph.gridcal`:

```bash
NUMBA_CACHE_DIR=/tmp/veragrid_numba \
MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH=/path/to/VeraGrid/src:$PWD \
python3 scripts/run_veragrid_stamp_wscc.py \
  --stamp-reference STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_eigenvalues.csv
```

The runner enforces topology, power-flow voltage/angle agreement, and the full
DAE residual at `t=0` before calculating or saving eigenvalues. It writes a
normalized VeraGrid eigenvalue CSV and performs a permutation-independent
nearest-mode comparison when the Multivac STAMP CSV is supplied.

The converter operating-point reconstruction in
`veragrid_stamp/initialization.py` is a direct port of STAMP's
`generate_initialization_VSC.m`. Model parameters and derived controller gains
are centralized in `veragrid_stamp/parameters.py`.

### Voltage-base boundary

VeraGrid exposes bus voltage as RMS line-to-line per unit. STAMP's converter
q/d circuit uses peak phase-to-neutral quantities, so only that model boundary
applies

`V_peak,LN = sqrt(2/3) * V_rms,LL`.

The conversion and inverse live in `veragrid_stamp/bases.py`. It must not be
applied globally to normal VeraGrid models, whose equations and power
normalization use their existing RMS convention.

To export the actual MATPOWER point used by STAMP (the workbook `PF` tab is not
the authoritative source when `fanals = 2`), run this on Multivac from the
`STAMP` directory and copy the three generated CSV files back:

```bash
matlab -batch "export_wscc_operating_point"
```

## Current validation status

The network importer, parameter transcription, converter operating-point
reconstruction, nonlinear GFOR/GFOL electrical and control blocks, and
end-to-end comparison harness are implemented. The GFOR has 21 states and the
GFOL has 20 states, matching the STAMP state list.

The synchronous generator is now a dedicated 17-state STAMP realization. It
contains the two dynamic generator-transformer currents; the six winding
currents (`is_q`, `is_d`, `if_d`, `ik_d`, `ik1_q`, and `ik2_q`); rotor speed;
three AC4A states; and five IEEEG1 governor/turbine states. Its fixed slack
rotor frame and STAMP `Il2g` current scaling/rotation are included explicitly.

`veragrid_stamp/rotor_circuit.py` ports STAMP's conversion from Xd/Xq and
open-circuit transient/subtransient time constants to the field winding and
d/q damper winding R/L parameters. `veragrid_stamp/nonlinear_generator.py`
uses those values in the complete coupled six-winding inductance matrix and
the generator-transformer R-L q/d equations. At the VeraGrid power-flow point,
the complete model currently initializes with maximum differential and
algebraic residuals of approximately `1.1e-9` and `1.5e-12`, respectively.

The full SSA spectrum is not yet equivalent. STAMP includes 30 dynamic
network states (line currents, bus capacitor voltages, and load currents),
whereas VeraGrid's standard RMS network is algebraic. Consequently this port
currently produces 58 finite dynamic modes rather than STAMP's 88-state
spectrum. The device dynamics are now state-count equivalent; dynamic network
components are the remaining structural difference before the full spectrum
is scientifically comparable.

DAE infinite eigenvalues associated with algebraic variables are reported and
excluded from the finite dynamic-mode CSV; they are not treated as physical
modes.

For a one-to-one spectrum assignment and an explicit list of the 30 unmatched
STAMP modes, run:

```bash
python3 scripts/compare_stamp_veragrid_modes.py \
  STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_eigenvalues.csv \
  STAMP/02_results/veragrid/WSCC_SG_GFOR_GFOL_eigenvalues.csv \
  --output STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_mode_matches.csv
```

The assignment uses a magnitude-scaled complex-plane distance and never reuses
a candidate mode. The `20 Hz` default classification is based on pole
magnitude, so fast nonoscillatory filter poles are classified as fast modes.

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
