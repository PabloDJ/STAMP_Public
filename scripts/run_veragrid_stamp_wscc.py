#!/usr/bin/env python3
"""Run and compare the VeraGrid port of STAMP's WSCC SG/GFOR/GFOL case."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from veragrid_stamp.wscc_case import STAMP_LINES, STAMP_LOADS, build_stamp_wscc_grid


def power_flow_options():
    import VeraGridEngine.api as gce
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions

    return PowerFlowOptions(
        solver_type=gce.SolverType.NR,
        retry_with_other_methods=False,
        tolerance=1.0e-10,
        max_iter=50,
        control_q=False,
        control_taps_modules=False,
        control_taps_phase=False,
        control_remote_voltage=False,
        distributed_slack=False,
        initialize_angles=True,
        use_stored_guess=False,
        verbose=0,
    )


def write_modes(path: Path, eigenvalues: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "real", "imaginary", "frequency_hz", "damping"))
        for index, value in enumerate(sorted(eigenvalues, key=lambda z: z.real, reverse=True), start=1):
            magnitude = abs(value)
            damping = -value.real / magnitude if magnitude and np.isfinite(magnitude) else np.nan
            writer.writerow((index, value.real, value.imag, abs(value.imag) / (2.0 * np.pi), damping))


def read_stamp_modes(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = {name.lower(): name for name in (reader.fieldnames or [])}
        real_key = names.get("real")
        imag_key = names.get("imaginary") or names.get("imag")
        if real_key is None or imag_key is None:
            raise ValueError(f"STAMP CSV must contain Real and Imaginary columns: {path}")
        return np.asarray([complex(float(row[real_key]), float(row[imag_key])) for row in reader], dtype=complex)


def compare_modes(reference: np.ndarray, candidate: np.ndarray) -> None:
    distances = np.abs(reference[:, None] - candidate[None, :])
    nearest = np.min(distances, axis=1)
    print(f"mode count: STAMP={reference.size}, VeraGrid={candidate.size}")
    print(f"rightmost STAMP:    {reference[np.argmax(reference.real)]}")
    print(f"rightmost VeraGrid: {candidate[np.argmax(candidate.real)]}")
    print(f"STAMP-to-nearest-VeraGrid error: median={np.median(nearest):.9g}, max={np.max(nearest):.9g}")


def audit_topology(grid) -> None:
    assert len([bus for bus in grid.buses if not bus.is_dc]) == 6
    assert len(grid.lines) == len(STAMP_LINES)
    assert len(grid.loads) == len(STAMP_LOADS)
    print("topology audit: 6 AC buses, 6 lines, 3 loads, 1 SG, 1 GFOR, 1 GFOL")
    converter_counts = {device.name: len(device.rms_model.state_vars) for device in grid.generators[1:]}
    assert converter_counts["STAMP GFOR1"] == 21
    assert converter_counts["STAMP GFOL2"] == 20
    print("converter state audit: GFOR=21, GFOL=20 (matching STAMP)")
    sg_states = sum(len(block.state_vars) for block in grid.generators[0].rms_model.get_all_blocks())
    print(f"generator state audit: VeraGrid SG assembly={sg_states}, STAMP SG reference=17")


def audit_initialization(grid, power_flow, small_signal) -> tuple[float, float, float, float]:
    voltage = np.asarray(power_flow.results.voltage)
    reference_path = REPOSITORY_ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_power_flow.csv"
    with reference_path.open(newline="", encoding="utf-8-sig") as stream:
        reference_rows = list(csv.DictReader(stream))
    expected_vm = np.asarray([float(row["Vm"]) for row in reference_rows])
    expected_va = np.deg2rad(np.asarray([float(row["theta"]) for row in reference_rows]))
    ac_indices = [index for index, bus in enumerate(grid.buses) if not bus.is_dc]
    actual = voltage[ac_indices]
    vm_error = float(np.max(np.abs(np.abs(actual) - expected_vm)))
    va_error = float(np.max(np.abs(np.angle(actual) - expected_va)))
    print(f"exact STAMP-PF max |dVm|: {vm_error:.9g}")
    print(f"exact STAMP-PF max |dVa|: {va_error:.9g} rad")

    problem = small_signal.problem
    x0 = problem.get_x0()
    dx0 = np.zeros_like(x0)
    state_residual = np.asarray(problem.rhs_state(x0, dx0), dtype=float)
    algebraic_residual = np.asarray(problem.rhs_algebraic(x0, dx0), dtype=float)
    state_error = float(np.max(np.abs(state_residual)))
    algebraic_error = float(np.max(np.abs(algebraic_residual)))
    print(f"t=0 max state residual: {state_error:.9g}")
    print(f"t=0 max algebraic residual: {algebraic_error:.9g}")
    if state_error > 1.0e-7:
        state_vars = getattr(problem, "_state_vars", getattr(problem, "state_vars", ()))
        for index in np.argsort(np.abs(state_residual))[-10:][::-1]:
            label = getattr(state_vars[index], "name", str(state_vars[index])) if len(state_vars) else str(index)
            print(f"  state residual {label}: {state_residual[index]:.9g}")
    return vm_error, va_error, state_error, algebraic_error


def run(output: Path, stamp_reference: Path | None) -> None:
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    import VeraGridEngine.api as vge

    grid = build_stamp_wscc_grid()
    audit_topology(grid)
    power_flow = PowerFlowDriver(grid, power_flow_options())
    power_flow.run()
    if not np.all(np.asarray(power_flow.results.converged)):
        raise RuntimeError("VeraGrid power flow did not converge")
    print("power flow converged")

    rms_options = RmsOptions(time_step=0.001, simulation_time=1.0, tolerance=1.0e-8, max_iter=1000, verbose=0)
    problem = RmsProblemDae(grid=grid, options=rms_options, pf_results=power_flow.results)
    problem.set_events_group(RmsEventsGroup("simulation1"))
    # Run SSA on this exact, already populated RMS problem.  Do not let the
    # small-signal driver silently rebuild a second formulation from the grid.
    small_signal = SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=problem.options,
        sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0.0),
        pf_results=power_flow.results,
    )
    small_signal.problem = problem
    small_signal.k = problem.get_states_number()
    vm_error, va_error, state_error, algebraic_error = audit_initialization(grid, power_flow, small_signal)
    if state_error > 1.0e-7 or algebraic_error > 1.0e-7:
        raise RuntimeError("VeraGrid RMS model is not consistently initialized at the power-flow point")
    small_signal.run()
    all_eigenvalues = np.asarray(small_signal.results.eigenvalues, dtype=complex).reshape(-1)
    finite = np.isfinite(all_eigenvalues.real) & np.isfinite(all_eigenvalues.imag)
    eigenvalues = all_eigenvalues[finite]
    print(f"finite dynamic modes: {eigenvalues.size}; algebraic/infinite modes: {np.count_nonzero(~finite)}")
    if eigenvalues.size == 0:
        raise RuntimeError("VeraGrid SSA returned no finite dynamic eigenvalues")
    write_modes(output, eigenvalues)
    print(f"wrote VeraGrid modes: {output}")
    if stamp_reference is not None:
        compare_modes(read_stamp_modes(stamp_reference), eigenvalues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("STAMP/02_results/veragrid/WSCC_SG_GFOR_GFOL_eigenvalues.csv"))
    parser.add_argument("--stamp-reference", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output.resolve(), args.stamp_reference.resolve() if args.stamp_reference else None)
