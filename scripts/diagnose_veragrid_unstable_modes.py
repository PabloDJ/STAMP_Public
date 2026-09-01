#!/usr/bin/env python3
"""Report state participation in unstable VeraGrid WSCC modes."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def subsystem(name: str) -> str:
    if "STAMP_SG1" in name:
        return "SG"
    if "STAMP_GFOR1" in name:
        return "GFOR"
    if "STAMP_GFOL2" in name:
        return "GFOL"
    return "other"


def main() -> None:
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions

    grid = build_stamp_wscc_grid()
    pf = PowerFlowDriver(grid, power_flow_options())
    pf.run()
    driver = SmallSignalStabilityRmsDriver(
        grid=grid,
        rms_options=RmsOptions(time_step=0.001, simulation_time=1.0, tolerance=1e-8, max_iter=1000),
        sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0.0),
        pf_results=pf.results,
    )
    driver.run()
    values = np.asarray(driver.results.eigenvalues)
    participation = np.asarray(driver.results.participation_factors)
    state_matrix = np.asarray(driver.results.state_matrix)
    names = [str(var) for var in driver.problem.state_vars]
    # Use VeraGrid's compiled partial df/dx for the device-only comparison.
    # ``get_x0`` is the complete [states, algebraics] DAE vector; get_j11
    # handles that layout and returns columns in state_vars order.
    x0 = driver.problem.get_x0()
    dx0 = np.zeros(driver.problem.get_diff_var_number())
    raw_state_jacobian = driver.problem.get_j11(
        x0, dx0, driver.problem.get_dt_value()
    ).toarray()
    reduced_state_jacobian = state_matrix
    stamp_matrix = np.loadtxt(ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv", delimiter=",")
    stamp_slices = {"SG": slice(30, 47), "GFOR": slice(47, 68), "GFOL": slice(68, 88)}
    print("Device-only Jacobian stability (bus voltage held fixed):")
    for group in ("SG", "GFOR", "GFOL"):
        indices = [i for i, name in enumerate(names) if subsystem(name) == group]
        local_values = np.linalg.eigvals(raw_state_jacobian[np.ix_(indices, indices)])
        stamp_values = np.linalg.eigvals(stamp_matrix[stamp_slices[group], stamp_slices[group]])
        print(f"  {group}: VeraGrid unstable={np.count_nonzero(local_values.real > 0)}, "
              f"rightmost={local_values[np.argmax(local_values.real)]:.9g}; "
              f"STAMP unstable={np.count_nonzero(stamp_values.real > 0)}, "
              f"rightmost={stamp_values[np.argmax(stamp_values.real)]:.9g}")
        if group == "GFOL":
            local_block = raw_state_jacobian[np.ix_(indices, indices)]
            stamp_block = stamp_matrix[stamp_slices[group], stamp_slices[group]]
            error = np.abs(local_block-stamp_block)
            print("  GFOL largest Jacobian entry differences:")
            for flat in np.argsort(error, axis=None)[-12:][::-1]:
                row, col = np.unravel_index(flat, error.shape)
                print(f"    d({names[indices[row]]})/d({names[indices[col]]}): "
                      f"VG={local_block[row,col]:.9g}, STAMP={stamp_block[row,col]:.9g}")
    print("Reduced-Jacobian diagonal-block stability (network feedback retained on each diagonal):")
    for group in ("SG", "GFOR", "GFOL"):
        indices = [i for i, name in enumerate(names) if subsystem(name) == group]
        block_values = np.linalg.eigvals(state_matrix[np.ix_(indices, indices)])
        print(f"  {group}: states={len(indices)}, unstable={np.count_nonzero(block_values.real > 0)}, "
              f"rightmost={block_values[np.argmax(block_values.real)]:.9g}")
    for mode in np.flatnonzero(values.real > 0):
        weights = np.abs(participation[:, mode])
        total = weights.sum()
        if total:
            weights /= total
        aggregate = {key: 0.0 for key in ("SG", "GFOR", "GFOL", "other")}
        for name, weight in zip(names, weights):
            aggregate[subsystem(name)] += float(weight)
        print(f"\nmode {mode + 1}: {values[mode].real:.9g} {values[mode].imag:+.9g}j")
        print("  subsystem participation: " + ", ".join(f"{key}={value:.1%}" for key, value in aggregate.items()))
        for index in np.argsort(weights)[-8:][::-1]:
            print(f"  {names[index]}: {weights[index]:.2%}")


if __name__ == "__main__":
    main()
