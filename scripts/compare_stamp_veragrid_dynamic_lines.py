#!/usr/bin/env python3
"""Compare dynamic-line VeraGrid DAE modes with the matching STAMP reduction."""

from pathlib import Path
import sys

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def main() -> None:
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions

    grid = build_stamp_wscc_grid(impedance_loads=True, dynamic_lines=True)
    pf = PowerFlowDriver(grid, power_flow_options()); pf.run()
    problem = RmsProblemDae(grid, RmsOptions(time_step=0.001), pf.results)
    problem.set_events_group(RmsEventsGroup("dynamic_line_ssa"))
    vector = problem.get_x0()
    state_index = {str(var): i for i, var in enumerate(problem.state_vars)}

    # RmsProblemDae currently skips explicit state initialization for Line
    # devices. Populate the two series-current states from the solved from-end
    # complex power, removing the pi-section shunt current.
    for branch, line in enumerate(grid.lines):
        f = grid.buses.index(line.bus_from)
        voltage = pf.results.voltage[f]
        vq, vd = voltage.real, -voltage.imag
        power = pf.results.Sf[branch]/grid.Sbase
        denom = vq*vq+vd*vd
        total_iq = (power.real*vq-power.imag*vd)/denom
        total_id = (power.real*vd+power.imag*vq)/denom
        bus_f = int(line.bus_from.name.removeprefix("Bus"))
        bus_t = int(line.bus_to.name.removeprefix("Bus"))
        name = f"NET.{min(bus_f, bus_t)}{max(bus_f, bus_t)}"
        vector[state_index[f"{name}.iq"]] = total_iq-line.B*vd/2.0
        vector[state_index[f"{name}.id"]] = total_id+line.B*vq/2.0

    dx = np.zeros(problem.get_diff_var_number())
    h = problem.get_dt_value()
    fx, fy = problem.get_j11(vector, dx, h).toarray(), problem.get_j12(vector, dx, h).toarray()
    gx, gy = problem.get_j21(vector, dx, h).toarray(), problem.get_j22(vector, dx, h).toarray()
    nx, ny = fx.shape[0], gy.shape[0]
    dae_jacobian = np.block([[fx, fy], [gx, gy]])
    descriptor = np.zeros_like(dae_jacobian); descriptor[:nx, :nx] = np.eye(nx)
    vg_all = la.eigvals(dae_jacobian, descriptor)
    vg_modes = vg_all[np.isfinite(vg_all)]

    stamp = np.loadtxt(ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv", delimiter=",")
    eliminated = np.arange(12, 30)  # bus-capacitor voltages and RL-load currents
    retained = np.r_[0:12, 30:88]
    stamp_70 = (stamp[np.ix_(retained, retained)]
                - stamp[np.ix_(retained, eliminated)]
                @ np.linalg.solve(stamp[np.ix_(eliminated, eliminated)],
                                  stamp[np.ix_(eliminated, retained)]))
    stamp_modes = np.linalg.eigvals(stamp_70)

    state_residual = problem.rhs_state(vector, np.zeros_like(vector))
    algebraic_residual = problem.rhs_algebraic(vector, np.zeros_like(vector))
    print(f"VeraGrid: states={nx}, algebraics={ny}, finite modes={vg_modes.size}, "
          f"infinite modes={vg_all.size-vg_modes.size}")
    print(f"initial residuals: state={np.max(np.abs(state_residual)):.6g}, "
          f"algebraic={np.max(np.abs(algebraic_residual)):.6g}")
    for label, modes in (("VeraGrid dynamic lines", vg_modes), ("STAMP dynamic lines", stamp_modes)):
        unstable = modes[modes.real > 1e-8]
        print(f"{label}: unstable={unstable.size}, rightmost={modes[np.argmax(modes.real)]:.12g}")
        for mode in sorted(unstable, key=lambda value: value.real, reverse=True)[:8]:
            print(f"  {mode:.12g}")

    output = ROOT/"STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_STAMP_dynamic_lines_A.csv"
    np.savetxt(output, stamp_70, delimiter=",")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
