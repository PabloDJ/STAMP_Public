#!/usr/bin/env python3
"""Compare VeraGrid's reduced RMS Jacobian with STAMP's device-state block."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def canonical_veragrid_name(name: str) -> str:
    result = (name.replace("STAMP_SG1.", "SG1.")
                  .replace("STAMP_GFOR1.", "GFOR1.")
                  .replace("STAMP_GFOL2.", "GFOL2."))
    if result == "SG1.ig_q":
        return "SG1.ig_qx"
    if result == "SG1.ig_d":
        return "SG1.ig_dx"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impedance-loads", action="store_true",
                        help="Use voltage-dependent impedance-equivalent RMS loads")
    args = parser.parse_args()
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import compute_state_matrix

    grid = build_stamp_wscc_grid(impedance_loads=args.impedance_loads)
    print(f"VeraGrid RMS load formulation: {'impedance' if args.impedance_loads else 'constant power'}")
    pf = PowerFlowDriver(grid, power_flow_options())
    pf.run()
    options = RmsOptions(time_step=0.001, simulation_time=1.0, tolerance=1e-8, max_iter=1000)
    problem = RmsProblemDae(grid=grid, options=options, pf_results=pf.results)
    problem.set_events_group(RmsEventsGroup("jacobian_comparison"))
    x = problem.get_x0()
    dx = np.zeros(problem.get_diff_var_number(), dtype=float)
    _, a_veragrid = compute_state_matrix(problem=problem, x=x, dx=dx)

    # Independent numerical reduction.  This is also a regression check for
    # VeraGrid's compiled symbolic partial Jacobians used above.
    nx = problem.get_states_number()
    ny = problem.get_algebraic_var_number()
    dx_full = np.zeros_like(x)
    f0 = np.asarray(problem.rhs_state(x, dx_full), dtype=float)
    g0 = np.asarray(problem.rhs_algebraic(x, dx_full), dtype=float)
    fx = np.empty((nx, nx)); gx = np.empty((ny, nx))
    fy = np.empty((nx, ny)); gy = np.empty((ny, ny))
    for column in range(nx + ny):
        step = 1e-7 * max(1.0, abs(x[column]))
        shifted = x.copy(); shifted[column] += step
        df = (np.asarray(problem.rhs_state(shifted, dx_full), dtype=float) - f0) / step
        dg = (np.asarray(problem.rhs_algebraic(shifted, dx_full), dtype=float) - g0) / step
        if column < nx:
            fx[:, column], gx[:, column] = df, dg
        else:
            fy[:, column - nx], gy[:, column - nx] = df, dg
    a_numerical = fx - fy @ np.linalg.solve(gy, gx)
    native_blocks = {
        "fx": np.asarray(problem.get_j11(x, dx, problem.get_dt_value()).toarray()),
        "fy": np.asarray(problem.get_j12(x, dx, problem.get_dt_value()).toarray()),
        "gx": np.asarray(problem.get_j21(x, dx, problem.get_dt_value()).toarray()),
        "gy": np.asarray(problem.get_j22(x, dx, problem.get_dt_value()).toarray()),
    }
    numerical_blocks = {"fx": fx, "fy": fy, "gx": gx, "gy": gy}
    row_names = {
        "fx": [str(v) for v in problem.state_vars],
        "fy": [str(v) for v in problem.state_vars],
        "gx": [f"g[{i}]" for i in range(ny)],
        "gy": [f"g[{i}]" for i in range(ny)],
    }
    column_names = {
        "fx": [str(v) for v in problem.state_vars],
        "fy": [str(v) for v in problem.algebraic_vars],
        "gx": [str(v) for v in problem.state_vars],
        "gy": [str(v) for v in problem.algebraic_vars],
    }
    print("native partial-Jacobian audit:")
    for label in ("fx", "fy", "gx", "gy"):
        error = native_blocks[label] - numerical_blocks[label]
        flat = int(np.argmax(np.abs(error)))
        row, column = np.unravel_index(flat, error.shape)
        print(f"  {label}: max={np.max(np.abs(error)):.12g}, "
              f"RMS={np.sqrt(np.mean(error*error)):.12g}, "
              f"at d({row_names[label][row]})/d({column_names[label][column]}): "
              f"native={native_blocks[label][row, column]:+.9g}, "
              f"FD={numerical_blocks[label][row, column]:+.9g}")
    target_row = next(i for i, v in enumerate(problem.state_vars) if str(v) == "STAMP_GFOL2.is_q")
    target_col = next(i for i, v in enumerate(problem.state_vars) if str(v) == "STAMP_GFOL2.Ke_P")
    from VeraGridEngine.Utils.Symbolic.symbolic import get_expression_vars
    target_eq = problem._state_eqs[target_row]
    target_var = problem.state_vars[target_col]
    expression_uids = {v.uid for v in get_expression_vars(target_eq)}
    target_derivative = target_eq.diff(target_var, dt=problem._dt)
    print("GFOL Ke_P dependency audit:")
    print(f"  Ke_P UID present in is_q RHS expression: {target_var.uid in expression_uids}")
    print(f"  symbolic derivative is constant zero: "
          f"{getattr(target_derivative, 'value', None) == 0}")
    def children(expr):
        result = []
        for attribute in ("left", "right", "operand", "arg", "arg1", "arg2"):
            child = getattr(expr, attribute, None)
            if child is not None:
                result.append(child)
        return result

    def is_zero_derivative(expr):
        derivative = expr.diff(target_var, dt=problem._dt)
        return getattr(derivative, "value", None) == 0

    losses = []
    def find_derivative_losses(expr, depth=0):
        node_children = children(expr)
        child_has_target = [target_var.uid in {v.uid for v in get_expression_vars(child)}
                            for child in node_children]
        if target_var.uid in {v.uid for v in get_expression_vars(expr)} and is_zero_derivative(expr):
            nonzero_children = [child for child, has_target in zip(node_children, child_has_target)
                                if has_target and not is_zero_derivative(child)]
            if nonzero_children:
                losses.append((depth, expr, nonzero_children))
        for child, has_target in zip(node_children, child_has_target):
            if has_target:
                find_derivative_losses(child, depth + 1)

    find_derivative_losses(target_eq)
    if losses:
        depth, loss_expr, surviving_children = max(losses, key=lambda item: item[0])
        print(f"  first derivative-loss node: {type(loss_expr).__name__}, depth={depth}")
        print(f"  expression: {loss_expr}")
        print(f"  child with nonzero derivative: {surviving_children[0]}")
        raw_derivative = loss_expr._diff1(target_var, problem._dt)
        print(f"  loss-node UID: {loss_expr.uid}; child UIDs: "
              f"{[getattr(child, 'uid', None) for child in children(loss_expr)]}")
        print(f"  raw derivative before simplify: {raw_derivative}")
        print(f"  raw derivative UID: {raw_derivative.uid}")
        if hasattr(loss_expr, "left"):
            left = loss_expr.left
            print(f"  coefficient simplified: {left.simplify()}; coefficient UID={left.uid}; "
                  f"factor UIDs={[getattr(child, 'uid', None) for child in children(left)]}; "
                  f"type={type(left).__name__}; value={getattr(left, 'value', 'n/a')!r}; "
                  f"name={getattr(left, 'name', 'n/a')!r}")
    else:
        print("  no parent/child derivative-loss transition found")
    print(f"native-vs-numerical reduced Jacobian max error: "
          f"{np.max(np.abs(a_veragrid-a_numerical)):.12g}")
    native_rightmost = np.linalg.eigvals(a_veragrid)
    numerical_rightmost = np.linalg.eigvals(a_numerical)
    print(f"rightmost native Jacobian mode: {native_rightmost[np.argmax(native_rightmost.real)]:.12g}")
    print(f"rightmost numerical Jacobian mode: {numerical_rightmost[np.argmax(numerical_rightmost.real)]:.12g}")

    stamp_names = (ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_state_names.txt").read_text().splitlines()
    a_stamp = np.loadtxt(ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv", delimiter=",")
    device_names = stamp_names[30:]
    vg_names = [canonical_veragrid_name(str(variable)) for variable in problem.state_vars]
    vg_index = {name: index for index, name in enumerate(vg_names)}
    missing = [name for name in device_names if name not in vg_index]
    if missing:
        raise RuntimeError(f"VeraGrid is missing STAMP states: {missing}")
    order = [vg_index[name] for name in device_names]
    # Use the independently checked numerical Jacobian for model comparison;
    # the native matrix above is retained and reported as a compiler audit.
    a_vg_ordered = a_numerical[np.ix_(order, order)]
    a_vg_native_ordered = a_veragrid[np.ix_(order, order)]
    a_stamp_devices = a_stamp[30:, 30:]
    # VeraGrid eliminates its network algebraically.  The apples-to-apples
    # STAMP matrix therefore sets the 30 network derivatives to zero and
    # eliminates those states by a Schur complement.
    a_stamp_network = a_stamp[:30, :30]
    a_stamp_quasistatic = (a_stamp_devices
                           - a_stamp[30:, :30]
                           @ np.linalg.solve(a_stamp_network, a_stamp[:30, 30:]))
    # STAMP shifts the global network q-d reference to the SG rotor. VeraGrid
    # keeps the slack-bus voltage at angle zero. Rotate only states expressed
    # in the global network frame before comparing matrix entries; controller
    # and machine-local states remain unchanged.
    sg_lp_path = ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_sg_linearization_point.csv"
    import csv
    with sg_lp_path.open(newline="", encoding="utf-8-sig") as stream:
        sg_lp = {row["field"]: float(row["value"]) for row in csv.DictReader(stream)}
    reference_shift = np.arctan2(-sg_lp["vd_bus0"], sg_lp["vq_bus0"])
    rotation = np.asarray([[np.cos(reference_shift), np.sin(reference_shift)],
                           [-np.sin(reference_shift), np.cos(reference_shift)]])
    transform = np.eye(len(device_names))
    global_pairs = (("SG1.ig_qx", "SG1.ig_dx"),
                    ("GFOR1.ig_q", "GFOR1.ig_d"),
                    ("GFOR1.is_q", "GFOR1.is_d"),
                    ("GFOR1.ucap_q", "GFOR1.ucap_d"),
                    ("GFOL2.ig_q", "GFOL2.ig_d"),
                    ("GFOL2.is_q", "GFOL2.is_d"),
                    ("GFOL2.ucap_q", "GFOL2.ucap_d"))
    name_index = {name: i for i, name in enumerate(device_names)}
    for q_name, d_name in global_pairs:
        pair = [name_index[q_name], name_index[d_name]]
        transform[np.ix_(pair, pair)] = rotation
    a_vg_ordered = transform @ a_vg_ordered @ transform.T
    a_vg_native_ordered = transform @ a_vg_native_ordered @ transform.T
    difference = a_vg_ordered - a_stamp_devices
    quasistatic_difference = a_vg_ordered - a_stamp_quasistatic
    native_quasistatic_difference = a_vg_native_ordered - a_stamp_quasistatic

    print(f"matrix shapes: VeraGrid={a_veragrid.shape}, STAMP={a_stamp.shape}, compared={difference.shape}")
    print(f"global q-d reference rotation applied: {reference_shift:.12g} rad")
    print(f"max absolute device-block error: {np.max(np.abs(difference)):.12g}")
    print(f"RMS device-block error: {np.sqrt(np.mean(difference * difference)):.12g}")
    print(f"entries within 1e-6: {np.mean(np.abs(difference) <= 1e-6):.1%}")
    print(f"quasi-static-network max error: {np.max(np.abs(quasistatic_difference)):.12g}")
    print(f"quasi-static-network RMS error: "
          f"{np.sqrt(np.mean(quasistatic_difference * quasistatic_difference)):.12g}")
    print(f"native quasi-static-network max error: "
          f"{np.max(np.abs(native_quasistatic_difference)):.12g}")
    print(f"native quasi-static-network RMS error: "
          f"{np.sqrt(np.mean(native_quasistatic_difference * native_quasistatic_difference)):.12g}")
    stamp_qs_modes = np.linalg.eigvals(a_stamp_quasistatic)
    veragrid_modes = np.linalg.eigvals(a_vg_ordered)
    print(f"STAMP quasi-static rightmost mode: "
          f"{stamp_qs_modes[np.argmax(stamp_qs_modes.real)]:.12g}")
    print(f"unstable modes: VeraGrid={np.count_nonzero(veragrid_modes.real > 1e-8)}, "
          f"STAMP quasi-static={np.count_nonzero(stamp_qs_modes.real > 1e-8)}")
    print("largest coefficient differences:")
    for flat in np.argsort(np.abs(difference), axis=None)[-20:][::-1]:
        row, column = np.unravel_index(flat, difference.shape)
        print(f"  d({device_names[row]})/d({device_names[column]}): "
              f"VG={a_vg_ordered[row, column]:+.9g}, "
              f"STAMP={a_stamp_devices[row, column]:+.9g}, "
              f"error={difference[row, column]:+.9g}")

    for label, section in (("SG", slice(0, 17)), ("GFOR", slice(17, 38)), ("GFOL", slice(38, 58))):
        err = difference[section, section]
        print(f"{label}: max={np.max(np.abs(err)):.9g}, RMS={np.sqrt(np.mean(err*err)):.9g}")

    suffix = "_impedance_loads" if args.impedance_loads else ""
    output = ROOT / f"STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_jacobian_difference{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, difference, delimiter=",")
    np.savetxt(output.with_name(f"WSCC_SG_GFOR_GFOL_veragrid_reduced_A{suffix}.csv"),
               a_vg_ordered, delimiter=",")
    np.savetxt(output.with_name("WSCC_SG_GFOR_GFOL_STAMP_quasistatic_A.csv"),
               a_stamp_quasistatic, delimiter=",")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
