#!/usr/bin/env python3
"""Benchmark VeraGrid's matched 88-state WSCC SSA on a compute node."""

from pathlib import Path
from time import perf_counter
import csv
import os
import sys

import numpy as np
import scipy.linalg as la
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid, STAMP_LOADS


def build_initialized_problem():
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions

    grid = build_stamp_wscc_grid(dynamic_lines=True, full_dynamic_network=True)
    power_flow = PowerFlowDriver(grid, power_flow_options())
    power_flow.run()
    problem = RmsProblemDae(grid, RmsOptions(time_step=0.001), power_flow.results)
    problem.set_events_group(RmsEventsGroup("full_dynamic_network_benchmark"))
    vector = problem.get_x0()
    nx = len(problem.state_vars)
    state_index = {str(variable): index for index, variable in enumerate(problem.state_vars)}
    buses = {int(bus.name.removeprefix("Bus")): index for index, bus in enumerate(grid.buses)}

    for branch, line in enumerate(grid.lines):
        f = grid.buses.index(line.bus_from)
        voltage = power_flow.results.voltage[f]
        vq, vd = voltage.real, -voltage.imag
        power = power_flow.results.Sf[branch] / grid.Sbase
        denominator = vq * vq + vd * vd
        total_iq = (power.real * vq - power.imag * vd) / denominator
        total_id = (power.real * vd + power.imag * vq) / denominator
        bus_f = int(line.bus_from.name.removeprefix("Bus"))
        bus_t = int(line.bus_to.name.removeprefix("Bus"))
        name = f"NET.{min(bus_f, bus_t)}{max(bus_f, bus_t)}"
        vector[state_index[f"{name}.iq"]] = total_iq - line.B * vd / 2.0
        vector[state_index[f"{name}.id"]] = total_id + line.B * vq / 2.0

    for load_number, (bus_number, p_mw, q_mvar) in enumerate(STAMP_LOADS, 1):
        voltage = power_flow.results.voltage[buses[bus_number]]
        vq, vd = voltage.real, -voltage.imag
        p, q = p_mw / grid.Sbase, q_mvar / grid.Sbase
        denominator = vq * vq + vd * vd
        conductance = p / denominator
        vector[state_index[f"Load{load_number}.ilq"]] = ((p * vq - q * vd) / denominator
                                                          - conductance * vq)
        vector[state_index[f"Load{load_number}.ild"]] = ((p * vd + q * vq) / denominator
                                                          - conductance * vd)
    for bus_number, bus_index in buses.items():
        voltage = power_flow.results.voltage[bus_index]
        vector[state_index[f"STAMP bus capacitor {bus_number}.vc_q"]] = voltage.real
        vector[state_index[f"STAMP bus capacitor {bus_number}.vc_d"]] = -voltage.imag

    solved = root(
        lambda y: problem.rhs_algebraic(
            np.r_[vector[:nx], y], np.zeros_like(vector)),
        vector[nx:].copy(),
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    vector[nx:] = solved.x
    return problem, vector


def matrices(problem, vector):
    dx = np.zeros(problem.get_diff_var_number())
    step = problem.get_dt_value()
    fx = problem.get_j11(vector, dx, step).toarray()
    fy = problem.get_j12(vector, dx, step).toarray()
    gx = problem.get_j21(vector, dx, step).toarray()
    gy = problem.get_j22(vector, dx, step).toarray()
    reduced = fx - fy @ np.linalg.solve(gy, gx)
    return reduced


def main() -> None:
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import (
        run_dense_small_signal_stability,
        run_sparse_small_signal_stability,
    )

    number_runs = int(os.environ.get("BENCHMARK_RUNS", "20"))
    sparse_k = int(os.environ.get("SPARSE_K", "10"))
    setup_start = perf_counter()
    problem, vector = build_initialized_problem()
    setup_seconds = perf_counter() - setup_start
    dx = np.zeros_like(vector)
    reduced = matrices(problem, vector)

    la.eigvals(reduced)
    dense_reference = run_dense_small_signal_stability(problem, vector, dx)[0]
    sparse_reference = run_sparse_small_signal_stability(
        problem, vector, dx, k=sparse_k)[0]
    eig_times = []
    dense_ssa_times = []
    sparse_ssa_times = []
    for _ in range(number_runs):
        start = perf_counter()
        la.eigvals(reduced)
        eig_times.append(perf_counter() - start)

        start = perf_counter()
        run_dense_small_signal_stability(problem, vector, dx)
        dense_ssa_times.append(perf_counter() - start)

        start = perf_counter()
        run_sparse_small_signal_stability(
            problem, vector, dx, k=sparse_k)
        sparse_ssa_times.append(perf_counter() - start)

    output = ROOT / "STAMP/02_results/comparison/benchmark_veragrid_multivac.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("implementation", "sparse_k", "run", "eig_seconds",
                         "dense_ssa_seconds", "sparse_ssa_seconds"))
        for index, values in enumerate(
                zip(eig_times, dense_ssa_times, sparse_ssa_times), 1):
            eig_time, dense_time, sparse_time = values
            writer.writerow(("VeraGrid", sparse_k, index, f"{eig_time:.12g}",
                             f"{dense_time:.12g}", f"{sparse_time:.12g}"))

    print(f"VeraGrid states: {problem.get_states_number()}")
    print(f"VeraGrid setup: {setup_seconds:.9f} s")
    print(f"VeraGrid eig median: {np.median(eig_times):.9f} s (min {np.min(eig_times):.9f} s)")
    print(f"VeraGrid dense SSA median: {np.median(dense_ssa_times):.9f} s "
          f"(min {np.min(dense_ssa_times):.9f} s)")
    nearest_errors = [np.min(np.abs(dense_reference - mode))
                      for mode in sparse_reference]
    print(f"VeraGrid sparse SSA k={sparse_k} median: "
          f"{np.median(sparse_ssa_times):.9f} s "
          f"(min {np.min(sparse_ssa_times):.9f} s)")
    print(f"Sparse/dense mode-match max error: {np.max(nearest_errors):.12g}")
    print(output)


if __name__ == "__main__":
    main()
