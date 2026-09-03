#!/usr/bin/env python3
"""Benchmark VeraGrid dense/sparse SSA scaling using replicated WSCC systems."""

from pathlib import Path
from time import perf_counter
import csv
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_veragrid_wscc_ssa import build_initialized_problem


class ReplicatedDescriptorProblem:
    """Minimal problem interface consumed by VeraGrid's SSA routines."""

    def __init__(self, static_matrix, descriptor_matrix, dynamic_states):
        self.static_matrix = static_matrix
        self.descriptor_matrix = descriptor_matrix
        self.dynamic_states = dynamic_states

    def get_static_state_matrix(self, _x, _dx):
        return self.static_matrix

    def get_E_matrix(self, _x, _dx):
        return self.descriptor_matrix

    def get_diff_var_number(self):
        return self.dynamic_states

    def get_states_number(self):
        return self.dynamic_states


def main() -> None:
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import (
        run_dense_small_signal_stability,
        run_sparse_small_signal_stability,
    )

    base_problem, vector = build_initialized_problem()
    dx = np.zeros(base_problem.get_diff_var_number())
    static_base = sp.csc_matrix(base_problem.get_static_state_matrix(vector, dx))
    descriptor_base = sp.csc_matrix(base_problem.get_E_matrix(vector, dx))
    base_states = base_problem.get_states_number()
    sparse_k = 10
    configurations = ((1, 10), (2, 8), (4, 5), (8, 3))
    rows = []

    for replicas, repetitions in configurations:
        static = sp.block_diag([static_base] * replicas, format="csc")
        descriptor = sp.block_diag([descriptor_base] * replicas, format="csc")
        problem = ReplicatedDescriptorProblem(
            static, descriptor, dynamic_states=base_states * replicas)
        placeholder = np.zeros(static.shape[0])

        dense_reference = run_dense_small_signal_stability(
            problem, placeholder, placeholder)[0]
        sparse_reference = run_sparse_small_signal_stability(
            problem, placeholder, placeholder, k=sparse_k)[0]
        match_error = max(np.min(np.abs(dense_reference - mode))
                          for mode in sparse_reference)

        for run_index in range(1, repetitions + 1):
            start = perf_counter()
            run_dense_small_signal_stability(problem, placeholder, placeholder)
            dense_seconds = perf_counter() - start

            start = perf_counter()
            run_sparse_small_signal_stability(
                problem, placeholder, placeholder, k=sparse_k)
            sparse_seconds = perf_counter() - start
            rows.append((replicas, base_states * replicas, static.shape[0],
                         sparse_k, run_index, dense_seconds, sparse_seconds,
                         match_error))

        print(f"states={base_states * replicas}, descriptor={static.shape[0]}, "
              f"dense={np.median([r[5] for r in rows if r[0] == replicas]):.6f}s, "
              f"sparse={np.median([r[6] for r in rows if r[0] == replicas]):.6f}s, "
              f"match={match_error:.3g}")

    output = ROOT / "STAMP/02_results/comparison/benchmark_veragrid_scaling_multivac.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wscc_replicas", "dynamic_states", "descriptor_dimension",
                         "sparse_k", "run", "dense_seconds", "sparse_seconds",
                         "mode_match_max_error"))
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
