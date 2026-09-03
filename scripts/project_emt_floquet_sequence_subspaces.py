#!/usr/bin/env python3
"""Project the EMT monodromy matrix onto balanced sequence state subspaces."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid


def _build_problem():
    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.enumerations import EmtInitializationMethod

    grid = build_stamp_wscc_emt_grid()
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid, pf_options); pf.run()
    pf3 = PowerFlowDriver3Ph(grid, pf_options); pf3.run()
    return EmtProblemDae(
        grid=grid, options=EmtOptions(initialization_method=EmtInitializationMethod.Explicit),
        pf_results=pf.results, pf_results_3ph=pf3.results)


def _sequence_basis(problem, orientation: int):
    """Return an orthonormal complex basis with each NABC-free triple reduced to one sequence."""
    states = problem.get_state_vars()
    columns = []
    used = set()
    a = np.exp(2j*np.pi/3.0)
    sequence = np.asarray([1.0, a**(-orientation), a**orientation], dtype=np.complex128)/np.sqrt(3.0)

    index = 0
    while index < len(states):
        if (index+2 < len(states)
                and states[index].name.endswith("_A")
                and states[index+1].name == states[index].name[:-1]+"B"
                and states[index+2].name == states[index].name[:-1]+"C"):
            column = np.zeros(len(states), dtype=np.complex128)
            column[index:index+3] = sequence
            columns.append(column)
            used.update((index, index+1, index+2))
            index += 3
        else:
            index += 1

    for index in range(len(states)):
        if index not in used:
            column = np.zeros(len(states), dtype=np.complex128)
            column[index] = 1.0
            columns.append(column)
    return np.column_stack(columns)


def main():
    problem = _build_problem()
    monodromy = np.load("scripts/stamp_wscc_emt_monodromy.npy")
    period = 1.0/50.0
    print(f"full EMT states={monodromy.shape[0]}")

    for orientation in (+1, -1):
        basis = _sequence_basis(problem, orientation)
        reduced = basis.conj().T @ monodromy @ basis
        leakage = np.linalg.norm((np.eye(monodromy.shape[0])-basis@basis.conj().T)
                                 @ monodromy @ basis, ord=2)
        multipliers = linalg.eigvals(reduced)
        exponents = np.log(multipliers.astype(np.complex128))/period
        order = np.argsort(exponents.real)[::-1]
        print(f"orientation={orientation:+d}: dimension={basis.shape[1]}, leakage_2={leakage:.6e}")
        for index in order[:12]:
            print(f"  lambda={exponents[index].real:+.9e} "
                  f"{exponents[index].imag:+.9e}j, mu_abs={abs(multipliers[index]):.9e}")


if __name__ == "__main__":
    main()
