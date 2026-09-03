#!/usr/bin/env python3
"""Run the STAMP WSCC abc EMT initialization check and Floquet SSA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid


def _print_continuous_residuals(label, solver, problem, x, dx, h, count=15):
    """Evaluate dx-f(x,y)=0 and g(x,y)=0 at one accepted point."""
    from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import (
        _build_structural_residual_debug_info,
    )

    n_state = problem.get_states_number()
    history = np.asarray(x).copy()
    # For trapezoidal discretization, history=x-h*dx and d_history=dx makes
    # the compiled derivative term exactly dx.  This lets the normal residual
    # kernel evaluate the continuous DAE rather than a step defect.
    history[:n_state] -= h*np.asarray(dx)
    residual = np.empty_like(np.asarray(x))
    if hasattr(solver, "_residual_assembler"):
        solver._residual_assembler.evaluate(
            np.asarray(x), solver._full_parameter_buffer, history,
            np.asarray(dx), h, history, residual)
    else:
        from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import (
            build_residual_evaluator,
            evaluate_vectorized_residual,
        )
        evaluator = build_residual_evaluator(solver.fused_residual,
                                             solver.vec_flat_args)
        evaluate_vectorized_residual(
            evaluator, np.asarray(x), solver._full_parameter_buffer, history,
            np.asarray(dx), h, history, residual)
    debug = _build_structural_residual_debug_info(
        problem.get_state_vars(), problem.get_algebraic_vars(),
        problem.get_state_eqs(), problem.get_algebraic_eqs(),
        problem.uid2idx_vars)
    order = np.argsort(np.abs(residual))[::-1]
    print(f"{label}: max continuous DAE residual={abs(residual[order[0]]):.12e}")
    for rank, index in enumerate(order[:min(count, len(order))], start=1):
        info = debug[int(index)]
        print(f"  {rank:02d} {info['kind']:<5} {residual[index]:+.12e}  {info['label']}")
        if int(index) < n_state:
            equation = problem.get_state_eqs()[int(index)]
        else:
            equation = problem.get_algebraic_eqs()[int(index)-n_state]
        equation_text = str(equation).replace("\n", " ")
        print(f"       equation: {equation_text[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-step", type=float, default=20e-6)
    parser.add_argument("--simulation-time", type=float, default=0.04)
    parser.add_argument("--assessment-time", type=float, default=0.06)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--skip-ssa", action="store_true")
    args = parser.parse_args()

    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
    from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes

    grid = build_stamp_wscc_emt_grid()
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid=grid, options=pf_options); pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=pf_options); pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError(f"Power flow failed: balanced={pf.results.converged}, abc={pf3.results.converged}")
    options = EmtOptions(time_step=args.time_step, simulation_time=args.simulation_time,
        tolerance=1e-7, solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit, verbose=0)
    problem = EmtProblemDae(grid=grid, options=options, pf_results=pf.results,
                            pf_results_3ph=pf3.results)
    report = problem.initialization_report
    if report is not None:
        print("EMT initialization: "
              f"method={report.method_used}, status={report.status.name}, "
              f"res0={report.initial_residual_inf:.9e}, "
              f"resf={report.final_residual_inf:.9e}, "
              f"newton={report.newton_iterations}, "
              f"ptc={report.pseudo_transient_steps}, "
              f"auto_dx0={report.automatic_dx0_count}")
    for parameter, value in zip(problem.get_variable_parameters(),
                                problem.event_params_values):
        if parameter.name.startswith(("Vnom_", "R_", "L_", "C_")):
            print(f"EMT runtime parameter {parameter.name}={value:.12e}")
    solver = build_emt_solver(options=options, problem=problem, t0=0.0,
        t_end=args.simulation_time, h=args.time_step, method=options.integration_method)
    time, values, derivatives, initialized, converged = solver.simulate(boundary_updater=cast(Any, problem))
    _print_continuous_residuals("t=0 explicit seed", solver, problem,
                                values[0], derivatives[0], args.time_step)
    if len(time) > 1:
        _print_continuous_residuals("first accepted endpoint", solver, problem,
                                    values[1], derivatives[1], args.time_step)
    finite = bool(np.isfinite(values).all())
    print(f"EMT check: initialized={initialized}, converged={converged}, finite={finite}, states={problem.get_states_number()}, steps={len(time)-1}")
    if not initialized or not converged or not finite:
        raise RuntimeError("Unperturbed EMT initialization check failed")

    # Reconstruct each initialized load's three-phase power on the same
    # peak-q/d convention used by the EMT device wrappers.  This exposes
    # current-base or sign errors directly at the DAE boundary.
    theta0 = values[0, problem.get_var_idx(next(
        variable for variable in grid.generators[0].emt_model.state_vars
        if "theta_grid" in variable.name))]
    shift = 2.0*np.pi/3.0
    for load in grid.loads:
        bus = load.bus
        bus_v = [values[0, problem.get_var_idx(variable)]
                 for variable in bus.emt_model.out_vars[:3]]
        load_i = [values[0, problem.get_var_idx(variable)]
                  for variable in load.emt_model.out_vars[:3]]
        vq = (2.0/3.0)*sum(np.sin(theta0+angle)*value
                            for angle, value in zip((0.0, -shift, shift), bus_v))
        vd = -(2.0/3.0)*sum(np.cos(theta0+angle)*value
                             for angle, value in zip((0.0, -shift, shift), bus_v))
        iq = (2.0/3.0)*sum(np.sin(theta0+angle)*value
                            for angle, value in zip((0.0, -shift, shift), load_i))
        id_ = -(2.0/3.0)*sum(np.cos(theta0+angle)*value
                              for angle, value in zip((0.0, -shift, shift), load_i))
        # Currents are three times the per-phase-base dq value in VeraGrid's
        # conventional three-phase convention, hence the 1/2 coefficient.
        p_inj = 0.5*(vq*iq+vd*id_)
        q_inj = 0.5*(vq*id_-vd*iq)
        print(f"initialized load injection {load.name}@{bus.name}: "
              f"P={p_inj:+.9e}, Q={q_inj:+.9e}; "
              f"target P={-load.P/grid.Sbase:+.9e}, Q={-load.Q/grid.Sbase:+.9e}")

    # A balanced steady EMT solution is periodic.  Compare the endpoints one
    # fundamental cycle apart instead of incorrectly requiring xdot == 0.
    cycle_steps = int(round((1.0/grid.fBase)/args.time_step))
    if len(values) > cycle_steps:
        cycle_delta = values[-1]-values[-1-cycle_steps]
        state_and_alg = problem.get_state_vars()+problem.get_algebraic_vars()
        # Unwrapped electrical/rotor angles advance by 2*pi per cycle and are
        # periodic modulo 2*pi.  Compare them on the circle.
        for idx, variable in enumerate(state_and_alg):
            if "theta_grid" in variable.name or "theta_abs" in variable.name:
                cycle_delta[idx] = (cycle_delta[idx]+np.pi)%(2.0*np.pi)-np.pi
        worst = int(np.argmax(np.abs(cycle_delta)))
        periodic_error = float(abs(cycle_delta[worst]))
        print(f"one-cycle endpoint max error={periodic_error:.9e} at {state_and_alg[worst].name}")
        state_delta = cycle_delta[:problem.get_states_number()]
        worst_state = int(np.argmax(np.abs(state_delta)))
        print(f"one-cycle state max error={abs(state_delta[worst_state]):.9e} at {problem.get_state_vars()[worst_state].name}")

    # Endpoint periodicity can hide an oscillation whose envelope grows inside
    # each fundamental cycle. Track GFOL2's PLL/angle state by cycle using both
    # peak deviation and endpoint deviation from the explicit initial value.
    gfol2 = next(generator for generator in grid.generators
                 if generator.name == "STAMP GFOL2")
    gfol2_angle = next(
        variable for variable in gfol2.emt_model.get_all_vars()
        if ".etheta_x" in variable.name and variable.uid in problem.uid2idx_vars
    )
    angle_trace = values[:, problem.get_var_idx(gfol2_angle)]
    complete_cycles = (len(angle_trace)-1)//cycle_steps
    for cycle in range(complete_cycles):
        start = cycle*cycle_steps
        stop = (cycle+1)*cycle_steps
        segment = angle_trace[start:stop+1]-angle_trace[0]
        print(f"GFOL2 angle cycle {cycle+1}: "
              f"peak_deviation={np.max(np.abs(segment)):.9e}, "
              f"endpoint_deviation={segment[-1]:+.9e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for generator in grid.generators:
        variables = generator.emt_model.get_all_vars()
        for axis, prefix in zip(axes, ("omega_", ".etheta_x", ".ig_q")):
            var = next((v for v in variables if prefix in v.name and v.uid in problem.uid2idx_vars), None)
            if var is not None:
                axis.plot(1e3*time, values[:, problem.get_var_idx(var)], label=generator.name)
    axes[0].set_ylabel("speed (p.u.)"); axes[1].set_ylabel("angle state"); axes[2].set_ylabel("grid i_q dev.")
    axes[2].set_xlabel("time (ms)")
    for axis in axes:
        axis.grid(True, alpha=.3)
        if axis.lines: axis.legend()
    plot_path = Path(__file__).with_name("stamp_wscc_emt_steady_state.png")
    fig.savefig(plot_path, dpi=180); plt.close(fig)
    print(f"plot={plot_path}")

    # Initialization audit: show the first Newton/integration endpoints as
    # deviations from the exact explicit initial condition.  Iteration index is
    # more useful than milliseconds here because it makes a first-step jump
    # immediately visible.
    first_count = min(31, len(time))
    first_steps = np.arange(first_count)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             constrained_layout=True)
    sg_theta_grid = next(
        variable for variable in grid.generators[0].emt_model.state_vars
        if "theta_grid" in variable.name
    )
    theta_trace = values[:first_count, problem.get_var_idx(sg_theta_grid)]
    phase_shift = 2.0*np.pi/3.0
    for bus in grid.buses:
        bus_indices = [problem.get_var_idx(variable) for variable in bus.emt_model.out_vars[:3]]
        va_trace, vb_trace, vc_trace = [
            values[:first_count, index] for index in bus_indices
        ]
        vq_trace = (2.0/3.0)*(
            np.sin(theta_trace)*va_trace
            + np.sin(theta_trace-phase_shift)*vb_trace
            + np.sin(theta_trace+phase_shift)*vc_trace
        )
        vd_trace = -(2.0/3.0)*(
            np.cos(theta_trace)*va_trace
            + np.cos(theta_trace-phase_shift)*vb_trace
            + np.cos(theta_trace+phase_shift)*vc_trace
        )
        axes[0].plot(first_steps, vq_trace-vq_trace[0], marker=".",
                     linewidth=.9, label=f"{bus.name} vq")
        axes[0].plot(first_steps, vd_trace-vd_trace[0], marker=".",
                     linewidth=.9, linestyle="--", label=f"{bus.name} vd")
        if len(vq_trace) > 1:
            print(f"first-step rotating derivative {bus.name}: "
                  f"dvq={(vq_trace[1]-vq_trace[0])/args.time_step:+.9e}, "
                  f"dvd={(vd_trace[1]-vd_trace[0])/args.time_step:+.9e}")
    axes[0].set_ylabel("bus dq voltage - value at x0")

    device_markers = (
        ("STAMP SG1", (".ig_q", ".ig_d", ".is_q", ".is_d")),
        ("STAMP GFOR1", (".ig_q", ".ig_d", ".is_q", ".is_d", ".ucap_q", ".ucap_d")),
        ("STAMP GFOL2", (".ig_q", ".ig_d", ".is_q", ".is_d", ".ucap_q", ".ucap_d")),
    )
    for axis, (device_name, suffixes) in zip(axes[1:], device_markers):
        device = next(generator for generator in grid.generators
                      if generator.name == device_name)
        for variable in device.emt_model.get_all_vars():
            if (variable.uid in problem.uid2idx_vars
                    and any(variable.name.endswith(suffix) for suffix in suffixes)):
                trace = values[:first_count, problem.get_var_idx(variable)]
                axis.plot(first_steps, trace-trace[0], marker=".", linewidth=.9,
                          label=variable.name.split(".")[-1])
        axis.set_ylabel(f"{device_name}\nstate - x0")
    axes[-1].set_xlabel(f"integration step (h={args.time_step:g} s)")
    for axis in axes:
        axis.grid(True, alpha=.3)
        if axis.lines:
            axis.legend(ncol=4, fontsize=7)
    axes[0].set_title("STAMP WSCC EMT — first integration steps")
    first_plot_path = Path(__file__).with_name("stamp_wscc_emt_first_iterations.png")
    fig.savefig(first_plot_path, dpi=180); plt.close(fig)
    print(f"first-iterations-plot={first_plot_path}")

    if not args.skip_ssa:
        sss_options = SmallSignalStabilityEmtOptions(k=min(args.modes, problem.get_states_number()-2),
            target_period=1.0/grid.fBase, ss_assessment_time=args.assessment_time, verbose=1)
        driver = SmallSignalStabilityEmtDriver(grid=grid, emt_options=options,
                                               sss_options=sss_options, pf_results=pf3.results)
        driver.run()
        exponents = np.asarray(driver.results.eigenvalues)
        order = np.argsort(exponents.real)[::-1]
        print("Floquet exponents (rightmost first):")
        for value in exponents[order]:
            print(f"{value.real:+.12e} {value.imag:+.12e}j")
        for mode in order[:min(3, len(order))]:
            participants = np.argsort(driver.results.participation_factors[:, mode])[-5:][::-1]
            labels = [f"{driver.results.stat_vars_array[idx]}={driver.results.participation_factors[idx, mode]:.3g}"
                      for idx in participants]
            print(f"mode {exponents[mode].real:+.6g}{exponents[mode].imag:+.6g}j participants: " + ", ".join(labels))


if __name__ == "__main__":
    main()
