#!/usr/bin/env python3
"""Run an unperturbed WSCC RMS simulation and plot device-state drift."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.parameters import STAMP_GFOL
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-time", type=float, default=2.0)
    parser.add_argument("--time-step", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("STAMP/02_results/veragrid/rms_initialization"))
    parser.add_argument("--pulse", action="store_true", help="Apply a tiny GFOL active-power reference pulse")
    parser.add_argument("--pulse-size", type=float, default=1e-7)
    parser.add_argument("--pulse-time", type=float, default=0.01)
    parser.add_argument("--pulse-duration", type=float, default=1e-4)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from VeraGridEngine.Devices.Events.rms_event import RmsEvent
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions

    grid = build_stamp_wscc_grid()
    group = RmsEventsGroup(name="tiny_pulse" if args.pulse else "unperturbed_initialization", active=True)
    grid.add_rms_events_group(group)
    if args.pulse:
        from VeraGridEngine.Utils.Symbolic.block import build_name_to_var_lookup
        gfol = next(device for device in grid.static_generators if device.name == "STAMP GFOL2")
        pref = build_name_to_var_lookup(gfol.rms_model).get("STAMP_GFOL2.P_ref")
        if pref is None:
            raise RuntimeError("Could not locate GFOL P_ref event parameter")
        base = STAMP_GFOL.p_pu_system
        grid.add_rms_event(RmsEvent(device=gfol, parameter=pref, time=args.pulse_time,
                                    value=base + args.pulse_size, group=group, force_step_alignment=True))
        grid.add_rms_event(RmsEvent(device=gfol, parameter=pref,
                                    time=args.pulse_time + args.pulse_duration,
                                    value=base, group=group, force_step_alignment=True))
    pf = PowerFlowDriver(grid, power_flow_options())
    pf.run()
    if not np.all(np.asarray(pf.results.converged)):
        raise RuntimeError("Power flow did not converge")

    driver = RmsSimulationDriver(
        grid=grid,
        options=RmsOptions(time_step=args.time_step, simulation_time=args.simulation_time,
                           tolerance=1e-9, max_iter=100, verbose=0),
        pf_results=pf.results,
    )
    driver.run()
    results = driver.results
    if results is None:
        raise RuntimeError("RMS driver returned no results")

    values = np.asarray(results.values[:, :, 0], dtype=float)
    if not bool(results.converged[0]):
        empty_rows = np.flatnonzero(np.all(values == 0.0, axis=1))
        empty_rows = empty_rows[empty_rows > 0]
        if empty_rows.size:
            values = values[:empty_rows[0]]
    time = np.arange(values.shape[0], dtype=float) * args.time_step
    state_uids = {var.uid for var in driver.problem.state_vars}
    selected = [i for i, var in enumerate(results.variables) if var.uid in state_uids]
    names = [str(results.variables[i]) for i in selected]
    state_values = values[:, selected]
    deviations = state_values - state_values[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "WSCC_tiny_pulse" if args.pulse else "WSCC_unperturbed"
    csv_path = args.output_dir / f"{stem}_{args.simulation_time:g}s_state_trajectories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *names])
        writer.writerows(np.column_stack((time, state_values)))

    groups = (("Synchronous generator", "STAMP_SG1"),
              ("Grid-forming converter", "STAMP_GFOR1"),
              ("Grid-following converter", "STAMP_GFOL2"))
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True, constrained_layout=True)
    for axis, (title, marker) in zip(axes, groups):
        indices = [i for i, name in enumerate(names) if marker in name]
        for index in indices:
            axis.plot(time, deviations[:, index], linewidth=0.9, label=names[index].split(".")[-1])
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title(f"{title}: state deviation from t=0")
        axis.set_ylabel("x(t) - x(0) [pu/internal]")
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=4, fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time [s]")
    figure_path = args.output_dir / f"{stem}_{args.simulation_time:g}s_state_deviations.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    finite = np.all(np.isfinite(state_values))
    max_drift = np.nanmax(np.abs(deviations), axis=0)
    print(f"well initialized: {bool(results.well_initialized[0])}")
    print(f"converged: {bool(results.converged[0])}")
    print(f"all samples finite: {finite}")
    print(f"last retained time: {time[-1]:.9g} s")
    for title, marker in groups:
        indices = [i for i, name in enumerate(names) if marker in name]
        worst = max(indices, key=lambda i: max_drift[i])
        print(f"{title}: max drift={max_drift[worst]:.9g} ({names[worst]})")
    print(f"wrote trajectories: {csv_path.resolve()}")
    print(f"wrote plot: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
