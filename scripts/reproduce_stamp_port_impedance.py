#!/usr/bin/env python3
"""Reproduce STAMP converter admittance and grid impedance with VeraGrid RMS."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VeraGridEngine.Simulations.SmallSignalStabilityRms.rms_port_analysis import (
    compute_device_admittance as compute_model_admittance,
    compute_grid_impedance_at_device_port as compute_grid_impedance_at_bus,
    prepare_port,
)
from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def operating_problem(*, nonlinear_converters: bool = False):
    """Build the STAMP WSCC case and its initialized VeraGrid RMS driver."""
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions

    grid = build_stamp_wscc_grid(dynamic_lines=True, full_dynamic_network=True,
                                 nonlinear_converters=nonlinear_converters)
    power_flow_driver = PowerFlowDriver(grid, power_flow_options())
    power_flow_driver.run()
    return SmallSignalStabilityRmsDriver(
        grid,
        rms_options=RmsOptions(time_step=.001),
        sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0),
        pf_results=power_flow_driver.results,
    )


def responses(frequencies_hz: np.ndarray, bus: int = 4,
              *, nonlinear_converters: bool = False):
    """Compute both sides of the selected STAMP converter's VeraGrid port."""
    from VeraGridEngine.enumerations import DeviceType

    driver = operating_problem(nonlinear_converters=nonlinear_converters)
    selected_bus = next((item for item in driver.grid.buses
                         if item.name == f"Bus{bus}"), None)
    if selected_bus is None:
        raise ValueError(f"Bus {bus} is not in the VeraGrid circuit")
    grouped_devices = driver.grid.get_injection_devices_grouped_by_bus()
    generators = grouped_devices.get(selected_bus, {}).get(
        DeviceType.GeneratorDevice,
        [],
    )
    if len(generators) != 1:
        raise ValueError(
            f"Expected one generator at {selected_bus.name}; found {len(generators)}"
        )

    response = driver.compute_device_port_response(
        device=generators[0],
        frequencies_hz=frequencies_hz,
    )
    return (
        response.device_admittance,
        response.grid_impedance,
        response.device.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus", type=int, choices=(4, 6), default=4)
    parser.add_argument("--nonlinear-converters", action="store_true")
    parser.add_argument("--frequencies", type=float, nargs="+",
                        default=[1, 10, 100, 1000])
    args = parser.parse_args()
    frequencies_hz = np.asarray(args.frequencies)
    converter_admittance, grid_impedance, device_name = responses(
        frequencies_hz,
        args.bus,
        nonlinear_converters=args.nonlinear_converters,
    )
    for frequency_hz, admittance, impedance in zip(
        frequencies_hz,
        converter_admittance,
        grid_impedance,
    ):
        print(f"{device_name} at bus {args.bus}, {frequency_hz:g} Hz")
        print("  converter Y_qd (injected current / voltage):",
              np.array2string(admittance, precision=6))
        print("  grid Z_qd (voltage / injected current):",
              np.array2string(impedance, precision=6))


if __name__ == "__main__":
    main()
