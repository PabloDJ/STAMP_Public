#!/usr/bin/env python3
"""Compare STAMP's saved bus-4 GFOR admittance with VeraGrid's RMS port."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_stamp_port_impedance import (
    compute_model_admittance,
    operating_problem,
    prepare_port,
)


DEFAULT_REFERENCE = (
    ROOT / "STAMP" / "02_results" / "comparison" /
    "example_data2_gfor_admittance.csv"
)


def read_stamp_reference(path: Path, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Read one factor's raw, generator-oriented qd admittance."""
    values: dict[float, np.ndarray] = {}
    seen: set[tuple[float, int, int]] = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not np.isclose(float(row["factor"]), factor, rtol=0, atol=1e-10):
                continue
            frequency = float(row["frequency_hz"])
            output_index = int(row["output_index"]) - 1
            input_index = int(row["input_index"]) - 1
            if not (0 <= output_index < 2 and 0 <= input_index < 2):
                raise ValueError(f"Invalid qd matrix entry: {row}")
            key = (frequency, output_index, input_index)
            if key in seen:
                raise ValueError(f"Duplicate STAMP response entry: {key}")
            seen.add(key)
            matrix = values.setdefault(frequency, np.full((2, 2), np.nan + 1j*np.nan))
            matrix[output_index, input_index] = (
                float(row["real"]) + 1j * float(row["imag"])
            )
    if not values:
        raise ValueError(f"Factor {factor:g} is absent from {path}")
    frequencies = np.array(sorted(values))
    response = np.array([values[frequency] for frequency in frequencies])
    if not np.all(np.isfinite(response)):
        raise ValueError("STAMP reference has missing or non-finite matrix entries")
    return frequencies, response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--factor", type=float, default=0.05,
                        help="Saved STAMP factor to compare; 0.05 is the first case")
    parser.add_argument("--linear-converter", action="store_true",
                        help="Use VeraGrid's source-linear converter instead of its nonlinear RMS model")
    args = parser.parse_args()

    if not args.reference.is_file():
        parser.error(
            f"STAMP reference CSV is missing: {args.reference}. "
            "Run export_example_impedance_reference in a licensed MATLAB session first."
        )
    frequencies, stamp_admittance = read_stamp_reference(args.reference, args.factor)
    driver = operating_problem(nonlinear_converters=not args.linear_converter)
    bus = next(bus for bus in driver.grid.buses if bus.name == "Bus4")
    devices = [device for device in driver.grid.generators if device.bus is bus]
    if len(devices) != 1:
        raise ValueError(f"Expected one generator at {bus.name}; found {len(devices)}")
    port = prepare_port(driver, devices[0])
    veragrid_admittance = compute_model_admittance(port, frequencies)

    difference = veragrid_admittance - stamp_admittance
    relative_error = np.linalg.norm(difference) / np.linalg.norm(stamp_admittance)
    print(f"STAMP example_data2 factor: {args.factor:g}")
    print(f"VeraGrid device: {port.device_name}")
    print("Both matrices use qd voltage input and generator-oriented injected current output.")
    print("Check that this saved STAMP factor represents the same converter parameters.")
    print(f"Relative Frobenius error across all frequencies: {relative_error:.6g}")
    for frequency, stamp, veragrid in zip(
        frequencies, stamp_admittance, veragrid_admittance
    ):
        print(f"\n{frequency:g} Hz")
        print("STAMP:\n", np.array2string(stamp, precision=6))
        print("VeraGrid:\n", np.array2string(veragrid, precision=6))
        print(f"Maximum entry difference: {np.max(np.abs(veragrid-stamp)):.6g}")


if __name__ == "__main__":
    main()
