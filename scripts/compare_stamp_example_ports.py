#!/usr/bin/env python3
"""Compare both sides of VeraGrid's bus-4 port with STAMP's saved example."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_stamp_port_impedance import responses


DEFAULT_REFERENCE_ANGLE = (
    ROOT / "STAMP" / "02_results" / "multivac" /
    "WSCC_SG_GFOR_GFOL_sg_linearization_point.csv"
)


def stamp_frame_rotation(path: Path) -> float:
    """Rotation from VeraGrid's slack-voltage frame to STAMP's rotor frame."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        values = {row["field"]: float(row["value"]) for row in csv.DictReader(stream)}
    return float(np.arctan2(values["vd_bus0"], values["vq_bus0"]))


def rotate_qd(response: np.ndarray, angle: float) -> np.ndarray:
    """Express every qd input/output response matrix in a rotated frame."""
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return np.asarray([rotation @ matrix @ rotation.T for matrix in response])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        type=Path,
        nargs="?",
        default=Path("STAMP/02_results/comparison/stamp_example_ports.mat"),
    )
    parser.add_argument("--nonlinear-converters", action="store_true")
    parser.add_argument("--reference-angle-data", type=Path,
                        default=DEFAULT_REFERENCE_ANGLE)
    args = parser.parse_args()

    reference = loadmat(args.reference)
    frequencies_hz = reference["frequencies_hz"].ravel()
    factors = reference["gfor_factors"].ravel()
    matching_factor = np.flatnonzero(np.isclose(factors, 0.05))
    if len(matching_factor) != 1:
        raise ValueError("Expected one STAMP GFOR case with droop factor 0.05")
    stamp_admittance = reference["gfor_admittance"][:, :, :, matching_factor[0]]
    stamp_grid_impedance = reference["gfor_grid_impedance"][:, :, :, matching_factor[0]]
    veragrid_admittance, veragrid_grid_impedance, device_name = responses(
        frequencies_hz, bus=4,
        nonlinear_converters=args.nonlinear_converters,
    )
    reference_angle = stamp_frame_rotation(args.reference_angle_data)
    veragrid_admittance = rotate_qd(veragrid_admittance, reference_angle)
    veragrid_grid_impedance = rotate_qd(veragrid_grid_impedance, reference_angle)

    print(f"STAMP case: {reference['gfor_case'].item()}, factor 0.05")
    print(f"VeraGrid device: {device_name}")
    print(f"VeraGrid-to-STAMP qd frame rotation: {reference_angle:.12g} rad")
    print("Hz   converter-admittance error   grid-impedance error")
    for frequency, stamp_y, veragrid_y, stamp_z, veragrid_z in zip(
        frequencies_hz,
        np.moveaxis(stamp_admittance, 2, 0),
        veragrid_admittance,
        np.moveaxis(stamp_grid_impedance, 2, 0),
        veragrid_grid_impedance,
    ):
        admittance_error = np.linalg.norm(veragrid_y - stamp_y) / np.linalg.norm(stamp_y)
        impedance_error = np.linalg.norm(veragrid_z - stamp_z) / np.linalg.norm(stamp_z)
        print(f"{frequency:4g} {admittance_error:28.6g} {impedance_error:22.6g}")


if __name__ == "__main__":
    main()
