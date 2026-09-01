#!/usr/bin/env python3
"""One-to-one comparison of STAMP and VeraGrid small-signal spectra."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_modes(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {field.lower(): field for field in reader.fieldnames or ()}
        real = fields["real"]
        imag = fields.get("imaginary", fields.get("imag"))
        if imag is None:
            raise ValueError(f"No imaginary eigenvalue column in {path}")
        return np.asarray([complex(float(row[real]), float(row[imag])) for row in reader])


def mode_class(value: complex, cutoff_hz: float) -> str:
    # Use pole magnitude rather than oscillation frequency alone so that very
    # fast real filter poles are not mislabeled as low-frequency modes.
    bandwidth = abs(value) / (2.0 * np.pi)
    if bandwidth <= cutoff_hz:
        return "slow_control_or_electromechanical"
    return "fast_electrical"


def compare(reference: np.ndarray, candidate: np.ndarray, cutoff_hz: float):
    # Scale each distance by the reference mode magnitude.  The 1 rad/s floor
    # prevents slow real modes close to zero from dominating the assignment.
    scale = np.maximum(1.0, np.abs(reference))
    cost = np.abs(reference[:, None] - candidate[None, :]) / scale[:, None]
    ref_indices, candidate_indices = linear_sum_assignment(cost)
    matches = []
    for ri, ci in zip(ref_indices, candidate_indices):
        absolute = abs(reference[ri] - candidate[ci])
        matches.append((ri, ci, absolute, cost[ri, ci]))
    unmatched = sorted(set(range(reference.size)) - set(ref_indices))
    return matches, unmatched


def write_report(path: Path, reference: np.ndarray, candidate: np.ndarray,
                 matches, unmatched, cutoff_hz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matched_by_reference = {ri: (ci, absolute, relative) for ri, ci, absolute, relative in matches}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stamp_index", "class", "stamp_real", "stamp_imag", "stamp_frequency_hz",
                         "veragrid_index", "veragrid_real", "veragrid_imag", "veragrid_frequency_hz",
                         "absolute_error", "scaled_error", "status"))
        for ri, ref in enumerate(reference):
            classification = mode_class(ref, cutoff_hz)
            if ri not in matched_by_reference:
                writer.writerow((ri + 1, classification, ref.real, ref.imag, abs(ref.imag)/(2*np.pi),
                                 "", "", "", "", "", "", "unmatched_stamp"))
                continue
            ci, absolute, relative = matched_by_reference[ri]
            cand = candidate[ci]
            writer.writerow((ri + 1, classification, ref.real, ref.imag, abs(ref.imag)/(2*np.pi),
                             ci + 1, cand.real, cand.imag, abs(cand.imag)/(2*np.pi),
                             absolute, relative, "matched"))


def summarize(reference: np.ndarray, candidate: np.ndarray, matches, unmatched, cutoff_hz: float) -> None:
    print(f"mode count: STAMP={reference.size}, VeraGrid={candidate.size}, unmatched STAMP={len(unmatched)}")
    print(f"unstable modes: STAMP={np.count_nonzero(reference.real > 0)}, VeraGrid={np.count_nonzero(candidate.real > 0)}")
    for classification in ("slow_control_or_electromechanical", "fast_electrical"):
        selected = [(absolute, relative) for ri, _, absolute, relative in matches
                    if mode_class(reference[ri], cutoff_hz) == classification]
        unmatched_count = sum(mode_class(reference[ri], cutoff_hz) == classification for ri in unmatched)
        if selected:
            absolute = np.asarray([item[0] for item in selected])
            relative = np.asarray([item[1] for item in selected])
            print(f"{classification}: matched={len(selected)}, unmatched={unmatched_count}, "
                  f"median_abs={np.median(absolute):.6g}, max_abs={np.max(absolute):.6g}, "
                  f"median_scaled={np.median(relative):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp", type=Path)
    parser.add_argument("veragrid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-frequency-cutoff-hz", type=float, default=20.0)
    args = parser.parse_args()
    reference, candidate = read_modes(args.stamp), read_modes(args.veragrid)
    matches, unmatched = compare(reference, candidate, args.low_frequency_cutoff_hz)
    write_report(args.output, reference, candidate, matches, unmatched, args.low_frequency_cutoff_hz)
    summarize(reference, candidate, matches, unmatched, args.low_frequency_cutoff_hz)
    print(f"wrote comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
