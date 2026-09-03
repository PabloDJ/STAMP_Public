#!/usr/bin/env python3
"""Plot apples-to-apples STAMP and VeraGrid WSCC eigenvalue comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "STAMP/02_results"
COMPARISON = RESULTS / "comparison"
OUTPUT = COMPARISON / "WSCC_SG_GFOR_GFOL_eigenvalue_comparison.png"


def eigenvalues(path: Path) -> np.ndarray:
    return np.linalg.eigvals(np.loadtxt(path, delimiter=","))


def complex_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    return values[:, 0] + 1j * values[:, 1]


def main() -> None:
    setups = [
        (
            "Full dynamic network",
            eigenvalues(RESULTS / "multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv"),
            complex_csv(COMPARISON / "WSCC_SG_GFOR_GFOL_veragrid_full_dynamic_eigenvalues.csv"),
        ),
        (
            "Dynamic lines only",
            eigenvalues(COMPARISON / "WSCC_SG_GFOR_GFOL_STAMP_dynamic_lines_A.csv"),
            complex_csv(COMPARISON / "WSCC_SG_GFOR_GFOL_veragrid_dynamic_lines_eigenvalues.csv"),
        ),
        (
            "Fully algebraic network",
            eigenvalues(COMPARISON / "WSCC_SG_GFOR_GFOL_STAMP_quasistatic_A.csv"),
            eigenvalues(COMPARISON / "WSCC_SG_GFOR_GFOL_veragrid_reduced_A_impedance_loads.csv"),
        ),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for axis, (title, stamp, veragrid) in zip(axes, setups):
        stamp = stamp[(stamp.real >= -700) & (stamp.real <= 350) & (np.abs(stamp.imag) <= 6000)]
        veragrid = veragrid[(veragrid.real >= -700) & (veragrid.real <= 350)
                            & (np.abs(veragrid.imag) <= 6000)]
        axis.scatter(stamp.real, stamp.imag, marker="o", s=30, facecolors="none",
                     edgecolors="#3568b8", linewidths=1.4, label="STAMP", zorder=2)
        axis.scatter(veragrid.real, veragrid.imag, marker="x", s=28, color="#d44c5c",
                     linewidths=1.15, label="VeraGrid", zorder=3)
        axis.axvline(0.0, color="#b62738", linewidth=1.0, linestyle="--", alpha=0.8)
        axis.axhline(0.0, color="#8a909c", linewidth=0.7, alpha=0.5)
        axis.set_title(title, fontsize=12, weight="semibold")
        axis.set_xlabel(r"Real part, Re($\lambda$) [rad/s]")
        axis.set_xlim(-700, 350)
        axis.set_ylim(-6000, 6000)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        axis.grid(color="#dfe3ea", linewidth=0.65)
    axes[0].set_ylabel(r"Imaginary part, Im($\lambda$) [rad/s]")
    fig.suptitle("WSCC eigenvalue comparison: STAMP vs VeraGrid", fontsize=15, weight="semibold")
    fig.text(0.5, 0.015,
             "Circles and crosses overlap at plot resolution in all three matched formulations.",
             ha="center", fontsize=9, color="#555d6b")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
