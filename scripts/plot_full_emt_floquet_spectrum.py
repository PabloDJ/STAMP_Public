#!/usr/bin/env python3
"""Plot all 124 EMT Floquet modes, without RMS tangent reduction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
PERIOD = 1.0 / 50.0


def floquet_exponents(matrix: np.ndarray) -> np.ndarray:
    multipliers = linalg.eigvals(matrix)
    return np.log(multipliers.astype(np.complex128)) / PERIOD


def main() -> None:
    full_emt = np.load(ROOT / "scripts/stamp_wscc_emt_monodromy.npy")
    rms_a = np.loadtxt(
        ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv",
        delimiter=",")
    emt_modes = floquet_exponents(full_emt)
    # Express RMS in the same principal Floquet frequency interval.
    rms_modes = floquet_exponents(linalg.expm(rms_a * PERIOD))

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), constrained_layout=True)
    panels = (
        (axes[0], "Complete 124-state EMT spectrum", None, None),
        (axes[1], "Rightmost full-EMT modes", (-5.0, 0.25), (-12.0, 12.0)),
    )
    for axis, title, x_limits, y_limits in panels:
        axis.scatter(emt_modes.real, emt_modes.imag, marker="o", s=31,
                     facecolors="none", edgecolors="#b2182b", linewidths=1.2,
                     label="Full EMT Floquet (124 states)")
        axis.scatter(rms_modes.real, rms_modes.imag, marker="x", s=38,
                     color="#2166ac", linewidths=1.25,
                     label="RMS / STAMP reference (88 states)")
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.axhline(0.0, color="0.7", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel(r"Real part, $\sigma$ [s$^{-1}$]")
        axis.set_ylabel(r"Imaginary part, $\omega$ [rad/s]")
        if x_limits is not None:
            axis.set_xlim(*x_limits)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
    axes[0].legend(loc="best")
    figure.suptitle("STAMP WSCC: complete EMT Floquet spectrum (no tangent reduction)")
    output = ROOT / "scripts/stamp_wscc_full_124_state_emt_floquet_spectrum.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    np.savetxt(
        ROOT / "scripts/stamp_wscc_full_124_state_emt_floquet_exponents.csv",
        np.column_stack((emt_modes.real, emt_modes.imag)), delimiter=",",
        header="real,imag", comments="")
    print(f"full EMT modes={emt_modes.size}, RMS reference modes={rms_modes.size}")
    print(f"unstable full EMT modes={np.count_nonzero(emt_modes.real > 1e-8)}")
    print(f"neutral full EMT modes={np.count_nonzero(np.abs(emt_modes) < 1e-8)}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
