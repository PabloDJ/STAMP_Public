#!/usr/bin/env python3
"""Compare RMS eigenvalues with EMT Floquet exponents modulo the base frequency."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
RMS_PATH = ROOT / "STAMP/02_results/comparison/WSCC_SG_GFOR_GFOL_veragrid_full_dynamic_eigenvalues.csv"
EMT_PATH = ROOT / "scripts/stamp_wscc_emt_floquet_exponents.csv"
OUT_PATH = ROOT / "scripts/stamp_wscc_rms_emt_spectral_matches.csv"
OMEGA_BASE = 2.0*np.pi*50.0


def fold_frequency(values):
    return values.real + 1j*((values.imag+OMEGA_BASE/2.0) % OMEGA_BASE-OMEGA_BASE/2.0)


def main():
    rms_data = np.loadtxt(RMS_PATH, delimiter=",", skiprows=1)
    emt_data = np.loadtxt(EMT_PATH, delimiter=",", skiprows=1)
    rms = fold_frequency(rms_data[:, 0]+1j*rms_data[:, 1])
    emt = emt_data[:, 0]+1j*emt_data[:, 1]
    rows, cols = linear_sum_assignment(np.abs(rms[:, None]-emt[None, :]))
    table = np.column_stack((rms[rows].real, rms[rows].imag,
                             emt[cols].real, emt[cols].imag,
                             np.abs(rms[rows]-emt[cols])))
    table = table[np.argsort(table[:, 0])[::-1]]
    np.savetxt(OUT_PATH, table, delimiter=",",
               header="rms_real,rms_imag_folded,emt_real,emt_imag,distance",
               comments="")
    for row in table[:25]:
        print(f"RMS {row[0]:+.9f}{row[1]:+.9f}j -> "
              f"EMT {row[2]:+.9f}{row[3]:+.9f}j; distance={row[4]:.3e}")
    unstable = emt[emt.real > 1.0e-8]
    print(f"unmatched/extra EMT unstable exponents: {len(unstable)}")
    for value in unstable:
        print(f"  {value.real:+.9f}{value.imag:+.9f}j")


if __name__ == "__main__":
    main()
