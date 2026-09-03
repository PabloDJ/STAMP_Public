#!/usr/bin/env python3
"""Visual evidence that the additional EMT poles are common-mode network modes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.project_emt_floquet_sequence_subspaces import _build_problem


PERIOD = 1.0 / 50.0


def exponents(multipliers: np.ndarray) -> np.ndarray:
    return np.log(multipliers.astype(np.complex128)) / PERIOD


def dominant_line_phase_vector(problem, vector: np.ndarray) -> tuple[str, np.ndarray]:
    best_name = ""
    best_values = None
    best_norm = -1.0
    for line in problem.grid.lines:
        child = next(block for block in line.emt_model.get_all_blocks()
                     if any(var.name == "i_ser_A" for var in block.state_vars))
        variables = [next(var for var in child.state_vars if var.name == f"i_ser_{phase}")
                     for phase in "ABC"]
        values = np.asarray([vector[problem.get_var_idx(var)] for var in variables])
        magnitude = linalg.norm(values)
        if magnitude > best_norm:
            best_norm = magnitude
            best_name = line.name
            best_values = values
    assert best_values is not None
    # Remove the arbitrary complex eigenvector phase and normalize phase A.
    best_values = best_values * np.exp(-1j*np.angle(best_values[0]))
    best_values /= np.max(np.abs(best_values))
    return best_name, best_values


def main() -> None:
    problem = _build_problem()
    monodromy = np.load(ROOT / "scripts/stamp_wscc_emt_monodromy.npy")
    lift = np.load(ROOT / "scripts/stamp_wscc_rms_emt_tangent_lift.npy")

    # Q spans the exact RMS-compatible tangent space; Q_perp is its
    # 36-dimensional EMT-only complement.
    q_full = linalg.qr(lift, mode="full")[0]
    q_rms = q_full[:, :lift.shape[1]]
    q_extra = q_full[:, lift.shape[1]:]
    shared_matrix = q_rms.T @ monodromy @ q_rms
    extra_matrix = q_extra.T @ monodromy @ q_extra
    shared_mu = linalg.eigvals(shared_matrix)
    extra_mu, extra_vectors_reduced = linalg.eig(extra_matrix)
    shared_lambda = exponents(shared_mu)
    extra_lambda = exponents(extra_mu)
    extra_vectors = q_extra @ extra_vectors_reduced

    targets = (94.28, 155.50)
    selected = [int(np.argmin(np.where(extra_lambda.imag > 0,
                                       np.abs(extra_lambda.imag-target), np.inf)))
                for target in targets]

    figure = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    spectrum = figure.add_subplot(grid[0, :])
    spectrum.scatter(shared_lambda.real, shared_lambda.imag, marker="x", s=42,
                     linewidths=1.4, color="#2166ac",
                     label="RMS-compatible EMT subspace (88 modes)")
    spectrum.scatter(extra_lambda.real, extra_lambda.imag, marker="o", s=48,
                     facecolors="#ef8a62", edgecolors="#b2182b", linewidths=0.9,
                     label="EMT-only complement (36 modes)")
    for index in selected:
        value = extra_lambda[index]
        spectrum.annotate(f"{value.real:.2f} + j{value.imag:.2f}",
                          xy=(value.real, value.imag), xytext=(12, 8),
                          textcoords="offset points", fontsize=9,
                          arrowprops={"arrowstyle": "->", "lw": 0.8})
    spectrum.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    spectrum.axhline(0.0, color="0.7", linewidth=0.7)
    spectrum.grid(True, alpha=0.25)
    spectrum.set_xlim(-180.0, 5.0)
    spectrum.set_ylim(-160.0, 160.0)
    spectrum.set_xlabel(r"Real part, $\sigma$ [s$^{-1}$]")
    spectrum.set_ylabel(r"Imaginary part, $\omega$ [rad/s]")
    spectrum.set_title("Full EMT Floquet spectrum separated by tangent subspace")
    spectrum.legend(loc="lower left")

    phases = np.arange(3)
    width = 0.36
    for panel, index in zip((figure.add_subplot(grid[1, 0]),
                             figure.add_subplot(grid[1, 1])), selected):
        line_name, values = dominant_line_phase_vector(problem, extra_vectors[:, index])
        panel.bar(phases-width/2, values.real, width, color="#4d9221", label="Real")
        panel.bar(phases+width/2, values.imag, width, color="#c51b7d", label="Imaginary")
        panel.axhline(0.0, color="black", linewidth=0.7)
        panel.set_xticks(phases, ("A", "B", "C"))
        panel.set_ylim(-1.15, 1.15)
        panel.grid(True, axis="y", alpha=0.25)
        value = extra_lambda[index]
        common = abs(np.mean(values)) / np.sqrt(np.mean(np.abs(values)**2))
        panel.set_title(
            f"{line_name} series-current shape\n"
            rf"$\lambda={value.real:.2f}+j{value.imag:.2f}$, common-mode score={common:.6f}")
        panel.set_ylabel("Normalized complex eigenvector component")
        panel.legend(loc="best")

    figure.suptitle("Why the additional EMT poles do not appear in balanced RMS")
    output = ROOT / "scripts/stamp_wscc_emt_only_common_mode_evidence.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    np.savetxt(ROOT / "scripts/stamp_wscc_emt_only_floquet_exponents.csv",
               np.column_stack((extra_lambda.real, extra_lambda.imag)), delimiter=",",
               header="real,imag", comments="")
    print(f"shared modes={shared_lambda.size}, EMT-only modes={extra_lambda.size}")
    for index in selected:
        line_name, values = dominant_line_phase_vector(problem, extra_vectors[:, index])
        common = abs(np.mean(values)) / np.sqrt(np.mean(np.abs(values)**2))
        print(f"{extra_lambda[index]=}, {line_name=}, common_mode_score={common:.12f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
