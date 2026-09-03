#!/usr/bin/env python3
"""Compare the exact balanced 88-state RMS tangent space with EMT Floquet."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.project_emt_floquet_sequence_subspaces import _build_problem


EDGES = ("12", "13", "24", "36", "45", "56")


def _stamp_transform(names: list[str]) -> np.ndarray:
    """Return the validated map from ordered VeraGrid RMS to STAMP states."""
    lp_path = ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_sg_linearization_point.csv"
    with lp_path.open(newline="", encoding="utf-8-sig") as stream:
        lp = {row["field"]: float(row["value"]) for row in csv.DictReader(stream)}
    shift = np.arctan2(-lp["vd_bus0"], lp["vq_bus0"])
    rotation = np.asarray([[np.cos(shift), np.sin(shift)],
                           [-np.sin(shift), np.cos(shift)]])
    transform = np.eye(len(names))
    ni = {name: index for index, name in enumerate(names)}
    pairs = [(f"NET.iq{edge}", f"NET.id{edge}") for edge in EDGES]
    pairs += [(f"vc_q{bus}", f"vc_d{bus}") for bus in range(1, 7)]
    pairs += [(f"Load{load}.ilq", f"Load{load}.ild") for load in range(1, 4)]
    pairs += [("SG1.ig_qx", "SG1.ig_dx")]
    pairs += [(f"{dev}.{base}_q", f"{dev}.{base}_d")
              for dev in ("GFOR1", "GFOL2") for base in ("ig", "is", "ucap")]
    network_pairs = set(pairs[:15])
    for q_name, d_name in pairs:
        scale = np.sqrt(2.0 / 3.0) if (q_name, d_name) in network_pairs else 1.0
        indices = [ni[q_name], ni[d_name]]
        transform[np.ix_(indices, indices)] = scale * rotation
    return transform


def _abc_from_qd() -> np.ndarray:
    """Peak abc perturbations from conventional RMS q,d at theta=0."""
    angles = np.asarray([0.0, -2.0*np.pi/3.0, 2.0*np.pi/3.0])
    return np.sqrt(2.0) * np.column_stack((np.sin(angles), -np.cos(angles)))


def _state_indices(problem, variables) -> list[int]:
    return [problem.get_var_idx(variable) for variable in variables]


def _line_child(line):
    return next(block for block in line.emt_model.get_all_blocks()
                if any(variable.name == "i_ser_A" for variable in block.state_vars))


def _capacitance_matrix(problem, child) -> np.ndarray:
    values = problem.get_parameters_values()
    result = np.empty((3, 3))
    for row, phase_row in enumerate("abc"):
        for column, phase_column in enumerate("abc"):
            name = f"C{phase_row}{phase_column}"
            variable = next(var for var in child.api_obj_mapping.values()
                            if var.name == name)
            index = problem.uid2idx_params[variable.uid]
            result[row, column] = float(values[index].value)
    return result


def _device_key(name: str) -> str | None:
    if "theta_grid" in name:
        return None
    key = name.replace("STAMP_SG1.", "SG1.")
    key = key.replace("STAMP_GFOR1_EMT.", "GFOR1.")
    key = key.replace("STAMP_GFOL2_EMT.", "GFOL2.")
    if key == "SG1.ig_q":
        return "SG1.ig_qx"
    if key == "SG1.ig_d":
        return "SG1.ig_dx"
    return key


def build_lift(problem, stamp_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build physical-RMS and STAMP-coordinate lifts into 124 EMT states."""
    n_emt = problem.get_states_number()
    n_rms = len(stamp_names)
    columns = {name: index for index, name in enumerate(stamp_names)}
    lift_vg = np.zeros((n_emt, n_rms))
    abc = _abc_from_qd()

    # Dynamic line series currents.
    for line in problem.grid.lines:
        edge = "".join(sorted((line.bus_from.name.removeprefix("Bus"),
                               line.bus_to.name.removeprefix("Bus"))))
        child = _line_child(line)
        indices = _state_indices(problem, child.state_vars[:3])
        lift_vg[np.ix_(indices, [columns[f"NET.iq{edge}"],
                                      columns[f"NET.id{edge}"]])] = abc

    # Each bus voltage drives every incident pi-line terminal charge through
    # that terminal's actual three-phase capacitance matrix.
    for line in problem.grid.lines:
        child = _line_child(line)
        capacitance = _capacitance_matrix(problem, child)
        for bus, charge_slice in ((line.bus_from, slice(3, 6)),
                                  (line.bus_to, slice(6, 9))):
            bus_number = bus.name.removeprefix("Bus")
            indices = _state_indices(problem, child.state_vars[charge_slice])
            lift_vg[np.ix_(indices, [columns[f"vc_q{bus_number}"],
                                      columns[f"vc_d{bus_number}"]])] = capacitance @ abc

    # Balanced load inductor currents.
    for load_number, load in enumerate(problem.grid.loads, 1):
        variables = [next(var for var in load.emt_model.get_all_vars()
                          if var.name == f"iL_{phase}") for phase in "ABC"]
        indices = _state_indices(problem, variables)
        lift_vg[np.ix_(indices, [columns[f"Load{load_number}.ilq"],
                                  columns[f"Load{load_number}.ild"]])] = abc

    # The 58 retained device dq/control states are common named coordinates.
    mapped_device_states = 0
    for device in problem.grid.generators:
        for variable in device.emt_model.state_vars:
            key = _device_key(variable.name)
            if key is None:
                continue
            if key not in columns:
                raise KeyError(f"No RMS state corresponding to EMT state {variable.name!r}")
            lift_vg[problem.get_var_idx(variable), columns[key]] = 1.0
            mapped_device_states += 1
    if mapped_device_states != 58:
        raise RuntimeError(f"Expected 58 named device states, mapped {mapped_device_states}")

    # x_stamp = T x_vg, hence x_emt = L_vg T^-1 x_stamp.
    transform = _stamp_transform(stamp_names)
    lift_stamp = lift_vg @ linalg.inv(transform)
    return lift_vg, lift_stamp


def _print_rightmost(label: str, matrix: np.ndarray, period: float, count: int = 12) -> None:
    multipliers = linalg.eigvals(matrix)
    exponents = np.log(multipliers.astype(np.complex128)) / period
    order = np.argsort(exponents.real)[::-1]
    print(label)
    for index in order[:count]:
        value = exponents[index]
        print(f"  lambda={value.real:+.12e} {value.imag:+.12e}j "
              f"mu_abs={abs(multipliers[index]):.12e}")


def _plot_spectra(reduced: np.ndarray, rms: np.ndarray, period: float,
                  output_path: Path) -> None:
    import matplotlib.pyplot as plt

    def exponents(matrix: np.ndarray) -> np.ndarray:
        multipliers = linalg.eigvals(matrix)
        return np.log(multipliers.astype(np.complex128)) / period

    emt_modes = exponents(reduced)
    rms_modes = exponents(rms)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    panels = (
        (axes[0], "Complete principal Floquet spectrum", None),
        (axes[1], "Rightmost modes", (-5.0, 0.25)),
    )
    for axis, title, x_limits in panels:
        axis.scatter(rms_modes.real, rms_modes.imag, marker="x", s=42,
                     linewidths=1.5, color="#2166ac", label="RMS / STAMP")
        axis.scatter(emt_modes.real, emt_modes.imag, marker="o", s=30,
                     facecolors="none", edgecolors="#b2182b", linewidths=1.2,
                     label="EMT Floquet (88-state reduction)")
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.axhline(0.0, color="0.7", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel(r"Real part, $\sigma$ [s$^{-1}$]")
        axis.set_ylabel(r"Imaginary part, $\omega$ [rad/s]")
        if x_limits is not None:
            axis.set_xlim(*x_limits)
            axis.set_ylim(-12.0, 12.0)
    axes[0].legend(loc="best")
    figure.suptitle("STAMP WSCC: corrected RMS–EMT small-signal comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    problem = _build_problem()
    monodromy = np.load(ROOT / "scripts/stamp_wscc_emt_monodromy.npy")
    stamp_a = np.loadtxt(
        ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv",
        delimiter=",")
    stamp_names = (ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_state_names.txt").read_text().splitlines()
    period = 1.0 / problem.grid.fBase
    _, lift = build_lift(problem, stamp_names)
    if lift.shape != (124, 88) or np.linalg.matrix_rank(lift) != 88:
        raise RuntimeError(f"Invalid lift shape/rank: {lift.shape}, rank={np.linalg.matrix_rank(lift)}")

    pinv = linalg.pinv(lift)
    reduced = pinv @ monodromy @ lift
    rms = linalg.expm(stamp_a * period)
    propagated = monodromy @ lift
    leaked = propagated - lift @ reduced
    relative_leakage_fro = linalg.norm(leaked, "fro") / linalg.norm(propagated, "fro")
    relative_leakage_2 = linalg.norm(leaked, 2) / linalg.norm(propagated, 2)
    difference = reduced - rms

    print(f"EMT states={monodromy.shape[0]}, RMS states={lift.shape[1]}, "
          f"lift_rank={np.linalg.matrix_rank(lift)}")
    print(f"lift condition_2={np.linalg.cond(lift):.12e}")
    print(f"leakage_fro={linalg.norm(leaked, 'fro'):.12e}, "
          f"relative_fro={relative_leakage_fro:.12e}")
    print(f"leakage_2={linalg.norm(leaked, 2):.12e}, "
          f"relative_2={relative_leakage_2:.12e}")
    print(f"M_red-M_RMS max={np.max(np.abs(difference)):.12e}, "
          f"relative_fro={linalg.norm(difference, 'fro')/linalg.norm(rms, 'fro'):.12e}")
    _print_rightmost("rightmost reduced EMT exponents:", reduced, period)
    _print_rightmost("rightmost RMS exponents:", rms, period)

    multipliers, right_vectors = linalg.eig(reduced)
    exponents = np.log(multipliers.astype(np.complex128)) / period
    dominant = int(np.argmax(exponents.real))
    shape = np.abs(right_vectors[:, dominant])
    shape /= np.max(shape)
    print("rightmost reduced EMT mode in STAMP RMS coordinates:")
    for index in np.argsort(shape)[::-1][:24]:
        print(f"  {shape[index]:.6e} {stamp_names[index]}")

    # Test how differently the two period maps propagate precisely the
    # unstable reduced direction, independent of coordinate-wise matrix norms.
    direction = right_vectors[:, dominant]
    rms_image = rms @ direction
    reduced_image = reduced @ direction
    print(f"rightmost-direction map mismatch="
          f"{linalg.norm(reduced_image-rms_image)/linalg.norm(reduced_image):.12e}")

    np.save(ROOT / "scripts/stamp_wscc_rms_emt_tangent_lift.npy", lift)
    np.save(ROOT / "scripts/stamp_wscc_emt_reduced_monodromy.npy", reduced)
    np.save(ROOT / "scripts/stamp_wscc_rms_monodromy.npy", rms)
    plot_path = ROOT / "scripts/stamp_wscc_corrected_rms_emt_spectrum.png"
    _plot_spectra(reduced, rms, period, plot_path)
    print(f"saved spectrum plot: {plot_path}")


if __name__ == "__main__":
    main()
