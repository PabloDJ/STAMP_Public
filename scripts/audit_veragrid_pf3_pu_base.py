#!/usr/bin/env python3
"""Audit VeraGrid balanced/three-phase PF data and its per-unit convention."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def _max_abs(values) -> float:
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def main() -> None:
    import VeraGridEngine
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph

    grid = build_stamp_wscc_grid()
    print(f"VeraGrid source: {Path(VeraGridEngine.__file__).resolve()}")
    options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid=grid, options=options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=options)
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("Balanced or native three-phase power flow did not converge")

    alpha = np.exp(1j*2.0*np.pi/3.0)
    v_bal = np.asarray(pf.results.voltage)
    v_reconstructed = np.vstack((v_bal, v_bal/alpha, v_bal*alpha))
    v_native = np.vstack((pf3.results.voltage_A,
                          pf3.results.voltage_B,
                          pf3.results.voltage_C))
    sbus_bal = np.asarray(pf.results.Sbus)
    sbus_reconstructed = np.tile(sbus_bal/3.0, (3, 1))
    sbus_native = np.vstack((pf3.results.Sbus_A,
                             pf3.results.Sbus_B,
                             pf3.results.Sbus_C))

    # PF3 stores phase powers as one third of total MVA, but phase currents on
    # the conventional three-phase current base.  Therefore I=3*conj(Sph/Vph).
    ibus_from_native_power = 3.0*np.conj(
        (sbus_native/grid.Sbase)/v_native)
    ibus_from_reconstructed_power = 3.0*np.conj(
        (sbus_reconstructed/grid.Sbase)/v_reconstructed)

    if_native = np.vstack((pf3.results.If_A, pf3.results.If_B, pf3.results.If_C))
    it_native = np.vstack((pf3.results.It_A, pf3.results.It_B, pf3.results.It_C))
    if_reconstructed = np.vstack((pf.results.If,
                                  pf.results.If/alpha,
                                  pf.results.If*alpha))
    it_reconstructed = np.vstack((pf.results.It,
                                  pf.results.It/alpha,
                                  pf.results.It*alpha))

    # Network KCL: the net bus injection current must equal the sum of branch
    # currents leaving that bus.  Use the native PF3 current arrays directly.
    branch_current_sum = np.zeros_like(v_native, dtype=complex)
    bus_index = {bus.idtag: index for index, bus in enumerate(grid.buses)}
    for index, line in enumerate(grid.lines):
        branch_current_sum[:, bus_index[line.bus_from.idtag]] += if_native[:, index]
        branch_current_sum[:, bus_index[line.bus_to.idtag]] += it_native[:, index]
    kcl_residual = ibus_from_native_power-branch_current_sum
    balanced_branch_balance = np.sum(np.asarray(pf.results.Sf)
                                     + np.asarray(pf.results.St))
    native_branch_balance = sum(np.sum(sf+st) for sf, st in (
        (pf3.results.Sf_A, pf3.results.St_A),
        (pf3.results.Sf_B, pf3.results.St_B),
        (pf3.results.Sf_C, pf3.results.St_C),
    ))

    print("Balanced PF versus native PF3")
    print(f"  max |Vabc(reconstructed)-Vabc(native)| = {_max_abs(v_reconstructed-v_native):.12e}")
    print(f"  max |Sph(reconstructed)-Sph(native)| = {_max_abs(sbus_reconstructed-sbus_native):.12e} MVA")
    print(f"  max |If(reconstructed)-If(native)| = {_max_abs(if_reconstructed-if_native):.12e} pu")
    print(f"  max |It(reconstructed)-It(native)| = {_max_abs(it_reconstructed-it_native):.12e} pu")
    print(f"  max native PF3 bus KCL residual = {_max_abs(kcl_residual):.12e} pu")
    print(f"  balanced total bus power = {np.sum(sbus_bal):+.12e} MVA")
    print(f"  balanced total branch loss/shunt = {balanced_branch_balance:+.12e} MVA")
    print(f"  balanced power-balance residual = {abs(np.sum(sbus_bal)-balanced_branch_balance):.12e} MVA")
    print(f"  native PF3 total bus power = {np.sum(sbus_native):+.12e} MVA")
    print(f"  native PF3 total branch loss/shunt = {native_branch_balance:+.12e} MVA")
    print(f"  native PF3 power-balance residual = {abs(np.sum(sbus_native)-native_branch_balance):.12e} MVA")
    print("\nPer-unit identity exposed by native PF3")
    print("  S_phase = S_total/3, V_phase_pu = V_positive_sequence_pu")
    print("  I_phase_pu = 3*conj(S_phase_pu/V_phase_pu) = conj(S_total_pu/V_pu)")

    rows = []
    for load in grid.loads:
        idx = bus_index[load.bus.idtag]
        voltage = v_native[0, idx]
        phase_power_pu = complex(load.P, load.Q)/(3.0*grid.Sbase)
        expected_consumption_current = 3.0*np.conj(phase_power_pu/voltage)
        # Candidate 1 is the legacy combined-template interpretation: R and L
        # are derived directly from P_phase,Q_phase. Candidate 2 uses total
        # three-phase power in those impedance formulas (equivalent to R,L /3).
        legacy_current = np.conj(phase_power_pu/voltage)
        conventional_current = 3.0*legacy_current
        divided_power_again_current = legacy_current/3.0
        rows.append({
            "load": load.name,
            "bus": load.bus.name,
            "expected_abs_I_pu": abs(expected_consumption_current),
            "per_phase_PQ_abs_I_pu": abs(legacy_current),
            "total_PQ_abs_I_pu": abs(conventional_current),
            "divide_PQ_by_3_again_abs_I_pu": abs(divided_power_again_current),
            "per_phase_ratio": abs(legacy_current/expected_consumption_current),
            "total_power_ratio": abs(conventional_current/expected_consumption_current),
        })

    print("\nR||L load interpretations (current magnitude ratios to native PF3)")
    for row in rows:
        print(f"  {row['load']}@{row['bus']}: "
              f"Pph/Qph -> {row['per_phase_ratio']:.6f}, "
              f"3*Pph/3*Qph -> {row['total_power_ratio']:.6f}")
    print("  Keeping R,L derived from P/3,Q/3 gives one-third current.")
    print("  Dividing P,Q by 3 again gives one-ninth current.")
    print("  The conventional-base formula must use total P,Q per phase equation, "
          "equivalently divide the legacy R,L values by 3.")

    output = Path(__file__).with_name("veragrid_pf3_pu_base_audit.csv")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {output}")


if __name__ == "__main__":
    main()
