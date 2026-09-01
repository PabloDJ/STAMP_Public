#!/usr/bin/env python3
"""Compare STAMP and VeraGrid WSCC operating points at full precision."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.bases import RMS_LL_TO_PEAK_LN
from veragrid_stamp.parameters import OMEGA_BASE, STAMP_GFOL, STAMP_GFOR
from veragrid_stamp.wscc_case import build_stamp_wscc_grid


def long_points(path: Path) -> dict[tuple[int, str], float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {(int(row["device"]), row["field"]): float(row["value"])
                for row in csv.DictReader(stream)}


def converter_point(vm: float, va: float, p0: float, q0: float) -> dict[str, float]:
    k = RMS_LL_TO_PEAK_LN
    vq, vd = k * vm * np.cos(va), -k * vm * np.sin(va)
    den = 1.5 * (vq * vq + vd * vd)
    igq = (p0 * vq - q0 * vd) / den
    igd = (p0 * vd + q0 * vq) / den
    uq = vq + 0.002 * igq + 0.1 * igd
    ud = vd + 0.002 * igd - 0.1 * igq
    cac = 0.15 / OMEGA_BASE
    a = 0.001 * cac * OMEGA_BASE
    ucapq = (uq - a * ud) / (1.0 + a * a)
    ucapd = (a * uq + ud) / (1.0 + a * a)
    isq = igq + cac * OMEGA_BASE * ucapd
    isd = igd - cac * OMEGA_BASE * ucapq
    theta = -np.arctan2(ud, uq)
    ct, st = np.cos(theta), np.sin(theta)
    local = lambda q, d: (ct * q - st * d, st * q + ct * d)
    uqc, udc = local(uq, ud)
    igqc, igdc = local(igq, igd)
    isqc, isdc = local(isq, isd)
    return {"u_qc0": uqc, "u_dc0": udc, "ig_qc0": igqc, "ig_dc0": igdc,
            "is_qc0": isqc, "is_dc0": isdc}


def main() -> None:
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae

    grid = build_stamp_wscc_grid()
    pf = PowerFlowDriver(grid, power_flow_options())
    pf.run()
    voltage = np.asarray(pf.results.voltage)
    with (ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_power_flow.csv").open(
            newline="", encoding="utf-8-sig") as stream:
        bus_ref = list(csv.DictReader(stream))
    vm_error = np.abs(np.abs(voltage) - np.asarray([float(row["Vm"]) for row in bus_ref]))
    va_error = np.abs(np.angle(voltage) - np.deg2rad([float(row["theta"]) for row in bus_ref]))
    print(f"bus Vm max error: {vm_error.max():.12g}")
    print(f"bus Va max error: {va_error.max():.12g} rad")

    problem = RmsProblemDae(grid=grid, options=RmsOptions(), pf_results=pf.results)
    x0 = problem.get_x0()
    sg_ref = long_points(ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_sg_linearization_point.csv")
    sg_map = {"is_q": "isq0", "is_d": "isd0", "if_d": "ifd0", "ik_d": "ikd0",
              "ik1_q": "ikq10", "ik2_q": "ikq20", "w_pu": "w0_pu"}
    sg_errors = []
    for index, variable in enumerate(problem.state_vars):
        prefix = "STAMP_SG1."
        if str(variable).startswith(prefix):
            short = str(variable)[len(prefix):]
            if short in sg_map:
                sg_errors.append(abs(x0[index] - sg_ref[(1, sg_map[short])]))
    print(f"SG winding/state max error: {max(sg_errors):.12g}")

    vsc_ref = long_points(ROOT / "STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_vsc_linearization_point.csv")
    bus_index = {bus.name: i for i, bus in enumerate(grid.buses)}
    vsc_errors = []
    for number, params in enumerate((STAMP_GFOR, STAMP_GFOL), start=1):
        i = bus_index[f"Bus{params.bus}"]
        point = converter_point(abs(voltage[i]), np.angle(voltage[i]),
                                params.p_pu_system, pf.results.Sbus[i].imag / grid.Sbase)
        for field, value in point.items():
            vsc_errors.append(abs(value - vsc_ref[(number, field)]))
    print(f"VSC local q-d max error: {max(vsc_errors):.12g}")


if __name__ == "__main__":
    main()
