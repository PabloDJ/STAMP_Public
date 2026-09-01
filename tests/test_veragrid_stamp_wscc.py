from __future__ import annotations

import math

from veragrid_stamp.bases import peak_ln_to_rms_ll, rms_ll_to_peak_ln
from veragrid_stamp.initialization import initialize_stamp_converter
from veragrid_stamp.parameters import STAMP_GFOL, STAMP_GFOR
from veragrid_stamp.parameters import STAMP_SG
from veragrid_stamp.rotor_circuit import derive_rotor_circuit
from veragrid_stamp.wscc_case import STAMP_LINES, STAMP_LOADS


def test_stamp_wscc_topology_constants() -> None:
    assert len(STAMP_LINES) == 6
    assert len(STAMP_LOADS) == 3
    assert {endpoint for line in STAMP_LINES for endpoint in line[:2]} == set(range(1, 7))
    assert sum(load[1] for load in STAMP_LOADS) == 315.0
    assert sum(load[2] for load in STAMP_LOADS) == 115.0


def test_stamp_converter_derived_gains() -> None:
    assert math.isclose(STAMP_GFOR.current_kp, (0.15 / (2 * math.pi * 50)) / 0.001)
    assert math.isclose(STAMP_GFOR.current_ki, 5.0)
    kp_pll, ki_pll = STAMP_GFOL.pll_gains
    assert kp_pll > 0.0
    assert ki_pll > kp_pll


def test_converter_initialization_reconstructs_terminal_power() -> None:
    point = initialize_stamp_converter(STAMP_GFOL, 1.03235, math.radians(4.1836), 0.85, -0.11)
    recovered = 3.0 * point.terminal_voltage_rms_phase * point.grid_current_rms.conjugate()
    assert abs(recovered.real - 0.85) < 1.0e-12
    assert abs(recovered.imag + 0.11) < 1.0e-12


def test_voltage_base_conversion_round_trip() -> None:
    assert math.isclose(peak_ln_to_rms_ll(rms_ll_to_peak_ln(1.025)), 1.025)


def test_stamp_rotor_circuit_parameters_are_physical() -> None:
    rotor = derive_rotor_circuit(STAMP_SG)
    assert all(value > 0.0 for value in (rotor.lfd, rotor.l1d, rotor.l1q, rotor.l2q,
                                         rotor.rf, rotor.r1d, rotor.r1q, rotor.r2q))
