"""STAMP steady-state initialization equations for converter filter states."""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math

from .parameters import OMEGA_BASE, StampConverterParameters


@dataclass(frozen=True)
class ConverterOperatingPoint:
    """Complex RMS and peak-q/d quantities used by STAMP at linearization."""

    terminal_voltage_rms_phase: complex
    grid_current_rms: complex
    capacitor_bus_voltage_rms: complex
    capacitor_branch_current_rms: complex
    capacitor_voltage_rms: complex
    converter_current_rms: complex
    converter_voltage_rms: complex
    local_angle: float
    vg_q_peak: float
    vg_d_peak: float
    u_q_peak: float
    u_d_peak: float
    ig_q_peak: float
    ig_d_peak: float
    is_q_peak: float
    is_d_peak: float
    vc_q_peak: float
    vc_d_peak: float


def _peak_qd(value: complex, absolute_angle: float) -> tuple[float, float]:
    magnitude = abs(value) * math.sqrt(2.0)
    return magnitude * math.cos(absolute_angle), -magnitude * math.sin(absolute_angle)


def initialize_stamp_converter(
    params: StampConverterParameters,
    voltage_pu_line_line: float,
    angle_rad: float,
    p_pu_system: float,
    q_pu_system: float,
) -> ConverterOperatingPoint:
    """Reproduce ``generate_initialization_VSC.m`` for GFOR/GFOL.

    STAMP converts the line-line per-unit voltage to phase RMS with ``1/sqrt(3)``
    and uses three-phase complex power. Converter and system bases are both
    100 MVA in this case, so no additional power-base scaling is required.
    """
    vg = complex(voltage_pu_line_line / math.sqrt(3.0), 0.0)
    grid_current = ((p_pu_system + 1j * q_pu_system) / (3.0 * vg)).conjugate()
    u = vg + grid_current * complex(params.transformer_r, params.transformer_x)
    local_angle = cmath.phase(u)
    capacitor_current = u / complex(params.damping_r, -1.0 / params.capacitor_b)
    capacitor_voltage = u - params.damping_r * capacitor_current
    converter_current = grid_current + capacitor_current
    converter_voltage = u + converter_current * complex(params.converter_r, params.converter_x)

    vg_q, vg_d = _peak_qd(vg, angle_rad)
    u_q, u_d = _peak_qd(u, angle_rad + cmath.phase(u))
    ig_q, ig_d = _peak_qd(grid_current, angle_rad + cmath.phase(grid_current))
    is_q, is_d = _peak_qd(converter_current, angle_rad + cmath.phase(converter_current))
    vc_q, vc_d = _peak_qd(converter_voltage, angle_rad + cmath.phase(converter_voltage))

    return ConverterOperatingPoint(
        terminal_voltage_rms_phase=vg,
        grid_current_rms=grid_current,
        capacitor_bus_voltage_rms=u,
        capacitor_branch_current_rms=capacitor_current,
        capacitor_voltage_rms=capacitor_voltage,
        converter_current_rms=converter_current,
        converter_voltage_rms=converter_voltage,
        local_angle=local_angle,
        vg_q_peak=vg_q,
        vg_d_peak=vg_d,
        u_q_peak=u_q,
        u_d_peak=u_d,
        ig_q_peak=ig_q,
        ig_d_peak=ig_d,
        is_q_peak=is_q,
        is_d_peak=is_d,
        vc_q_peak=vc_q,
        vc_d_peak=vc_d,
    )

