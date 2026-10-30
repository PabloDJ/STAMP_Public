"""The source-linear converter must be the tangent of its nonlinear RMS model."""

import numpy as np
import pytest

pytest.importorskip("VeraGridEngine")

from scripts.reproduce_stamp_port_impedance import (compute_grid_impedance_at_bus,
                                                     compute_model_admittance,
                                                     operating_problem,
                                                     prepare_port)


def test_converter_port_linearizations_agree() -> None:
    frequencies = np.array([1.0, 10.0, 100.0])
    results = {}
    for nonlinear in (False, True):
        driver = operating_problem(nonlinear_converters=nonlinear)
        for device in driver.grid.generators[1:]:
            port = prepare_port(driver, device)
            results[(nonlinear, device.name)] = (
                compute_model_admittance(port, frequencies),
                compute_grid_impedance_at_bus(port, frequencies),
            )

    for device_name in ("STAMP GFOR1", "STAMP GFOL2"):
        linear = results[(False, device_name)]
        nonlinear = results[(True, device_name)]
        for linear_response, nonlinear_response in zip(linear, nonlinear):
            np.testing.assert_allclose(linear_response, nonlinear_response,
                                       rtol=1e-9, atol=1e-9)
