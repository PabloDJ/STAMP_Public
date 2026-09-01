"""Build the STAMP WSCC/GFOR/GFOL network directly in VeraGrid."""

from __future__ import annotations

from typing import Any

from .models import get_stamp_gfol_rms, get_stamp_gfor_rms, get_stamp_synchronous_generator_rms
from .parameters import STAMP_GFOL, STAMP_GFOR, STAMP_SG, SYSTEM_BASE_MVA


VERAGRID_IEEE9_PATH = (
    "/home/pablo/Desktop/eroots/VeraGrid_TenSyGrid/Grids_and_profiles/grids/"
    "IEEE_9_Christoph.gridcal"
)


STAMP_LINES = (
    (1, 2, 0.0100, 0.0850, 0.1760),
    (1, 3, 0.0170, 0.0920, 0.1580),
    (2, 4, 0.0320, 0.1610, 0.3060),
    (3, 6, 0.0390, 0.1700, 0.3580),
    (4, 5, 0.0085, 0.0720, 0.1490),
    (5, 6, 0.0119, 0.1008, 0.2090),
)

STAMP_LOADS = (
    (2, 125.0, 50.0),
    (3, 90.0, 30.0),
    (5, 100.0, 35.0),
)


def _assign_network_rms_models(grid: Any) -> None:
    from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
    from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
    from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model

    for bus in grid.buses:
        initialize_bus_rms(bus, vf=grid.var_factory)
    for index, line in enumerate(grid.lines, start=1):
        set_rms_model(line, get_line_rms_template(grid.var_factory, name=f"STAMP_line_{index}").block, grid.var_factory)
    for index, load in enumerate(grid.loads, start=1):
        set_rms_model(load, get_load_rms_template(grid.var_factory, name=f"STAMP_load_{index}").block, grid.var_factory)


def build_stamp_wscc_grid() -> Any:
    """Import VeraGrid's six-bus IEEE9 case and populate STAMP RMS models."""
    import VeraGridEngine.api as gce
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model

    grid = gce.open_file(VERAGRID_IEEE9_PATH)
    grid.name = "STAMP WSCC SG-GFOR-GFOL"
    grid.Sbase = SYSTEM_BASE_MVA
    grid.fBase = 50.0
    buses = {int(bus.name.removeprefix("Bus")): bus for bus in grid.buses}

    # This legacy gridcal stores the STAMP dispatch in pu although VeraGrid's
    # device API expects MW/MVAr.  Apply the authoritative case values.
    for load, (_, p_mw, q_mvar) in zip(grid.loads, STAMP_LOADS):
        load.P, load.Q = p_mw, q_mvar
    generator = grid.generators[0]
    generator.name = "STAMP SG1"
    generator.P = STAMP_SG.p_pu_system * SYSTEM_BASE_MVA
    generator.Vset = STAMP_SG.voltage_pu
    generator.Snom = STAMP_SG.rated_mva

    # The topology file predates the GFOR/GFOL injections, so add those two
    # devices after importing the grid rather than recreating the network.
    converter_devices = []
    for params in (STAMP_GFOR, STAMP_GFOL):
        # STAMP places both VSCs in MATPOWER's generator table: P and |V| are
        # specified and Q is solved.  They must therefore be PV devices here,
        # even though their attached RMS blocks are converter models.
        converter = gce.Generator(
            name=f"STAMP {params.mode}{params.number}",
            P=params.p_pu_system * SYSTEM_BASE_MVA,
            vset=params.voltage_pu,
            Snom=params.rated_mva,
        )
        grid.add_generator(buses[params.bus], converter)
        converter_devices.append((params, converter))

    _assign_network_rms_models(grid)
    set_rms_model(generator, get_stamp_synchronous_generator_rms(grid.var_factory).block, grid.var_factory)
    for params, converter in converter_devices:
        template = get_stamp_gfor_rms(grid.var_factory) if params.mode == "GFOR" else get_stamp_gfol_rms(grid.var_factory)
        set_rms_model(converter, template.block, grid.var_factory)
    return grid
