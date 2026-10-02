#!/usr/bin/env python3
"""Compare nonlinear VeraGrid GFOR/GFOL equations with the STAMP WSCC point and spectrum."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_stamp_veragrid_full_dynamic_network import main


if __name__ == "__main__":
    main(nonlinear_converters=True)
