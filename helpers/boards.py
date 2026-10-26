"""
boards.py
─────────────────────────────────────────────────────────────────────────────
FPGA resource budgets, selectable by name (e.g. --board Zynq-7015).

To add a board (or a reduced budget of an existing one), add one entry to
BOARDS. Resource counts are LUT, DSP, BRAM (36Kb blocks) and FF.
"""

import argparse
from typing import Dict

BOARDS: Dict[str, Dict[str, int]] = {
    "XC7S25":    {"LUT": 14600,   "DSP": 80,    "BRAM": 45,   "FF": 29200},
    # XC7S25 with a reduced resource budget
    "XC7S25-65": {"LUT": 9490,    "DSP": 52,    "BRAM": 29,   "FF": 18980},
    "Zynq-7010": {"LUT": 17600,   "DSP": 80,    "BRAM": 60,   "FF": 35200},
    "Zynq-7015": {"LUT": 46200,   "DSP": 160,   "BRAM": 95,   "FF": 92400},
    "Zynq-7020": {"LUT": 53200,   "DSP": 220,   "BRAM": 140,  "FF": 106400},
    "Zynq-7030": {"LUT": 78600,   "DSP": 400,   "BRAM": 265,  "FF": 157200},
    "ZU3EG":     {"LUT": 70500,   "DSP": 360,   "BRAM": 216,  "FF": 141000},
    "U250":      {"LUT": 1727040, "DSP": 12288, "BRAM": 2688, "FF": 3454080},
}

# Board used when --board is not given (the one that was active in the scripts)
DEFAULT_BOARD = "Zynq-7015"


def _norm(name: str) -> str:
    """Case/separator-insensitive key: 'zynq_7015' == 'Zynq-7015' == 'ZYNQ7015'."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_board(name: str) -> str:
    """Return the canonical BOARDS key for `name`, or raise ValueError."""
    key = _norm(name)
    for canonical in BOARDS:
        if _norm(canonical) == key:
            return canonical
    raise ValueError(f"unknown board '{name}'. Available: {', '.join(BOARDS)}")


def board_arg(name: str) -> str:
    """argparse `type=` helper: canonicalises the name and gives a clear error."""
    try:
        return resolve_board(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
