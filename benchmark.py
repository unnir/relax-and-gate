"""Deterministic benchmark access for the macro_placement task.  FIXED — do not edit.

Owns three things the rest of the task must not re-decide:
  * the frozen dev / val / test split (SPLIT.json),
  * cached ProxyEngine construction (loading an engine costs ~0.5-3 s),
  * the legality predicate (zero hard-macro overlap + inside the canvas).
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mpx.engine import ProxyEngine  # noqa: E402

with open(os.path.join(ROOT, "SPLIT.json")) as f:
    SPLIT = json.load(f)

DEV, VAL, TEST, ALL = SPLIT["dev"], SPLIT["val"], SPLIT["test"], SPLIT["all"]

_ENGINES = {}


def get_engine(name, device="cuda", dtype=torch.float64):
    key = (name, str(device), str(dtype))
    if key not in _ENGINES:
        _ENGINES[key] = ProxyEngine(os.path.join(ROOT, "data", f"{name}.npz"),
                                    device=device, dtype=dtype)
    return _ENGINES[key]


def split_names(split):
    return {"dev": DEV, "val": VAL, "test": TEST, "all": ALL}[split]


def legality(eng, pos):
    """(is_legal, detail) for a single [N,2] placement, using the challenge's rules."""
    pos = pos.to(eng.device, eng.dtype)
    st = eng.overlap_stats(pos)
    nh = eng.n_hard
    hw = eng.sizes[:nh, 0] * 0.5
    hh = eng.sizes[:nh, 1] * 0.5
    x, y = pos[:nh, 0], pos[:nh, 1]
    oob = int(((x - hw < -1e-9) | (x + hw > eng.W + 1e-9) |
               (y - hh < -1e-9) | (y + hh > eng.H + 1e-9)).sum())
    finite = bool(torch.isfinite(pos).all())
    n_ov = int(st["overlap_count"])
    return (n_ov == 0 and oob == 0 and finite,
            {"overlap_count": n_ov, "overlap_area": float(st["overlap_area"]),
             "out_of_bounds": oob, "finite": finite})


def baseline_table():
    """Published Tier-1 numbers, for reference only (reported, never reproduced-by-us)."""
    return {
        "SA":        {"avg": 2.1251, "source": "challenge README (TILOS paper)"},
        "RePlAce":   {"avg": 1.4578, "source": "challenge README (TILOS paper)"},
        "GreedyRow": {"avg": 2.2109, "source": "challenge README demo placer"},
        "Archgen":   {"avg": 0.9507, "source": "official leaderboard rank 1"},
        "Carrotato": {"avg": 0.9522, "source": "official leaderboard rank 2"},
        "JaneRT":    {"avg": 0.9694, "source": "official leaderboard rank 3"},
    }
