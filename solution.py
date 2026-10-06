"""THE FILE THE AGENT EDITS — the macro-placement algorithm.

Contract:
    DEFAULT_PARAMS   current constants
    SEARCH_SPACE     what Optuna may tune
    place(eng, params, rng, budget_s, seed) -> [N, 2] macro centers
    construct(rng, params, seed)            -> evolve.py's unit of evolution

v3 — RELAX-AND-GATE, in cycles.

    legalized initial placement
      -> [ ANALYTICAL : Adam on a differentiable RELAXATION of the exact proxy,
                        moving all ~2600 macros at once
           BPX        : batched exact-proxy local search, accepting only
                        exactly-verified improvements                    ] x n_cycles
      -> best exactly-scored iterate ever seen

Two facts drove this shape, both measured rather than assumed:

  * the search is BASIN-limited, not budget-limited. On ibm01, 4x the budget buys 1.4%
    (0.78976 -> 0.77890) and 13x buys 3.2%. Single-macro moves cannot leave the basin
    they start in, so the fix has to be a move that displaces everything at once.
  * the exact objective is piecewise CONSTANT (positions are binned into grid cells before
    any routing demand is laid down), so its true gradient is zero almost everywhere.
    `mpx/soft.py` therefore dequantizes the objective rather than replacing it with a
    hand-designed surrogate, and one backward pass then moves every macro.

The relaxation only ever PROPOSES. Every kept iterate is scored by the bit-exact engine,
so a mis-modelled term in the relaxation costs time and can never corrupt the result.
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mpx.analytical import analytical_place                # noqa: E402
from mpx.analytical_batch import analytical_replicas        # noqa: E402
from mpx.legalize import legalize_batch, clamp_bounds, pad_small   # noqa: E402
from mpx.search import BatchedProxySearch                  # noqa: E402

DEFAULT_PARAMS = {
    "method": 2,             # 0 = initial only, 1 = BPX only, 2 = relax-and-gate cycles
    "legal_eps": 1e-4,
    "legal_iters": 400,
    "pad_pctl": 80.0,        # spread the smaller macros, anchor the largest
    "pad_frac": 0.0,         # 0 disables; see mpx/legalize.pad_small
    "pad_iters": 8,
    "pad_alpha": 0.20,

    # --- schedule ---
    "n_cycles": 1,
    "an_frac": 0.5,          # share of each cycle spent in the analytical stage
    "n_basins": 1,           # diversified short analytical probes before committing
    "probe_frac": 0.25,      # share of the budget spent on those probes
    "n_rep": 1,              # analytical replicas descended in parallel (see soft_batch.py)

    # --- analytical stage (differentiable relaxation) ---
    "an_lr": 0.06,           # Adam step, in units of a grid cell
    "an_steps": 1000000,     # effectively unbounded; the budget is the real limit
    "an_check_every": 25,    # exact-verified checkpoints
    "an_overlap_w": 40.0,    # hard-macro repulsion inside the relaxation
    "an_gamma": 0.5,         # log-sum-exp temperature, in units of a grid cell
    "an_legal_iters": 120,
    # Objective-weight SCHEDULE for the relaxation: descend a congestion-heavy weighting
    # early, anneal to the competition weights. Measured +0.005 on the dev subset and the
    # best ibm17 result observed. The weighting is a search direction; what is KEPT is
    # always decided by the exact engine under the real weights (1, 0.5, 0.5).
    "an_w0": (1.0, 0.5, 1.5),
    "an_w1": (1.0, 0.5, 0.5),

    # --- BPX stage (batched exact-proxy search) ---
    "adaptive_ops": 0,       # measured LOSS: see mpx/search.py and the KB verdict
    "screen_fp32": 0,        # measured NEUTRAL (+0.0008): the throughput does not convert
    "incremental": 1,        # exact incremental scoring of single-macro moves (52x on ibm17)
    "batch": 2048,
    "sigma0": 0.05,
    "sigma1": 0.003,
    "hot_k": 32,
    "refresh_every": 8,
    "w_gauss": 1.0,
    "w_axis": 1.0,
    "w_jump": 0.3,
    "w_toward": 0.7,
    "w_away": 0.7,
    "w_coldrow": 0.0,
    "w_coldcol": 0.0,
    "w_toward_x": 0.0,
    "w_toward_y": 0.0,
}

SEARCH_SPACE = {
    "an_frac": ("float", 0.15, 0.85),
    "an_lr": ("log", 0.005, 0.4),
    "an_overlap_w": ("log", 1.0, 400.0),
    "an_gamma": ("log", 0.1, 3.0),
    "sigma0": ("log", 0.005, 0.3),
    "sigma1": ("log", 0.0005, 0.02),
    "n_cycles": ("int", 1, 4),
    "n_basins": ("int", 1, 6),
    "probe_frac": ("float", 0.1, 0.5),
    "n_rep": ("int", 1, 8),
}
CONSTRUCT_SPACE = SEARCH_SPACE


def initial_placement(eng, p):
    pos = eng.pos0.clone().unsqueeze(0)
    pos = clamp_bounds(pos, eng)
    if float(p.get("pad_frac", 0.0)) > 0:
        pos = pad_small(pos, eng.sizes, eng.n_hard, eng.W, eng.H,
                        pctl=float(p.get("pad_pctl", 80.0)),
                        frac=float(p.get("pad_frac", 0.14)),
                        iters=int(p.get("pad_iters", 8)),
                        alpha=float(p.get("pad_alpha", 0.20)),
                        fixed_mask=eng.fixed)
    pos = legalize_batch(pos, eng, eps=float(p["legal_eps"]), iters=int(p["legal_iters"]))
    return pos[0]


def place(eng, params, rng, budget_s=60.0, seed=0, log=None):
    """Return [N, 2] macro centers for one benchmark."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    torch.manual_seed(int(seed))

    pos = initial_placement(eng, p)
    method = int(p.get("method", 2))
    if method == 0 or budget_s <= 0:
        return pos

    best_pos = pos.clone()
    best = float(eng.evaluate(pos.unsqueeze(0))["proxy"][0])

    if method == 1:
        pos, info = BatchedProxySearch(eng, p, rng).run(pos, budget_s, log=log)
        place.last_info = info
        return pos

    # --- basin selection by successive halving --------------------------------
    # The measured failure mode is basin-limitation, and the analytical stage is what
    # chooses the basin. So spend a small slice of the budget running several SHORT,
    # diversified analytical starts, keep the one the EXACT engine likes best, and give
    # the rest of the budget to that basin. Diversity comes from the objective weighting
    # and the step size, not just the seed: different weightings land in structurally
    # different layouts, while a different seed alone mostly re-finds the same one.
    n_basins = max(1, int(p.get("n_basins", 1)))
    if n_basins > 1:
        probe_frac = float(p.get("probe_frac", 0.25))
        probe_t = budget_s * probe_frac / n_basins
        cw = [(1.0, 0.5, 0.5), (1.0, 0.5, 1.5), (0.3, 0.5, 0.5),
              (1.0, 1.5, 0.5), (2.0, 0.5, 0.5), (1.0, 0.25, 1.0)]
        cand_best, cand_pos = None, None
        for b in range(n_basins):
            pb = dict(p)
            pb["an_w0"] = cw[b % len(cw)]
            pb["an_w1"] = (1.0, 0.5, 0.5)
            pb["an_lr"] = float(p.get("an_lr", 0.06)) * (1.0 + 0.5 * (b % 3 - 1))
            torch.manual_seed(int(seed) * 977 + b)
            pp, ii = analytical_place(eng, pos, probe_t, pb, log=log)
            if cand_best is None or ii["proxy"] < cand_best:
                cand_best, cand_pos = ii["proxy"], pp.clone()
                p = dict(p)
                p["an_w0"] = pb["an_w0"]
                p["an_w1"] = pb["an_w1"]
                p["an_lr"] = pb["an_lr"]
        pos = cand_pos
        if cand_best < best:
            best, best_pos = cand_best, pos.clone()
        budget_s = budget_s * (1.0 - probe_frac)

    n_cycles = max(1, int(p.get("n_cycles", 1)))
    an_frac = float(p.get("an_frac", 0.5))
    per = budget_s / n_cycles
    infos = []
    for c in range(n_cycles):
        if an_frac > 0:
            n_rep = int(p.get("n_rep", 1))
            if n_rep > 1:
                pos, ia = analytical_replicas(eng, pos, per * an_frac, p,
                                              n_rep=n_rep, log=log)
            else:
                pos, ia = analytical_place(eng, pos, per * an_frac, p, log=log)
            if ia["proxy"] < best:
                best, best_pos = ia["proxy"], pos.clone()
            infos.append(("an", ia))
        if an_frac < 1:
            pos, ib = BatchedProxySearch(eng, p, rng).run(pos, per * (1 - an_frac), log=log)
            if ib["proxy"] < best:
                best, best_pos = ib["proxy"], pos.clone()
            infos.append(("bpx", ib))
    place.last_info = {"stages": infos, "proxy": best}
    return best_pos


def construct(rng, params, seed=None):
    """evolve.py unit: build one placement for one circuit and report its proxy."""
    import benchmark as B
    prm = dict(params or {})
    name = prm.get("circuit", "ibm01")
    eng = B.get_engine(name)
    pos = place(eng, prm, rng, budget_s=float(prm.get("budget_s", 60.0)),
                seed=int(prm.get("seed", 0)))
    c = eng.evaluate(pos.unsqueeze(0))
    return {"score": -float(c["proxy"][0]),
            "candidate": pos.detach().cpu().numpy().round(6).tolist(),
            "behavior": f"wl{float(c['wirelength'][0]):.2f}_cong{float(c['congestion'][0]):.2f}"}
