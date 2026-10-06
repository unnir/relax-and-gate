"""Gate: our fast engine must agree with the OFFICIAL evaluator.

AGENTS.md: calibrate the instrument on a known answer before trusting it. Here the
known answer is the official Python evaluator itself, so we compare component by
component (wirelength / density / congestion / proxy) on several placements per
benchmark -- the initial placement, a greedy-row placement, and random legal ones --
because a bug that only shows up off the initial placement is exactly the bug that
would silently mis-rank every candidate the search generates.

    python mpx/validate.py ibm01 [ibm02 ...]
"""

import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "challenge"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "external", "MacroPlacement",
                                "CodeElements", "Plc_client"))

from mpx.engine import ProxyEngine       # noqa: E402
from mpx.extract import load_plc         # noqa: E402


def official(plc, pos_np, n_hard):
    """Score a placement with the untouched official evaluator."""
    import math
    # apply the challenge's boundary clamp patch, exactly as macro_place.objective does
    from plc_client_os import PlacementCost

    def patched(self, x_pos, y_pos):
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)
        row = math.floor(y_pos / self.grid_height)
        col = math.floor(x_pos / self.grid_width)
        return max(0, min(row, self.grid_row - 1)), max(0, min(col, self.grid_col - 1))

    PlacementCost._PlacementCost__get_grid_cell_location = patched

    if not hasattr(plc, "_pin_map"):
        pm = {}
        for i, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
                pm.setdefault(mod.get_macro_name(), []).append(i)
        plc._pin_map = pm

    order = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)
    for i, midx in enumerate(order):
        node = plc.modules_w_pins[midx]
        x, y = float(pos_np[i, 0]), float(pos_np[i, 1])
        node.set_pos(x, y)
        for pi in plc._pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pi]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    n = plc.grid_col * plc.grid_row
    plc.V_routing_cong = [0] * n
    plc.H_routing_cong = [0] * n
    plc.V_macro_routing_cong = [0] * n
    plc.H_macro_routing_cong = [0] * n
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True

    wl = plc.get_cost()
    den = plc.get_density_cost()
    cong = plc.get_congestion_cost()
    return dict(wirelength=wl, density=den, congestion=cong, proxy=wl + 0.5 * den + 0.5 * cong)


def make_variants(eng, seed=0):
    """Initial + a few structurally different placements (the ones a search would visit)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = eng.N
    out = {"initial": eng.pos0.cpu().numpy()}

    hw = (eng.sizes[:, 0] * 0.5).cpu().numpy()
    hh = (eng.sizes[:, 1] * 0.5).cpu().numpy()
    for tag in ("random", "clustered", "corner"):
        p = eng.pos0.cpu().numpy().copy()
        if tag == "random":
            p[:, 0] = np.random.RandomState(seed).uniform(hw, eng.W - hw)
            p[:, 1] = np.random.RandomState(seed + 1).uniform(hh, eng.H - hh)
        elif tag == "clustered":
            p[:, 0] = np.clip(p[:, 0] * 0.5 + eng.W * 0.25, hw, eng.W - hw)
            p[:, 1] = np.clip(p[:, 1] * 0.5 + eng.H * 0.25, hh, eng.H - hh)
        else:
            p[:, 0] = np.where(np.arange(N) % 2 == 0, hw, eng.W - hw)
            p[:, 1] = np.clip(p[:, 1], hh, eng.H - hh)
        out[tag] = p
    return out


def main(names):
    ok_all = True
    for name in names:
        npz = os.path.join(ROOT, "data", f"{name}.npz")
        eng = ProxyEngine(npz, device="cuda" if torch.cuda.is_available() else "cpu")
        plc = load_plc(name)
        variants = make_variants(eng)

        # batch every variant through our engine in ONE call (also tests batching)
        batch = torch.tensor(np.stack(list(variants.values())), dtype=torch.float64)
        mine = eng.evaluate(batch)

        print(f"\n=== {name}  (grid {eng.R}x{eng.C}, {eng.n_hard} hard + {eng.n_soft} soft, "
              f"{eng.n_nets} nets, net_cnt {eng.net_cnt:.0f})")
        for i, (tag, p) in enumerate(variants.items()):
            ref = official(plc, p, eng.n_hard)
            row, bad = [], False
            for k in ("wirelength", "density", "congestion", "proxy"):
                a, b = float(mine[k][i]), float(ref[k])
                rel = abs(a - b) / max(abs(b), 1e-12)
                if rel > 1e-9:
                    bad = True
                row.append(f"{k[:4]} {a:.9f}/{b:.9f} rel={rel:.2e}")
            ok_all &= not bad
            print(("  OK   " if not bad else "  FAIL ") + tag.ljust(10) + " | " + "  ".join(row))
    print("\nALL MATCH" if ok_all else "\nMISMATCH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["ibm01"]))
