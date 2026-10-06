"""Standalone re-verification of a placement artifact.  FIXED — do not edit.

Imports NOTHING from the search code. It reads the artifact's coordinates, reloads each
benchmark from the original ICCAD04 netlist with the UNTOUCHED official TILOS evaluator,
recomputes the proxy cost and the overlap check from those coordinates alone, and
cross-checks the recomputed numbers against the values the artifact claims.

That last check is the point: a validity check alone cannot catch an artifact that has
drifted from the score it is reported under. Exit 0 iff every circuit is legal AND every
claimed proxy matches the official recomputation.

    python verify.py [artifacts/best_placement.json] [--tol 1e-6]
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PLC_DIR = os.path.join(ROOT, "challenge", "external", "MacroPlacement",
                       "CodeElements", "Plc_client")
TESTCASES = os.path.join(ROOT, "challenge", "external", "MacroPlacement",
                         "Testcases", "ICCAD04")
sys.path.insert(0, PLC_DIR)

from plc_client_os import PlacementCost   # noqa: E402


def _patch_bounds():
    """The challenge's own boundary-clamp patch to __get_grid_cell_location."""
    def patched(self, x_pos, y_pos):
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)
        row = math.floor(y_pos / self.grid_height)
        col = math.floor(x_pos / self.grid_width)
        return max(0, min(row, self.grid_row - 1)), max(0, min(col, self.grid_col - 1))
    PlacementCost._PlacementCost__get_grid_cell_location = patched


def official_score(name, positions):
    """Recompute (proxy, wl, den, cong, overlaps) from coordinates using the real evaluator."""
    _patch_bounds()
    d = os.path.join(TESTCASES, name)
    plc = PlacementCost(os.path.join(d, "netlist.pb.txt"))
    plc.restore_placement(os.path.join(d, "initial.plc"), ifInital=True, ifReadComment=True)

    order = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)
    if len(positions) != len(order):
        raise ValueError(f"{name}: artifact has {len(positions)} macros, netlist has {len(order)}")

    pin_map = {}
    for i, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
            pin_map.setdefault(mod.get_macro_name(), []).append(i)

    W, H = plc.get_canvas_width_height()
    n_hard = len(plc.hard_macro_indices)
    boxes = []
    for i, midx in enumerate(order):
        node = plc.modules_w_pins[midx]
        x, y = float(positions[i][0]), float(positions[i][1])
        if node.get_fix_flag():
            ox, oy = node.get_pos()
            if abs(ox - x) > 1e-9 or abs(oy - y) > 1e-9:
                raise ValueError(f"{name}: fixed macro {node.get_name()} was moved")
        node.set_pos(x, y)
        for pi in pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pi]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)
        if i < n_hard:
            boxes.append((x, y, node.get_width(), node.get_height(), node.get_name()))

    # bounds
    oob = [b[4] for b in boxes
           if b[0] - b[2] / 2 < -1e-9 or b[0] + b[2] / 2 > W + 1e-9
           or b[1] - b[3] / 2 < -1e-9 or b[1] + b[3] / 2 > H + 1e-9]

    # exhaustive pairwise hard-macro overlap, recomputed from coordinates alone
    n_ov, ov_area = 0, 0.0
    for i in range(len(boxes)):
        xi, yi, wi, hi, _ = boxes[i]
        for j in range(i + 1, len(boxes)):
            xj, yj, wj, hj, _ = boxes[j]
            ox = (wi + wj) / 2 - abs(xi - xj)
            oy = (hi + hj) / 2 - abs(yi - yj)
            if ox > 0 and oy > 0:
                n_ov += 1
                ov_area += ox * oy

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
    return dict(proxy=wl + 0.5 * den + 0.5 * cong, wirelength=wl, density=den,
                congestion=cong, overlap_count=n_ov, overlap_area=ov_area,
                out_of_bounds=len(oob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="?",
                    default=os.path.join(ROOT, "artifacts", "best_placement.json"))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    with open(a.artifact) as f:
        art = json.load(f)
    # accept both artifact schemas: evaluate.py writes {"circuits": {name: {...}}},
    # run_full.py writes {"rows": [{"circuit":..., "seed":..., "positions":...}, ...]}
    if "circuits" in art:
        circuits = art["circuits"]
    else:
        circuits = {}
        for r in art["rows"]:
            key = r["circuit"] if r.get("seed", 0) == 0 else f"{r['circuit']}#s{r['seed']}"
            circuits[key] = r
    names = a.only or list(circuits)

    ok = True
    proxies = []
    print(f"verifying {a.artifact}  ({len(names)} circuits)")
    for name in names:
        rec = circuits[name]
        got = official_score(name.split("#")[0], rec["positions"])
        claimed = float(rec["proxy"])
        drift = abs(got["proxy"] - claimed)
        legal = got["overlap_count"] == 0 and got["out_of_bounds"] == 0
        good = legal and drift <= a.tol
        ok &= good
        proxies.append(got["proxy"])
        print(f"  {'OK  ' if good else 'FAIL'} {name:8} official={got['proxy']:.6f} "
              f"claimed={claimed:.6f} drift={drift:.2e} overlaps={got['overlap_count']} "
              f"oob={got['out_of_bounds']}", flush=True)

    print(f"official mean proxy = {sum(proxies)/len(proxies):.6f}")
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
