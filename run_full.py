"""Run the frozen method over a split (or all 17 circuits) and emit the paper's tables.

    python run_full.py --split all --budget 900 --seeds 0 1 2 --tag final

Writes, under artifacts/<tag>/:
    placements_<circuit>_seed<k>.npy   the raw coordinates (the thing verify.py re-checks)
    results.json                       every run, every component, runtime, legality
    table_main.md / table_components.md
Reports mean / std / best / median across seeds, never only the lucky run.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import benchmark as B                                     # noqa: E402
import solution                                           # noqa: E402
from mpx.legalize import legalize_batch, clamp_bounds     # noqa: E402

# published Tier-1 numbers -- REPORTED by the challenge/leaderboard, never reproduced here
PUBLISHED = {
    "SA": 2.1251, "RePlAce": 1.4578, "GreedyRow": 2.2109,
    "Archgen(rank1)": 0.9507, "Carrotato(rank2)": 0.9522, "JaneRT(rank3)": 0.9694,
}


def run(name, params, budget, seed, out_dir, tag):
    eng = B.get_engine(name)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pos = solution.place(eng, params, rng, budget_s=budget, seed=seed)
    elapsed = time.time() - t0

    pos = clamp_bounds(torch.as_tensor(pos, device=eng.device, dtype=eng.dtype).unsqueeze(0), eng)
    ok, det = B.legality(eng, pos[0])
    if not ok:
        pos = legalize_batch(pos, eng)
        ok, det = B.legality(eng, pos[0])
    c = eng.evaluate(pos)
    arr = pos[0].detach().cpu().numpy()
    np.save(os.path.join(out_dir, f"placement_{name}_seed{seed}.npy"), arr)
    return {
        "circuit": name, "seed": seed, "budget_s": budget,
        "proxy": float(c["proxy"][0]), "wirelength": float(c["wirelength"][0]),
        "density": float(c["density"][0]), "congestion": float(c["congestion"][0]),
        "legal": bool(ok), "elapsed_s": elapsed, **det,
        "positions": arr.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--circuits", nargs="*", default=None)
    a = ap.parse_args()

    names = a.circuits or B.split_names(a.split)
    out_dir = os.path.join(ROOT, "artifacts", a.tag)
    os.makedirs(out_dir, exist_ok=True)

    params = dict(solution.DEFAULT_PARAMS)
    bp = os.path.join(ROOT, "best_params.json")
    if os.path.exists(bp):
        params.update(json.load(open(bp)))

    rows = []
    for n in names:
        for s in a.seeds:
            r = run(n, params, a.budget, s, out_dir, a.tag)
            rows.append(r)
            print(f"{n:8} seed{s} proxy={r['proxy']:.5f} wl={r['wirelength']:.4f} "
                  f"den={r['density']:.4f} cong={r['congestion']:.4f} "
                  f"legal={r['legal']} {r['elapsed_s']:.0f}s", flush=True)

    # ---- aggregate ------------------------------------------------------
    per = {}
    for r in rows:
        per.setdefault(r["circuit"], []).append(r["proxy"])
    summary = {}
    for n, v in per.items():
        summary[n] = {"mean": statistics.mean(v), "best": min(v),
                      "median": statistics.median(v),
                      "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
                      "n_seeds": len(v)}

    means = [s["mean"] for s in summary.values()]
    bests = [s["best"] for s in summary.values()]
    agg = {"mean_of_means": statistics.mean(means), "mean_of_bests": statistics.mean(bests),
           "split": a.split, "budget_s": a.budget, "seeds": a.seeds,
           "circuits": names, "all_legal": all(r["legal"] for r in rows)}

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary, "aggregate": agg,
                   "params": params, "published": PUBLISHED}, f)

    lines = ["| Circuit | split | seeds | mean | std | best | median |",
             "|---|---|--:|--:|--:|--:|--:|"]
    for n in names:
        s = summary[n]
        sp = "dev" if n in B.DEV else ("val" if n in B.VAL else "test")
        lines.append(f"| {n} | {sp} | {s['n_seeds']} | {s['mean']:.4f} | {s['std']:.4f} | "
                     f"{s['best']:.4f} | {s['median']:.4f} |")
    lines.append(f"| **mean** | {a.split} | | **{agg['mean_of_means']:.4f}** | | "
                 f"**{agg['mean_of_bests']:.4f}** | |")
    with open(os.path.join(out_dir, "table_main.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nmean-of-means={agg['mean_of_means']:.4f}  mean-of-bests={agg['mean_of_bests']:.4f}"
          f"  all_legal={agg['all_legal']}")
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
