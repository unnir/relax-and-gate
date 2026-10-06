"""Replica-parallel analytical placement: R basins descended simultaneously.

One Adam optimizer over a [R, N, 2] tensor, one shared set of kernel launches, R different
objective weightings. Because the relaxation's kernels are small and latency-bound (the
single-replica stage leaves the GPU at ~23% utilization), R replicas cost far less than R
times the wall clock -- so every replica gets the FULL analytical time rather than 1/R of
it. That is precisely what the earlier successive-halving experiment could not afford, and
it is why breadth is worth re-testing here even though it lost before.

Selection is by the EXACT engine, never by the relaxed loss.
"""

import time

import torch

from mpx.legalize import push_legalize
from mpx.soft_batch import BatchedSoftProxy

# weightings spread across the (wirelength, density, congestion) simplex. The competition
# weights are (1, .5, .5); the others deliberately over- and under-emphasize the terms so
# the replicas settle into structurally different layouts instead of re-finding one.
WEIGHT_MENU = [
    (1.0, 0.5, 0.5),
    (1.0, 0.5, 1.5),
    (1.0, 0.5, 2.5),
    (1.0, 1.0, 2.5),
    (0.3, 0.5, 1.5),
    (2.0, 0.5, 1.0),
    (1.0, 1.5, 0.5),
    (1.0, 0.25, 3.5),
]


def analytical_replicas(eng, pos, budget_s, params, n_rep=4, log=None):
    p = params
    lr = float(p.get("an_lr", 0.06)) * max(eng.gw, eng.gh)
    check_every = int(p.get("an_check_every", 25))
    ov_w = float(p.get("an_overlap_w", 40.0))
    gamma_f = float(p.get("an_gamma", 0.5))
    legal_iters = int(p.get("an_legal_iters", 120))
    anneal_to = tuple(float(v) for v in p.get("an_w1", (1.0, 0.5, 0.5)))

    menu = [WEIGHT_MENU[i % len(WEIGHT_MENU)] for i in range(n_rep)]
    W0 = torch.tensor(menu, device=eng.device, dtype=torch.float32)
    W1 = torch.tensor([anneal_to] * n_rep, device=eng.device, dtype=torch.float32)

    sp = BatchedSoftProxy(eng, gamma=gamma_f * min(eng.gw, eng.gh))
    x = pos.unsqueeze(0).repeat(n_rep, 1, 1).to(torch.float32)
    # break the tie between replicas so they can separate even under equal weights
    if n_rep > 1:
        jitter = torch.randn_like(x) * (0.05 * min(eng.gw, eng.gh))
        jitter[0] = 0
        x = x + jitter
    x = x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    hw = eng.sizes[:, 0].to(torch.float32) * 0.5
    hh = eng.sizes[:, 1].to(torch.float32) * 0.5
    best = float(eng.evaluate(pos.unsqueeze(0))["proxy"][0])
    best_pos = pos.clone()
    t0 = time.time()
    it = 0
    traj = [(0.0, best)]

    while time.time() - t0 < budget_s:
        frac = min(1.0, (time.time() - t0) / max(budget_s, 1e-9))
        w = W0 + (W1 - W0) * frac
        opt.zero_grad(set_to_none=True)
        loss, _ = sp(x, overlap_w=ov_w, w=w)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            x[:, :, 0] = torch.maximum(torch.minimum(x[:, :, 0], eng.W - hw - 1e-5), hw + 1e-5)
            x[:, :, 1] = torch.maximum(torch.minimum(x[:, :, 1], eng.H - hh - 1e-5), hh + 1e-5)
        it += 1

        if it % check_every == 0:
            with torch.no_grad():
                cand = x.detach().to(eng.dtype)
                cand, _ = push_legalize(cand, eng.sizes, eng.n_hard, eng.W, eng.H,
                                        eps=1e-4, iters=legal_iters)
                c = eng.evaluate(cand)["proxy"]
                k = int(torch.argmin(c))
                if float(c[k]) < best:
                    best, best_pos = float(c[k]), cand[k].clone()
                traj.append((time.time() - t0, best))
                if log:
                    log(f"    anR it{it:4d} best={best:.5f} "
                        f"spread={float(c.max() - c.min()):.4f} winner=rep{k}")

    return best_pos, {"proxy": best, "iters": it, "traj": traj, "n_rep": n_rep}
