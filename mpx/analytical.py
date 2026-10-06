"""Analytical global placement on the differentiable relaxation of the true proxy.

Adam on `SoftProxy`, which is the competition objective with only its two quantization steps
dequantized. One backward pass moves every macro, so this reshapes the layout globally in a
way single-macro local search cannot reach in any budget.

The relaxation is used to PROPOSE only. Every few steps the exact engine scores the current
iterate (after legalization), and only exactly-verified improvements are kept -- so a
mis-modelled term in the relaxation can cost time but can never corrupt the result.
"""

import time

import torch

from mpx.legalize import push_legalize, clamp_bounds
from mpx.soft import SoftProxy


def analytical_place(eng, pos, budget_s, params, log=None):
    p = params
    lr = float(p.get("an_lr", 0.06)) * max(eng.gw, eng.gh)
    steps = int(p.get("an_steps", 400))
    check_every = int(p.get("an_check_every", 25))
    ov_w = float(p.get("an_overlap_w", 40.0))
    gamma_f = float(p.get("an_gamma", 0.5))
    legal_iters = int(p.get("an_legal_iters", 120))

    # objective-weight schedule: (w_wl, w_den, w_cong) linearly interpolated from
    # `an_w0` to `an_w1` over the run. The competition weights are (1, .5, .5); the
    # hypothesis under test is that a different weighting EARLY reaches a better basin.
    w0 = tuple(float(v) for v in p.get("an_w0", (1.0, 0.5, 0.5)))
    w1 = tuple(float(v) for v in p.get("an_w1", w0))
    # Graduated non-convexity: the relaxation starts SMOOTH and sharpens toward the exact
    # objective. Early on a soft log-sum-exp HPWL and a wide effective footprint let macros
    # slide past each other and reorganize globally; by the end gamma is small enough that
    # the relaxation and the real objective agree closely, so the last iterates are the
    # ones the exact gate is most likely to keep.
    gamma_end = float(p.get("an_gamma_end", gamma_f))
    sp = SoftProxy(eng, gamma=gamma_f * min(eng.gw, eng.gh))
    x = pos.clone().to(torch.float32).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    best_pos = pos.clone()
    best = float(eng.evaluate(pos.unsqueeze(0))["proxy"][0])
    t0 = time.time()
    hw = eng.sizes[:, 0].to(torch.float32) * 0.5
    hh = eng.sizes[:, 1].to(torch.float32) * 0.5

    for it in range(steps):
        if time.time() - t0 > budget_s:
            break
        frac = min(1.0, (time.time() - t0) / max(budget_s, 1e-9))
        w = tuple(a + (b - a) * frac for a, b in zip(w0, w1))
        sp.gamma = (gamma_f * (gamma_end / gamma_f) ** frac) * min(eng.gw, eng.gh)
        opt.zero_grad(set_to_none=True)
        loss, parts = sp(x, overlap_w=ov_w, w=w)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x[:, 0].clamp_(min=0).clamp_(max=eng.W)
            x[:, 1].clamp_(min=0).clamp_(max=eng.H)
            x[:, 0] = torch.maximum(torch.minimum(x[:, 0], eng.W - hw - 1e-5), hw + 1e-5)
            x[:, 1] = torch.maximum(torch.minimum(x[:, 1], eng.H - hh - 1e-5), hh + 1e-5)

        if (it + 1) % check_every == 0:
            with torch.no_grad():
                cand = x.detach().to(eng.dtype).unsqueeze(0)
                cand, _ = push_legalize(cand, eng.sizes, eng.n_hard, eng.W, eng.H,
                                        eps=1e-4, iters=legal_iters)
                c = float(eng.evaluate(cand)["proxy"][0])
                if c < best:
                    best, best_pos = c, cand[0].clone()
                if log:
                    log(f"    an it{it+1:4d} soft={float(loss):.4f} exact={c:.5f} best={best:.5f}")
    return best_pos, {"proxy": best, "steps": it + 1}
