"""A differentiable relaxation of the EXACT proxy cost.

Not a hand-designed surrogate. This is the same objective, structurally term for term, with
exactly one thing changed: the two places where the reference quantizes a continuous
position into a grid cell are replaced by their continuous counterparts.

  reference                                    relaxation
  ---------                                    ----------
  pin lands in cell floor(x/gw)                pin splits between the two adjacent cells by
                                               its fractional position (bilinear splat)
  route occupies integer columns               route occupies the CONTINUOUS interval
    [min(cs,cd), max(cs,cd))                   [x_s/gw, x_d/gw); a column gets the length of
                                               its overlap with that interval
  HPWL = max - min over pins                   log-sum-exp softmax/softmin at temperature g

Everything else -- the 0.5 weights, the top-10% density mean, the top-5% ABU over concat(V,H),
the macro blockage with its partial-overlap correction, the +-2 cell smoothing, the H/V
capacity normalization -- is carried over unchanged and stays differentiable as written.

Both relaxations preserve the TOTAL demand a route lays down (the overlap lengths sum to the
interval length, the splat weights sum to one), so the relaxation is unbiased: it does not
systematically under- or over-count congestion, it only spreads it smoothly.

Why it is worth building. The exact objective is piecewise constant in the macro positions,
so its true gradient is zero almost everywhere and carries no information. The relaxation has
an informative gradient everywhere, and one backward pass yields it for ALL ~2600 macros at
once -- versus one batched evaluation per macro per axis for finite differences. It is used to
PROPOSE (global shaping); the exact engine still decides what is ACCEPTED, so the relaxation
never gets to be wrong about the score.

The soft column profile uses a second-difference trick: the overlap of column c with the
interval [a, b] is a trapezoid in c whose second difference is 4 sparse deltas, so each route
costs O(1) scatters plus two cumulative sums over the grid -- the same complexity as the exact
engine, and differentiable in the deltas' weights.
"""

import torch


def _splat_rows(y, gh, R):
    """Bilinear row weights: returns (r0, w0, r1, w1) with w0 + w1 = 1."""
    t = torch.clamp(y / gh - 0.5, 0.0, R - 1.0)
    r0 = torch.floor(t).long().clamp(0, R - 1)
    frac = t - r0.to(t.dtype)
    r1 = (r0 + 1).clamp(max=R - 1)
    return r0, 1.0 - frac, r1, frac


def _interval_deltas(a, b, n):
    """First-difference deltas for f(c) = |[c, c+1] ∩ [a, b]| over c = 0..n-1.

    f is a trapezoid: it ramps up across the column straddling a, sits at 1, and ramps down
    across the column straddling b. Its first difference is four weighted deltas, so ONE
    cumulative sum rebuilds f. Cell indices are integers (non-differentiable, as in any
    splat); the WEIGHTS carry the gradient.
    """
    lo = torch.minimum(a, b).clamp(0.0, float(n))
    hi = torch.maximum(a, b).clamp(0.0, float(n))
    i0 = torch.floor(lo).long().clamp(0, n)
    f0 = lo - i0.to(lo.dtype)
    i1 = torch.floor(hi).long().clamp(0, n)
    f1 = hi - i1.to(hi.dtype)
    # ramp up at lo, ramp down at hi (each spread over the two straddled columns)
    idx = torch.stack([i0, (i0 + 1).clamp(max=n), i1, (i1 + 1).clamp(max=n)], dim=-1)
    val = torch.stack([1.0 - f0, f0, -(1.0 - f1), -f1], dim=-1)
    return idx, val


class SoftProxy:
    """Differentiable relaxation of ProxyEngine.evaluate, sharing its data."""

    def __init__(self, eng, gamma=None, dtype=torch.float32):
        self.e = eng
        self.dt = dtype
        self.gamma = gamma if gamma is not None else 0.5 * min(eng.gw, eng.gh)

    # -------------------------------------------------------------- wirelength
    def wirelength(self, pin_xy):
        e = self.e
        g = self.gamma
        n = e.n_nets
        idx = e.pin_net
        out = 0.0
        for ax in (0, 1):
            z = pin_xy[:, ax] / g
            mx = torch.full((n,), -1e30, device=z.device, dtype=z.dtype)
            mx.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
            mn = torch.full((n,), 1e30, device=z.device, dtype=z.dtype)
            mn.scatter_reduce_(0, idx, z, reduce="amin", include_self=True)
            ep = torch.zeros(n, device=z.device, dtype=z.dtype)
            em = torch.zeros(n, device=z.device, dtype=z.dtype)
            ep.index_add_(0, idx, torch.exp(z - mx[idx]))
            em.index_add_(0, idx, torch.exp(mn[idx] - z))
            out = out + g * ((mx + torch.log(ep)) + (-mn + torch.log(em)))
        return (out * e.net_weight.to(self.dt)).sum() / ((e.W + e.H) * e.net_cnt)

    # ------------------------------------------------------------------ density
    def _axis_overlaps(self, pos):
        e = self.e
        hw = e.sizes[:, 0].to(self.dt) * 0.5
        hh = e.sizes[:, 1].to(self.dt) * 0.5
        x0, x1 = pos[:, 0] - hw, pos[:, 0] + hw
        y0, y1 = pos[:, 1] - hh, pos[:, 1] + hh
        cols = torch.arange(e.C, device=pos.device, dtype=self.dt).view(1, -1)
        rows = torch.arange(e.R, device=pos.device, dtype=self.dt).view(1, -1)
        x_ov = (torch.minimum(x1.view(-1, 1), (cols + 1) * e.gw)
                - torch.maximum(x0.view(-1, 1), cols * e.gw)).clamp(min=0)
        y_ov = (torch.minimum(y1.view(-1, 1), (rows + 1) * e.gh)
                - torch.maximum(y0.view(-1, 1), rows * e.gh)).clamp(min=0)
        return x_ov, y_ov

    def density(self, x_ov, y_ov):
        e = self.e
        occ = torch.einsum("mr,mc->rc", y_ov, x_ov) / (e.gw * e.gh)
        flat = occ.reshape(-1)
        top = torch.topk(flat, e.top_density_k).values
        return 0.5 * top.sum() / e.top_density_k

    # --------------------------------------------------------------- congestion
    def congestion(self, pin_xy, x_ov, y_ov):
        e = self.e
        dev = pin_xy.device
        R, C = e.R, e.C

        # ---- net routing, relaxed ------------------------------------------
        # source pin of every net, and every other pin as a sink (star model). The exact
        # engine deduplicates gcells first; the relaxation cannot, so it keeps every pin --
        # which over-counts coincident pins slightly but is smooth, and acceptance is
        # decided by the exact engine anyway.
        src = e.driver_pin
        sx = (pin_xy[src, 0] / e.gw)[e.pin_net]
        sy = pin_xy[src, 1]
        dx = pin_xy[:, 0] / e.gw
        dy = pin_xy[:, 1]
        w = e.net_weight.to(self.dt)[e.pin_net]
        keep = torch.ones_like(w)
        keep[src] = 0.0                                  # a net's driver is not its own sink
        w = w * keep

        Hd = torch.zeros(R, C + 2, device=dev, dtype=self.dt)
        r0, w0, r1, w1 = _splat_rows(sy[e.pin_net], e.gh, R)
        cidx, cval = _interval_deltas(sx, dx, C)
        for rr, ww in ((r0, w0), (r1, w1)):
            flat = (rr.view(-1, 1) * (C + 2) + cidx).reshape(-1)
            vals = (cval * (w * ww).view(-1, 1)).reshape(-1)
            Hd = Hd.reshape(-1).index_add(0, flat, vals).reshape(R, C + 2)
        Hm = torch.cumsum(Hd, dim=1)[:, :C]

        Vd = torch.zeros(C, R + 2, device=dev, dtype=self.dt)
        c0, v0, c1, v1 = _splat_rows(pin_xy[:, 0], e.gw, C)
        ridx, rval = _interval_deltas(sy[e.pin_net] / e.gh, dy / e.gh, R)
        for cc, ww in ((c0, v0), (c1, v1)):
            flat = (cc.view(-1, 1) * (R + 2) + ridx).reshape(-1)
            vals = (rval * (w * ww).view(-1, 1)).reshape(-1)
            Vd = Vd.reshape(-1).index_add(0, flat, vals).reshape(C, R + 2)
        Vm = torch.cumsum(Vd, dim=1)[:, :R].transpose(0, 1)

        Hm = Hm / e.grid_h_routes
        Vm = Vm / e.grid_v_routes

        # ---- smoothing (identical to the reference) -------------------------
        sr = e.smooth_range
        if sr > 0:
            cols = torch.arange(C, device=dev)
            lp = torch.clamp(cols - sr, min=0)
            rp = torch.clamp(cols + sr, max=C - 1)
            cnt = (rp - lp + 1).to(self.dt)
            Vs = torch.zeros(R, C + 1, device=dev, dtype=self.dt)
            Vs = Vs.reshape(-1).index_add(
                0, (torch.arange(R, device=dev).view(-1, 1) * (C + 1) + lp.view(1, -1)).reshape(-1),
                (Vm / cnt.view(1, -1)).reshape(-1)).reshape(R, C + 1)
            Vs = Vs.reshape(-1).index_add(
                0, (torch.arange(R, device=dev).view(-1, 1) * (C + 1) + (rp + 1).view(1, -1)).reshape(-1),
                (-Vm / cnt.view(1, -1)).reshape(-1)).reshape(R, C + 1)
            Vm = torch.cumsum(Vs, dim=1)[:, :C]

            rws = torch.arange(R, device=dev)
            lpr = torch.clamp(rws - sr, min=0)
            upr = torch.clamp(rws + sr, max=R - 1)
            cntr = (upr - lpr + 1).to(self.dt)
            Hs = torch.zeros(R + 1, C, device=dev, dtype=self.dt)
            Hs = Hs.reshape(-1).index_add(
                0, (lpr.view(-1, 1) * C + torch.arange(C, device=dev).view(1, -1)).reshape(-1),
                (Hm / cntr.view(-1, 1)).reshape(-1)).reshape(R + 1, C)
            Hs = Hs.reshape(-1).index_add(
                0, ((upr + 1).view(-1, 1) * C + torch.arange(C, device=dev).view(1, -1)).reshape(-1),
                (-Hm / cntr.view(-1, 1)).reshape(-1)).reshape(R + 1, C)
            Hm = torch.cumsum(Hs, dim=0)[:R, :]

        # ---- hard-macro blockage (already differentiable, separable) ---------
        nh = e.n_hard
        xg, yg = x_ov[:nh], y_ov[:nh]
        Vb = torch.einsum("mr,mc->rc", (yg > 0).to(self.dt), xg) * e.v_alloc / e.grid_v_routes
        Hb = torch.einsum("mr,mc->rc", yg, (xg > 0).to(self.dt)) * e.h_alloc / e.grid_h_routes

        both = torch.cat([(Vm + Vb).reshape(-1), (Hm + Hb).reshape(-1)])
        top = torch.topk(both, e.top_cong_k).values
        return top.sum() / e.top_cong_k

    # -------------------------------------------------------------------- total
    def __call__(self, pos, overlap_w=0.0, eps=1e-4, w=(1.0, 0.5, 0.5)):
        """w = (w_wl, w_den, w_cong). The competition weights are (1, 0.5, 0.5); the
        optimizer is free to descend a DIFFERENT weighting, because what gets kept is
        decided by the exact engine under the real weights either way. That makes the
        weighting a search direction rather than a definition of success."""
        e = self.e
        pin_xy = e.pin_positions(pos.unsqueeze(0).to(e.dtype))[0].to(self.dt)
        x_ov, y_ov = self._axis_overlaps(pos)
        wl = self.wirelength(pin_xy)
        den = self.density(x_ov, y_ov)
        cong = self.congestion(pin_xy, x_ov, y_ov)
        loss = w[0] * wl + w[1] * den + w[2] * cong
        if overlap_w > 0:
            nh = e.n_hard
            p = pos[:nh]
            s = e.sizes[:nh].to(self.dt)
            dx = (p[:, None, 0] - p[None, :, 0]).abs()
            dy = (p[:, None, 1] - p[None, :, 1]).abs()
            ox = ((s[:, 0][:, None] + s[:, 0][None, :]) * 0.5 + eps - dx).clamp(min=0)
            oy = ((s[:, 1][:, None] + s[:, 1][None, :]) * 0.5 + eps - dy).clamp(min=0)
            iu = torch.triu(torch.ones(nh, nh, device=pos.device, dtype=torch.bool), 1)
            loss = loss + overlap_w * (ox * oy * iu).sum() / (e.W * e.H)
        return loss, {"wl": wl, "den": den, "cong": cong}
