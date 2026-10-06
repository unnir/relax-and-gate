"""Replica-batched version of the differentiable relaxation.

Motivation, from a measured failure. Basin selection lost (0.93634 with 3 probes vs 0.92747
with one deep run) for a specific reason: the analytical stage needs ~30 s to converge, so
affording several probes meant making each of them too short to rank honestly. The probe
budget came straight out of the stage that was paying.

But the analytical stage leaves the GPU at ~23% utilization -- its kernels are small and
latency-bound. So R replicas can share one set of kernel launches and cost far less than
R times the wall clock, which lets every replica run for the FULL time instead of 1/R of
it. That removes the exact reason breadth lost, so breadth is worth re-testing rather than
assumed dead.

Each replica may carry its own objective weighting, which is what makes them land in
structurally different layouts rather than re-finding the same one.
"""

import torch


def _splat_rows(y, gh, R):
    t = torch.clamp(y / gh - 0.5, 0.0, R - 1.0)
    r0 = torch.floor(t).long().clamp(0, R - 1)
    frac = t - r0.to(t.dtype)
    r1 = (r0 + 1).clamp(max=R - 1)
    return r0, 1.0 - frac, r1, frac


def _interval_deltas(a, b, n):
    lo = torch.minimum(a, b).clamp(0.0, float(n))
    hi = torch.maximum(a, b).clamp(0.0, float(n))
    i0 = torch.floor(lo).long().clamp(0, n)
    f0 = lo - i0.to(lo.dtype)
    i1 = torch.floor(hi).long().clamp(0, n)
    f1 = hi - i1.to(hi.dtype)
    idx = torch.stack([i0, (i0 + 1).clamp(max=n), i1, (i1 + 1).clamp(max=n)], dim=-1)
    val = torch.stack([1.0 - f0, f0, -(1.0 - f1), -f1], dim=-1)
    return idx, val


class BatchedSoftProxy:
    """Relaxed proxy for [B, N, 2] placements at once. Returns [B] losses."""

    def __init__(self, eng, gamma=None, dtype=torch.float32):
        self.e = eng
        self.dt = dtype
        self.gamma = gamma if gamma is not None else 0.5 * min(eng.gw, eng.gh)

    def _wirelength(self, pin_xy):
        e, g = self.e, self.gamma
        B, P, _ = pin_xy.shape
        n = e.n_nets
        idx = e.pin_net.view(1, -1).expand(B, -1)
        out = 0.0
        for ax in (0, 1):
            z = pin_xy[:, :, ax] / g
            mx = torch.full((B, n), -1e30, device=z.device, dtype=z.dtype)
            mx.scatter_reduce_(1, idx, z, reduce="amax", include_self=True)
            mn = torch.full((B, n), 1e30, device=z.device, dtype=z.dtype)
            mn.scatter_reduce_(1, idx, z, reduce="amin", include_self=True)
            ep = torch.zeros(B, n, device=z.device, dtype=z.dtype)
            em = torch.zeros(B, n, device=z.device, dtype=z.dtype)
            ep.scatter_add_(1, idx, torch.exp(z - mx.gather(1, idx)))
            em.scatter_add_(1, idx, torch.exp(mn.gather(1, idx) - z))
            out = out + g * ((mx + torch.log(ep)) + (-mn + torch.log(em)))
        return (out * e.net_weight.to(self.dt).view(1, -1)).sum(1) / ((e.W + e.H) * e.net_cnt)

    def _axis_overlaps(self, pos):
        e = self.e
        hw = e.sizes[:, 0].to(self.dt) * 0.5
        hh = e.sizes[:, 1].to(self.dt) * 0.5
        x0, x1 = pos[:, :, 0] - hw, pos[:, :, 0] + hw
        y0, y1 = pos[:, :, 1] - hh, pos[:, :, 1] + hh
        cols = torch.arange(e.C, device=pos.device, dtype=self.dt).view(1, 1, -1)
        rows = torch.arange(e.R, device=pos.device, dtype=self.dt).view(1, 1, -1)
        x_ov = (torch.minimum(x1.unsqueeze(2), (cols + 1) * e.gw)
                - torch.maximum(x0.unsqueeze(2), cols * e.gw)).clamp(min=0)
        y_ov = (torch.minimum(y1.unsqueeze(2), (rows + 1) * e.gh)
                - torch.maximum(y0.unsqueeze(2), rows * e.gh)).clamp(min=0)
        return x_ov, y_ov

    def _density(self, x_ov, y_ov):
        e = self.e
        occ = torch.einsum("bmr,bmc->brc", y_ov, x_ov) / (e.gw * e.gh)
        top = torch.topk(occ.reshape(occ.shape[0], -1), e.top_density_k, dim=1).values
        return 0.5 * top.sum(1) / e.top_density_k

    def _congestion(self, pin_xy, x_ov, y_ov):
        e = self.e
        dev = pin_xy.device
        B = pin_xy.shape[0]
        R, C = e.R, e.C

        src = e.driver_pin
        pn = e.pin_net
        sx = (pin_xy[:, src, 0] / e.gw).gather(1, pn.view(1, -1).expand(B, -1))
        sy = pin_xy[:, src, 1].gather(1, pn.view(1, -1).expand(B, -1))
        dx = pin_xy[:, :, 0] / e.gw
        dy = pin_xy[:, :, 1]
        w = e.net_weight.to(self.dt)[pn].view(1, -1).expand(B, -1).clone()
        w[:, src] = 0.0

        rep = torch.arange(B, device=dev).view(-1, 1, 1)

        # --- H demand: source's row, continuous column interval -----------------
        Hd = torch.zeros(B * R * (C + 2), device=dev, dtype=self.dt)
        r0, w0, r1, w1 = _splat_rows(sy, e.gh, R)
        cidx, cval = _interval_deltas(sx, dx, C)
        for rr, ww in ((r0, w0), (r1, w1)):
            flat = (rep * (R * (C + 2)) + rr.unsqueeze(2) * (C + 2) + cidx).reshape(-1)
            Hd = Hd.index_add(0, flat, (cval * (w * ww).unsqueeze(2)).reshape(-1))
        Hm = torch.cumsum(Hd.view(B, R, C + 2), dim=2)[:, :, :C] / e.grid_h_routes

        # --- V demand: sink's column, continuous row interval --------------------
        Vd = torch.zeros(B * C * (R + 2), device=dev, dtype=self.dt)
        c0, v0, c1, v1 = _splat_rows(pin_xy[:, :, 0], e.gw, C)
        ridx, rval = _interval_deltas(sy / e.gh, dy / e.gh, R)
        for cc, ww in ((c0, v0), (c1, v1)):
            flat = (rep * (C * (R + 2)) + cc.unsqueeze(2) * (R + 2) + ridx).reshape(-1)
            Vd = Vd.index_add(0, flat, (rval * (w * ww).unsqueeze(2)).reshape(-1))
        Vm = torch.cumsum(Vd.view(B, C, R + 2), dim=2)[:, :, :R].transpose(1, 2)
        Vm = Vm / e.grid_v_routes

        # --- smoothing (same kernel as the reference) ----------------------------
        sr = e.smooth_range
        if sr > 0:
            cols = torch.arange(C, device=dev)
            lp = torch.clamp(cols - sr, min=0)
            rp = torch.clamp(cols + sr, max=C - 1)
            cnt = (rp - lp + 1).to(self.dt)
            base = torch.zeros(B, R, C + 1, device=dev, dtype=self.dt)
            src_v = Vm / cnt.view(1, 1, -1)
            base = base.scatter_add(2, lp.view(1, 1, -1).expand(B, R, C), src_v)
            base = base.scatter_add(2, (rp + 1).view(1, 1, -1).expand(B, R, C), -src_v)
            Vm = torch.cumsum(base, dim=2)[:, :, :C]

            rws = torch.arange(R, device=dev)
            lpr = torch.clamp(rws - sr, min=0)
            upr = torch.clamp(rws + sr, max=R - 1)
            cntr = (upr - lpr + 1).to(self.dt)
            baseh = torch.zeros(B, R + 1, C, device=dev, dtype=self.dt)
            src_h = Hm / cntr.view(1, -1, 1)
            baseh = baseh.scatter_add(1, lpr.view(1, -1, 1).expand(B, R, C), src_h)
            baseh = baseh.scatter_add(1, (upr + 1).view(1, -1, 1).expand(B, R, C), -src_h)
            Hm = torch.cumsum(baseh, dim=1)[:, :R, :]

        nh = e.n_hard
        xg, yg = x_ov[:, :nh], y_ov[:, :nh]
        Vb = torch.einsum("bmr,bmc->brc", (yg > 0).to(self.dt), xg) * e.v_alloc / e.grid_v_routes
        Hb = torch.einsum("bmr,bmc->brc", yg, (xg > 0).to(self.dt)) * e.h_alloc / e.grid_h_routes

        both = torch.cat([(Vm + Vb).reshape(B, -1), (Hm + Hb).reshape(B, -1)], dim=1)
        top = torch.topk(both, e.top_cong_k, dim=1).values
        return top.sum(1) / e.top_cong_k

    def __call__(self, pos, overlap_w=0.0, eps=1e-4, w=None):
        """pos [B,N,2]; w is [B,3] (per-replica weighting) or a 3-tuple shared by all."""
        e = self.e
        B = pos.shape[0]
        pin_xy = e.pin_positions(pos.to(e.dtype)).to(self.dt)
        x_ov, y_ov = self._axis_overlaps(pos)
        wl = self._wirelength(pin_xy)
        den = self._density(x_ov, y_ov)
        cong = self._congestion(pin_xy, x_ov, y_ov)
        if w is None:
            w = (1.0, 0.5, 0.5)
        W = torch.as_tensor(w, device=pos.device, dtype=self.dt)
        if W.dim() == 1:
            W = W.view(1, 3).expand(B, 3)
        loss = W[:, 0] * wl + W[:, 1] * den + W[:, 2] * cong
        if overlap_w > 0:
            nh = e.n_hard
            p = pos[:, :nh]
            s = e.sizes[:nh].to(self.dt)
            dx = (p[:, :, None, 0] - p[:, None, :, 0]).abs()
            dy = (p[:, :, None, 1] - p[:, None, :, 1]).abs()
            ox = ((s[:, 0][:, None] + s[:, 0][None, :]) * 0.5 + eps - dx).clamp(min=0)
            oy = ((s[:, 1][:, None] + s[:, 1][None, :]) * 0.5 + eps - dy).clamp(min=0)
            iu = torch.triu(torch.ones(nh, nh, device=pos.device, dtype=torch.bool), 1)
            loss = loss + overlap_w * (ox * oy * iu).sum(dim=(1, 2)) / (e.W * e.H)
        return loss, {"wl": wl, "den": den, "cong": cong}
