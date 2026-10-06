"""Incremental exact scoring for single-macro moves.

The batched engine scores B candidate placements from scratch. But in the discrete stage
every candidate differs from the incumbent in exactly ONE macro, so almost all of that work
is recomputation of things that did not change. The dominant cost is the segmented sort over
all pins ([B, n_pins]); on ibm17 that is 133,186 pins per candidate when only ~40 of them
moved.

This module keeps the incumbent's demand DIFFERENCE arrays and, per candidate, subtracts the
routes of the affected nets at their old positions and adds them back at the new ones. Cost
per candidate drops from O(n_pins) to O(affected_pins + R*C), where R*C is a few thousand
cells -- roughly a 40x reduction on the largest circuits, and those are exactly the circuits
where the method is weakest.

The result is EXACT, not approximate: the same routing rules, the same cumulative sums, the
same top-k statistics. It is verified against the full engine in `selftest()`, and the search
still re-scores every accepted move with the full engine, so drift cannot accumulate.
"""

import numpy as np
import torch

from mpx.engine import _clamped_cell


class IncrementalScorer:
    """Score single-macro moves against a fixed incumbent placement."""

    def __init__(self, eng):
        self.e = eng
        self._build_macro_nets()

    # ---------------------------------------------------------------- static
    def _build_macro_nets(self):
        """macro -> the pins of every net it touches (padded), built once."""
        e = self.e
        owner = e.pin_owner_macro.detach().cpu().numpy()
        net = e.pin_net.detach().cpu().numpy()
        isport = e.pin_is_port.detach().cpu().numpy()

        nets_of = [[] for _ in range(e.N)]
        for o, n, ip in zip(owner, net, isport):
            if not ip:
                nets_of[int(o)].append(int(n))
        nets_of = [sorted(set(v)) for v in nets_of]

        order = np.argsort(net, kind="stable")
        net_sorted = net[order]
        starts = np.searchsorted(net_sorted, np.arange(e.n_nets), side="left")
        ends = np.searchsorted(net_sorted, np.arange(e.n_nets), side="right")

        pins_of, netid_of = [], []
        for m in range(e.N):
            pins, nids = [], []
            for n in nets_of[m]:
                idx = order[starts[n]:ends[n]]
                pins.extend(idx.tolist())
                nids.extend([n] * len(idx))
            pins_of.append(pins)
            netid_of.append(nids)

        self.max_pins = max((len(p) for p in pins_of), default=0)
        self.max_nets = max((len(v) for v in nets_of), default=0)
        P = max(self.max_pins, 1)
        pad_pin = np.zeros((e.N, P), dtype=np.int64)
        pad_net = np.zeros((e.N, P), dtype=np.int64)
        pad_msk = np.zeros((e.N, P), dtype=bool)
        for m in range(e.N):
            k = len(pins_of[m])
            if k:
                pad_pin[m, :k] = pins_of[m]
                pad_net[m, :k] = netid_of[m]
                pad_msk[m, :k] = True
        dev = e.device
        self.aff_pin = torch.as_tensor(pad_pin, device=dev)
        self.aff_net = torch.as_tensor(pad_net, device=dev)
        self.aff_msk = torch.as_tensor(pad_msk, device=dev)
        self.n_aff = torch.as_tensor(np.array([len(p) for p in pins_of]), device=dev)

        # local (per-macro) net numbering, plus where each net's DRIVER pin sits inside the
        # macro's padded affected-pin list -- both needed to keep the per-candidate work in
        # [B, max_pins] instead of [B, n_nets].
        Kn = max(self.max_nets, 1)
        loc = np.zeros((e.N, P), dtype=np.int64)
        drvpos = np.zeros((e.N, Kn), dtype=np.int64)
        nwt = np.zeros((e.N, Kn), dtype=np.float64)
        nmask = np.zeros((e.N, Kn), dtype=bool)
        drv_np = e.driver_pin.detach().cpu().numpy()
        w_np = e.net_weight.detach().cpu().numpy()
        for m in range(e.N):
            idx_of = {n: i for i, n in enumerate(nets_of[m])}
            for j, n in enumerate(netid_of[m]):
                loc[m, j] = idx_of[n]
            for i, n in enumerate(nets_of[m]):
                nmask[m, i] = True
                nwt[m, i] = w_np[n]
                dp = int(drv_np[n])
                drvpos[m, i] = pins_of[m].index(dp)
        self.aff_loc = torch.as_tensor(loc, device=dev)
        self.aff_drvpos = torch.as_tensor(drvpos, device=dev)
        self.aff_nwt = torch.as_tensor(nwt, device=dev, dtype=e.dtype)
        self.aff_nmask = torch.as_tensor(nmask, device=dev)
        self.aff_netid = torch.zeros((e.N, Kn), dtype=torch.int64, device=dev)
        for m in range(e.N):
            if nets_of[m]:
                self.aff_netid[m, :len(nets_of[m])] = torch.as_tensor(
                    nets_of[m], device=dev)

    # ------------------------------------------------------------- incumbent
    def set_base(self, pos):
        """Cache everything about the incumbent that a single-macro move does not change."""
        e = self.e
        self.pos = pos.clone()
        out = e.evaluate(pos.unsqueeze(0), need_maps=True)
        self.base_proxy = float(out["proxy"][0])
        self.pin_xy = e.pin_positions(pos.unsqueeze(0))[0]
        row, col = _clamped_cell(self.pin_xy[:, 0], self.pin_xy[:, 1],
                                 e.gw_t, e.gh_t, e.R, e.C)
        self.pin_row, self.pin_col = row, col
        self.pin_g = row * e.C + col
        # unsmoothed net-demand difference arrays for the incumbent
        self.Hd, self.Vd = self._route_diff_full(self.pin_g)
        self.wl_net = self._net_hpwl(self.pin_xy)
        self.macroV, self.macroH = self._macro_maps(pos)
        return self.base_proxy

    def _net_hpwl(self, pin_xy):
        e = self.e
        big = torch.finfo(e.dtype).max / 4
        idx = e.pin_net.view(-1, 1).expand(-1, 2)
        mx = torch.full((e.n_nets, 2), -big, device=e.device, dtype=e.dtype)
        mn = torch.full((e.n_nets, 2), big, device=e.device, dtype=e.dtype)
        mx.scatter_reduce_(0, idx, pin_xy, reduce="amax", include_self=True)
        mn.scatter_reduce_(0, idx, pin_xy, reduce="amin", include_self=True)
        return (mx - mn).sum(1) * e.net_weight

    def _macro_maps(self, pos):
        e = self.e
        xo, yo, blr, urr, blc, urc, inr, inc = e._axis_overlaps(pos.unsqueeze(0))
        V, H = e.macro_routing(xo, yo, blr, urr, blc, urc, inr, inc)
        self._dens_occ = torch.einsum("bmr,bmc->brc", yo.clamp(min=0), xo.clamp(min=0))[0]
        self._xo, self._yo = xo[0], yo[0]
        return V[0] / e.grid_v_routes, H[0] / e.grid_h_routes

    # ------------------------------------------------------------- routing
    def _route_diff_full(self, g):
        """Difference arrays for the whole netlist (used once, for the incumbent)."""
        e = self.e
        Hd = torch.zeros(e.R, e.C + 1, device=e.device, dtype=e.dtype)
        Vd = torch.zeros(e.R + 1, e.C, device=e.device, dtype=e.dtype)
        self._emit_routes(g, torch.arange(e.n_nets, device=e.device), Hd, Vd, +1.0)
        return Hd, Vd

    def _emit_routes(self, g, nets, Hd, Vd, sign):
        """Add (sign=+1) or remove (sign=-1) the routes of `nets`, given pin gcells `g`."""
        e = self.e
        sel = torch.isin(e.pin_net, nets)
        pn = e.pin_net[sel]
        pg = g[sel]
        key = pn * (e.n_cells + 1) + pg
        skey, _ = torch.sort(key)
        sg = skey % (e.n_cells + 1)
        snet = skey // (e.n_cells + 1)
        first = torch.ones_like(sg, dtype=torch.bool)
        first[1:] = skey[1:] != skey[:-1]

        k = torch.zeros(e.n_nets, device=e.device, dtype=e.dtype)
        k.scatter_add_(0, snet, first.to(e.dtype))
        src_g = g[e.driver_pin]
        w = e.net_weight

        s_row = (src_g // e.C)[snet]
        s_col = (src_g % e.C)[snet]
        d_row = sg // e.C
        d_col = sg % e.C
        star = first & ((k[snet] == 2) | (k[snet] > 3)) & (sg != src_g[snet])
        wt = w[snet] * sign

        self._add_H(Hd, s_row, s_col, d_col, wt, star)
        self._add_V(Vd, d_col, s_row, d_row, wt, star)

        three = (k == 3) & torch.isin(torch.arange(e.n_nets, device=e.device), nets)
        if bool(three.any()):
            self._three(Hd, Vd, sg, snet, first, three, w, sign)

    def _add_H(self, Hd, rows, c0, c1, wt, mask):
        e = self.e
        lo = torch.minimum(c0, c1)
        hi = torch.maximum(c0, c1)
        v = wt * mask.to(e.dtype)
        flat = Hd.view(-1)
        base = rows * (e.C + 1)
        flat.scatter_add_(0, base + lo, v)
        flat.scatter_add_(0, base + hi, -v)

    def _add_V(self, Vd, cols, r0, r1, wt, mask):
        e = self.e
        lo = torch.minimum(r0, r1)
        hi = torch.maximum(r0, r1)
        v = wt * mask.to(e.dtype)
        flat = Vd.view(-1)
        flat.scatter_add_(0, lo * e.C + cols, v)
        flat.scatter_add_(0, hi * e.C + cols, -v)

    def _three(self, Hd, Vd, sg, snet, first, three, w, sign):
        e = self.e
        idx = torch.nonzero(three).flatten()
        remap = torch.full((e.n_nets,), -1, device=e.device, dtype=torch.int64)
        remap[idx] = torch.arange(idx.numel(), device=e.device)
        rank = torch.cumsum(first.to(torch.int64), 0) - 1
        kk = torch.zeros(e.n_nets, device=e.device, dtype=torch.int64)
        kk.scatter_add_(0, snet, first.to(torch.int64))
        before = torch.cumsum(kk, 0) - kk
        pos_in_net = rank - before[snet]
        m = first & (remap[snet] >= 0) & (pos_in_net < 3)
        cells = torch.zeros(idx.numel(), 3, device=e.device, dtype=torch.int64)
        cells[remap[snet][m], pos_in_net[m]] = sg[m]

        rr = cells // e.C
        cc = cells % e.C
        okey = cc * (e.R + 1) + rr
        oi = torch.argsort(okey, dim=1)
        y = torch.gather(rr, 1, oi)
        x = torch.gather(cc, 1, oi)
        y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        wt = w[idx] * sign
        one = torch.ones_like(y1, dtype=torch.bool)

        A = (x1 < x2) & (x2 < x3) & (torch.minimum(y1, y3) < y2) & (torch.maximum(y1, y3) > y2)
        Bc = (~A) & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
        Cc = (~A) & (~Bc) & (y2 == y3)
        D = (~A) & (~Bc) & (~Cc)

        self._add_H(Hd, y1, x1, x2, wt, A)
        self._add_H(Hd, y2, x2, x3, wt, A)
        self._add_V(Vd, x2, y1, y2, wt, A)
        self._add_V(Vd, x3, y2, y3, wt, A)
        self._add_H(Hd, y1, x1, x2, wt, Bc)
        self._add_V(Vd, x2, y1, torch.maximum(y2, y3), wt, Bc)
        self._add_H(Hd, y1, x1, x2, wt, Cc)
        self._add_H(Hd, y2, x2, x3, wt, Cc)
        self._add_V(Vd, x2, y2, y1, wt, Cc)
        ty1, ty2, ty3 = rr[:, 0], rr[:, 1], rr[:, 2]
        tx1, tx2, tx3 = cc[:, 0], cc[:, 1], cc[:, 2]
        self._add_H(Hd, ty2, torch.minimum(torch.minimum(tx1, tx2), tx3),
                    torch.maximum(torch.maximum(tx1, tx2), tx3), wt, D)
        self._add_V(Vd, tx1, ty1, ty2, wt, D)
        self._add_V(Vd, tx3, ty2, ty3, wt, D)
        del one

    # ------------------------------------------------------------ selftest
    def selftest(self, n=8, seed=0):
        """Every incremental score must equal a full recompute of the same placement."""
        e = self.e
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.set_base(self.pos if hasattr(self, "pos") else e.pos0)
        movable = (~e.fixed).nonzero().flatten()
        errs = []
        for _ in range(n):
            m = int(movable[torch.randint(movable.numel(), (1,), generator=g)])
            newxy = torch.tensor([
                float(torch.rand(1, generator=g)) * e.W,
                float(torch.rand(1, generator=g)) * e.H], device=e.device, dtype=e.dtype)
            inc = self.score_moves(torch.tensor([m], device=e.device), newxy.view(1, 2))
            full = self.pos.clone()
            full[m] = newxy
            ref = float(e.evaluate(full.unsqueeze(0))["proxy"][0])
            errs.append(abs(float(inc[0]) - ref))
        return max(errs), errs

    # ------------------------------------------------------------ batched core
    def _emit_batch(self, g, loc, msk, nwt, nmsk, drvpos, Hd, Vd, sign):
        """Emit (sign=+1) or retract (sign=-1) the affected nets' routes, per candidate.

        Same routing rules as the full engine, but the segmented sort runs over the padded
        AFFECTED-pin list ([B, max_pins]) instead of every pin in the design.
        """
        e = self.e
        B, P = g.shape
        MUL = e.n_cells + 1
        BIG = (loc.max().item() + 2) * MUL

        key = torch.where(msk, loc * MUL + g, torch.full_like(g, BIG))
        skey, _ = torch.sort(key, dim=1)
        valid = skey < BIG
        sg = skey % MUL
        sloc = skey // MUL
        first = torch.ones_like(sg, dtype=torch.bool)
        first[:, 1:] = skey[:, 1:] != skey[:, :-1]
        first = first & valid

        Kn = nmsk.shape[1]
        k = torch.zeros(B, Kn, device=e.device, dtype=e.dtype)
        k.scatter_add_(1, sloc.clamp(max=Kn - 1), first.to(e.dtype) * valid.to(e.dtype))

        src_g = g.gather(1, drvpos)                       # [B, Kn]
        slocc = sloc.clamp(max=Kn - 1)
        s_all = src_g.gather(1, slocc)
        s_row = s_all // e.C
        s_col = s_all % e.C
        d_row = sg // e.C
        d_col = sg % e.C
        kk = k.gather(1, slocc)
        wt = nwt.gather(1, slocc) * sign
        star = first & ((kk == 2) | (kk > 3)) & (sg != s_all)

        self._addH_b(Hd, s_row, s_col, d_col, wt, star)
        self._addV_b(Vd, d_col, s_row, d_row, wt, star)

        three = (k == 3) & nmsk
        if bool(three.any()):
            self._three_b(Hd, Vd, sg, slocc, first, three, nwt, sign, Kn)

    def _addH_b(self, Hd, rows, c0, c1, wt, mask):
        e = self.e
        B = Hd.shape[0]
        lo = torch.minimum(c0, c1)
        hi = torch.maximum(c0, c1)
        v = wt * mask.to(e.dtype)
        flat = Hd.view(B, -1)
        base = rows * (e.C + 1)
        flat.scatter_add_(1, base + lo, v)
        flat.scatter_add_(1, base + hi, -v)

    def _addV_b(self, Vd, cols, r0, r1, wt, mask):
        e = self.e
        B = Vd.shape[0]
        lo = torch.minimum(r0, r1)
        hi = torch.maximum(r0, r1)
        v = wt * mask.to(e.dtype)
        flat = Vd.view(B, -1)
        flat.scatter_add_(1, lo * e.C + cols, v)
        flat.scatter_add_(1, hi * e.C + cols, -v)

    def _three_b(self, Hd, Vd, sg, sloc, first, three, nwt, sign, Kn):
        e = self.e
        B = sg.shape[0]
        rank = torch.cumsum(first.to(torch.int64), dim=1) - 1
        kint = torch.zeros(B, Kn, device=e.device, dtype=torch.int64)
        kint.scatter_add_(1, sloc, first.to(torch.int64))
        before = torch.cumsum(kint, dim=1) - kint
        pos_in = rank - before.gather(1, sloc)
        sel = first & three.gather(1, sloc) & (pos_in >= 0) & (pos_in < 3)

        cells = torch.zeros(B, Kn, 3, device=e.device, dtype=torch.int64)
        bidx = torch.arange(B, device=e.device).view(-1, 1).expand_as(sg)
        cells[bidx[sel], sloc[sel], pos_in[sel]] = sg[sel]

        rr = cells // e.C
        cc = cells % e.C
        oi = torch.argsort(cc * (e.R + 1) + rr, dim=2)
        y = torch.gather(rr, 2, oi)
        x = torch.gather(cc, 2, oi)
        y1, y2, y3 = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        x1, x2, x3 = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        wt = nwt * sign

        A = three & (x1 < x2) & (x2 < x3) & (torch.minimum(y1, y3) < y2) & (torch.maximum(y1, y3) > y2)
        Bc = three & (~A) & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
        Cc = three & (~A) & (~Bc) & (y2 == y3)
        D = three & (~A) & (~Bc) & (~Cc)

        self._addH_b(Hd, y1, x1, x2, wt, A)
        self._addH_b(Hd, y2, x2, x3, wt, A)
        self._addV_b(Vd, x2, y1, y2, wt, A)
        self._addV_b(Vd, x3, y2, y3, wt, A)
        self._addH_b(Hd, y1, x1, x2, wt, Bc)
        self._addV_b(Vd, x2, y1, torch.maximum(y2, y3), wt, Bc)
        self._addH_b(Hd, y1, x1, x2, wt, Cc)
        self._addH_b(Hd, y2, x2, x3, wt, Cc)
        self._addV_b(Vd, x2, y2, y1, wt, Cc)
        ty1, ty2, ty3 = rr[:, :, 0], rr[:, :, 1], rr[:, :, 2]
        tx1, tx2, tx3 = cc[:, :, 0], cc[:, :, 1], cc[:, :, 2]
        self._addH_b(Hd, ty2, torch.minimum(torch.minimum(tx1, tx2), tx3),
                     torch.maximum(torch.maximum(tx1, tx2), tx3), wt, D)
        self._addV_b(Vd, tx1, ty1, ty2, wt, D)
        self._addV_b(Vd, tx3, ty2, ty3, wt, D)

    def _one_macro_maps(self, m, xy):
        """Blockage + density contribution of ONE macro at [B,2] positions. m is [B]."""
        e = self.e
        B = m.numel()
        hw = e.sizes[m, 0] * 0.5
        hh = e.sizes[m, 1] * 0.5
        x0, x1 = xy[:, 0] - hw, xy[:, 0] + hw
        y0, y1 = xy[:, 1] - hh, xy[:, 1] + hh
        ur_row, ur_col = _clamped_cell(x1, y1, e.gw_t, e.gh_t, e.R, e.C)
        bl_row, bl_col = _clamped_cell(x0, y0, e.gw_t, e.gh_t, e.R, e.C)
        cols = torch.arange(e.C, device=e.device, dtype=e.dtype).view(1, -1)
        rows = torch.arange(e.R, device=e.device, dtype=e.dtype).view(1, -1)
        x_ov = torch.minimum(x1.view(-1, 1), (cols + 1) * e.gw) - torch.maximum(x0.view(-1, 1), cols * e.gw)
        y_ov = torch.minimum(y1.view(-1, 1), (rows + 1) * e.gh) - torch.maximum(y0.view(-1, 1), rows * e.gh)
        ci = torch.arange(e.C, device=e.device).view(1, -1)
        ri = torch.arange(e.R, device=e.device).view(1, -1)
        in_c = (ci >= bl_col.view(-1, 1)) & (ci <= ur_col.view(-1, 1))
        in_r = (ri >= bl_row.view(-1, 1)) & (ri <= ur_row.view(-1, 1))
        x_ov = torch.where(in_c, x_ov, torch.zeros((), device=e.device, dtype=e.dtype))
        y_ov = torch.where(in_r, y_ov, torch.zeros((), device=e.device, dtype=e.dtype))

        occ = torch.einsum("br,bc->brc", y_ov.clamp(min=0), x_ov.clamp(min=0))

        is_hard = (m < e.n_hard).view(-1, 1, 1).to(e.dtype)
        cvalid = x_ov > 0
        rvalid = y_ov > 0
        xg = torch.where(cvalid, x_ov, torch.zeros_like(x_ov))
        yg = torch.where(rvalid, y_ov, torch.zeros_like(y_ov))
        V = torch.einsum("br,bc->brc", rvalid.to(e.dtype), xg) * e.v_alloc
        H = torch.einsum("br,bc->brc", yg, cvalid.to(e.dtype)) * e.h_alloc

        gh_close = (y_ov - e.gh).abs() <= 1e-5
        gw_close = (x_ov - e.gw).abs() <= 1e-5
        any_col_invalid = (in_c & ~cvalid).any(dim=1)
        any_row_invalid = (in_r & ~rvalid).any(dim=1)
        bl_ok_v = gh_close.gather(1, bl_row.view(-1, 1)).squeeze(1) & ~any_col_invalid
        ur_ok_v = gh_close.gather(1, ur_row.view(-1, 1)).squeeze(1) & ~any_col_invalid
        PV = (ur_row != bl_row) & ~(bl_ok_v & ur_ok_v)
        bl_ok_h = gw_close.gather(1, bl_col.view(-1, 1)).squeeze(1) & ~any_row_invalid
        ur_ok_h = gw_close.gather(1, ur_col.view(-1, 1)).squeeze(1) & ~any_row_invalid
        PH = (ur_col != bl_col) & ~(bl_ok_h & ur_ok_h)

        if bool(PV.any()):
            wv = rvalid.gather(1, ur_row.view(-1, 1)).squeeze(1).to(e.dtype) * PV.to(e.dtype)
            V.scatter_add_(1, ur_row.view(-1, 1, 1).expand(-1, 1, e.C),
                           (-xg * wv.view(-1, 1) * e.v_alloc).unsqueeze(1))
        if bool(PH.any()):
            wh = cvalid.gather(1, ur_col.view(-1, 1)).squeeze(1).to(e.dtype) * PH.to(e.dtype)
            Ht = H.transpose(1, 2).contiguous()
            Ht.scatter_add_(1, ur_col.view(-1, 1, 1).expand(-1, 1, e.R),
                            (-yg * wh.view(-1, 1) * e.h_alloc).unsqueeze(1))
            H = Ht.transpose(1, 2).contiguous()
        return V * is_hard, H * is_hard, occ

    def score_moves(self, midx, newxy):
        """Exact proxy for B single-macro moves against the cached incumbent. -> [B]"""
        e = self.e
        B = midx.numel()
        pins = self.aff_pin[midx]
        msk = self.aff_msk[midx]
        loc = self.aff_loc[midx]
        nwt = self.aff_nwt[midx]
        nmsk = self.aff_nmask[midx]
        drvpos = self.aff_drvpos[midx]
        netid = self.aff_netid[midx]

        owner = e.pin_owner_macro[pins]
        owned = (owner == midx.view(-1, 1)) & msk & (~e.pin_is_port[pins])
        off = e.pin_off[pins]
        old_xy = self.pin_xy[pins]
        new_xy = torch.where(owned.unsqueeze(2), newxy.unsqueeze(1) + off, old_xy)

        g_old = self.pin_g[pins]
        r, c = _clamped_cell(new_xy[..., 0], new_xy[..., 1], e.gw_t, e.gh_t, e.R, e.C)
        g_new = r * e.C + c

        Hd = self.Hd.unsqueeze(0).repeat(B, 1, 1)
        Vd = self.Vd.unsqueeze(0).repeat(B, 1, 1)
        self._emit_batch(g_old, loc, msk, nwt, nmsk, drvpos, Hd, Vd, -1.0)
        self._emit_batch(g_new, loc, msk, nwt, nmsk, drvpos, Hd, Vd, +1.0)

        Hm = torch.cumsum(Hd, dim=2)[:, :, :e.C] / e.grid_h_routes
        Vm = torch.cumsum(Vd, dim=1)[:, :e.R, :] / e.grid_v_routes
        Vs, Hs = e.smooth(Vm, Hm)

        Vo, Ho, occ_o = self._one_macro_maps(midx, self.pos[midx])
        Vn, Hn, occ_n = self._one_macro_maps(midx, newxy)
        V = Vs + self.macroV.unsqueeze(0) + (Vn - Vo) / e.grid_v_routes
        H = Hs + self.macroH.unsqueeze(0) + (Hn - Ho) / e.grid_h_routes

        both = torch.cat([V.reshape(B, -1), H.reshape(B, -1)], dim=1)
        cong = torch.topk(both, e.top_cong_k, dim=1).values.sum(1) / e.top_cong_k

        occ = self._dens_occ.unsqueeze(0) + (occ_n - occ_o)
        cells = (occ / (e.gw * e.gh)).reshape(B, -1)
        vals = torch.where(cells != 0, cells, torch.full_like(cells, -1.0))
        top = torch.topk(vals, e.top_density_k, dim=1).values
        top = torch.where(top < 0, torch.zeros_like(top), top)
        den = 0.5 * top.sum(1) / e.top_density_k

        big = torch.finfo(e.dtype).max / 4
        Kn = nmsk.shape[1]
        mx = torch.full((B, Kn, 2), -big, device=e.device, dtype=e.dtype)
        mn = torch.full((B, Kn, 2), big, device=e.device, dtype=e.dtype)
        li = loc.unsqueeze(2).expand(-1, -1, 2)
        src = torch.where(msk.unsqueeze(2), new_xy, torch.zeros_like(new_xy))
        srcmx = torch.where(msk.unsqueeze(2), new_xy, torch.full_like(new_xy, -big))
        srcmn = torch.where(msk.unsqueeze(2), new_xy, torch.full_like(new_xy, big))
        mx.scatter_reduce_(1, li, srcmx, reduce="amax", include_self=True)
        mn.scatter_reduce_(1, li, srcmn, reduce="amin", include_self=True)
        new_hpwl = ((mx - mn).sum(2) * nwt) * nmsk.to(e.dtype)
        old_hpwl = self.wl_net[netid] * nmsk.to(e.dtype)
        wl_total = self.wl_net.sum() + (new_hpwl - old_hpwl).sum(1)
        wl = wl_total / ((e.W + e.H) * e.net_cnt)
        del src
        return wl + 0.5 * den + 0.5 * cong
