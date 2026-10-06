"""Exact, GPU-batched reimplementation of the TILOS/Partcl proxy cost.

WHY. The official evaluator (`plc_client_os.PlacementCost`) is pure Python and takes
~1.6 s per full evaluation on ibm01. A search that must be honest about its objective
therefore gets ~2000 evaluations per CPU-hour. That is the binding constraint on this
whole problem, so the first thing we build is our own evaluator: numerically identical
to the official one, but evaluating a BATCH of B candidate placements at once on the GPU.

The port is exact, not approximate. Every branch of the reference is reproduced:

  wirelength : per-net HPWL over PIN positions (macro center + pin offset), weighted by
               the net's DRIVER pin weight, normalized by (W + H) * plc.net_cnt.
               Note net_cnt is the netlist's declared count (7269 on ibm01), NOT the
               number of driver-keyed nets (5993).
  density    : per-cell occupied area / cell area over ALL macros (hard AND soft);
               cost = 0.5 * mean of the top floor(0.1*n_cells) values among the NONZERO
               cells (the reference divides by the count, not by how many it found).
  congestion : H/V routing demand from a gcell-quantized net model
                 - k distinct gcells == 2 or k > 3  -> star: source -> each other gcell
                 - k == 3                           -> the reference's L / T / two special
                                                       cases, with their exact tie ordering
                 - each route lays H demand along the SOURCE's row and V demand along the
                   SINK's column (half-open ranges, matching the reference loops)
               plus hard-macro routing blockage with the reference's partial-overlap
               correction, then 1-D smoothing over +/- smooth_range, then
               cost = mean of top floor(0.05 * 2 * n_cells) of concat(V, H).

  proxy      = 1.0 * wirelength + 0.5 * density + 0.5 * congestion

THE TRICK THAT MAKES IT BATCHABLE. Both expensive parts collapse to dense linear algebra:

  * Routing demand is a union of axis-aligned segments. Instead of writing each segment
    cell by cell (the reference's inner loops), we write its two endpoints into a
    difference array and take one cumulative sum over the grid at the end -- O(1) scatter
    per route instead of O(length).
  * Macro blockage and density are separable in x and y (the overlap of a rectangle with
    a cell is width(c) * height(r)), so each becomes a single [R,M] @ [M,C] matmul.

Everything else is masked arithmetic on padded tensors, so there is no per-net Python
loop anywhere in the hot path.
"""

import math
import numpy as np
import torch


def _clamped_cell(x, y, gw, gh, R, C):
    """The evaluator's __get_grid_cell_location, with the challenge's bounds-clamp patch.

    gw/gh MUST arrive as 0-dim tensors. Dividing a tensor by a Python float makes PyTorch
    fold the division into a multiply-by-reciprocal on CUDA, which is not correctly rounded:
    9.69 / 0.51 comes back as 18.999999999999996 instead of 19.0 and silently moves a pin
    into the neighbouring grid cell. Grid membership is a hard branch in the reference
    evaluator, so that one ULP changes the congestion number. A tensor divisor forces a
    true IEEE division and makes GPU and CPU agree bit for bit with the reference.
    """
    col = torch.clamp(torch.floor(torch.div(x, gw)).long(), 0, C - 1)
    row = torch.clamp(torch.floor(torch.div(y, gh)).long(), 0, R - 1)
    return row, col


class ProxyEngine:
    """Batched exact proxy-cost evaluator for one benchmark.

    evaluate(pos) takes [B, N, 2] macro centers (hard first, then soft) and returns a dict
    of [B] tensors: proxy, wirelength, density, congestion.
    """

    def __init__(self, npz_path, device="cuda", dtype=torch.float64):
        d = np.load(npz_path, allow_pickle=True)
        self.device = torch.device(device)
        self.dtype = dtype
        self.name = str(npz_path).split("/")[-1].replace(".npz", "")

        self.W, self.H = float(d["canvas"][0]), float(d["canvas"][1])
        self.R, self.C = int(d["grid"][0]), int(d["grid"][1])
        self.hrpm, self.vrpm = float(d["routes_per_micron"][0]), float(d["routes_per_micron"][1])
        self.h_alloc, self.v_alloc = float(d["routing_alloc"][0]), float(d["routing_alloc"][1])
        self.smooth_range = int(d["scalars"][0])
        self.overlap_thres = float(d["scalars"][1])
        self.net_cnt = float(d["scalars"][2])
        self.n_hard, self.n_soft, self.n_ports = (int(v) for v in d["counts"])
        self.N = self.n_hard + self.n_soft

        self.gw = self.W / self.C
        self.gh = self.H / self.R
        self.gw_t = torch.tensor(self.gw, device=self.device, dtype=dtype)
        self.gh_t = torch.tensor(self.gh, device=self.device, dtype=dtype)
        self.grid_v_routes = self.gw * self.vrpm
        self.grid_h_routes = self.gh * self.hrpm
        self.n_cells = self.R * self.C

        t = lambda a: torch.as_tensor(a, device=self.device, dtype=dtype)
        self.pos0 = t(d["pos0"])
        self.sizes = t(d["sizes"])
        self.fixed = torch.as_tensor(d["fixed"], device=self.device)
        self.port_pos = t(d["port_pos"])
        self.macro_names = list(d["macro_names"])

        self.pin_owner = torch.as_tensor(d["pin_owner"].astype(np.int64), device=self.device)
        self.pin_off = t(d["pin_off"])
        net_ptr = torch.as_tensor(d["net_ptr"].astype(np.int64), device=self.device)
        self.net_ptr = net_ptr
        self.net_weight = t(d["net_weight"])
        self.n_nets = int(self.net_weight.numel())
        self.n_pins = int(self.pin_owner.numel())

        # pin -> net id, and a flag for "this pin is its net's driver" (first pin of the net)
        counts = net_ptr[1:] - net_ptr[:-1]
        self.pin_net = torch.repeat_interleave(
            torch.arange(self.n_nets, device=self.device), counts)
        self.net_deg = counts
        self.driver_pin = net_ptr[:-1]

        # pins whose owner is a port never move: precompute their absolute positions
        self.pin_is_port = self.pin_owner >= self.N
        self.port_pin_pos = torch.zeros(self.n_pins, 2, device=self.device, dtype=dtype)
        if self.pin_is_port.any():
            pidx = self.pin_owner[self.pin_is_port] - self.N
            self.port_pin_pos[self.pin_is_port] = self.port_pos[pidx]
        # owner index clamped into macro range so we can gather unconditionally
        self.pin_owner_macro = torch.clamp(self.pin_owner, max=self.N - 1)

        self.top_density_k = max(1, math.floor(self.n_cells * 0.1))
        self.top_cong_k = max(1, math.floor(2 * self.n_cells * 0.05))

        self._sort_key_mul = self.n_cells + 1
        self.max_batch = self._autosize()

    def _autosize(self, frac=0.55, cap=4096):
        """Largest batch that fits in `frac` of FREE VRAM.

        AGENTS.md: size the batch to the hardware, never to a hardcoded guess -- but the
        other half of that rule is that an OOM mid-search is a lost experiment, so the
        engine chunks internally instead of asking the caller to know this number. The
        dominant term is the segmented sort over [B, n_pins], which materializes both
        values and indices in int64.
        """
        per = self.n_pins * 96 + self.n_cells * 40 + self.N * 64   # bytes per candidate
        if self.device.type != "cuda":
            return max(1, min(cap, int(8e9 // max(per, 1))))
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except Exception:                                          # noqa: BLE001
            free = 8 << 30
        return max(1, min(cap, int(free * frac // max(per, 1))))

    # ------------------------------------------------------------------ pins
    def pin_positions(self, pos):
        """[B, n_pins, 2] absolute pin coordinates for macro centers pos [B, N, 2]."""
        owner_xy = pos[:, self.pin_owner_macro, :]                    # [B, P, 2]
        p = owner_xy + self.pin_off.unsqueeze(0)
        return torch.where(self.pin_is_port.view(1, -1, 1), self.port_pin_pos.unsqueeze(0), p)

    # ------------------------------------------------------------ wirelength
    def wirelength(self, pin_xy):
        B = pin_xy.shape[0]
        big = torch.finfo(self.dtype).max / 4
        idx = self.pin_net.view(1, -1, 1).expand(B, -1, 2)
        mx = torch.full((B, self.n_nets, 2), -big, device=self.device, dtype=self.dtype)
        mn = torch.full((B, self.n_nets, 2), big, device=self.device, dtype=self.dtype)
        mx.scatter_reduce_(1, idx, pin_xy, reduce="amax", include_self=True)
        mn.scatter_reduce_(1, idx, pin_xy, reduce="amin", include_self=True)
        hpwl = ((mx - mn).sum(dim=2) * self.net_weight.unsqueeze(0)).sum(dim=1)
        return hpwl / ((self.W + self.H) * self.net_cnt)

    # --------------------------------------------------- separable overlaps
    def _axis_overlaps(self, pos):
        """Per-macro overlap extent with every grid column / row.

        Returns x_ov [B, M, C], y_ov [B, M, R] (clamped at 0), plus the reference's
        bl/ur cell indices. Cells outside [bl, ur] are forced to exactly 0 so the
        reference's iteration bounds are reproduced rather than approximated.
        """
        B, M, _ = pos.shape
        hw = self.sizes[:, 0] * 0.5
        hh = self.sizes[:, 1] * 0.5
        x_min = pos[:, :, 0] - hw
        x_max = pos[:, :, 0] + hw
        y_min = pos[:, :, 1] - hh
        y_max = pos[:, :, 1] + hh

        ur_row, ur_col = _clamped_cell(x_max, y_max, self.gw_t, self.gh_t, self.R, self.C)
        bl_row, bl_col = _clamped_cell(x_min, y_min, self.gw_t, self.gh_t, self.R, self.C)

        cols = torch.arange(self.C, device=self.device, dtype=self.dtype).view(1, 1, -1)
        rows = torch.arange(self.R, device=self.device, dtype=self.dtype).view(1, 1, -1)
        cell_x0 = cols * self.gw
        cell_x1 = (cols + 1) * self.gw
        cell_y0 = rows * self.gh
        cell_y1 = (rows + 1) * self.gh

        x_ov = torch.minimum(x_max.unsqueeze(2), cell_x1) - torch.maximum(x_min.unsqueeze(2), cell_x0)
        y_ov = torch.minimum(y_max.unsqueeze(2), cell_y1) - torch.maximum(y_min.unsqueeze(2), cell_y0)

        ci = torch.arange(self.C, device=self.device).view(1, 1, -1)
        ri = torch.arange(self.R, device=self.device).view(1, 1, -1)
        in_c = (ci >= bl_col.unsqueeze(2)) & (ci <= ur_col.unsqueeze(2))
        in_r = (ri >= bl_row.unsqueeze(2)) & (ri <= ur_row.unsqueeze(2))
        x_ov = torch.where(in_c, x_ov, torch.zeros((), device=self.device, dtype=self.dtype))
        y_ov = torch.where(in_r, y_ov, torch.zeros((), device=self.device, dtype=self.dtype))
        return x_ov, y_ov, bl_row, ur_row, bl_col, ur_col, in_r, in_c

    # ---------------------------------------------------------------- density
    def density_cost(self, x_ov, y_ov):
        """0.5 * mean of the top 10% NONZERO cell densities (reference semantics)."""
        xc = x_ov.clamp(min=0.0)
        yc = y_ov.clamp(min=0.0)
        occ = torch.einsum("bmr,bmc->brc", yc, xc)          # [B, R, C] occupied area
        cells = (occ / (self.gw * self.gh)).reshape(occ.shape[0], -1)
        # reference sorts only the nonzero cells, then sums the first K and divides by K
        vals = torch.where(cells != 0, cells, torch.full_like(cells, -1.0))
        top = torch.topk(vals, self.top_density_k, dim=1).values
        top = torch.where(top < 0, torch.zeros_like(top), top)
        return 0.5 * top.sum(dim=1) / self.top_density_k

    # ------------------------------------------------------- macro blockage
    def macro_routing(self, x_ov, y_ov, bl_row, ur_row, bl_col, ur_col, in_r, in_c):
        """H/V blockage demand from HARD macros, with the partial-overlap correction."""
        B = x_ov.shape[0]
        nh = self.n_hard
        xh, yh = x_ov[:, :nh, :], y_ov[:, :nh, :]
        blr, urr = bl_row[:, :nh], ur_row[:, :nh]
        blc, urc = bl_col[:, :nh], ur_col[:, :nh]
        in_r, in_c = in_r[:, :nh, :], in_c[:, :nh, :]

        # __overlap_dist gates BOTH extents on x_diff > 0 AND y_diff > 0 jointly
        cvalid = (xh > 0)                                    # [B, M, C]
        rvalid = (yh > 0)                                    # [B, M, R]
        xg = torch.where(cvalid, xh, torch.zeros_like(xh))
        yg = torch.where(rvalid, yh, torch.zeros_like(yh))

        # V[r,c] += x_dist(m,c) * v_alloc for every covered (r,c);  x_dist is gated by rvalid
        V = torch.einsum("bmr,bmc->brc", rvalid.to(self.dtype), xg) * self.v_alloc
        H = torch.einsum("bmr,bmc->brc", yg, cvalid.to(self.dtype)) * self.h_alloc

        gh_close = (yh - self.gh).abs() <= 1e-5
        gw_close = (xh - self.gw).abs() <= 1e-5
        any_col_invalid = (in_c & ~cvalid).any(dim=2)        # [B, M]
        any_row_invalid = (in_r & ~rvalid).any(dim=2)

        def _gather_row(t, idx):                             # t [B,M,R] -> [B,M]
            return t.gather(2, idx.unsqueeze(2)).squeeze(2)

        bl_ok_v = _gather_row(gh_close, blr) & ~any_col_invalid
        ur_ok_v = _gather_row(gh_close, urr) & ~any_col_invalid
        PV = (urr != blr) & ~(bl_ok_v & ur_ok_v)

        bl_ok_h = _gather_row(gw_close, blc) & ~any_row_invalid
        ur_ok_h = _gather_row(gw_close, urc) & ~any_row_invalid
        PH = (urc != blc) & ~(bl_ok_h & ur_ok_h)

        # correction: strip the whole ur_row from V, and the whole ur_col from H
        if PV.any():
            w = _gather_row(rvalid.to(self.dtype), urr).unsqueeze(2) * PV.unsqueeze(2).to(self.dtype)
            corr = xg * w                                     # [B, M, C]
            V.scatter_add_(1, urr.unsqueeze(2).expand(-1, -1, self.C), -corr * self.v_alloc)
        if PH.any():
            w = _gather_row(cvalid.to(self.dtype), urc).unsqueeze(2) * PH.unsqueeze(2).to(self.dtype)
            corr = yg * w                                     # [B, M, R]
            Ht = H.transpose(1, 2).contiguous()               # [B, C, R]
            Ht.scatter_add_(1, urc.unsqueeze(2).expand(-1, -1, self.R), -corr * self.h_alloc)
            H = Ht.transpose(1, 2).contiguous()
        return V, H

    # ------------------------------------------------------------ net routing
    def net_routing(self, pin_xy):
        """H/V routing demand from the gcell-quantized net model.

        Emits axis-aligned segments into difference arrays; one cumsum turns them into
        the demand map. Star routing covers k==2 and k>3 identically (the reference's
        __split_net on a 2-gcell net yields exactly the same single route); k==3 gets
        the reference's four-way L / T / special-case dispatch.
        """
        B = pin_xy.shape[0]
        dev, dt = self.device, self.dtype
        row, col = _clamped_cell(pin_xy[:, :, 0], pin_xy[:, :, 1], self.gw_t, self.gh_t, self.R, self.C)
        g = row * self.C + col                                # [B, P]

        # ---- segmented unique of gcells within each net -----------------
        key = self.pin_net.view(1, -1) * self._sort_key_mul + g
        skey, order = torch.sort(key, dim=1)
        sg = skey % self._sort_key_mul
        snet = skey // self._sort_key_mul
        first = torch.ones_like(sg, dtype=torch.bool)
        first[:, 1:] = (skey[:, 1:] != skey[:, :-1])
        uniq = first                                          # [B, P] mask of distinct (net, gcell)

        k = torch.zeros(B, self.n_nets, device=dev, dtype=dt)
        k.scatter_add_(1, snet, uniq.to(dt))                  # distinct gcell count per net

        src_g = g.gather(1, self.driver_pin.view(1, -1).expand(B, -1))   # [B, n_nets]
        src_row = src_g // self.C
        src_col = src_g % self.C
        w = self.net_weight.view(1, -1).expand(B, -1)

        H_diff = torch.zeros(B, self.R, self.C + 1, device=dev, dtype=dt)
        V_diff = torch.zeros(B, self.R + 1, self.C, device=dev, dtype=dt)

        def add_H(rows, c0, c1, wt, mask):
            """H demand on row `rows`, columns [min(c0,c1), max(c0,c1))."""
            lo = torch.minimum(c0, c1)
            hi = torch.maximum(c0, c1)
            wt = wt * mask.to(dt)
            flat = H_diff.view(B, -1)
            base = rows * (self.C + 1)
            flat.scatter_add_(1, base + lo, wt)
            flat.scatter_add_(1, base + hi, -wt)

        def add_V(cols, r0, r1, wt, mask):
            """V demand on column `cols`, rows [min(r0,r1), max(r0,r1))."""
            lo = torch.minimum(r0, r1)
            hi = torch.maximum(r0, r1)
            wt = wt * mask.to(dt)
            flat = V_diff.view(B, -1)
            flat.scatter_add_(1, lo * self.C + cols, wt)
            flat.scatter_add_(1, hi * self.C + cols, -wt)

        # ---- star routing for every net with k == 2 or k > 3 -------------
        net_of = snet
        k_of_pin = k.gather(1, net_of)
        star_net = (k_of_pin == 2) | (k_of_pin > 3)
        s_row = src_row.gather(1, net_of)
        s_col = src_col.gather(1, net_of)
        d_row = sg // self.C
        d_col = sg % self.C
        star = uniq & star_net & (sg != src_g.gather(1, net_of))
        wt = w.gather(1, net_of)
        add_H(s_row, s_col, d_col, wt, star)
        add_V(d_col, s_row, d_row, wt, star)

        # ---- k == 3 nets: the reference's explicit dispatch ---------------
        three = (k == 3)
        if bool(three.any()):
            self._k_int = k.to(torch.int64)
            self._route_three(B, sg, snet, uniq, three, w, add_H, add_V)

        Hm = torch.cumsum(H_diff[:, :, :-1], dim=2)
        Vm = torch.cumsum(V_diff[:, :-1, :], dim=1)
        return Vm, Hm

    def _route_three(self, B, sg, snet, uniq, three, w, add_H, add_V):
        """Vectorized port of __three_pin_net_routing (+ __l_routing / __t_routing).

        The three distinct gcells arrive already sorted by (row, col) because the
        segmented sort keyed on gcell = row*C + col. The reference sorts them by
        (col, row) instead, so we re-sort; __t_routing then sorts by (row, col),
        which is the order we already have.
        """
        dev, dt = sg.device, w.dtype
        # gather the 3 distinct gcells of each k==3 net into [B, n3, 3].
        # cumsum(uniq) is a GLOBAL rank among distinct (net, gcell) pairs; subtracting the
        # count of distinct pairs in all preceding nets turns it into the rank WITHIN the
        # net, which is what indexes the 3 slots.
        rank = torch.cumsum(uniq.to(torch.int64), dim=1) - 1
        k_int = self._k_int
        before = torch.cumsum(k_int, dim=1) - k_int                    # [B, n_nets]
        pos_in_net = rank - before.gather(1, snet)
        sel = uniq & three.gather(1, snet)
        n3_idx = torch.nonzero(three.any(dim=0)).flatten()     # nets that are k==3 in ANY batch
        remap = torch.full((self.n_nets,), -1, device=dev, dtype=torch.int64)
        remap[n3_idx] = torch.arange(n3_idx.numel(), device=dev)
        n3 = int(n3_idx.numel())
        cells = torch.zeros(B, n3, 3, device=dev, dtype=torch.int64)
        b_idx = torch.arange(B, device=dev).view(-1, 1).expand_as(sg)
        m = sel & (remap[snet] >= 0) & (pos_in_net < 3)
        cells[b_idx[m], remap[snet][m], pos_in_net[m]] = sg[m]

        valid = three.gather(1, n3_idx.view(1, -1).expand(B, -1))
        wt = w.gather(1, n3_idx.view(1, -1).expand(B, -1))

        rr = cells // self.C
        cc = cells % self.C
        # --- order by (col, row) as the reference's three_pin sort does
        okey = cc * (self.R + 1) + rr
        oidx = torch.argsort(okey, dim=2)
        y = torch.gather(rr, 2, oidx)
        x = torch.gather(cc, 2, oidx)
        y1, y2, y3 = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        x1, x2, x3 = x[:, :, 0], x[:, :, 1], x[:, :, 2]

        caseA = (x1 < x2) & (x2 < x3) & (torch.minimum(y1, y3) < y2) & (torch.maximum(y1, y3) > y2)
        caseB = (~caseA) & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
        caseC = (~caseA) & (~caseB) & (y2 == y3)
        caseD = (~caseA) & (~caseB) & (~caseC)

        # A: L routing
        mA = valid & caseA
        add_H(y1, x1, x2, wt, mA)
        add_H(y2, x2, x3, wt, mA)
        add_V(x2, y1, y2, wt, mA)
        add_V(x3, y2, y3, wt, mA)
        # B
        mB = valid & caseB
        add_H(y1, x1, x2, wt, mB)
        add_V(x2, y1, torch.maximum(y2, y3), wt, mB)
        # C
        mC = valid & caseC
        add_H(y1, x1, x2, wt, mC)
        add_H(y2, x2, x3, wt, mC)
        add_V(x2, y2, y1, wt, mC)
        # D: T routing, re-sorted by (row, col) -- the raw gcell order
        mD = valid & caseD
        ty1, ty2, ty3 = rr[:, :, 0], rr[:, :, 1], rr[:, :, 2]
        tx1, tx2, tx3 = cc[:, :, 0], cc[:, :, 1], cc[:, :, 2]
        xmin = torch.minimum(torch.minimum(tx1, tx2), tx3)
        xmax = torch.maximum(torch.maximum(tx1, tx2), tx3)
        add_H(ty2, xmin, xmax, wt, mD)
        add_V(tx1, ty1, ty2, wt, mD)
        add_V(tx3, ty2, ty3, wt, mD)

    # --------------------------------------------------------------- smoothing
    def smooth(self, V, H):
        """Reference __smooth_routing_cong: V spreads along columns, H along rows."""
        sr = self.smooth_range
        if sr <= 0:
            return V, H
        C, R = self.C, self.R
        cols = torch.arange(C, device=self.device)
        lp = torch.clamp(cols - sr, min=0)
        rp = torch.clamp(cols + sr, max=C - 1)
        cnt_c = (rp - lp + 1).to(self.dtype)
        Vd = V / cnt_c.view(1, 1, -1)
        Vc = torch.cumsum(
            torch.nn.functional.pad(Vd, (0, 1)) * 0 + torch.nn.functional.pad(Vd, (0, 1)), dim=2)
        # spread: out[c] = sum over source columns s with lp[s] <= c <= rp[s]
        diff = torch.zeros(V.shape[0], R, C + 1, device=self.device, dtype=self.dtype)
        diff.scatter_add_(2, lp.view(1, 1, -1).expand_as(Vd), Vd)
        diff.scatter_add_(2, (rp + 1).view(1, 1, -1).expand_as(Vd), -Vd)
        Vs = torch.cumsum(diff, dim=2)[:, :, :C]

        rows = torch.arange(R, device=self.device)
        lpr = torch.clamp(rows - sr, min=0)
        upr = torch.clamp(rows + sr, max=R - 1)
        cnt_r = (upr - lpr + 1).to(self.dtype)
        Hd = H / cnt_r.view(1, -1, 1)
        diffh = torch.zeros(H.shape[0], R + 1, C, device=self.device, dtype=self.dtype)
        diffh.scatter_add_(1, lpr.view(1, -1, 1).expand_as(Hd), Hd)
        diffh.scatter_add_(1, (upr + 1).view(1, -1, 1).expand_as(Hd), -Hd)
        Hs = torch.cumsum(diffh, dim=1)[:, :R, :]
        del Vc
        return Vs, Hs

    # ---------------------------------------------------------------- overall
    def evaluate(self, pos, need_maps=False):
        """pos: [B, N, 2] or [N, 2]. Returns dict of [B] tensors.

        Splits the batch into VRAM-sized chunks so callers can pass any B.
        """
        squeeze = pos.dim() == 2
        if squeeze:
            pos = pos.unsqueeze(0)
        B = pos.shape[0]
        if B > self.max_batch:
            outs = []
            for s0 in range(0, B, self.max_batch):
                outs.append(self._evaluate(pos[s0:s0 + self.max_batch], need_maps))
            keys = outs[0].keys()
            return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}
        out = self._evaluate(pos, need_maps)
        if squeeze:
            out = {k: (v[0] if v.dim() >= 1 else v) for k, v in out.items()}
        return out

    def _evaluate(self, pos, need_maps=False):
        pos = pos.to(self.device, self.dtype)

        pin_xy = self.pin_positions(pos)
        wl = self.wirelength(pin_xy)

        x_ov, y_ov, blr, urr, blc, urc, in_r, in_c = self._axis_overlaps(pos)
        den = self.density_cost(x_ov, y_ov)

        Vn, Hn = self.net_routing(pin_xy)
        Vn = Vn / self.grid_v_routes
        Hn = Hn / self.grid_h_routes
        Vm, Hm = self.macro_routing(x_ov, y_ov, blr, urr, blc, urc, in_r, in_c)
        Vm = Vm / self.grid_v_routes
        Hm = Hm / self.grid_h_routes
        Vs, Hs = self.smooth(Vn, Hn)
        V = Vs + Vm
        Hc = Hs + Hm

        both = torch.cat([V.reshape(V.shape[0], -1), Hc.reshape(Hc.shape[0], -1)], dim=1)
        top = torch.topk(both, self.top_cong_k, dim=1).values
        cong = top.sum(dim=1) / self.top_cong_k

        proxy = wl + 0.5 * den + 0.5 * cong
        out = {"proxy": proxy, "wirelength": wl, "density": den, "congestion": cong}
        if need_maps:
            out["V"], out["H"] = V, Hc
        return out

    # ---------------------------------------------------------------- overlap
    def overlap_stats(self, pos):
        """Hard-macro pairwise overlap count / area, matching compute_overlap_metrics."""
        squeeze = pos.dim() == 2
        if squeeze:
            pos = pos.unsqueeze(0)
        pos = pos.to(self.device, self.dtype)
        nh = self.n_hard
        p = pos[:, :nh, :]
        s = self.sizes[:nh]
        dx = (p[:, :, None, 0] - p[:, None, :, 0]).abs()
        dy = (p[:, :, None, 1] - p[:, None, :, 1]).abs()
        mx = (s[:, 0][:, None] + s[:, 0][None, :]) * 0.5
        my = (s[:, 1][:, None] + s[:, 1][None, :]) * 0.5
        ox = (mx - dx).clamp(min=0)
        oy = (my - dy).clamp(min=0)
        ov = (ox > 0) & (oy > 0)
        iu = torch.triu(torch.ones(nh, nh, device=self.device, dtype=torch.bool), diagonal=1)
        ov = ov & iu
        cnt = ov.sum(dim=(1, 2))
        area = (ox * oy * ov).sum(dim=(1, 2))
        res = {"overlap_count": cnt, "overlap_area": area}
        if squeeze:
            res = {k: v[0] for k, v in res.items()}
        return res
