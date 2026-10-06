"""Batched exact-proxy local search.

The idea this whole task is built on: because `ProxyEngine` scores B full placements
exactly in one GPU call, we can afford to ask "what does the TRUE objective do if I move
macro m to position p?" for a thousand different (m, p) at once, every few milliseconds.
Every other placer has to answer that question with a surrogate or an incremental
approximation; we answer it exactly.

One ROUND is:

  1. propose  -- take the next B macros in a shuffled cycle (so coverage is uniform, not
                 random-with-replacement) and sample one displacement each, from a mixture
                 of move operators.
  2. screen   -- reject proposals that leave the canvas or overlap a hard macro. This keeps
                 the incumbent legal at every instant instead of legalizing afterwards,
                 which is what lets accepted moves be trusted.
  3. score    -- one batched exact evaluation gives the exact delta of each single move.
  4. accept   -- apply ALL improving moves at once, after removing collisions among the
                 accepted movers themselves. Single-move deltas are not additive, so the
                 combined placement is re-scored exactly and the batch is halved until it
                 genuinely beats the incumbent (or is dropped).

Step 4 is what makes it fast: a round costs one batched evaluation and typically banks
tens to hundreds of improving moves, instead of the one move a classical accept/reject
loop would bank.
"""

import math
import time

import numpy as np
import torch


class MoveMix:
    """Displacement generator. The mixture weights are part of the search space."""

    NAMES = ("gauss", "axis", "jump", "toward", "away", "coldrow", "coldcol",
             "toward_x", "toward_y")

    def __init__(self, weights, sigma, W, H):
        w = np.asarray([max(0.0, float(x)) for x in weights], dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones(len(self.NAMES))
        self.w = w / w.sum()
        self.sigma = sigma
        self.W, self.H = W, H


def _hard_conflict(newxy, midx, pos, sizes, n_hard):
    """[B] bool: does hard macro midx[b] at newxy[b] overlap any OTHER hard macro?

    Soft macros are cluster abstractions and may overlap, so they are never screened.
    """
    B = newxy.shape[0]
    hw = sizes[:n_hard, 0] * 0.5
    hh = sizes[:n_hard, 1] * 0.5
    px = pos[:n_hard, 0].view(1, -1)
    py = pos[:n_hard, 1].view(1, -1)
    mi = midx.clamp(max=n_hard - 1)
    mw = (sizes[mi, 0] * 0.5).view(-1, 1)
    mh = (sizes[mi, 1] * 0.5).view(-1, 1)
    dx = (newxy[:, 0:1] - px).abs()
    dy = (newxy[:, 1:2] - py).abs()
    hit = (dx < mw + hw.view(1, -1)) & (dy < mh + hh.view(1, -1))
    self_col = torch.zeros_like(hit)
    self_col.scatter_(1, mi.view(-1, 1), True)
    hit = hit & ~self_col
    conflict = hit.any(dim=1)
    return conflict & (midx < n_hard)


def _select_compatible(order, midx, newxy, pos, sizes, n_hard, sep=0.0, net_sig=None):
    """Pick a mutually compatible subset of improving moves, best delta first.

    Two things can make a batch of individually-improving moves fail together:

      * they collide -- two hard macros land on top of each other. Always screened.
      * they interact -- the single-move deltas are only additive when the moves do not
        touch the same part of the objective. Congestion and density couple through the
        grid (locally, within the smoothing radius) and wirelength couples through shared
        nets. Requiring accepted movers to be `sep` apart AND net-disjoint keeps the
        deltas near-additive, so the combined move usually survives the exact re-check
        instead of being thrown away by the halving backoff.

    sep=0 and net_sig=None reduces this to the plain collision filter.
    """
    keep = []
    kx, ky, kw, kh = [], [], [], []
    used_nets = set()
    for i in order:
        m = int(midx[i])
        x, y = float(newxy[i, 0]), float(newxy[i, 1])
        w = float(sizes[m, 0]) * 0.5 if m < n_hard else 0.0
        h = float(sizes[m, 1]) * 0.5 if m < n_hard else 0.0
        bad = False
        if sep > 0.0:
            ox, oy = float(pos[m, 0]), float(pos[m, 1])
            for j in range(len(kx)):
                # keep both the destination and the origin away from accepted movers
                if (abs(x - kx[j]) < sep and abs(y - ky[j]) < sep) or \
                   (abs(ox - kx[j]) < sep and abs(oy - ky[j]) < sep):
                    bad = True
                    break
        if not bad and m < n_hard:
            for j in range(len(kx)):
                if abs(x - kx[j]) < w + kw[j] and abs(y - ky[j]) < h + kh[j]:
                    bad = True
                    break
        if not bad and net_sig is not None:
            nets = net_sig[m]
            if used_nets and not used_nets.isdisjoint(nets):
                bad = True
        if bad:
            continue
        keep.append(i)
        kx.append(x)
        ky.append(y)
        kw.append(w)
        kh.append(h)
        if net_sig is not None:
            used_nets.update(net_sig[m])
    return keep


class BatchedProxySearch:
    """Batched exact-proxy local search.

    SCREENING PRECISION. The batch of B candidate moves is only a RANKING problem: the
    accepted subset is re-scored in full float64 before it is kept, and the incumbent is
    always a float64 number. So the batch can be screened in float32, which costs nothing
    in correctness and buys 1.5-1.65x throughput plus double the batch that fits in VRAM --
    and it buys the most on the largest circuits, which is exactly where the method is
    weakest. A float32 mis-ranking can only cost us a good move, never admit a bad one.
    """

    def __init__(self, eng, params, rng):
        self.eng = eng
        self.eng32 = None
        if bool(int(params.get("screen_fp32", 0))) and eng.dtype == torch.float64:
            try:
                from mpx.engine import ProxyEngine
                import os as _os
                self.eng32 = ProxyEngine(_os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "data", f"{eng.name}.npz"), device=str(eng.device),
                    dtype=torch.float32)
            except Exception:                                  # noqa: BLE001
                self.eng32 = None
        # incremental exact scorer: every candidate differs from the incumbent in ONE
        # macro, so only that macro's nets need re-routing. Verified bit-exact against
        # the full engine; up to 52x faster on the largest circuits.
        self.inc = None
        if bool(int(params.get("incremental", 1))):
            try:
                from mpx.incremental import IncrementalScorer
                self.inc = IncrementalScorer(eng)
            except Exception:                                  # noqa: BLE001
                self.inc = None
        self.p = params
        self.rng = rng
        self.dev = eng.device
        self.dt = eng.dtype
        N, nh = eng.N, eng.n_hard
        self.N, self.nh = N, nh
        self.movable = (~eng.fixed).nonzero().flatten()
        self._cycle = None
        self._ptr = 0
        # per-macro connected-pin structure, for the directed operators
        self._build_net_force()
        self._net_sig = None
        self._row_h = None
        self._col_v = None

    def _build_net_force(self):
        eng = self.eng
        self.pin_macro = eng.pin_owner_macro
        self.pin_is_port = eng.pin_is_port

    def net_sig(self):
        """macro -> frozenset of nets it touches (built once; used by the compat filter)."""
        if self._net_sig is None:
            eng = self.eng
            owner = eng.pin_owner_macro.detach().cpu().numpy()
            net = eng.pin_net.detach().cpu().numpy()
            isport = eng.pin_is_port.detach().cpu().numpy()
            sig = [set() for _ in range(eng.N)]
            for o, n, ip in zip(owner, net, isport):
                if not ip:
                    sig[int(o)].add(int(n))
            self._net_sig = [frozenset(s) for s in sig]
        return self._net_sig

    def _next_macros(self, B):
        """Next B movable macros in a reshuffled cycle: uniform coverage, no repeats."""
        out, have = [], 0
        while have < B:
            if self._cycle is None or self._ptr >= self._cycle.numel():
                perm = torch.randperm(self.movable.numel(), device=self.dev)
                self._cycle = self.movable[perm]
                self._ptr = 0
            take = min(B - have, self._cycle.numel() - self._ptr)
            out.append(self._cycle[self._ptr:self._ptr + take])
            self._ptr += take
            have += take
        return torch.cat(out)

    def _net_centroid(self, pos, midx):
        """Centroid of every pin on the nets that touch each selected macro.

        Cheap approximation of "where wirelength wants this macro": the mean position of
        all pins sharing a net with it. Used by the `toward` operator.
        """
        eng = self.eng
        pin_xy = eng.pin_positions(pos.unsqueeze(0))[0]
        # sum pin positions per net, then per macro over its nets
        net_sum = torch.zeros(eng.n_nets, 2, device=self.dev, dtype=self.dt)
        net_cnt = torch.zeros(eng.n_nets, device=self.dev, dtype=self.dt)
        net_sum.index_add_(0, eng.pin_net, pin_xy)
        net_cnt.index_add_(0, eng.pin_net, torch.ones_like(net_cnt[eng.pin_net]))
        macro_sum = torch.zeros(eng.N, 2, device=self.dev, dtype=self.dt)
        macro_cnt = torch.zeros(eng.N, device=self.dev, dtype=self.dt)
        pn = eng.pin_net
        po = eng.pin_owner_macro
        keep = ~eng.pin_is_port
        macro_sum.index_add_(0, po[keep], net_sum[pn[keep]])
        macro_cnt.index_add_(0, po[keep], net_cnt[pn[keep]])
        c = macro_sum / macro_cnt.clamp(min=1).unsqueeze(1)
        return c[midx]

    def _hot_cells(self, pos, k=32):
        """Centers of the k most congested grid cells, plus the row/column demand profiles.

        The demand model is anisotropic and STRIPED, not isotropic: a route lays H demand
        along its SOURCE's row and V demand along its SINK's column. So the way to relieve
        a hot H cell is to move the responsible macro to a different ROW (perpendicular to
        the stripe), and a hot V cell to a different COLUMN -- not to push it radially away,
        which is the wrong direction half the time.

        The suite makes this worse in one direction: hroutes_per_micron is 65.96 and
        vroutes_per_micron 106.96 on every circuit, so horizontal capacity is 1.08-1.87x
        scarcer and the top-5% cells are overwhelmingly horizontal.
        """
        eng = self.eng
        o = eng.evaluate(pos.unsqueeze(0), need_maps=True)
        V, H = o["V"][0], o["H"][0]
        dm = torch.maximum(V, H)
        flat = dm.reshape(-1)
        idx = torch.topk(flat, min(k, flat.numel())).indices
        r = (idx // eng.C).to(self.dt)
        c = (idx % eng.C).to(self.dt)
        hot = torch.stack([(c + 0.5) * eng.gw, (r + 0.5) * eng.gh], dim=1)
        self._row_h = H.mean(dim=1)          # H demand per row
        self._col_v = V.mean(dim=0)          # V demand per column
        return hot

    def propose(self, pos, B, sigma, mix, hot=None, centroid_cache=None):
        eng = self.eng
        midx = self._next_macros(B)
        cur = pos[midx]
        u = torch.rand(B, device=self.dev, dtype=self.dt)
        cw = torch.tensor(np.cumsum(mix.w), device=self.dev, dtype=self.dt)
        kind = torch.searchsorted(cw, u).clamp(max=len(mix.w) - 1)

        aniso = float(self.p.get("aniso", 1.0))     # >1 = prefer vertical displacement
        scale = torch.tensor([1.0 / aniso, aniso], device=self.dev, dtype=self.dt)
        g = torch.randn(B, 2, device=self.dev, dtype=self.dt) * sigma * scale
        ax = torch.zeros(B, 2, device=self.dev, dtype=self.dt)
        horiz = torch.rand(B, device=self.dev) < 0.5
        sgn = torch.where(torch.rand(B, device=self.dev) < 0.5, -1.0, 1.0).to(self.dt)
        ax[horiz, 0] = (sgn * sigma / aniso)[horiz]
        ax[~horiz, 1] = (sgn * sigma * aniso)[~horiz]

        jump_xy = torch.stack([
            torch.rand(B, device=self.dev, dtype=self.dt) * eng.W,
            torch.rand(B, device=self.dev, dtype=self.dt) * eng.H], dim=1)
        jump = jump_xy - cur

        if centroid_cache is not None:
            tgt = centroid_cache[midx]
        else:
            tgt = self._net_centroid(pos, midx)
        frac = torch.rand(B, 1, device=self.dev, dtype=self.dt)
        toward = (tgt - cur) * frac
        # single-axis pulls: they change less per move, so they clear the exact gate more
        # often than the full 2-D pull once the layout is roughly settled
        zero = torch.zeros(B, device=self.dev, dtype=self.dt)
        toward_x = torch.stack([(tgt[:, 0] - cur[:, 0]) * frac[:, 0], zero], dim=1)
        toward_y = torch.stack([zero, (tgt[:, 1] - cur[:, 1]) * frac[:, 0]], dim=1)

        # cold-row / cold-column relocation: sample a destination row (resp. column)
        # inversely to its current H (resp. V) demand, keeping the other coordinate.
        if getattr(self, "_row_h", None) is not None:
            w_row = torch.softmax(-self._row_h / (self._row_h.std() + 1e-9)
                                  * float(self.p.get("cold_temp", 2.0)), dim=0)
            rsel = torch.multinomial(w_row, B, replacement=True).to(self.dt)
            jit = torch.rand(B, device=self.dev, dtype=self.dt)
            coldrow = torch.stack([torch.zeros(B, device=self.dev, dtype=self.dt),
                                   (rsel + jit) * eng.gh - cur[:, 1]], dim=1)
            w_col = torch.softmax(-self._col_v / (self._col_v.std() + 1e-9)
                                  * float(self.p.get("cold_temp", 2.0)), dim=0)
            csel = torch.multinomial(w_col, B, replacement=True).to(self.dt)
            jit2 = torch.rand(B, device=self.dev, dtype=self.dt)
            coldcol = torch.stack([(csel + jit2) * eng.gw - cur[:, 0],
                                   torch.zeros(B, device=self.dev, dtype=self.dt)], dim=1)
        else:
            coldrow = coldcol = None

        if hot is not None and hot.numel() > 0:
            d = cur.unsqueeze(1) - hot.unsqueeze(0)                # [B, K, 2]
            dist = d.norm(dim=2).clamp(min=1e-6)
            nearest = dist.argmin(dim=1)
            dirv = d[torch.arange(B, device=self.dev), nearest]
            dirv = dirv / dirv.norm(dim=1, keepdim=True).clamp(min=1e-6)
            away = dirv * sigma * (1.0 + torch.rand(B, 1, device=self.dev, dtype=self.dt))
        else:
            away = g

        if coldrow is None:
            coldrow, coldcol = g, g
        d = torch.where((kind == 0).view(-1, 1), g,
            torch.where((kind == 1).view(-1, 1), ax,
            torch.where((kind == 2).view(-1, 1), jump,
            torch.where((kind == 3).view(-1, 1), toward,
            torch.where((kind == 4).view(-1, 1), away,
            torch.where((kind == 5).view(-1, 1), coldrow,
            torch.where((kind == 6).view(-1, 1), coldcol,
            torch.where((kind == 7).view(-1, 1), toward_x, toward_y))))))))

        new = cur + d
        hw = eng.sizes[midx, 0] * 0.5
        hh = eng.sizes[midx, 1] * 0.5
        eps = 1e-6
        new[:, 0] = new[:, 0].clamp(min=0) .minimum(torch.full_like(new[:, 0], eng.W))
        new[:, 1] = new[:, 1].clamp(min=0).minimum(torch.full_like(new[:, 1], eng.H))
        new[:, 0] = torch.maximum(torch.minimum(new[:, 0], eng.W - hw - eps), hw + eps)
        new[:, 1] = torch.maximum(torch.minimum(new[:, 1], eng.H - hh - eps), hh + eps)

        bad = _hard_conflict(new, midx, pos, eng.sizes, eng.nh if hasattr(eng, "nh") else eng.n_hard)
        return midx, new, ~bad, kind

    def run(self, pos, budget_s, log=None):
        eng, p = self.eng, self.p
        B = int(p.get("batch", 512))
        sigma0 = float(p.get("sigma0", 0.15)) * max(eng.W, eng.H)
        sigma1 = float(p.get("sigma1", 0.004)) * max(eng.W, eng.H)
        mix = MoveMix([p.get(f"w_{k}", 1.0) for k in MoveMix.NAMES], sigma0, eng.W, eng.H)
        hot_k = int(p.get("hot_k", 32))
        refresh = int(p.get("refresh_every", 8))
        single_accept = bool(int(p.get("single_accept", 0)))
        sep = float(p.get("accept_sep", 0.0)) * max(eng.gw, eng.gh)
        use_sig = bool(int(p.get("accept_net_disjoint", 0)))

        pos = pos.clone()
        base = float(eng.evaluate(pos.unsqueeze(0))["proxy"][0])
        best, best_pos = base, pos.clone()
        t0 = time.time()
        rounds = 0
        accepted = 0
        traj = [(0.0, base)]
        hot = None
        cent = None
        nops = len(MoveMix.NAMES)
        op_try = np.zeros(nops); op_hit = np.zeros(nops); op_gain = np.zeros(nops)
        # Adaptive operator selection (probability matching over measured yield).
        # Every operator competes for the same scarce resource -- slots in the batch --
        # and the exact gate tells us, for free and every round, what each slot bought.
        # So the mixture is not a hyperparameter to tune offline; it is estimated online,
        # per circuit, from the yield the operator is actually delivering right now.
        adapt = bool(int(p.get("adaptive_ops", 1)))
        alpha = float(p.get("adapt_alpha", 0.15))
        floor = float(p.get("adapt_floor", 0.04))
        q = np.array(mix.w, dtype=np.float64)
        q = q / max(q.sum(), 1e-12)

        while time.time() - t0 < budget_s:
            frac = min(1.0, (time.time() - t0) / max(budget_s, 1e-9))
            sigma = sigma0 * (sigma1 / sigma0) ** frac
            if rounds % refresh == 0:
                hot = self._hot_cells(pos, hot_k) if hot_k > 0 else None
                cent = self._net_centroid(pos, torch.arange(eng.N, device=self.dev))
            if adapt:
                mix.w = (1.0 - floor * nops) * (q / max(q.sum(), 1e-12)) + floor
            midx, new, okmask, kind = self.propose(pos, B, sigma, mix, hot, cent)
            if not bool(okmask.any()):
                rounds += 1
                continue

            if self.inc is not None:
                base_i = self.inc.set_base(pos)
                costs = self.inc.score_moves(midx, new)
                delta = costs - base_i
                delta = torch.where(okmask, delta, torch.full_like(delta, float("inf")))
                cand = None
            else:
                cand = pos.unsqueeze(0).repeat(B, 1, 1)
                bidx = torch.arange(B, device=self.dev)
                cand[bidx, midx] = new
            if cand is not None and self.eng32 is not None:
                c32 = self.eng32.evaluate(cand.to(torch.float32))["proxy"]
                base32 = float(self.eng32.evaluate(pos.unsqueeze(0).to(torch.float32))["proxy"][0])
                costs = c32.to(self.dt)
                delta = costs - base32
                delta = torch.where(okmask, delta, torch.full_like(delta, float("inf")))
            elif cand is not None:
                costs = eng.evaluate(cand)["proxy"]
                delta = costs - base
                delta = torch.where(okmask, delta, torch.full_like(delta, float("inf")))

            imp = torch.nonzero(delta < -1e-12).flatten()
            rounds += 1
            # per-operator yield: proposals offered, and how many were exactly improving.
            # In a batched exact-gated search the scarce resource is BATCH SLOTS, so the
            # figure of merit for an operator is its improving-hit-rate per slot, not how
            # well-motivated it looks.
            op_try += torch.bincount(kind, minlength=len(MoveMix.NAMES)).cpu().numpy()
            if imp.numel():
                op_hit += torch.bincount(kind[imp], minlength=len(MoveMix.NAMES)).cpu().numpy()
                op_gain += torch.bincount(kind[imp], weights=-delta[imp].to(torch.float64),
                                          minlength=len(MoveMix.NAMES)).cpu().numpy()
            if imp.numel() == 0:
                continue
            order = imp[torch.argsort(delta[imp])].tolist()
            if single_accept:          # ablation: classical one-move-per-round acceptance
                order = order[:1]
            keep = _select_compatible(order, midx, new, pos, eng.sizes, eng.n_hard,
                                      sep=sep, net_sig=self.net_sig() if use_sig else None)
            if not keep:
                continue

            # single-move deltas are not additive: verify the combined move, halve on failure
            k = len(keep)
            while k >= 1:
                sel = keep[:k]
                trial = pos.clone()
                si = midx[sel]
                trial[si] = new[sel]
                c = float(eng.evaluate(trial.unsqueeze(0))["proxy"][0])
                if c < base - 1e-12:
                    pos = trial
                    base = c
                    accepted += len(sel)
                    if c < best:
                        best, best_pos = c, pos.clone()
                    break
                k //= 2
            if adapt:
                tr = np.bincount(kind.cpu().numpy(), minlength=nops).astype(np.float64)
                gn = np.zeros(nops)
                if imp.numel():
                    gn = np.bincount(kind[imp].cpu().numpy(),
                                     weights=(-delta[imp]).to(torch.float64).cpu().numpy(),
                                     minlength=nops)
                yld = gn / np.maximum(tr, 1.0)
                if yld.sum() > 0:
                    yld = yld / yld.sum()
                    q = (1 - alpha) * q + alpha * yld
            traj.append((time.time() - t0, base))
            if log and rounds % 50 == 0:
                log(f"    r{rounds:5d} t={time.time()-t0:6.1f}s proxy={base:.5f} "
                    f"acc={accepted} sigma={sigma:.3f}")

        return best_pos, {"proxy": best, "rounds": rounds, "accepted": accepted,
                          "evals": rounds * B, "traj": traj,
                          "op_try": op_try, "op_hit": op_hit, "op_gain": op_gain,
                          "op_names": list(MoveMix.NAMES), "op_final_w": q.tolist()}
