"""Extract each ICCAD04 benchmark from the OFFICIAL TILOS evaluator into a compact
npz our fast engine can consume.

We do NOT use `benchmarks/processed/public/*.pt`: that loader hardcodes
net_weights=1.0, while the official `get_wirelength()` weights every net by its
DRIVER PIN's weight (values up to 10 in ibm01), and normalizes by `plc.net_cnt`
(7269 for ibm01) rather than by the number of driver-keyed nets (5993). Either
discrepancy silently shifts the wirelength term, so we read the ground truth
straight out of the PlacementCost object.

Layout of the emitted npz (all float32/int32 unless noted):

  canvas            [2]      (W, H)
  grid              [2] i32  (grid_row, grid_col)
  routes_per_micron [2]      (hroutes, vroutes)
  routing_alloc     [2]      (hrouting_alloc, vrouting_alloc)
  scalars           [3]      (smooth_range, overlap_thres, net_cnt_normalizer)
  counts            [3] i32  (n_hard, n_soft, n_ports)
  pos0              [N,2]    initial macro centers (hard first, then soft)
  sizes             [N,2]
  fixed             [N] bool
  port_pos          [P,2]
  pin_owner         [M] i32  owner index: macros [0,N), ports [N, N+P)
  pin_off           [M,2]    pin offset from owner center (0 for ports/soft)
  net_ptr           [n_nets+1] i32   pins of net i are [net_ptr[i], net_ptr[i+1]);
                                     the FIRST pin of every net is its driver
  net_weight        [n_nets]
  macro_names / port_names  (object arrays, for artifact traceability)
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(ROOT, "challenge")
sys.path.insert(0, os.path.join(CHALLENGE, "external", "MacroPlacement", "CodeElements", "Plc_client"))

BENCHMARKS = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
              "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
              "ibm16", "ibm17", "ibm18"]


def testcase_dir(name):
    return os.path.join(CHALLENGE, "external", "MacroPlacement", "Testcases", "ICCAD04", name)


def load_plc(name):
    from plc_client_os import PlacementCost
    d = testcase_dir(name)
    plc = PlacementCost(os.path.join(d, "netlist.pb.txt"))
    plc.restore_placement(os.path.join(d, "initial.plc"), ifInital=True, ifReadComment=True)
    return plc


def extract(name, out_dir):
    plc = load_plc(name)

    hard = list(plc.hard_macro_indices)
    soft = list(plc.soft_macro_indices)
    ports = list(plc.port_indices)
    n_hard, n_soft, n_ports = len(hard), len(soft), len(ports)
    N = n_hard + n_soft

    pos0, sizes, fixed, macro_names = [], [], [], []
    for idx in hard + soft:
        m = plc.modules_w_pins[idx]
        x, y = m.get_pos()
        pos0.append([x, y])
        sizes.append([m.get_width(), m.get_height()])
        fixed.append(bool(m.get_fix_flag()))
        macro_names.append(m.get_name())

    port_pos, port_names = [], []
    for idx in ports:
        m = plc.modules_w_pins[idx]
        x, y = m.get_pos()
        port_pos.append([x, y])
        port_names.append(m.get_name())

    # plc module index -> our owner index
    owner_of = {}
    for i, idx in enumerate(hard):
        owner_of[idx] = i
    for i, idx in enumerate(soft):
        owner_of[idx] = n_hard + i
    for i, idx in enumerate(ports):
        owner_of[idx] = N + i

    def pin_entry(pin_name):
        """(owner_index, off_x, off_y) for a pin name, exactly as __get_pin_position does."""
        pin_idx = plc.mod_name_to_indices[pin_name]
        node = plc.modules_w_pins[pin_idx]
        if node.get_type() == "PORT":
            return owner_of[pin_idx], 0.0, 0.0
        ref = plc.get_ref_node_id(pin_idx)
        ox, oy = node.get_offset()
        return owner_of[ref], float(ox), float(oy)

    pin_owner, pin_off, net_ptr, net_weight = [], [], [0], []
    for driver, sinks in plc.nets.items():
        w = plc.modules_w_pins[plc.mod_name_to_indices[driver]].get_weight()
        for pn in [driver] + list(sinks):
            o, ox, oy = pin_entry(pn)
            pin_owner.append(o)
            pin_off.append([ox, oy])
        net_ptr.append(len(pin_owner))
        net_weight.append(float(w))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        out,
        canvas=np.array([plc.width, plc.height], dtype=np.float64),
        grid=np.array([plc.grid_row, plc.grid_col], dtype=np.int32),
        routes_per_micron=np.array([plc.hroutes_per_micron, plc.vroutes_per_micron], dtype=np.float64),
        routing_alloc=np.array([plc.hrouting_alloc, plc.vrouting_alloc], dtype=np.float64),
        scalars=np.array([plc.smooth_range, plc.overlap_thres, plc.net_cnt], dtype=np.float64),
        counts=np.array([n_hard, n_soft, n_ports], dtype=np.int32),
        pos0=np.array(pos0, dtype=np.float64),
        sizes=np.array(sizes, dtype=np.float64),
        fixed=np.array(fixed, dtype=bool),
        port_pos=np.array(port_pos, dtype=np.float64).reshape(-1, 2),
        pin_owner=np.array(pin_owner, dtype=np.int32),
        pin_off=np.array(pin_off, dtype=np.float64).reshape(-1, 2),
        net_ptr=np.array(net_ptr, dtype=np.int32),
        net_weight=np.array(net_weight, dtype=np.float64),
        macro_names=np.array(macro_names, dtype=object),
        port_names=np.array(port_names, dtype=object),
    )
    return out, dict(n_hard=n_hard, n_soft=n_soft, n_ports=n_ports,
                     n_nets=len(net_weight), n_pins=len(pin_owner),
                     grid=(plc.grid_row, plc.grid_col), net_cnt=plc.net_cnt,
                     canvas=(plc.width, plc.height))


if __name__ == "__main__":
    names = sys.argv[1:] or BENCHMARKS
    out_dir = os.path.join(ROOT, "data")
    for n in names:
        path, info = extract(n, out_dir)
        print(f"{n}: {info}", flush=True)
