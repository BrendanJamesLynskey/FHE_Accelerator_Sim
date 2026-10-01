"""Command-line front end.

    fhe-sim                                  # one ARK-like bootstrap on the ARK-class digital design
    fhe-sim --hw small                       # an NTT-bound design
    fhe-sim --min-ks --seeded-keys --otf-pt  # the key-traffic reduction techniques
    fhe-sim --sweep-sram 128 256 512 1024    # key traffic against scratchpad size
    fhe-sim --hw small --optical ideal       # hypothetical precision-free optical NTT
    fhe-sim --hw small --optical hybrid --enob 14 --block 16
    fhe-sim --counts                         # operation counts per stage, no timing
    fhe-sim --trace boot.json                # open in https://ui.perfetto.dev
    fhe-sim --dump-trace t.json / --replay t.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .hardware import ACCELERATORS, OpticalEngine
from .metrics import format_report, summarise
from .params import PARAMS
from .search import analytic_bound, sram_sweep
from .sim import SimConfig, simulate
from .trace import write_trace
from .workload import (BootOptions, bootstrap_trace, dump_trace, he_op_trace, load_trace,
                       summarise_trace)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fhe-sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", choices=PARAMS, default="ark")
    p.add_argument("--hw", choices=ACCELERATORS, default="ark")
    p.add_argument("--op", choices=["bootstrap", "hmult", "hrot"], default="bootstrap")
    p.add_argument("--n-boot", type=int, default=1, help="bootstraps issued back to back")
    p.add_argument("--no-hoist", action="store_true", help="disable hoisted baby-step rotations")
    p.add_argument("--min-ks", action="store_true", help="one key per BSGS loop (ARK Min-KS)")
    p.add_argument("--seeded-keys", action="store_true", help="regenerate evk 'a' halves on chip")
    p.add_argument("--otf-pt", action="store_true", help="generate DFT plaintexts on chip")
    p.add_argument("--sram", type=int, metavar="MiB")
    p.add_argument("--hbm", type=float, metavar="GB/s")
    p.add_argument("--ntt", type=float, metavar="BFLY/CYCLE")
    p.add_argument("--mac", type=float, metavar="LANES")
    p.add_argument("--tdp", type=float, metavar="W")
    p.add_argument("--no-tdp", action="store_true", help="do not enforce the TDP")
    p.add_argument("--dvfs", action="store_true", help="lower the clock when memory-bound")
    p.add_argument("--optical", choices=["off", "hybrid", "ideal"])
    p.add_argument("--enob", type=int)
    p.add_argument("--block", type=int)
    p.add_argument("--sweep-sram", type=int, nargs="+", metavar="MiB")
    p.add_argument("--counts", action="store_true", help="print operation counts per stage")
    p.add_argument("--trace", metavar="FILE", help="write a Chrome trace-event JSON")
    p.add_argument("--dump-trace", metavar="FILE", help="write the operation trace as JSON")
    p.add_argument("--replay", metavar="FILE", help="simulate an operation trace from JSON")
    p.add_argument("--json", action="store_true")
    return p


def hardware_from_args(a):
    hw = ACCELERATORS[a.hw]
    kw = {k: v for k, v in (("sram_mib", a.sram), ("hbm_gbps", a.hbm), ("ntt_bfly_per_cycle", a.ntt),
                            ("mac_lanes", a.mac), ("tdp_w", a.tdp)) if v is not None}
    if a.no_tdp:
        kw["enforce_tdp"] = False
    if a.optical == "off":
        kw["optical"] = None
    elif a.optical in ("hybrid", "ideal") or a.enob or a.block:
        base = hw.optical or (ACCELERATORS["ideal-optical"].optical if a.optical == "ideal"
                              else OpticalEngine())
        o = replace(base, ideal=(a.optical == "ideal") or (a.optical is None and base.ideal))
        if a.enob:
            o = replace(o, enob=a.enob)
        if a.block:
            o = replace(o, block=a.block)
        kw["optical"] = o
    return replace(hw, **kw)


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    hw = hardware_from_args(a)
    params = PARAMS[a.params]
    opts = BootOptions(n_boot=a.n_boot, hoisting=not a.no_hoist, min_ks=a.min_ks,
                       seeded_keys=a.seeded_keys, otf_plaintexts=a.otf_pt)
    if a.replay:
        trace = load_trace(a.replay)
    elif a.op == "bootstrap":
        trace = bootstrap_trace(params, opts)
    else:
        trace = he_op_trace(params, a.op)
    if a.dump_trace:
        dump_trace(trace, a.dump_trace)
    if a.counts:
        for stage, c in summarise_trace(trace).items():
            print(f"{stage:<9} ops {c['ops']:4d}  hmult {c['hmult']:3d}  hrot {c['hrot']:3d}  "
                  f"pmult {c['pmult']:4d}  keys {c['distinct_keys']:3d}  NTT limbs {c['ntt_limbs'] + c['intt_limbs']:6d}  "
                  f"key {c['key_bytes'] / 1e9:6.2f} GB  pt {c['pt_bytes'] / 1e9:5.2f} GB")
        return
    if a.sweep_sram:
        print(f"{'SRAM MiB':>9} {'latency':>10} {'keys GB':>8} {'HBM GB':>8} {'key %':>6}  verdict")
        for r in sram_sweep(params, hw, a.sweep_sram, opts):
            print(f"{r['sram_mib']:9d} {1e3 * r['latency_s']:8.2f}ms {r['key_GB']:8.2f} {r['hbm_GB']:8.2f} "
                  f"{100 * r['key_share']:5.0f}%  {r['bound']}")
        return
    res = simulate(trace, SimConfig(hw, trace=bool(a.trace), dvfs=a.dvfs))
    if a.trace and res.tracer:
        write_trace(res.tracer, a.trace)
    m = summarise(res)
    m["analytic_bound_s"] = analytic_bound(trace, hw)["bound_s"]
    if a.json:
        print(json.dumps(m, indent=2))
    else:
        print(format_report(m))
        print(f"analytic lower bound {1e3 * m['analytic_bound_s']:.2f} ms "
              f"(simulated / bound = {m['latency_s'] / m['analytic_bound_s']:.2f})")


if __name__ == "__main__":
    main()
