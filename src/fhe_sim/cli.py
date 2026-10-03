"""Command-line front end.

    fhe-sim                                  # one ARK-like bootstrap on the ARK-class digital design
    fhe-sim --hw small                       # an NTT-bound design
    fhe-sim --min-ks --seeded-keys --otf-pt  # the key-traffic reduction techniques
    fhe-sim --sweep-sram 128 256 512 1024    # latency, energy and area against scratchpad size
    fhe-sim --area                           # area breakdown (7 nm), yield, cost and PPA metrics
    fhe-sim --hw small --optical ideal       # hypothetical precision-free optical NTT
    fhe-sim --hw small --optical hybrid --enob 14 --block 16
    fhe-sim --counts                         # operation counts per stage, no timing
    fhe-sim --power-mode worst-case          # one fixed TDP clock instead of the power manager
    fhe-sim --trace boot.json                # open in https://ui.perfetto.dev
    fhe-sim --dump-trace t.json / --replay t.json
    fhe-sim --openfhe-log calibration/openfhe_trace/sparse16.log.gz --hw cpu   # a real OpenFHE bootstrap
    fhe-sim --heir calibration/heir/lola.ckks.mlir.gz                    # a program compiled by HEIR
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .hardware import ACCELERATORS, OpticalEngine
from .metrics import format_report, summarise
from .params import PARAMS
from .ppa import format_area, ppa_metrics
from .search import analytic_bound, pareto_nd, sram_sweep
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
    p.add_argument("--stc-first", action="store_true",
                   help="SlotToCoeff first, at the bottom of the modulus chain (OpenFHE's BTSlotsEncoding)")
    p.add_argument("--lazy-moddown", action="store_true",
                   help="OpenFHE's BSGS: rotations stay in Q*P, one ModDown per DFT level")
    p.add_argument("--sram", type=int, metavar="MiB")
    p.add_argument("--hbm", type=float, metavar="GB/s")
    p.add_argument("--memsim", action="store_true",
                   help="time HBM chunks with Memory_System_Sim's command-level HBM model (pip install it)")
    p.add_argument("--ntt", type=float, metavar="BFLY/CYCLE")
    p.add_argument("--mac", type=float, metavar="LANES")
    p.add_argument("--tdp", type=float, metavar="W")
    p.add_argument("--no-tdp", action="store_true", help="do not enforce the TDP")
    p.add_argument("--power-mode", choices=["dynamic", "worst-case"],
                   help="dynamic power manager (default) or one worst-case TDP clock")
    p.add_argument("--dvfs", action="store_true", help="lower the clock when memory-bound")
    p.add_argument("--optical", choices=["off", "hybrid", "ideal"])
    p.add_argument("--enob", type=int)
    p.add_argument("--block", type=int)
    p.add_argument("--sweep-sram", type=int, nargs="+", metavar="MiB")
    p.add_argument("--area", action="store_true",
                   help="also print the area breakdown (7 nm, illustrative), die yield, cost and PPA metrics")
    p.add_argument("--counts", action="store_true", help="print operation counts per stage")
    p.add_argument("--trace", metavar="FILE", help="write a Chrome trace-event JSON")
    p.add_argument("--dump-trace", metavar="FILE", help="write the operation trace as JSON")
    p.add_argument("--replay", metavar="FILE", help="simulate an operation trace from JSON")
    p.add_argument("--heir", metavar="FILE",
                   help="simulate the server function of HEIR ckks-dialect output (.mlir or .mlir.gz)")
    p.add_argument("--openfhe-log", metavar="FILE",
                   help="simulate a kernel stream recorded from instrumented OpenFHE (.log or .log.gz)")
    p.add_argument("--json", action="store_true")
    return p


def hardware_from_args(a):
    hw = ACCELERATORS[a.hw]
    kw = {k: v for k, v in (("sram_mib", a.sram), ("hbm_gbps", a.hbm), ("ntt_bfly_per_cycle", a.ntt),
                            ("mac_lanes", a.mac), ("tdp_w", a.tdp)) if v is not None}
    if a.no_tdp:
        kw["enforce_tdp"] = False
    if getattr(a, "memsim", False):
        from memsim.fhe import HBMChunkModel     # optional dependency: Memory_System_Sim
        kw["memory"] = HBMChunkModel()
    if a.power_mode:
        kw["power_mode"] = a.power_mode
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
                       seeded_keys=a.seeded_keys, otf_plaintexts=a.otf_pt, lazy_moddown=a.lazy_moddown,
                       stc_first=a.stc_first)
    if a.heir:
        from .heir_frontend import compile_ir
        prog = compile_ir(a.heir, opts)
        trace = prog.trace
        for note in prog.notes:
            print("note:", note)
    elif a.openfhe_log:
        from .openfhe_trace import log_to_trace, read_log
        trace = log_to_trace(read_log(a.openfhe_log))
    elif a.replay:
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
        rows = sram_sweep(params, hw, a.sweep_sram, opts)
        front = pareto_nd(rows)
        print(f"{'SRAM MiB':>9} {'latency':>10} {'keys GB':>8} {'HBM GB':>8} {'key %':>6} {'mJ':>7} {'mm²':>7} "
              f"{'/s/mm²':>7} {'/J':>6}  Pareto  verdict")
        for r in rows:
            print(f"{r['sram_mib']:9d} {1e3 * r['latency_s']:8.2f}ms {r['key_GB']:8.2f} {r['hbm_GB']:8.2f} "
                  f"{100 * r['key_share']:5.0f}% {1e3 * r['energy_J']:7.0f} {r['area_mm2']:7.1f} "
                  f"{r['perf_per_mm2']:7.3f} {r['perf_per_W']:6.2f}  {'  *   ' if r in front else '      '}  {r['bound']}")
        print("Pareto: not dominated in (latency, energy per bootstrap, area). Area is 7 nm and illustrative.")
        return
    res = simulate(trace, SimConfig(hw, trace=bool(a.trace), dvfs=a.dvfs))
    if a.trace and res.tracer:
        write_trace(res.tracer, a.trace)
    m = summarise(res)
    m["analytic_bound_s"] = analytic_bound(trace, hw)["bound_s"]
    q = ppa_metrics(m, hw)
    if a.json:
        print(json.dumps(dict(m, ppa=q), indent=2))
    else:
        print(format_report(m))
        print(f"analytic lower bound {1e3 * m['analytic_bound_s']:.2f} ms "
              f"(simulated / bound = {m['latency_s'] / m['analytic_bound_s']:.2f})")
        if a.area:
            print(format_area(q))


if __name__ == "__main__":
    main()
