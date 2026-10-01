"""Design-space exploration: sweeps, an analytic bound, bisection and a Pareto front.

* ``analytic_bound``  a roofline-style lower bound on bootstrap latency computed
  straight from the trace, with no event simulation. It is used to prune a
  search and is checked against the simulator in the tests (sim >= bound).
* ``sram_sweep``      key traffic and latency as the scratchpad grows.
* ``min_sram``        bisection for the smallest scratchpad that brings a
  traffic metric (HBM or key GB per bootstrap) under a target; valid because
  both fall monotonically with SRAM size, which the behavioural tests check.
* ``design_sweep``    a grid over hardware knobs, run in parallel processes.
* ``pareto``          non-dominated (latency, energy) points.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product

from .hardware import Accelerator, CostModel
from .metrics import summarise
from .params import CKKSParams
from .sim import simulate
from .workload import BootOptions, Trace, bootstrap_trace


def analytic_bound(trace: Trace, hw: Accelerator) -> dict:
    """Busy time each resource *must* accumulate: compute from the kernels, and HBM from the
    compulsory traffic (each distinct key and plaintext once, each external input once)."""
    p = trace.params
    cm = CostModel(hw, p.log_n, p.q_bits, hw.tdp_clock())
    busy = {"ntt": 0.0, "mac": 0.0, "auto": 0.0, "optical": 0.0}
    seen: dict[str, int] = {}
    for o in trace.ops:
        for k in o.kernels:
            for seg in cm.segments(k):
                busy[seg.unit] += seg.time
        for name, b in ([o.key] if o.key else []) + list(o.pts):
            seen[name] = max(seen.get(name, 0), b)
    compulsory = sum(seen.values()) + sum(trace.sizes[x] for x in trace.external)
    busy["hbm"] = compulsory / (hw.hbm_gbps * 1e9)
    top = max(busy, key=busy.get)
    return {"bound_s": busy[top], "resource": top, "busy_s": busy}


def run_point(args) -> dict:
    p, hw, opts = args
    m = summarise(simulate(bootstrap_trace(p, opts), hw))
    return {"latency_s": m["per_bootstrap_s"], "energy_J": m["energy"]["per_bootstrap_J"],
            "key_GB": m["hbm_bytes"]["key"] / 1e9, "hbm_GB": m["hbm_bytes"]["total"] / 1e9,
            "key_share": m["hbm_bytes"]["key_share"], "bound": m["bound"],
            "peak_W": m["energy"]["peak_power_W"]}


def sram_sweep(p: CKKSParams, hw: Accelerator, sizes_mib: list[int],
               opts: BootOptions | None = None) -> list[dict]:
    out = []
    for mib in sizes_mib:
        r = run_point((p, replace(hw, sram_mib=mib), opts or BootOptions()))
        out.append({"sram_mib": mib, **r})
    return out


def min_sram(p: CKKSParams, hw: Accelerator, target_gb: float, lo_mib: int, hi_mib: int,
             opts: BootOptions | None = None, metric: str = "hbm_GB", tol_mib: int = 16) -> dict:
    """Smallest scratchpad (to within tol) whose ``metric`` per bootstrap is <= target_gb."""
    runs = 0

    def gb(mib):
        nonlocal runs
        runs += 1
        return run_point((p, replace(hw, sram_mib=mib), opts or BootOptions()))[metric]

    if gb(hi_mib) > target_gb:
        return {"sram_mib": None, "runs": runs}
    while hi_mib - lo_mib > tol_mib:
        mid = (lo_mib + hi_mib) // 2
        if gb(mid) <= target_gb:
            hi_mib = mid
        else:
            lo_mib = mid
    return {"sram_mib": hi_mib, "runs": runs}


def design_sweep(p: CKKSParams, base: Accelerator, grid: dict[str, list],
                 opts: BootOptions | None = None, workers: int = 1) -> list[dict]:
    keys = list(grid)
    points = [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]
    jobs = [(p, replace(base, **pt), opts or BootOptions()) for pt in points]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(run_point, jobs))
    else:
        res = [run_point(j) for j in jobs]
    return [{**pt, **r} for pt, r in zip(points, res)]


def pareto(rows: list[dict], x: str = "latency_s", y: str = "energy_J") -> list[dict]:
    """Points not dominated in both x and y (lower is better), sorted by x."""
    front = []
    for r in sorted(rows, key=lambda r: (r[x], r[y])):
        if not front or r[y] < front[-1][y]:
            front.append(r)
    return front
