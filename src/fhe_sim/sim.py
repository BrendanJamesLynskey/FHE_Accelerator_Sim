"""The SimPy discrete-event model of an FHE accelerator running an operation trace.

Structure::

    trace (program order) -> issuer (window W) -> scratchpad (functional, at issue)
                                    |                     | misses / spills
                                    v                     v
                      per-op process: wait deps -> HBM loads (shared, FCFS chunks)
                                    -> kernels on NTT / MAC / AUTO / OPTICAL units

* **Functional units** are ``simpy.Resource`` objects (capacity 1, aggregate
  throughput); a kernel holds its unit for the time the cost model gives it.
* **HBM** is one shared ``simpy.Resource``; every transfer is split into chunks
  so concurrent transfers interleave and share the bandwidth.
* **The scratchpad** is a capacity-limited LRU store of ciphertexts, keys and
  plaintexts. Hits and misses are decided at issue, in program order, so the
  byte counts are independent of timing (and therefore deterministic). Misses
  become HBM loads; evicting a live, dirty ciphertext becomes an HBM write.
* **Dependencies**: an op starts its kernels when every producer of its inputs
  has finished. Key and plaintext loads do not depend on data, so they are
  prefetched as soon as the op is issued (decoupled access/execute, as in F1).

The JavaScript port in ``web/sim_engine.js`` reproduces this file, including
SimPy's event ordering, and is tested to match it exactly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import simpy

from .hardware import UNITS, Accelerator, CostModel
from .params import MiB
from .trace import Tracer
from .workload import STAGES, Trace


@dataclass
class SimConfig:
    hw: Accelerator
    trace: bool = False          # record a Chrome/Perfetto trace
    dvfs: bool = False           # lower the clock when the run is memory-bound (two passes)
    clock: float | None = None   # force a clock fraction (otherwise TDP-limited)


# ──────────────────────────────────────────────────────────── scratchpad ──
class Scratchpad:
    """LRU over named objects of known size. Pinned objects are never evicted."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.used = 0
        self.items: OrderedDict[str, list] = OrderedDict()     # name -> [size, cls, dirty]

    def hit(self, name: str, size: int) -> bool:
        it = self.items.get(name)
        if it is None or it[0] < size:
            return False
        self.items.move_to_end(name)
        return True

    def free(self, name: str) -> None:
        it = self.items.pop(name, None)
        if it is not None:
            self.used -= it[0]

    def alloc(self, name: str, size: int, cls: str, dirty: bool, pinned: set, evicted: list) -> bool:
        """Insert (replacing any smaller copy). Returns False if it cannot fit (bypass)."""
        self.free(name)
        if size > self.cap:
            return False
        while self.used + size > self.cap:
            victim = None
            for n in self.items:
                if n not in pinned:
                    victim = n
                    break
            if victim is None:
                return False
            vsize, vcls, vdirty = self.items[victim]
            self.free(victim)
            evicted.append((victim, vsize, vcls, vdirty))
        self.items[name] = [size, cls, dirty]
        self.used += size
        return True


@dataclass
class OpPlan:
    """What the scratchpad decided for one op at issue time."""

    key_load: int = 0
    pt_load: int = 0
    ct_load: list = field(default_factory=list)     # (object, bytes) to reload from HBM
    writebacks: list = field(default_factory=list)  # (object, bytes) spilled to HBM
    out_write: int = 0                              # output written straight to HBM


# ──────────────────────────────────────────────────────────── simulation ──
@dataclass
class Stats:
    busy: dict = field(default_factory=lambda: {u: 0.0 for u in UNITS + ["hbm"]})
    work: dict = field(default_factory=lambda: {u: 0.0 for u in UNITS})
    energy: dict = field(default_factory=lambda: {u: 0.0 for u in UNITS + ["hbm"]})
    bytes: dict = field(default_factory=lambda: {"key": 0, "pt": 0, "ct_read": 0, "ct_write": 0})
    dac: float = 0.0
    adc: float = 0.0
    stage_busy: dict = field(default_factory=dict)  # stage -> unit -> seconds
    stage_first: dict = field(default_factory=dict)
    stage_last: dict = field(default_factory=dict)
    peak_w: float = 0.0


@dataclass
class SimResult:
    trace: Trace
    hw: Accelerator
    clock: float
    horizon: float
    stats: Stats
    op_start: list
    op_end: list
    static_w: float
    tracer: dict | None = None


class Simulation:
    def __init__(self, trace: Trace, cfg: SimConfig, clock: float):
        self.t, self.cfg, self.hw = trace, cfg, cfg.hw
        self.cost = CostModel(self.hw, trace.params.log_n, trace.params.q_bits, clock)
        self.env = simpy.Environment()
        self.units = {u: simpy.Resource(self.env, capacity=1) for u in UNITS}
        self.hbm = simpy.Resource(self.env, capacity=1)
        self.st = Stats()
        self.tracer = Tracer(cfg.trace)
        n = len(trace.ops)
        self.producer = {o.output: o.id for o in trace.ops}
        self.last_use: dict[str, int] = {}
        for o in trace.ops:
            for x in o.inputs:
                self.last_use[x] = o.id
        reserve = trace.params.ks_working_set()
        if self.hw.sram_bytes < reserve:
            raise ValueError(f"scratchpad {self.hw.sram_mib} MiB is smaller than one key switch's "
                             f"working set ({reserve / MiB:.0f} MiB)")
        self.spad = Scratchpad(self.hw.sram_bytes - reserve)
        self.done = [False] * n
        self.done_ev = [None] * n
        self.wb_ev: dict[str, simpy.Event] = {}
        self.op_start = [0.0] * n
        self.op_end = [0.0] * n
        self.inflight = 0
        self.slot_ev = None
        self.p_now = 0.0
        self.chunk = self.hw.hbm_chunk_mib * MiB
        self.hbm_w = self.hw.hbm_gbps * 1e9 * self.hw.pj_hbm_byte * 1e-12
        self.env.process(self.issuer())

    # ── scratchpad decisions (functional, program order) ─────────────
    def plan(self, o) -> OpPlan:
        sp, pl, sizes = self.spad, OpPlan(), self.t.sizes
        pinned = set(o.inputs)
        pinned.add(o.output)
        if o.key:
            pinned.add(o.key[0])
        for pid, _ in o.pts:
            pinned.add(pid)
        ev: list = []
        for x in o.inputs:
            if not sp.hit(x, sizes[x]):
                pl.ct_load.append((x, sizes[x]))
                sp.alloc(x, sizes[x], "ct", False, pinned, ev)
        if o.key:
            kid, kb = o.key
            if not sp.hit(kid, kb):
                pl.key_load += kb
                sp.alloc(kid, kb, "key", False, pinned, ev)
        for pid, pb in o.pts:
            if not sp.hit(pid, pb):
                pl.pt_load += pb
                sp.alloc(pid, pb, "pt", False, pinned, ev)
        out_size = sizes[o.output]
        if o.output not in self.last_use:             # final result: goes to HBM
            pl.out_write = out_size
        elif not sp.alloc(o.output, out_size, "ct", True, pinned, ev):
            pl.out_write = out_size
        for name, size, cls, dirty in ev:
            if dirty and self.last_use.get(name, -1) > o.id:
                pl.writebacks.append((name, size))
        for x in o.inputs:
            if self.last_use[x] == o.id:
                sp.free(x)
        return pl

    # ── processes ────────────────────────────────────────────────────
    def issuer(self):
        W = self.hw.window
        for o in self.t.ops:
            while self.inflight >= W:
                self.slot_ev = self.env.event()
                yield self.slot_ev
            pl = self.plan(o)
            self.inflight += 1
            self.done_ev[o.id] = self.env.event()
            self.env.process(self.run_op(o, pl))

    def wait_done(self, i: int):
        if not self.done[i]:
            yield self.done_ev[i]

    def writeback(self, name: str, size: int):
        p = self.producer.get(name)
        if p is not None:
            yield from self.wait_done(p)
        yield from self.xfer(size, "ct_write", "spill")

    def prefetch(self, o, pl: OpPlan):
        if pl.key_load:
            yield from self.xfer(pl.key_load, "key", o.stage)
        if pl.pt_load:
            yield from self.xfer(pl.pt_load, "pt", o.stage)

    def xfer(self, nbytes: int, cls: str, stage: str):
        bw, left = self.hw.hbm_gbps * 1e9, nbytes
        while left > 0:
            sz = left if left < self.chunk else self.chunk
            req = self.hbm.request()
            yield req
            start = self.env.now
            self.power(self.hbm_w)
            dt = sz / bw
            yield self.env.timeout(dt)
            self.power(-self.hbm_w)
            self.hbm.release(req)
            self.st.busy["hbm"] += dt
            self.st.energy["hbm"] += sz * self.hw.pj_hbm_byte * 1e-12
            self.stage_busy(stage, "hbm", dt)
            self.tracer.span("hbm", cls, f"{cls} {sz / MiB:.1f} MiB", start, dt)
            left -= sz
        self.st.bytes[cls] += nbytes

    def power(self, dp: float) -> None:
        self.p_now += dp
        if self.p_now > self.st.peak_w:
            self.st.peak_w = self.p_now

    def stage_busy(self, stage: str, unit: str, dt: float) -> None:
        d = self.st.stage_busy.setdefault(stage, {})
        d[unit] = d.get(unit, 0.0) + dt

    def run_op(self, o, pl: OpPlan):
        env = self.env
        pf = env.process(self.prefetch(o, pl)) if (pl.key_load or pl.pt_load) else None
        for name, size in pl.writebacks:
            ev = env.process(self.writeback(name, size))
            self.wb_ev[name] = ev
        for x in o.inputs:
            p = self.producer.get(x)
            if p is not None:
                yield from self.wait_done(p)
        for x, size in pl.ct_load:
            w = self.wb_ev.get(x)
            if w is not None and not w.processed:
                yield w
            yield from self.xfer(size, "ct_read", o.stage)
        if pf is not None and not pf.processed:
            yield pf
        self.op_start[o.id] = env.now
        if o.stage not in self.st.stage_first:
            self.st.stage_first[o.stage] = env.now
        for k in o.kernels:
            for seg in self.cost.segments(k):
                req = self.units[seg.unit].request()
                yield req
                start = env.now
                pw = seg.energy / seg.time if seg.time > 0 else 0.0
                self.power(pw)
                yield env.timeout(seg.time)
                self.power(-pw)
                self.units[seg.unit].release(req)
                st = self.st
                st.busy[seg.unit] += seg.time
                st.work[seg.unit] += seg.work
                st.energy[seg.unit] += seg.energy
                st.dac += seg.dac
                st.adc += seg.adc
                self.stage_busy(o.stage, seg.unit, seg.time)
                self.tracer.span(seg.unit, o.stage, f"{o.op}.{k.kind} L{o.level}", start, seg.time)
        if pl.out_write:
            yield from self.xfer(pl.out_write, "ct_write", o.stage)
        self.op_end[o.id] = env.now
        self.st.stage_last[o.stage] = env.now
        self.done[o.id] = True
        self.done_ev[o.id].succeed()
        self.inflight -= 1
        if self.slot_ev is not None and not self.slot_ev.triggered:
            self.slot_ev.succeed()

    def run(self) -> SimResult:
        self.env.run()
        return SimResult(self.t, self.hw, self.cost.s, self.env.now, self.st, self.op_start,
                         self.op_end, self.cost.static_w,
                         self.tracer.export() if self.cfg.trace else None)


def simulate(trace: Trace, cfg: SimConfig | Accelerator) -> SimResult:
    """Run a trace. The clock is the TDP-limited maximum unless ``cfg.clock`` forces one.

    With ``cfg.dvfs`` a memory-bound run is re-run at the clock that just keeps the
    busiest compute unit as busy as HBM (never below ``s_min``): same work, less
    dynamic energy, a small latency cost.
    """
    if isinstance(cfg, Accelerator):
        cfg = SimConfig(cfg)
    s = cfg.clock if cfg.clock is not None else cfg.hw.tdp_clock()
    res = Simulation(trace, cfg, s).run()
    if cfg.dvfs and cfg.clock is None:
        b = res.stats.busy
        top = max(b["ntt"], b["mac"], b["auto"])
        if b["hbm"] > top:
            s2 = s * top / b["hbm"]
            if s2 < cfg.hw.s_min:
                s2 = cfg.hw.s_min
            if s2 < s:
                res = Simulation(trace, cfg, s2).run()
    return res


__all__ = ["SimConfig", "SimResult", "Simulation", "Scratchpad", "simulate", "STAGES"]
