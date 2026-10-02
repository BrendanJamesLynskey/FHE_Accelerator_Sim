"""Levels 2-5 of the verification ladder for the event engine.

2. invariant  – dependencies respected, work conserved, determinism, passive probes
3. analytic   – serial chain and two-stage flow shop solved exactly; roofline bound
4. behaviour  – the effects the model exists to show (SRAM, key reuse, optics, power)
5. external   – the precision rule against a functional model; calibration against
                OpenFHE; the JavaScript port against this package
"""

import json
import random
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from fhe_sim import (ACCELERATORS, PARAMS, BootOptions, SimConfig, bootstrap_trace, he_op_trace,
                     simulate, summarise)
from fhe_sim.hardware import CPU_LIKE, CostModel, OpticalEngine
from fhe_sim.precision import bits_needed, find_psi, ntt_reference, optical_block_ntt
from fhe_sim.search import analytic_bound, min_sram, pareto, sram_sweep
from fhe_sim.workload import HEOp, Kernel, Trace

ARK, SMALL = PARAMS["ark"], PARAMS["small"]
HW = ACCELERATORS["ark"]
ROOT = Path(__file__).parent.parent


def run(p=ARK, hw=HW, **opts):
    t = bootstrap_trace(p, BootOptions(**opts))
    return t, simulate(t, hw)


# ── 2. invariants ────────────────────────────────────────────────────────
@pytest.mark.parametrize("hw", ["ark", "small", "ideal-optical"])
def test_dependencies_respected(hw):
    t, r = run(SMALL, ACCELERATORS[hw], n_boot=2)
    producer = {o.output: o.id for o in t.ops}
    for o in t.ops:
        assert r.op_end[o.id] >= r.op_start[o.id]
        for x in o.inputs:
            if x in producer:
                assert r.op_start[o.id] >= r.op_end[producer[x]]


def test_work_is_conserved():
    """Every kernel's work is executed exactly once on its unit."""
    t, r = run(SMALL, n_boot=2)
    cm = CostModel(HW, SMALL.log_n, SMALL.q_bits, r.clock)
    want = {"ntt": 0.0, "mac": 0.0, "auto": 0.0, "optical": 0.0}
    for o in t.ops:
        for k in o.kernels:
            for s in cm.segments(k):
                want[s.unit] += s.work
    for u, w in want.items():
        assert r.stats.work[u] == pytest.approx(w, rel=1e-12)


def test_traffic_at_least_compulsory_and_reads_follow_writes():
    t, r = run(n_boot=2)
    b = analytic_bound(t, HW)["busy_s"]["hbm"] * HW.hbm_gbps * 1e9
    st = r.stats.bytes
    assert st["key"] + st["pt"] + st["ct_read"] >= b * (1 - 1e-12)
    ext = sum(t.sizes[x] for x in t.external)
    assert st["ct_read"] <= st["ct_write"] + ext


def test_same_trace_same_answer():
    a, b = run()[1], run()[1]
    assert a.horizon == b.horizon and a.op_end == b.op_end and a.stats.bytes == b.stats.bytes


def test_probes_are_passive():
    t = bootstrap_trace(ARK)
    quiet = simulate(t, SimConfig(HW, trace=False))
    noisy = simulate(t, SimConfig(HW, trace=True))
    assert quiet.op_end == noisy.op_end
    names = {e["args"]["name"] for e in noisy.tracer["traceEvents"] if e["ph"] == "M"}
    assert {"ntt", "mac", "hbm"} <= names


# ── 3. analytic checks ──────────────────────────────────────────────────
def synthetic(n, key_bytes, mac_ops, chained):
    """n ops, each loading one key and running one MAC kernel; optionally a dependency chain."""
    ops, sizes = [], {}
    for i in range(n):
        out = f"o{i}"
        sizes[out] = 0 if (not chained or i == n - 1) else 0
        ins = [f"o{i - 1}"] if (chained and i) else []
        ops.append(HEOp(i, "pmac", "cts", 0, ins, out, [Kernel("mac", mac_ops, 0)],
                        key=(f"k{i}", key_bytes) if key_bytes else None))
    return Trace(ARK, ops, sizes, [])


def test_serial_chain_time_is_the_sum():
    hw = HW.with_(enforce_tdp=False)
    r = simulate(synthetic(50, 0, 10_000_000, chained=True), hw)
    assert r.horizon == pytest.approx(50 * 10_000_000 / hw.rate("mac"), rel=1e-12)


@pytest.mark.parametrize("a_mib,mac_ops", [(1, 10_000_000), (3, 1_000_000)])
def test_two_stage_flow_shop(a_mib, mac_ops):
    """Independent ops, one HBM load then one compute each, unlimited window: the makespan of a
    deterministic two-machine flow shop is a + b + (n - 1) max(a, b)."""
    n = 40
    hw = HW.with_(enforce_tdp=False, window=n, sram_mib=4096)
    r = simulate(synthetic(n, a_mib * (1 << 20), mac_ops, chained=False), hw)
    a = a_mib * (1 << 20) / (hw.hbm_gbps * 1e9)
    b = mac_ops / hw.rate("mac")
    assert r.horizon == pytest.approx(a + b + (n - 1) * max(a, b), rel=1e-12)


@pytest.mark.parametrize("hw", ["ark", "small", "cpu"])
def test_simulation_respects_roofline_bound(hw):
    p = PARAMS["openfhe-sparse"] if hw == "cpu" else ARK
    t = bootstrap_trace(p)
    r = simulate(t, ACCELERATORS[hw])
    m = summarise(r)
    assert r.horizon >= m["lower_bound_s"] >= analytic_bound(t, ACCELERATORS[hw])["bound_s"] * (1 - 1e-12)


# ── 4. behaviour ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("opts", [dict(), dict(min_ks=True, seeded_keys=True), dict(n_boot=2)])
def test_more_sram_never_increases_key_traffic(opts):
    rows = sram_sweep(ARK, HW, [128, 192, 256, 384, 512, 768, 1024, 2048], BootOptions(**opts))
    keys = [r["key_GB"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(keys, keys[1:]))
    hbm = [r["hbm_GB"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(hbm, hbm[1:]))
    assert hbm[-1] < hbm[0]


def test_baseline_is_memory_bound_and_key_traffic_dominates():
    m = summarise(run()[1])
    assert m["bound"] == "memory-bound"
    assert m["hbm_bytes"]["key"] > max(m["hbm_bytes"]["pt"], m["hbm_bytes"]["ct_read"])
    assert m["hotspots"]["stage"] == "cts" and m["hotspots"]["resource"] == "hbm"


def test_key_reuse_techniques_cut_key_traffic_and_shift_the_bound():
    base = summarise(run()[1])
    opt = summarise(run(min_ks=True, seeded_keys=True, otf_plaintexts=True)[1])
    assert opt["hbm_bytes"]["key"] * 5 < base["hbm_bytes"]["key"]
    assert opt["per_bootstrap_s"] < base["per_bootstrap_s"]
    assert opt["bound"] != "memory-bound"


def test_key_reuse_hurts_an_ntt_bound_design():
    """Min-KS trades compute for bandwidth: on a design already short of NTT throughput it loses."""
    small = ACCELERATORS["small"]
    base = summarise(run(hw=small)[1])
    opt = summarise(run(hw=small, min_ks=True, otf_plaintexts=True)[1])
    assert base["bound"] == "NTT-bound"
    assert opt["per_bootstrap_s"] > base["per_bootstrap_s"]


def test_ideal_optical_engine_helps_ntt_bound_not_memory_bound():
    eng = ACCELERATORS["ideal-optical"].optical
    small = ACCELERATORS["small"]
    ntt_bound = summarise(run(hw=small)[1])["per_bootstrap_s"]
    with_opt = summarise(run(hw=small.with_(optical=eng, tdp_w=300.0))[1])["per_bootstrap_s"]
    assert ntt_bound / with_opt > 1.2
    mem = HW.with_(sram_mib=256)
    mem_base = summarise(run(hw=mem)[1])
    assert mem_base["bound"] == "memory-bound"
    mem_opt = summarise(run(hw=mem.with_(optical=eng, tdp_w=350.0))[1])["per_bootstrap_s"]
    assert mem_base["per_bootstrap_s"] / mem_opt < 1.05


def test_realistic_precision_makes_the_optical_engine_lose():
    small = ACCELERATORS["small"]
    digital = summarise(run(hw=small)[1])
    hybrid = summarise(run(hw=ACCELERATORS["hybrid"])[1])
    assert hybrid["per_bootstrap_s"] > 5 * digital["per_bootstrap_s"]
    assert hybrid["energy"]["per_bootstrap_J"] > 5 * digital["energy"]["per_bootstrap_J"]


def test_trace_export_and_hotspots_per_stage():
    t = bootstrap_trace(SMALL)
    r = simulate(t, SimConfig(HW, trace=True))
    spans = [e for e in r.tracer["traceEvents"] if e["ph"] == "X"]
    assert spans and all(e["dur"] >= 0 for e in spans)
    m = summarise(r)
    assert set(m["hotspots"]["per_stage"]) == {"modraise", "cts", "evalmod", "stc"}


# ── power and energy ─────────────────────────────────────────────────────
def test_energy_accounts_add_up():
    _, r = run()
    m = summarise(r)
    assert sum(m["energy"]["breakdown"].values()) == pytest.approx(1.0)
    assert m["energy"]["total_J"] == pytest.approx(r.static_w * r.horizon + sum(r.stats.energy.values()))
    assert r.stats.energy["hbm"] == pytest.approx(sum(r.stats.bytes.values()) * HW.pj_hbm_byte * 1e-12)


@pytest.mark.parametrize("mode", ["worst-case", "dynamic"])
@pytest.mark.parametrize("ntt,tdp", [(4096, 250.0), (16384, 250.0), (32768, 250.0), (16384, 150.0)])
def test_tdp_is_never_exceeded(ntt, tdp, mode):
    hw = HW.with_(ntt_bfly_per_cycle=ntt, mac_lanes=2 * ntt, tdp_w=tdp, power_mode=mode)
    m = summarise(run(min_ks=True, seeded_keys=True, otf_plaintexts=True, hw=hw)[1])
    assert m["energy"]["peak_power_W"] <= tdp * (1 + 1e-12)
    if mode == "worst-case" and hw.peak_power(1.0) > tdp:
        assert m["clock"] < 1.0 and m["bound"].startswith("power-bound")


def test_dynamic_power_manager_beats_worst_case_clocking():
    """Worst-case clocking reserves power for HBM and every unit at once; the manager does not,
    so over-provisioned designs and very fast HBM stop being penalised."""
    opts = dict(min_ks=True, seeded_keys=True, otf_plaintexts=True)
    for over in (dict(ntt_bfly_per_cycle=16384, mac_lanes=32768), dict(hbm_gbps=4000.0)):
        wc = summarise(run(hw=HW.with_(power_mode="worst-case", **over), **opts)[1])
        dy = summarise(run(hw=HW.with_(**over), **opts)[1])
        assert wc["clock"] < 1.0
        assert dy["per_bootstrap_s"] < 0.9 * wc["per_bootstrap_s"]
        assert dy["energy"]["peak_power_W"] <= HW.tdp_w * (1 + 1e-12)
    # the fast-HBM anomaly: worst-case 4 TB/s is slower than 2 TB/s; dynamic is not
    lat = {(mode, bw): summarise(run(hw=HW.with_(power_mode=mode, hbm_gbps=bw), **opts)[1])["per_bootstrap_s"]
           for mode in ("worst-case", "dynamic") for bw in (2000.0, 4000.0)}
    assert lat[("worst-case", 4000.0)] > lat[("worst-case", 2000.0)]
    assert lat[("dynamic", 4000.0)] <= lat[("dynamic", 2000.0)] * 1.01


def test_dynamic_manager_throttles_under_a_tight_tdp():
    """At 100 W worst-case clocking cannot run this design at all (even s_min reserves too much);
    the manager runs it, throttling when the real draw nears the limit."""
    hw = HW.with_(ntt_bfly_per_cycle=16384, mac_lanes=32768, tdp_w=100.0)
    with pytest.raises(ValueError):
        run(hw=hw.with_(power_mode="worst-case"), min_ks=True, seeded_keys=True, otf_plaintexts=True)
    _, r = run(hw=hw, min_ks=True, seeded_keys=True, otf_plaintexts=True)
    m = summarise(r)
    assert r.stats.power_loss > 0.1 * r.horizon and m["bound"].startswith("power-bound")
    assert m["clock"] < 1.0 and m["energy"]["peak_power_W"] <= 100.0 * (1 + 1e-12)


def test_unknown_power_mode_is_an_error():
    with pytest.raises(ValueError):
        simulate(bootstrap_trace(SMALL), HW.with_(power_mode="turbo"))


def test_cap_below_static_is_an_error():
    with pytest.raises(ValueError):
        simulate(bootstrap_trace(SMALL), HW.with_(tdp_w=50.0))


def test_dvfs_saves_energy_when_memory_bound():
    t = bootstrap_trace(ARK)
    a = summarise(simulate(t, SimConfig(HW)))
    b = summarise(simulate(t, SimConfig(HW, dvfs=True)))
    assert b["clock"] < a["clock"]
    assert b["energy"]["per_bootstrap_J"] < a["energy"]["per_bootstrap_J"]
    assert b["per_bootstrap_s"] < 1.25 * a["per_bootstrap_s"]


def test_scratchpad_smaller_than_a_key_switch_is_an_error():
    with pytest.raises(ValueError):
        simulate(bootstrap_trace(ARK), HW.with_(sram_mib=64))


# ── search utilities ─────────────────────────────────────────────────────
def test_min_sram_bisection_agrees_with_the_sweep():
    opts = BootOptions(min_ks=True, seeded_keys=True, otf_plaintexts=True)
    found = min_sram(ARK, HW, 1.0, 128, 1024, opts, tol_mib=16)
    assert found["sram_mib"] is not None and found["runs"] < 10
    below = sram_sweep(ARK, HW, [found["sram_mib"] - 16, found["sram_mib"]], opts)
    assert below[0]["hbm_GB"] > 1.0 >= below[1]["hbm_GB"]


def test_pareto_front_is_non_dominated():
    rows = [{"latency_s": x, "energy_J": y} for x, y in [(1, 5), (2, 3), (3, 4), (4, 1), (2, 6)]]
    front = pareto(rows)
    assert [(r["latency_s"], r["energy_J"]) for r in front] == [(1, 5), (2, 3), (4, 1)]


# ── 5. external references ───────────────────────────────────────────────
@pytest.mark.parametrize("b", [1, 2, 3])
def test_precision_rule_is_exact_and_tight(b):
    """The functional model rounds correctly at the predicted ENOB and fails one bit below."""
    q, q_bits, n = 12289, 14, 16
    w = pow(find_psi(q, 2 * n), 2, q)
    rng = random.Random(b)
    x = [rng.randrange(q) for _ in range(n)]
    d = -(-q_bits // b)
    e = bits_needed(n, b, d)
    ok, worst = optical_block_ntt(x, q, q_bits, b, e)
    assert ok == ntt_reference(x, q, w) and worst < 0.5
    bad, worst = optical_block_ntt(x, q, q_bits, b, e - 1)
    assert worst >= 0.5 and bad != ntt_reference(x, q, w)


def test_hardware_precision_rule_matches_functional_model():
    for enob in range(6, 20):
        for block in (8, 16, 64):
            try:
                b, d = OpticalEngine(block=block, enob=enob).digits(14)
            except ValueError:
                assert bits_needed(block, 1, 14) > enob
                continue
            assert bits_needed(block, b, d) <= enob
            if b < 14:
                assert bits_needed(block, b + 1, -(-14 // (b + 1))) > enob


def test_calibration_against_openfhe():
    meas = json.loads((ROOT / "calibration" / "openfhe_measured.json").read_text())
    prim = meas["primitives_N65536_limbs24_dnum4"]["threads8"]
    hm = simulate(he_op_trace(ARK, "hmult"), CPU_LIKE).horizon
    hr = simulate(he_op_trace(ARK, "hrot"), CPU_LIKE).horizon
    boot = simulate(bootstrap_trace(PARAMS["openfhe-sparse"]), CPU_LIKE).horizon
    measured_boot = meas["bootstrap_N65536_slots8_levelBudget11_dnum3_depth18_threads8"]["boot_s"]
    assert hm == pytest.approx(prim["hmult_relin_ms"] / 1e3, rel=0.01)        # the fitted quantity
    assert hr == pytest.approx(prim["hrot_ms"] / 1e3, rel=0.15)                # predicted
    # predicted by the scheme model; its rescale-once EvalMod does fewer transforms than OpenFHE's
    # rescale-on-use schedule (tests/test_openfhe_trace.py), so it runs fast. The replayed real trace
    # is the better predictor (within 15%, tested there).
    assert boot == pytest.approx(sum(measured_boot) / 3, rel=0.35)


JS_NAMES = {"power_mode": "powerMode", "tdp_w": "tdpW", "hbm_gbps": "hbmGbps",
            "ntt_bfly_per_cycle": "nttBflyPerCycle", "mac_lanes": "macLanes"}


def js_cases():
    """(parameter set, hardware preset, algorithm options, hardware overrides)."""
    hot = dict(ntt_bfly_per_cycle=16384, mac_lanes=32768)
    return [
        ("ark", "ark", dict(), {}),
        ("ark", "ark", dict(), dict(power_mode="worst-case")),
        ("ark", "small", dict(min_ks=True, seeded_keys=True), {}),
        ("small", "ark", dict(n_boot=2, hoisting=False), {}),
        ("ark", "ark", dict(otf_plaintexts=True), dict(hbm_gbps=4000.0)),
        ("openfhe-sparse", "cpu", dict(), {}),
        ("ark", "hybrid", dict(), {}),
        ("ark", "ideal-optical", dict(n_boot=2), {}),
        ("ark", "ark", dict(min_ks=True, seeded_keys=True, otf_plaintexts=True), dict(hot, power_mode="worst-case")),
        ("ark", "ark", dict(min_ks=True, seeded_keys=True, otf_plaintexts=True), dict(hot, tdp_w=150.0)),
        ("ark", "ark", dict(), dict(hbm_gbps=4000.0, tdp_w=150.0)),
        ("ark", "small", dict(lazy_moddown=True), {}),
        ("openfhe-full14", "ark", dict(lazy_moddown=True, otf_plaintexts=True), {}),
    ]


def test_javascript_port_matches_python():
    """The browser simulator in deck 03 must reproduce this package exactly, in both power modes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = ROOT / "web" / "sim_engine.js"
    payload, expect = [], []
    for pname, hname, opts, over in js_cases():
        hw = ACCELERATORS[hname].with_(**over)
        for dvfs in (False, True):
            t = bootstrap_trace(PARAMS[pname], BootOptions(**opts))
            r = simulate(t, SimConfig(hw, dvfs=dvfs))
            m = summarise(r)
            payload.append({"params": pname, "hw": hname, "opts": opts, "dvfs": dvfs,
                            "over": {JS_NAMES[k]: v for k, v in over.items()}})
            expect.append({"horizon": r.horizon, "end": r.op_end, "clock": r.clock,
                           "bytes": r.stats.bytes, "energy": m["energy"]["total_J"],
                           "peak": m["energy"]["peak_power_W"], "bound": m["bound"],
                           "hot": m["hotspots"]["resource"], "loss": r.stats.power_loss})
    script = (f"require({json.dumps(str(engine))});"
              "const F=globalThis.FheSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=F.simulateNamed(c.params,c.hw,c.opts,c.dvfs,c.over);"
              "const m=F.summarise(r);return {horizon:r.horizon,end:r.opEnd,clock:r.clock,bytes:r.stats.bytes,"
              "energy:m.energy.totalJ,peak:m.energy.peakW,bound:m.bound,hot:m.hotspots.resource,loss:r.stats.powerLoss};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(payload), capture_output=True,
                         text=True, check=True)
    got = json.loads(out.stdout)
    assert len(got) == len(expect) == 26
    for e, g, c in zip(expect, got, payload):
        assert g["horizon"] == e["horizon"], c                    # bit-identical: no transcendentals
        assert g["end"] == e["end"], c
        assert g["clock"] == e["clock"] and g["bytes"] == e["bytes"], c
        assert g["energy"] == e["energy"] and g["peak"] == e["peak"], c
        assert g["loss"] == e["loss"], c
        assert (g["bound"], g["hot"]) == (e["bound"], e["hot"]), c
