"""Turning a finished simulation into numbers an architect can act on.

* **latency**      bootstrap latency and the time spent in each stage
* **utilisation**  busy fraction of every functional unit and of HBM
* **attribution**  is the design NTT-bound, memory-bound or power-bound?
* **hot-spots**    for each stage, the resource that was busiest
* **traffic**      HBM bytes by class: evaluation keys, plaintexts, ciphertexts
* **power**        static + dynamic energy, average and peak power, energy per bootstrap
"""

from __future__ import annotations

from .hardware import UNITS
from .sim import SimResult
from .workload import STAGES

BOUND_NAME = {"hbm": "memory-bound", "ntt": "NTT-bound", "optical": "NTT-bound (optical engine)",
              "mac": "MAC-bound", "auto": "permutation-bound"}


def summarise(res: SimResult) -> dict:
    st, H, hw = res.stats, res.horizon, res.hw
    n_boot = max(o.boot for o in res.trace.ops) + 1
    resources = UNITS + ["hbm"]
    util = {u: st.busy[u] / H for u in resources}

    top = "hbm"
    for u in resources:                          # deterministic tie-break: list order
        if util[u] > util[top]:
            top = u
    bound = BOUND_NAME[top]
    dynamic = hw.enforce_tdp and hw.power_mode == "dynamic"
    if top != "hbm" and hw.enforce_tdp and (
            (dynamic and st.power_loss > 0.1 * H) or (not dynamic and res.clock < 1.0)):
        bound = "power-bound (" + BOUND_NAME[top] + " at a TDP-limited clock)"

    stages, hot = {}, {}
    for s in STAGES:
        if s not in st.stage_first:
            continue
        span = st.stage_last[s] - st.stage_first[s]
        busy = st.stage_busy.get(s, {})
        r = None
        for u in resources:
            if busy.get(u, 0.0) > 0 and (r is None or busy[u] > busy[r]):
                r = u
        stages[s] = {"span_s": span, "busy_s": {u: busy.get(u, 0.0) for u in resources}}
        hot[s] = r
    total_span = 0.0               # plain loops, not sum(): Python 3.12's sum() of floats is
    for v in stages.values():      # compensated, which the JavaScript port does not reproduce
        total_span += v["span_s"]
    for v in stages.values():
        v["share"] = v["span_s"] / total_span if total_span > 0 else 0.0
    hottest = None
    for s in stages:
        if hottest is None or stages[s]["span_s"] > stages[hottest]["span_s"]:
            hottest = s

    b = st.bytes
    hbm_total = b["key"] + b["pt"] + b["ct_read"] + b["ct_write"]

    static = res.static_w * H
    dyn = 0.0
    for u in resources:
        dyn += st.energy[u]
    total = static + dyn
    energy = {"total_J": total, "per_bootstrap_J": total / n_boot, "avg_power_W": total / H,
              "peak_power_W": res.static_w + st.peak_w, "tdp_W": hw.tdp_w,
              "breakdown": {"static": static / total, **{u: st.energy[u] / total for u in resources}},
              "dac_samples": st.dac, "adc_samples": st.adc}

    return {
        "params": res.trace.params.name, "hardware": hw.name, "n_boot": n_boot,
        "clock": res.clock, "latency_s": H, "per_bootstrap_s": H / n_boot,
        "utilisation": util, "bound": bound, "bound_resource": top,
        "stages": stages, "hotspots": {"stage": hottest, "resource": hot.get(hottest),
                                       "per_stage": hot},
        "hbm_bytes": dict(b, total=hbm_total,
                          key_share=b["key"] / hbm_total if hbm_total else 0.0),
        "energy": energy,
        "lower_bound_s": lower_bound(res),
        "power_mode": hw.power_mode if hw.enforce_tdp else "none",
        "power_loss_s": st.power_loss,
    }


def lower_bound(res: SimResult) -> float:
    """No schedule can beat the busiest single resource: max over units of its busy time."""
    return max(res.stats.busy.values())


def format_report(m: dict) -> str:
    ms = lambda x: f"{1e3 * x:,.2f} ms"
    gb = lambda x: f"{x / 1e9:.2f} GB"
    u = m["utilisation"]
    lines = [f"── {m['params']} on {m['hardware']}",
             f"bootstraps {m['n_boot']}   latency {ms(m['latency_s'])}   per bootstrap {ms(m['per_bootstrap_s'])}"
             f"   clock {100 * m['clock']:.0f}% ({m['power_mode']})",
             "utilisation  " + "  ".join(f"{k} {100 * v:.0f}%" for k, v in u.items() if v > 0 or k != "optical"),
             f"verdict      {m['bound']}",
             "stages       " + "  ".join(f"{s} {ms(v['span_s'])} ({100 * v['share']:.0f}%)"
                                         for s, v in m["stages"].items()),
             "hot-spots    " + "  ".join(f"{s}->{r}" for s, r in m["hotspots"]["per_stage"].items())
             + f"   (dominant: {m['hotspots']['stage']} -> {m['hotspots']['resource']})"]
    h = m["hbm_bytes"]
    lines.append(f"HBM traffic  keys {gb(h['key'])}  plaintexts {gb(h['pt'])}  ct read {gb(h['ct_read'])}"
                 f"  ct write {gb(h['ct_write'])}  (keys {100 * h['key_share']:.0f}%)")
    e = m["energy"]
    lines.append(f"power        avg {e['avg_power_W']:.0f} W  peak {e['peak_power_W']:.0f} W (TDP {e['tdp_W']:.0f} W)"
                 f"  {1e3 * e['per_bootstrap_J']:.1f} mJ/bootstrap  energy: "
                 + "  ".join(f"{k} {100 * v:.0f}%" for k, v in e["breakdown"].items() if v >= 0.005))
    if e["dac_samples"]:
        lines.append(f"converters   DAC {e['dac_samples']:.3g} samples  ADC {e['adc_samples']:.3g} samples")
    return "\n".join(lines)
