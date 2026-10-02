"""Every number quoted in the README and the FHE Accelerator Simulators decks.

    python examples/results.py            # writes examples/results.md and prints it

Re-run this after any model change and update the quoted tables from its output.
All hardware coefficients are illustrative (see hardware.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from fhe_sim import (ACCELERATORS, PARAMS, BootOptions, SimConfig, bootstrap_trace, he_op_trace,
                     simulate, summarise, summarise_trace)
from fhe_sim.hardware import OpticalEngine
from fhe_sim.search import analytic_bound, design_sweep, pareto, sram_sweep
from fhe_sim.workload import output_level

OUT = []
ARK, HW, SMALL = PARAMS["ark"], ACCELERATORS["ark"], ACCELERATORS["small"]
ALL = dict(min_ks=True, seeded_keys=True, otf_plaintexts=True)


def h(title):
    OUT.append(f"\n## {title}\n")


def table(head, rows):
    OUT.append("| " + " | ".join(head) + " |")
    OUT.append("|" + "---|" * len(head))
    for r in rows:
        OUT.append("| " + " | ".join(str(x) for x in r) + " |")


def ms(x):
    return f"{1e3 * x:,.2f} ms"


def gb(x):
    return f"{x / 1e9:.2f}"


def run(p=ARK, hw=HW, dvfs=False, **opts):
    return summarise(simulate(bootstrap_trace(p, BootOptions(**opts)), SimConfig(hw, dvfs=dvfs)))


# 1 ── parameter sets ────────────────────────────────────────────────────
h("1. Parameter sets (sizes from params.py; levels from the trace generator)")
rows = []
for name, p in PARAMS.items():
    t = bootstrap_trace(p)
    rows.append([name, f"2^{p.log_n}", p.L, p.dnum, p.alpha, f"{p.ct_bytes(p.L) / 2**20:.1f}",
                 f"{p.evk_bytes() / 2**20:.1f}", p.log_pq(), p.L - output_level(t), output_level(t), len(t.ops)])
table(["set", "N", "L", "dnum", "alpha", "ct MiB (top)", "evk MiB", "log PQ (approx)",
       "levels used by bootstrap", "levels left", "HE ops / bootstrap"], rows)

# 2 ── single operations ─────────────────────────────────────────────────
h("2. One HMult and one HRotate at the top level (ark set)")
rows = []
for op in ("hmult", "hrot"):
    s = summarise_trace(he_op_trace(ARK, op))[op]
    t = simulate(he_op_trace(ARK, op), HW).horizon
    rows.append([op, s["ntt_limbs"] + s["intt_limbs"], f"{s['bconv'] / 1e6:.1f} M", f"{s['mac'] / 1e6:.1f} M",
                 f"{s['key_bytes'] / 2**20:.0f} MiB", f"{1e6 * t:.1f} us"])
table(["op", "NTT+iNTT limbs", "BConv mul-adds", "other mul-adds", "key loaded", "time on ARK-class"], rows)

# 3 ── per-stage anatomy ─────────────────────────────────────────────────
h("3. Anatomy of one bootstrap (ark set, baseline algorithm)")
s = summarise_trace(bootstrap_trace(ARK))
table(["stage", "HE ops", "HMult", "HRot", "PMult", "distinct keys", "NTT+iNTT limbs", "key GB requested",
       "plaintext GB requested"],
      [[k, v["ops"], v["hmult"], v["hrot"], v["pmult"], v["distinct_keys"], v["ntt_limbs"] + v["intt_limbs"],
        gb(v["key_bytes"]), gb(v["pt_bytes"])] for k, v in s.items()])

# 4 ── the default run ───────────────────────────────────────────────────
h("4. Default run: ark set on the ARK-class digital design")
m = run()
OUT.append("```")
from fhe_sim.metrics import format_report  # noqa: E402
OUT.append(format_report(m))
OUT.append(f"analytic lower bound {ms(analytic_bound(bootstrap_trace(ARK), HW)['bound_s'])}")
OUT.append("```")

# 5 ── acceleration techniques ───────────────────────────────────────────
h("5. Acceleration techniques (algorithmic), on two designs")
ladder = [("baseline (hoisted BSGS)", {}), ("no hoisting", dict(hoisting=False)),
          ("OpenFHE's BSGS (lazy ModDown)", dict(lazy_moddown=True)),
          ("SlotToCoeff first", dict(stc_first=True)),
          ("+ Min-KS", dict(min_ks=True)), ("+ Min-KS + seeded keys", dict(min_ks=True, seeded_keys=True)),
          ("+ Min-KS + seeded keys + OTF plaintexts", ALL),
          ("+ all three + SlotToCoeff first", dict(ALL, stc_first=True))]
for hwname, hw in (("ARK-class (memory-rich compute)", HW), ("small digital (NTT-starved)", SMALL)):
    rows = []
    for name, o in ladder:
        r = run(hw=hw, **o)
        rows.append([name, ms(r["per_bootstrap_s"]), gb(r["hbm_bytes"]["key"]), gb(r["hbm_bytes"]["total"]),
                     f"{1e3 * r['energy']['per_bootstrap_J']:.0f}", r["bound"]])
    OUT.append(f"\n**{hwname}**\n")
    table(["algorithm", "bootstrap", "key GB", "HBM GB", "mJ", "verdict"], rows)

# 6 ── SRAM ──────────────────────────────────────────────────────────────
h("6. Scratchpad size against traffic (ARK-class design)")
sizes = [128, 256, 384, 512, 768, 1024, 2048, 4096]
for label, o in (("baseline algorithm", BootOptions()), ("Min-KS + seeded keys + OTF plaintexts", BootOptions(**ALL))):
    OUT.append(f"\n**{label}**\n")
    table(["SRAM MiB", "bootstrap", "key GB", "HBM GB", "key share", "verdict"],
          [[r["sram_mib"], ms(r["latency_s"]), f"{r['key_GB']:.2f}", f"{r['hbm_GB']:.2f}",
            f"{100 * r['key_share']:.0f}%", r["bound"]] for r in sram_sweep(ARK, HW, sizes, o)])
OUT.append("\n**two bootstraps back to back, baseline algorithm (keys can be reused across bootstraps only if they all fit)**\n")
table(["SRAM MiB", "per bootstrap", "key GB per bootstrap", "verdict"],
      [[r["sram_mib"], ms(r["latency_s"]), f"{r['key_GB'] / 2:.2f}", r["bound"]]
       for r in sram_sweep(ARK, HW, [512, 2048, 8192, 16384], BootOptions(n_boot=2))])

# 7 ── regimes ───────────────────────────────────────────────────────────
h("7. NTT-bound against memory-bound: verdict and bootstrap latency")
ntts, hbms = [512, 1024, 2048, 4096, 8192], [500.0, 1000.0, 2000.0, 4000.0]
for label, o in (("baseline algorithm", {}), ("Min-KS + seeded keys + OTF plaintexts", ALL)):
    OUT.append(f"\n**{label}** (MAC lanes = 2 x NTT butterflies; cells: latency, verdict)\n")
    rows = []
    for n in ntts:
        cells = [f"{n}"]
        for bw in hbms:
            r = run(hw=HW.with_(ntt_bfly_per_cycle=n, mac_lanes=2 * n, hbm_gbps=bw), **o)
            cells.append(f"{1e3 * r['per_bootstrap_s']:.1f} ms, {r['bound'].split(' ')[0]}")
        rows.append(cells)
    table(["NTT bfly/cycle \\ HBM GB/s"] + [f"{b:.0f}" for b in hbms], rows)

# 8 ── power ─────────────────────────────────────────────────────────────
h("8. Power and energy (TDP 250 W; dynamic = the power manager, worst-case = one fixed TDP clock)")
X2 = dict(ntt_bfly_per_cycle=8192, mac_lanes=16384)
X4 = dict(ntt_bfly_per_cycle=16384, mac_lanes=32768)
rows = []
for name, hw, o, dv in [("ARK-class, baseline", HW, {}, False), ("ARK-class, baseline, DVFS", HW, {}, True),
                        ("ARK-class, all techniques", HW, ALL, False),
                        ("2x NTT + MAC, all techniques", HW.with_(**X2), ALL, False),
                        ("4x NTT + MAC, all techniques", HW.with_(**X4), ALL, False),
                        ("4x, TDP not enforced", HW.with_(enforce_tdp=False, **X4), ALL, False)]:
    for mode in (("dynamic", "worst-case") if hw.enforce_tdp else ("none",)):
        r = run(hw=hw if mode == "none" else hw.with_(power_mode=mode), dvfs=dv, **o)
        e = r["energy"]
        rows.append([name, mode, ms(r["per_bootstrap_s"]), f"{100 * r['clock']:.0f}%", f"{e['avg_power_W']:.0f}",
                     f"{e['peak_power_W']:.0f}", f"{1e3 * e['per_bootstrap_J']:.0f}",
                     f"{100 * e['breakdown']['static']:.0f}% / {100 * e['breakdown']['hbm']:.0f}%", r["bound"]])
table(["configuration", "power mode", "bootstrap", "mean clock", "avg W", "peak W", "mJ / bootstrap",
       "static / HBM energy", "verdict"], rows)
OUT.append(f"\nWorst-case power of the 4x design at full clock: {HW.with_(**X4).peak_power(1.0):.0f} W")

# 9 ── optics ────────────────────────────────────────────────────────────
h("9. Optical NTT engine: the precision tax")
rows = []
for q in (50, 36, 28):
    for blk in (16, 256, 4096):
        for e in (8, 12, 16, 20):
            eng = OpticalEngine(block=blk, enob=e)
            try:
                b, d = eng.digits(q)
                dac, adc = eng.planes(q)
                rows.append([q, blk, e, b, d, (dac + adc) * 2, f"{(blk.bit_length() - 1) / 2:.1f}"])
            except ValueError:
                rows.append([q, blk, e, "-", "infeasible", "-", f"{(blk.bit_length() - 1) / 2:.1f}"])
table(["limb bits", "block", "ENOB", "digit bits b", "digits d", "conversions per point",
       "butterflies per point offloaded"], rows)

h("10. Optical NTT engine in the full system (ark set, baseline algorithm)")
ideal = OpticalEngine(block=4096, enob=8, samples_per_s=5e11, ideal=True)
cases = [("small digital, no optics", SMALL, None),
         ("+ realistic engine (block 16, ENOB 12)", SMALL.with_(tdp_w=300.0), OpticalEngine(samples_per_s=5e11)),
         ("+ realistic engine, ENOB 16, 36-bit limbs*", SMALL.with_(tdp_w=300.0),
          OpticalEngine(enob=16, samples_per_s=5e10)),
         ("+ ideal engine (exact at any precision, block 4096)", SMALL.with_(tdp_w=300.0), ideal),
         ("+ ideal engine, 4x converter rate", SMALL.with_(tdp_w=400.0), replace(ideal, samples_per_s=2e12)),
         ("ARK-class (memory-bound), no optics", HW, None),
         ("ARK-class + ideal engine", HW.with_(tdp_w=350.0), ideal)]
rows = []
for name, hw, eng in cases:
    p = ARK.with_(q_bits=36) if "36-bit" in name else ARK
    r = run(p=p, hw=hw.with_(optical=eng) if eng else hw)
    e = r["energy"]
    rows.append([name, ms(r["per_bootstrap_s"]), f"{1e3 * e['per_bootstrap_J']:.0f}",
                 f"{e['dac_samples'] + e['adc_samples']:.3g}", f"{100 * r['utilisation']['optical']:.0f}%", r["bound"]])
table(["design", "bootstrap", "mJ", "conversions", "optical busy", "verdict"], rows)
OUT.append("\n*36-bit limbs (SHARP's word size) with the same level structure: illustrative only. At ENOB 16 a "
           "conversion costs ~2 nJ (Walden), so the converter rate is cut to 5e10 samples/s to fit the 300 W TDP; "
           "at 5e11 the converters alone would need ~1 kW.")

OUT.append("\n**Break-even converter energy for the ideal engine (energy per bootstrap equal to the small digital design)**\n")
base_e = run(hw=SMALL)["energy"]["per_bootstrap_J"]
rows = []
for laser in (20.0, 5.0, 0.0):
    lo, hi = 0.0, 10000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        eng = replace(ideal, fom_dac_fj=mid, fom_adc_fj=mid, laser_w=laser / 2, tuning_w=laser / 2)
        if run(hw=SMALL.with_(enforce_tdp=False, optical=eng))["energy"]["per_bootstrap_J"] < base_e:
            lo = mid
        else:
            hi = mid
    if lo > 9999:
        lo = float("inf")
    rows.append([f"{laser:.0f} W", f"{lo:.1f} fJ/step" if 0 < lo < 1e9 else ("always" if lo > 0 else "never"),
                 f"{lo * 2 ** 8 * 1e-3:.1f} pJ/sample" if 0 < lo < 1e9 else "-"])
table(["laser + tuning static", "break-even Walden FoM (DAC = ADC, ENOB 8)", "= energy per conversion"], rows)
OUT.append("\nEnergy question only: the TDP is not enforced in this search, so the clock stays at 100%.")

# 11 ── design sweep ─────────────────────────────────────────────────────
h("11. Design-space sweep and Pareto front (all techniques)")
grid = {"ntt_bfly_per_cycle": [1024, 2048, 4096, 8192], "sram_mib": [256, 512, 1024], "hbm_gbps": [500.0, 1000.0, 2000.0]}
t0 = time.perf_counter()
rows = design_sweep(ARK, HW, grid, BootOptions(**ALL), workers=8)
sweep_s = time.perf_counter() - t0
front = pareto(rows)
table(["NTT bfly/cycle", "SRAM MiB", "HBM GB/s", "bootstrap", "mJ", "verdict"],
      [[r["ntt_bfly_per_cycle"], r["sram_mib"], f"{r['hbm_gbps']:.0f}", ms(r["latency_s"]),
        f"{1e3 * r['energy_J']:.0f}", r["bound"]] for r in front])
OUT.append(f"\n{len(rows)} design points simulated in {sweep_s:.1f} s on 8 processes; {len(front)} Pareto-optimal.")

# 12 ── calibration and speed ────────────────────────────────────────────
h("12. Calibration against OpenFHE (this machine) and simulator speed")
out = subprocess.run(["python", str(Path(__file__).parent / "calibrate_openfhe.py")], capture_output=True, text=True)
OUT.append("```\n" + out.stdout.strip() + "\n```")
t = bootstrap_trace(ARK)
reps = 10
t0 = time.perf_counter()
for _ in range(reps):
    simulate(t, HW)
py = (time.perf_counter() - t0) / reps
node = shutil.which("node")
js = None
if node:
    eng = Path(__file__).parent.parent / "web" / "sim_engine.js"
    script = (f"require({json.dumps(str(eng))});const F=globalThis.FheSim;const p=F.mkParams(F.PARAMS.ark),"
              "hw=F.ACCELERATORS.ark(),t=F.bootstrapTrace(p,{});F.simulate(t,hw);const t0=Date.now();"
              "for(let i=0;i<20;i++)F.simulate(t,hw);console.log((Date.now()-t0)/20/1e3);")
    js = float(subprocess.run([node, "-e", script], capture_output=True, text=True).stdout)
OUT.append(f"\nOne ARK-set bootstrap ({len(t.ops)} HE ops, "
           f"{sum(len(o.kernels) for o in t.ops)} kernels): Python/SimPy {1e3 * py:.0f} ms"
           + (f", JavaScript (node) {1e3 * js:.0f} ms" if js else "") + " of wall-clock time.")

# 13 ── dnum trade-off ───────────────────────────────────────────────────
h("13. The dnum trade-off (N = 2^16, L = 23, top level; log PQ uses 50-bit scaling and 60-bit special primes)")
rows = []
for dn in (1, 2, 3, 4, 6, 8, 12, 24):
    p = ARK.with_(dnum=dn)
    hr = summarise_trace(he_op_trace(p, "hrot"))["hrot"]
    rows.append([dn, p.alpha, p.log_pq(), f"{p.evk_bytes() / 2**20:.0f}", hr["ntt_limbs"] + hr["intt_limbs"],
                 f"{hr['bconv'] / 1e6:.0f} M", f"{1e6 * simulate(he_op_trace(p, 'hrot'), HW).horizon:.0f} us"])
table(["dnum", "alpha = k", "log PQ", "evk MiB", "NTT+iNTT limbs per HRot", "BConv mul-adds", "HRot on ARK-class"], rows)

# 14 ── functional precision check ───────────────────────────────────────
h("14. Functional check of the exact-rounding rule (q = 12289, 14-bit limbs, 16-point block)")
import random  # noqa: E402
from fhe_sim.precision import bits_needed, find_psi, ntt_reference, optical_block_ntt  # noqa: E402
q, n = 12289, 16
w = pow(find_psi(q, 2 * n), 2, q)
rng = random.Random(1)
x = [rng.randrange(q) for _ in range(n)]
rows = []
for b in (1, 2, 3):
    d = -(-14 // b)
    e = bits_needed(n, b, d)
    for enob in (e, e - 1, e - 3):
        y, worst = optical_block_ntt(x, q, 14, b, enob)
        rows.append([b, d, e, enob, f"{worst:.3f}", "exact" if y == ntt_reference(x, q, w) else "wrong"])
table(["digit bits b", "digits d", "predicted ENOB", "ENOB used", "worst analogue error", "result"], rows)

# 15 ── converter energy ─────────────────────────────────────────────────
h("15. Converter energy per sample (Walden FoM: DAC 10 fJ/step, ADC 20 fJ/step)")
rows = []
for e in (6, 8, 10, 12, 14, 16, 20):
    eng = OpticalEngine(enob=e)
    rows.append([e, f"{eng.pj_dac():.2f}", f"{eng.pj_adc():.2f}", f"{eng.pj_dac() + eng.pj_adc():.1f}"])
table(["ENOB", "DAC pJ/sample", "ADC pJ/sample", "DAC + ADC pJ"], rows)

# 16 ── dynamic power manager against worst-case clocking ──────────────
h("16. The dynamic power manager against worst-case clocking (all techniques)")
rows = []
for label, over in [("ARK-class, HBM 1 TB/s", {}), ("ARK-class, HBM 2 TB/s", dict(hbm_gbps=2000.0)),
                    ("ARK-class, HBM 4 TB/s", dict(hbm_gbps=4000.0)), ("4x NTT + MAC", X4),
                    ("4x NTT + MAC, TDP 150 W", dict(X4, tdp_w=150.0)), ("4x NTT + MAC, TDP 100 W", dict(X4, tdp_w=100.0)),
                    ("4x NTT + MAC, TDP 80 W", dict(X4, tdp_w=80.0))]:
    cells = [label]
    for mode in ("worst-case", "dynamic"):
        try:
            r = run(hw=HW.with_(power_mode=mode, **over), **ALL)
            cells += [ms(r["per_bootstrap_s"]), f"{100 * r['clock']:.0f}%", f"{r['energy']['peak_power_W']:.0f} W",
                      r["bound"].split(" (")[0]]
        except ValueError:
            cells += ["cannot run", "-", "-", "TDP below worst case at s_min"]
    rows.append(cells)
table(["design", "worst-case: bootstrap", "clock", "peak", "verdict", "dynamic: bootstrap", "mean clock", "peak",
       "verdict"], rows)

# 17 ── the model against real OpenFHE kernel streams ───────────────────
h("17. The scheme model against recorded OpenFHE v1.5.1 bootstraps (calibration/openfhe_trace)")
from fhe_sim.openfhe_trace import log_to_trace, read_log, summarise_log  # noqa: E402
TD = Path(__file__).parent.parent / "calibration" / "openfhe_trace"
for fname, preset in (("full14.log.gz", "openfhe-full14"), ("sparse16.log.gz", "openfhe-sparse")):
    lg = read_log(TD / fname)
    o = summarise_log(lg)
    eager = summarise_trace(bootstrap_trace(PARAMS[preset]))
    lazy = summarise_trace(bootstrap_trace(PARAMS[preset], BootOptions(lazy_moddown=True)))
    hd = lg.header
    OUT.append(f"\n**{fname}**: N = {hd['N']}, {hd['slots']} slots, {hd['towersQ']} limbs, dnum {hd['dnum']}; "
               f"OpenFHE bootstrap depth {hd['bootDepth']}, model {PARAMS[preset].L - output_level(bootstrap_trace(PARAMS[preset]))}\n")
    rows = []
    for st in ("modraise", "cts", "evalmod", "stc"):
        a, e, z = o.get(st, {}), eager.get(st, {}), lazy.get(st, {})
        tr = lambda x: x.get("ntt_limbs", 0) + x.get("intt_limbs", 0)
        rows.append([st, f"{a.get('hrot', 0)} / {e.get('hrot', 0)} / {z.get('hrot', 0)}",
                     f"{a.get('hmult', 0)} / {e.get('hmult', 0)}",
                     f"{a.get('distinct_keys', 0)} / {e.get('distinct_keys', 0)} / {z.get('distinct_keys', 0)}",
                     f"{tr(a):,} / {tr(e):,} / {tr(z):,}",
                     f"{a.get('key_bytes', 0) / 1e9:.2f} / {e.get('key_bytes', 0) / 1e9:.2f} / {z.get('key_bytes', 0) / 1e9:.2f}",
                     f"{a.get('rescale_polys', 0)}"])
    table(["stage", "rotations (OpenFHE / model / model lazy)", "HMults (OpenFHE / model)",
           "distinct rotation keys", "NTT + iNTT limbs", "key GB requested", "OpenFHE polynomial rescales"], rows)
meas = json.loads((Path(__file__).parent.parent / "calibration" / "openfhe_measured.json").read_text())
mb = meas["bootstrap_N65536_slots8_levelBudget11_dnum3_depth18_threads8"]["boot_s"]
mb = sum(mb) / len(mb)
from fhe_sim.hardware import CPU_LIKE  # noqa: E402
rp = simulate(log_to_trace(read_log(TD / "sparse16.log.gz")), CPU_LIKE).horizon
md = simulate(bootstrap_trace(PARAMS["openfhe-sparse"]), CPU_LIKE).horizon
OUT.append(f"\nSparse bootstrap on the CPU-like model: measured {mb:.2f} s; scheme model {md:.2f} s "
           f"({100 * (md - mb) / mb:+.0f}%); replayed OpenFHE trace {rp:.2f} s ({100 * (rp - mb) / mb:+.0f}%).")

# 18 ── programs compiled by HEIR ─────────────────────────────────────────
h("18. Programs compiled by HEIR v2026.10.01, simulated through the HEIR front end (calibration/heir)")
from fhe_sim.heir_frontend import compile_ir  # noqa: E402
HD = Path(__file__).parent.parent / "calibration" / "heir"
rows = []
for name, label in (("lola", "LoLa (MNIST CNN, square activations)"), ("mnist_mlp", "MNIST MLP (polynomial ReLU)"),
                    ("lola_bootstrap", "LoLa, level budget 2 (HEIR places a bootstrap)")):
    prog = compile_ir(HD / f"{name}.ckks.mlir.gz")
    pp = prog.params
    for mib in ((512, 2048) if name == "lola_bootstrap" else (512,)):
        r = summarise(simulate(prog.trace, HW.with_(sram_mib=mib)))
        hb = r["hbm_bytes"]
        rows.append([label, f"2^{pp.log_n}, {pp.L + 1} limbs, {pp.k} special", len(prog.trace.ops),
                     prog.counts.get("rotate", 0), prog.counts.get("mul_plain", 0),
                     prog.counts.get("relinearize", 0) + prog.counts.get("eval_chebyshev", 0) * 0,
                     prog.counts.get("bootstrap", 0), mib, ms(r["latency_s"]),
                     f"{hb['key'] / 1e9:.2f} / {hb['pt'] / 1e9:.2f} / {(hb['ct_read'] + hb['ct_write']) / 1e9:.2f}",
                     f"{1e3 * r['energy']['total_J']:.0f}", r["bound"]])
table(["program", "N, limbs", "HE ops", "rotations", "pt mults", "relins", "bootstraps", "SRAM MiB", "latency",
       "keys / pt / ct GB", "mJ", "verdict"], rows)
lgh = read_log(HD / "lola_openfhe.log.gz")
oh = summarise_log(lgh)["app"]
progl = compile_ir(HD / "lola.ckks.mlir.gz")
mh = summarise_trace(progl.trace)["app"]
rk = len({op.key[0] for op in progl.trace.ops if op.key and op.key[0] != "relin"})
OUT.append("\n**The same LoLa after HEIR's OpenFHE code generation, run on the instrumented OpenFHE**\n")
table(["", "HEIR IR (front end)", "OpenFHE execution"],
      [["rotations", progl.counts["rotate"], oh["hrot"]],
       ["distinct rotation keys", rk, oh["distinct_keys"]],
       ["relinearisations", progl.counts["relinearize"], oh["hmult"]],
       ["limbs at entry", progl.trace.levels[progl.trace.external[0]] + 1, lgh.header["in_towers"]],
       ["polynomial rescales", 2 * progl.counts["rescale"], oh["rescale_polys"]],
       ["NTT + iNTT limbs", f"{mh['ntt_limbs'] + mh['intt_limbs']:,}", f"{oh['ntt_limbs'] + oh['intt_limbs']:,}"],
       ["key GB requested", f"{mh['key_bytes'] / 1e9:.2f}", f"{oh['key_bytes'] / 1e9:.2f}"],
       ["ARK-class latency", ms(simulate(progl.trace, HW).horizon), ms(simulate(log_to_trace(lgh), HW).horizon)],
       ["small-digital latency", ms(simulate(progl.trace, SMALL).horizon), ms(simulate(log_to_trace(lgh), SMALL).horizon)]])
for note in compile_ir(HD / "mnist_mlp.ckks.mlir.gz").notes + compile_ir(HD / "lola_bootstrap.ckks.mlir.gz").notes:
    OUT.append(f"\nfront-end note: {note}")

# 19 ── bootstrap ordering ───────────────────────────────────────────────
h("19. SlotToCoeff-first against the conventional order (ark set)")
rows = []
for hwname, hw in (("ARK-class", HW), ("small digital", SMALL)):
    for name, o in (("conventional", {}), ("SlotToCoeff first", dict(stc_first=True)),
                    ("all three", ALL), ("all three + SlotToCoeff first", dict(ALL, stc_first=True))):
        t = bootstrap_trace(ARK, BootOptions(**o))
        r = summarise(simulate(t, hw))
        st = summarise_trace(t)
        lev = output_level(t)
        rows.append([hwname, name, ms(r["per_bootstrap_s"]), lev, f"{1e3 * r['per_bootstrap_s'] / lev:.2f} ms",
                     sum(v["hmult"] for v in st.values()), f"{sum(v['ntt_limbs'] + v['intt_limbs'] for v in st.values()):,}",
                     gb(r["hbm_bytes"]["key"]), f"{1e3 * r['energy']['per_bootstrap_J']:.0f}", r["bound"]])
table(["design", "order", "bootstrap", "levels left", "per useful level", "HMults", "NTT + iNTT limbs", "key GB",
       "mJ", "verdict"], rows)
OUT.append("\n**OpenFHE v1.5.1, both orders (calibration/openfhe_trace)**\n")
rows = []
for base in ("full14", "sparse16"):
    lc, lf = read_log(TD / f"{base}.log.gz"), read_log(TD / f"{base}_stcfirst.log.gz")
    oc, of = summarise_log(lc), summarise_log(lf)
    rows.append([base, f"{lc.header['in_towers']} -> {lc.header['out_towers']}", f"{lf.header['in_towers']} -> {lf.header['out_towers']}",
                 f"{oc['evalmod']['hmult']} / {of['evalmod']['hmult']}",
                 f"{oc['stc']['key_bytes'] / 1e9:.2f} / {of['stc']['key_bytes'] / 1e9:.2f}",
                 f"{sum(v['ntt_limbs'] + v['intt_limbs'] for v in oc.values()):,} / {sum(v['ntt_limbs'] + v['intt_limbs'] for v in of.values()):,}"])
table(["recording", "towers in -> out (conventional)", "towers in -> out (StC first)", "EvalMod HMults (conv / StC first)",
       "SlotToCoeff key GB (conv / StC first)", "NTT + iNTT limbs (conv / StC first)"], rows)

text = "# Results (generated by examples/results.py)\n\nAll hardware coefficients are illustrative.\n" + "\n".join(OUT) + "\n"
(Path(__file__).parent / "results.md").write_text(text)
print(text)
