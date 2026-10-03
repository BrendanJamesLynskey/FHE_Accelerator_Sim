"""The area, yield and cost model (ppa.py), the N-dimensional Pareto front, and the JavaScript port."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fhe_sim import ACCELERATORS, PARAMS, BootOptions, simulate, summarise
from fhe_sim.cli import main as cli_main
from fhe_sim.hardware import OpticalEngine
from fhe_sim.ppa import (AREA_7NM, CACTI_LSTP_22NM, AreaModel, DieCost, area_mm2, die_cost, dies_per_wafer,
                         murphy_yield, poisson_yield, ppa_metrics)
from fhe_sim.search import dominates, pareto_nd
from fhe_sim.workload import bootstrap_trace

ROOT = Path(__file__).resolve().parent.parent
ARK_HW = ACCELERATORS["ark"]
KNOBS = {"ntt_bfly_per_cycle": [256, 512, 1024, 4096, 8192, 16384],
         "mac_lanes": [512, 2048, 8192, 32768],
         "auto_words_per_cycle": [256, 1024, 4096],
         "sram_mib": [16, 64, 100, 128, 200, 256, 384, 512, 768, 1024, 1500, 2048, 4096, 8192],
         "hbm_gbps": [250.0, 500.0, 501.0, 1000.0, 2000.0, 4000.0]}


def test_cacti_table_matches_the_recorded_cacti_run():
    rec = json.loads((ROOT / "calibration" / "cacti" / "out" / "results.json").read_text())
    lstp = [(r["size_mib"], r["mm2_per_mib"]) for r in rec["rows"] if r["cell"] == "itrs-lstp"]
    assert tuple(lstp) == CACTI_LSTP_22NM
    assert rec["node_nm"] == 22 and rec["bank_mib"] == 4


def test_ark_as_published_reproduces_its_table():
    """ARK's own unit counts give back ARK's Table IV (418.2 mm²; the paper rounds the sum to 418.3)."""
    a = area_mm2(ARK_HW.with_(ntt_bfly_per_cycle=8192, auto_words_per_cycle=1024))
    for k, v in (("ntt", 57.2), ("mac", 9.3 + 8.9), ("auto", 20.6), ("sram", 229.2), ("uncore", 42.8 + 20.6),
                 ("hbm_phy", 29.6)):
        assert a[k] == pytest.approx(v, rel=1e-9), k
    assert a["die"] == pytest.approx(418.2, abs=1e-9)


@pytest.mark.parametrize("knob", list(KNOBS))
def test_area_is_monotone_in_each_resource(knob):
    prev = None
    for v in KNOBS[knob]:
        a = area_mm2(ARK_HW.with_(**{knob: v}))
        if prev is not None:
            assert a["die"] >= prev["die"], (knob, v)
            assert a["total"] >= prev["total"]
        prev = a


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 8192), st.integers(1, 8192))
def test_sram_area_is_monotone_on_any_capacities(a, b):
    lo, hi = min(a, b), max(a, b)
    assert ARK_HW.with_(sram_mib=lo).sram_mib * AREA_7NM.sram_mm2_per_mib(lo) <= hi * AREA_7NM.sram_mm2_per_mib(hi)


def test_optical_engine_adds_area_and_a_photonic_die():
    small = area_mm2(ACCELERATORS["small"])
    hyb = area_mm2(ACCELERATORS["hybrid"])
    assert small["optical_electronic"] == small["photonic_die"] == 0.0
    assert hyb["optical_electronic"] > 0 and hyb["photonic_die"] == AREA_7NM.photonic_die_mm2
    assert hyb["total"] == hyb["die"] + hyb["photonic_die"]
    faster = area_mm2(ACCELERATORS["hybrid"].with_(optical=OpticalEngine(samples_per_s=2e12)))
    assert faster["optical_electronic"] > hyb["optical_electronic"]


@settings(max_examples=300, deadline=None)
@given(st.floats(0.0, 5000.0), st.floats(0.0, 5000.0), st.floats(0.001, 2.0))
def test_yield_is_in_0_1_and_decreasing_in_area(a1, a2, d0):
    lo, hi = min(a1, a2), max(a1, a2)
    for y in (poisson_yield, murphy_yield):
        assert 0.0 < y(lo, d0) <= 1.0 and 0.0 < y(hi, d0) <= 1.0
        assert y(hi, d0) <= y(lo, d0)


@settings(max_examples=300, deadline=None)
@given(st.floats(1e-3, 5000.0), st.floats(0.001, 2.0))
def test_poisson_is_below_murphy(a, d0):
    assert poisson_yield(a, d0) <= murphy_yield(a, d0)


def test_yields_agree_for_small_dies_and_diverge_for_large():
    assert murphy_yield(10.0, 0.1) - poisson_yield(10.0, 0.1) < 1e-4
    assert murphy_yield(800.0, 0.1) / poisson_yield(800.0, 0.1) > 1.05
    assert murphy_yield(800.0, 0.5) / poisson_yield(800.0, 0.5) > 3.0


def test_dies_per_wafer():
    # pi 150^2 / 100 - pi 300 / sqrt(200) = 706.86 - 66.64
    assert dies_per_wafer(100.0) == 640
    assert dies_per_wafer(1e6) == 0
    prev = math.inf
    for a in (50, 100, 200, 400, 800, 1600):
        n = dies_per_wafer(float(a))
        assert n < prev
        prev = n


def test_die_cost_rises_faster_than_area():
    c1, c2 = die_cost(200.0), die_cost(800.0)
    assert c2["usd_per_good_die"] > 4 * c1["usd_per_good_die"]
    assert c1["fits_reticle"] and not die_cost(900.0)["fits_reticle"]
    assert die_cost(200.0, DieCost(yield_model="poisson"))["yield"] == c1["poisson"]


def test_ppa_metrics_from_a_run():
    m = summarise(simulate(bootstrap_trace(PARAMS["ark"]), ARK_HW))
    q = ppa_metrics(m, ARK_HW)
    t, e = m["per_bootstrap_s"], m["energy"]["per_bootstrap_J"]
    assert q["perf_per_W"] == 1.0 / e                    # bootstraps per joule
    assert q["perf_per_mm2"] == (1.0 / t) / q["area_mm2"]["total"]
    assert q["edp_Js"] == e * t and q["ed2p_Js2"] == e * t * t


def test_area_does_not_change_equality_or_results():
    custom = ARK_HW.with_(area=AreaModel(mm2_per_bfly=1.0))
    assert custom == ARK_HW
    assert area_mm2(custom)["ntt"] == ARK_HW.ntt_bfly_per_cycle
    t = bootstrap_trace(PARAMS["ark"])
    assert simulate(t, custom).horizon == simulate(t, ARK_HW).horizon


def _rows():
    rows = []
    for i, (lat, en, ar) in enumerate([(1, 9, 5), (2, 2, 5), (3, 1, 1), (2, 2, 6), (1, 9, 5), (4, 4, 4), (9, 9, 0.5)]):
        rows.append({"id": i, "latency_s": lat, "energy_J": en, "area_mm2": ar})
    return rows


def test_pareto_set_is_non_dominated_and_covers_the_rest():
    rows = _rows()
    front = pareto_nd(rows)
    assert [r["id"] for r in front] == [0, 1, 2, 4, 6]    # equal points do not dominate each other
    for r in front:
        assert not any(dominates(o, r, ("latency_s", "energy_J", "area_mm2")) for o in rows)
    for r in rows:
        if r not in front:
            assert any(dominates(f, r, ("latency_s", "energy_J", "area_mm2")) for f in front)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 5), st.integers(0, 5), st.integers(0, 5)), min_size=1, max_size=25))
def test_pareto_property(points):
    rows = [{"latency_s": a, "energy_J": b, "area_mm2": c} for a, b, c in points]
    front = pareto_nd(rows)
    keys = ("latency_s", "energy_J", "area_mm2")
    assert front
    for r in rows:
        assert (r in front) == (not any(dominates(o, r, keys) for o in rows))


def test_cli_area_and_json(capsys):
    cli_main(["--area"])
    out = capsys.readouterr().out
    assert "latency 13.94 ms" in out and "area (7 nm" in out and "Murphy" in out
    cli_main(["--json"])
    d = json.loads(capsys.readouterr().out)
    assert round(d["per_bootstrap_s"] * 1e3, 2) == 13.94
    assert d["ppa"]["area_mm2"]["die"] == area_mm2(ARK_HW)["die"]


def test_default_text_output_has_no_area(capsys):
    cli_main([])
    assert "area (" not in capsys.readouterr().out


# ── the JavaScript port ────────────────────────────────────────────────

JS_KEYS = {"ntt_bfly_per_cycle": "nttBflyPerCycle", "mac_lanes": "macLanes", "auto_words_per_cycle": "autoWordsPerCycle",
           "sram_mib": "sramMib", "hbm_gbps": "hbmGbps"}
AREA_KEYS = {"ntt": "ntt", "mac": "mac", "auto": "auto", "sram": "sram", "uncore": "uncore", "hbm_phy": "hbmPhy",
             "optical_electronic": "opticalElectronic", "die": "die", "photonic_die": "photonicDie", "total": "total"}


def test_javascript_ppa_port_matches_python():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = ROOT / "web" / "sim_engine.js"
    cases = [("ark", {}), ("small", {}), ("hybrid", {}), ("ideal-optical", {}),
             ("ark", {"sram_mib": 96}), ("ark", {"sram_mib": 300}), ("ark", {"sram_mib": 777}), ("ark", {"sram_mib": 3000}),
             ("ark", {"ntt_bfly_per_cycle": 16384, "mac_lanes": 32768, "hbm_gbps": 1500.0}),
             ("ark", {"auto_words_per_cycle": 1024, "ntt_bfly_per_cycle": 8192})]
    payload, expect = [], []
    for name, over in cases:
        hw = ACCELERATORS[name].with_(**over)
        m = summarise(simulate(bootstrap_trace(PARAMS["ark"], BootOptions(min_ks=True)), hw))
        q = ppa_metrics(m, hw)
        payload.append({"hw": name, "over": {JS_KEYS[k]: v for k, v in over.items()}})
        expect.append(q)
    areas = [10.0, 123.4, 457.87, 858.0, 1500.0]
    script = (f"require({json.dumps(str(engine))});"
              "const F=globalThis.FheSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "const out=cases.map(c=>{const r=F.simulateNamed('ark',c.hw,{min_ks:true},false,c.over);"
              "const hw=F.withHw(F.ACCELERATORS[c.hw](),c.over);return F.ppaMetrics(F.summarise(r),hw);});"
              f"const ys={json.dumps(areas)}.map(a=>[F.poissonYield(a,0.1),F.murphyYield(a,0.1),F.diesPerWafer(a)]);"
              "const pts=JSON.parse(" + json.dumps(json.dumps([{k[0]: r[k] for k in ('latency_s', 'energy_J', 'area_mm2')}
                                                               for r in _rows()])) + ");"
              "const pf=F.paretoNd(pts.map(p=>({latencyS:p.l,energyJ:p.e,areaMm2:p.a}))).map(p=>pts.findIndex(q=>q.l===p.latencyS&&q.e===p.energyJ&&q.a===p.areaMm2));"
              "console.log(JSON.stringify({out,ys,pf}));")
    res = json.loads(subprocess.run([node, "-e", script], input=json.dumps(payload), capture_output=True, text=True,
                                    check=True).stdout)
    assert len(res["out"]) == len(expect)
    for e, g, c in zip(expect, res["out"], payload):
        for pk, jk in AREA_KEYS.items():
            assert g["areaMm2"][jk] == e["area_mm2"][pk], (c, pk)        # exact: no transcendentals
        assert g["dieCost"]["diesPerWafer"] == e["die_cost"]["dies_per_wafer"], c
        assert g["perfPerW"] == e["perf_per_W"] and g["perfPerMm2"] == e["perf_per_mm2"], c
        assert g["edpJs"] == e["edp_Js"] and g["ed2pJs2"] == e["ed2p_Js2"], c
        assert g["dieCost"]["murphy"] == pytest.approx(e["die_cost"]["murphy"], rel=1e-12), c   # exp: tolerance
        assert g["usdPerUnit"] == pytest.approx(e["usd_per_unit"], rel=1e-12), c
    for a, (yp, ym, dpw) in zip(areas, res["ys"]):
        assert yp == pytest.approx(poisson_yield(a, 0.1), rel=1e-12)
        assert ym == pytest.approx(murphy_yield(a, 0.1), rel=1e-12)
        assert dpw == dies_per_wafer(a)
    # paretoNd: the same front (the JS indices find the first equal point, so 4 maps to 0)
    assert sorted(set(res["pf"])) == sorted({0 if r["id"] == 4 else r["id"] for r in pareto_nd(_rows())})


def test_readme_ppa_numbers_come_from_results_md(capsys):
    """Every table in the README's PPA section is copied verbatim from examples/results.md, and the
    quoted `fhe-sim --area` lines are what the CLI prints."""
    readme = (ROOT / "README.md").read_text()
    sec = readme.split("### Power, performance and area (PPA)")[1].split("\n### ")[0]
    results = (ROOT / "examples" / "results.md").read_text()
    tables, cur = [], []
    for line in sec.splitlines() + [""]:
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append("\n".join(cur))
            cur = []
    assert len(tables) == 5
    for t in tables:
        assert t in results
    cli_main(["--area"])
    out = capsys.readouterr().out
    block = sec.split("```\n")[1].split("```")[0].strip()
    assert block in out
