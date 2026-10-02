"""The scheme model against real OpenFHE kernel streams.

Two bootstraps were recorded with an instrumented OpenFHE v1.5.1
(calibration/openfhe_trace): N = 2^14 with 8,192 slots and level budget {3, 3},
and N = 2^16 with 8 slots and level budget {1, 1} (the timing-calibration
configuration). These tests pin down where the model agrees with the library,
where it differs and why, and that the recorded stream replays on the engine.
"""

import json
from pathlib import Path

import pytest

from fhe_sim import (ACCELERATORS, PARAMS, BootOptions, bootstrap_trace, dump_trace, load_trace,
                     simulate, summarise, summarise_trace)
from fhe_sim.hardware import CPU_LIKE
from fhe_sim.openfhe_trace import log_to_trace, read_log, stage_order, summarise_log
from fhe_sim.workload import output_level

ROOT = Path(__file__).parent.parent
D = ROOT / "calibration" / "openfhe_trace"
FULL = read_log(D / "full14.log.gz")
SPARSE = read_log(D / "sparse16.log.gz")


def transforms(s):
    return s["ntt_limbs"] + s["intt_limbs"]


def test_logs_parse():
    assert FULL.header["N"] == 1 << 14 and FULL.header["slots"] == 1 << 13
    assert SPARSE.header["N"] == 1 << 16 and SPARSE.header["slots"] == 8
    assert stage_order(FULL) == ["modraise", "cts", "evalmod", "stc", "post"][:len(stage_order(FULL))]
    assert {"cts", "evalmod", "stc"} <= set(stage_order(SPARSE))
    assert FULL.relin and SPARSE.relin


@pytest.mark.parametrize("log,preset", [(FULL, "openfhe-full14"), (SPARSE, "openfhe-sparse")])
def test_preset_shape_and_levels_match_openfhe(log, preset):
    p, h = PARAMS[preset], log.header
    assert (p.N, p.L + 1, p.dnum, p.alpha, p.k) == (h["N"], h["towersQ"], h["dnum"], h["alpha"], h["sizeP"])
    assert p.L - output_level(bootstrap_trace(p)) == h["bootDepth"]


def test_key_switching_matches_openfhe_exactly_where_the_algorithms_agree():
    """SubSum (sparse packing) is 12 plain rotations in both: same keys, same bytes, same transforms."""
    o = summarise_log(SPARSE)["modraise"]
    m = summarise_trace(bootstrap_trace(PARAMS["openfhe-sparse"]))["modraise"]
    assert (o["hrot"], o["distinct_keys"], o["key_bytes"]) == (m["hrot"], m["distinct_keys"], m["key_bytes"])
    assert transforms(m) == pytest.approx(transforms(o), rel=0.03)
    assert m["bconv"] == pytest.approx(o["bconv"], rel=0.06)


def test_lazy_moddown_reproduces_openfhes_dft():
    """With OpenFHE's BSGS split and Q*P accumulation the model's DFT stages land within 20% of
    OpenFHE on transforms and rotations; the balanced-split baseline is more than 2x off."""
    o = summarise_log(FULL)
    eager = summarise_trace(bootstrap_trace(PARAMS["openfhe-full14"]))
    lazy = summarise_trace(bootstrap_trace(PARAMS["openfhe-full14"], BootOptions(lazy_moddown=True)))
    for st in ("cts", "stc"):
        assert transforms(eager[st]) > 2 * transforms(o[st])
        assert transforms(lazy[st]) == pytest.approx(transforms(o[st]), rel=0.20)
        assert lazy[st]["hrot"] == pytest.approx(o[st]["hrot"], rel=0.15)


def test_openfhe_evalmod_rescales_on_use():
    """OpenFHE (FLEXIBLEAUTO) rescales copies of each input before every multiplication, so its
    EvalMod does far more transforms per HMult than a rescale-once schedule. The model keeps the
    rescale-once schedule an accelerator compiler would use; this pins the difference down."""
    for log, preset in ((FULL, "openfhe-full14"), (SPARSE, "openfhe-sparse")):
        o = summarise_log(log)["evalmod"]
        m = summarise_trace(bootstrap_trace(PARAMS[preset]))["evalmod"]
        assert o["rescale_polys"] / o["hmult"] > 4 * 2            # model: 2 polynomials per HMult
        assert 1.5 < transforms(o) / transforms(m) < 2.0


def test_replayed_trace_is_a_valid_trace(tmp_path):
    t = log_to_trace(SPARSE)
    dump_trace(t, tmp_path / "t.json")
    back = load_trace(tmp_path / "t.json")
    assert summarise_trace(back) == summarise_trace(t)
    r = simulate(t, ACCELERATORS["ark"])
    producer = {o.output: o.id for o in t.ops}
    for o in t.ops:
        for x in o.inputs:
            if x in producer:
                assert r.op_start[o.id] >= r.op_end[producer[x]]
    assert summarise(r)["energy"]["peak_power_W"] <= ACCELERATORS["ark"].tdp_w * (1 + 1e-12)


def test_replayed_trace_beats_the_scheme_model_on_calibration():
    meas = json.loads((ROOT / "calibration" / "openfhe_measured.json").read_text())
    boot = meas["bootstrap_N65536_slots8_levelBudget11_dnum3_depth18_threads8"]["boot_s"]
    measured = sum(boot) / len(boot)
    replay = simulate(log_to_trace(SPARSE), CPU_LIKE).horizon
    model = simulate(bootstrap_trace(PARAMS["openfhe-sparse"]), CPU_LIKE).horizon
    assert abs(replay - measured) / measured < 0.15
    assert abs(replay - measured) < abs(model - measured)


def test_openfhes_cpu_tuned_bsgs_costs_a_memory_bound_accelerator():
    """OpenFHE's split saves NTTs (good on a CPU) but needs more rotation keys; on the
    memory-bound ARK-class design it is slower."""
    base = bootstrap_trace(PARAMS["ark"])
    lazy = bootstrap_trace(PARAMS["ark"], BootOptions(lazy_moddown=True))
    tb, tl = summarise_trace(base), summarise_trace(lazy)
    assert sum(map(transforms, tl.values())) < 0.75 * sum(map(transforms, tb.values()))
    mb, ml = summarise(simulate(base, ACCELERATORS["ark"])), summarise(simulate(lazy, ACCELERATORS["ark"]))
    assert ml["hbm_bytes"]["key"] > mb["hbm_bytes"]["key"]
    assert ml["per_bootstrap_s"] > 1.2 * mb["per_bootstrap_s"]
