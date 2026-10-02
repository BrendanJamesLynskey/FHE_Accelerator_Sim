"""Property-based tests: invariants checked over randomly generated parameters and hardware."""

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from fhe_sim import ACCELERATORS, BootOptions, CKKSParams, bootstrap_trace, simulate, summarise  # noqa: E402
from fhe_sim.hardware import CostModel  # noqa: E402


@st.composite
def configs(draw):
    log_n = draw(st.integers(10, 13))
    cts, stc = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    deg = draw(st.sampled_from([7, 15, 31]))
    r = draw(st.integers(0, 2))
    p0 = CKKSParams("h", log_n=log_n, L=1, dnum=1, cts_levels=cts, stc_levels=stc,
                    evalmod_degree=deg, double_angle=r)
    b, g = p0.evalmod_baby, p0.evalmod_giant
    need = cts + stc + 2 + (b - 1).bit_length() + (g - 1).bit_length() + r
    L = need + draw(st.integers(0, 4))
    p = p0.with_(L=L, dnum=draw(st.integers(1, L + 1)),
                 log_slots=draw(st.sampled_from([None, log_n - 3])))
    opts = BootOptions(n_boot=draw(st.integers(1, 2)), hoisting=draw(st.booleans()),
                       min_ks=draw(st.booleans()), seeded_keys=draw(st.booleans()),
                       otf_plaintexts=draw(st.booleans()), lazy_moddown=draw(st.booleans()))
    base = ACCELERATORS[draw(st.sampled_from(["ark", "small", "ideal-optical"]))]
    ws = p.ks_working_set() >> 20
    hw = base.with_(window=draw(st.integers(1, 8)),
                    sram_mib=ws + 1 + draw(st.integers(0, 64)),
                    hbm_gbps=draw(st.sampled_from([200.0, 1000.0, 4000.0])),
                    hbm_chunk_mib=draw(st.sampled_from([1, 4])),
                    power_mode=draw(st.sampled_from(["dynamic", "worst-case"])),
                    tdp_w=draw(st.sampled_from([base.tdp_w, base.tdp_w + 100.0])))
    if hw.optical is not None and hw.optical.block > p.N:
        hw = hw.with_(optical=None)
    return p, opts, hw


@settings(max_examples=60, deadline=None)
@given(configs())
def test_invariants_for_any_config(cfg):
    p, opts, hw = cfg
    t = bootstrap_trace(p, opts)
    r = simulate(t, hw)
    producer = {o.output: o.id for o in t.ops}
    for o in t.ops:                                   # dependencies
        for x in o.inputs:
            if x in producer:
                assert r.op_start[o.id] >= r.op_end[producer[x]]
    cm = CostModel(hw, p.log_n, p.q_bits, r.clock)    # conservation of work
    work = {u: 0.0 for u in r.stats.work}
    for o in t.ops:
        for k in o.kernels:
            for s in cm.segments(k):
                work[s.unit] += s.work
    for u in work:
        assert abs(r.stats.work[u] - work[u]) <= 1e-9 * max(1.0, work[u])
    m = summarise(r)
    assert r.horizon >= m["lower_bound_s"]
    assert m["energy"]["peak_power_W"] <= hw.tdp_w * (1 + 1e-12)
    r2 = simulate(t, hw)                              # determinism
    assert r2.op_end == r.op_end and r2.stats.bytes == r.stats.bytes
