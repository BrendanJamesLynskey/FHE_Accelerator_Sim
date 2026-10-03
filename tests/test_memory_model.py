"""The optional detailed memory model (Accelerator.memory).

A model is any object with ``chunk_time(nbytes, write, prev_write, peak_gbps) -> seconds``.
Memory_System_Sim's HBMChunkModel is the real one; these tests use stubs to pin the
interface, and run the real one only if it is installed.
"""

import pytest

from fhe_sim import ACCELERATORS, PARAMS, BootOptions, SimConfig, bootstrap_trace, simulate, summarise

ARK, HW = PARAMS["ark"], ACCELERATORS["ark"]


class Flat:
    """bytes / (peak x efficiency), recording every call."""

    def __init__(self, eff=1.0):
        self.eff, self.calls = eff, []

    def chunk_time(self, nbytes, write, prev_write, peak_gbps):
        self.calls.append((nbytes, write, prev_write))
        return nbytes / (peak_gbps * self.eff * 1e9)


def run(hw, **opts):
    return summarise(simulate(bootstrap_trace(ARK, BootOptions(**opts)), SimConfig(hw)))


def test_default_has_no_memory_model():
    assert HW.memory is None


@pytest.mark.parametrize("mode", ["worst-case", "dynamic"])
def test_peak_stub_reproduces_the_default(mode):
    hw = HW.with_(power_mode=mode)
    a, b = run(hw), run(hw.with_(memory=Flat(1.0)))
    assert b["per_bootstrap_s"] == pytest.approx(a["per_bootstrap_s"], rel=1e-12)
    assert b["hbm_bytes"] == a["hbm_bytes"]
    assert b["energy"]["per_bootstrap_J"] == pytest.approx(a["energy"]["per_bootstrap_J"], rel=1e-12)


def test_flat_stub_matches_derated_bandwidth():
    """A flat-efficiency model is exactly 'bandwidth x efficiency' (worst-case clocking)."""
    hw = HW.with_(power_mode="worst-case")
    a = run(hw.with_(hbm_gbps=HW.hbm_gbps * 0.5))
    b = run(hw.with_(memory=Flat(0.5)))
    assert b["per_bootstrap_s"] == pytest.approx(a["per_bootstrap_s"], rel=1e-12)


def test_halving_efficiency_doubles_hbm_busy_time():
    hw = HW.with_(power_mode="worst-case")
    a = simulate(bootstrap_trace(ARK), SimConfig(hw.with_(memory=Flat(1.0))))
    b = simulate(bootstrap_trace(ARK), SimConfig(hw.with_(memory=Flat(0.5))))
    assert b.stats.busy["hbm"] == pytest.approx(2 * a.stats.busy["hbm"], rel=1e-12)
    assert b.horizon > a.horizon


def test_model_sees_sizes_directions_and_history():
    m = Flat(1.0)
    res = simulate(bootstrap_trace(ARK), SimConfig(HW.with_(memory=m)))
    st = res.stats.bytes
    assert sum(n for n, _, _ in m.calls) == st["key"] + st["pt"] + st["ct_read"] + st["ct_write"]
    assert sum(n for n, w, _ in m.calls if w) == st["ct_write"]
    assert max(n for n, _, _ in m.calls) <= HW.hbm_chunk_mib * 2 ** 20
    assert m.calls[0][2] is None                                   # nothing before the first chunk
    assert all(m.calls[k][2] == m.calls[k - 1][1] for k in range(1, len(m.calls)))


def test_memory_model_is_not_part_of_equality():
    assert HW.with_(memory=Flat()) == HW


def test_memsim_hbm_model():
    fhe = pytest.importorskip("memsim.fhe")
    m = fhe.HBMChunkModel()
    a = run(HW)
    b = run(HW.with_(memory=m))
    # a 4 MiB sequential chunk runs at 85-95 % of peak in this HBM model (refresh is most of the loss)
    eff = m.efficiency(4 << 20)
    assert 0.85 < eff < 0.95
    assert b["per_bootstrap_s"] > a["per_bootstrap_s"]
    assert b["hbm_bytes"] == a["hbm_bytes"]
