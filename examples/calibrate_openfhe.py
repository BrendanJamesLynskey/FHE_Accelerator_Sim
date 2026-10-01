"""Calibrate the CPU-like configuration against OpenFHE and check it out of sample.

Measurements (calibration/openfhe_measured.json) come from openfhe-python on an
Intel i7-3770 (4 cores / 8 threads), via calibration/prims.py and sboot.py.

* One parameter is fitted: the modular-operation rate per cycle, shared by the
  NTT and MAC units, chosen by bisection so the simulated HMult matches OpenFHE.
* Two quantities are then *predicted*: HRotate at the same parameters, and a
  sparse-slot bootstrap at N = 2^16.

    python examples/calibrate_openfhe.py
"""

import json
from pathlib import Path

from fhe_sim import PARAMS, bootstrap_trace, he_op_trace, simulate
from fhe_sim.hardware import CPU_LIKE

meas = json.loads((Path(__file__).parent.parent / "calibration" / "openfhe_measured.json").read_text())
prim = meas["primitives_N65536_limbs24_dnum4"]["threads8"]
boot = meas["bootstrap_N65536_slots8_levelBudget11_dnum3_depth18_threads8"]


def cpu(rate):
    return CPU_LIKE.with_(ntt_bfly_per_cycle=rate, mac_lanes=rate, auto_words_per_cycle=4 * rate)


def t(hw, trace):
    return simulate(trace, hw).horizon


ark = PARAMS["ark"]                      # N = 2^16, L = 23 (24 limbs), dnum = 4: the measured shape
lo, hi = 0.01, 100.0
for _ in range(60):
    mid = (lo + hi) / 2
    if t(cpu(mid), he_op_trace(ark, "hmult")) > prim["hmult_relin_ms"] / 1e3:
        lo = mid
    else:
        hi = mid
hw = cpu(lo)
rows = [("HMult (fitted)", prim["hmult_relin_ms"] / 1e3, t(hw, he_op_trace(ark, "hmult"))),
        ("HRotate (predicted)", prim["hrot_ms"] / 1e3, t(hw, he_op_trace(ark, "hrot"))),
        ("Sparse bootstrap (predicted)", sum(boot["boot_s"]) / len(boot["boot_s"]),
         t(hw, bootstrap_trace(PARAMS["openfhe-sparse"])))]
print(f"fitted rate: {lo:.4f} modular ops per cycle per unit = {lo * hw.freq_ghz:.3f} G/s")
print(f"{'quantity':<30}{'OpenFHE':>12}{'simulated':>12}{'error':>9}")
for name, m, s in rows:
    print(f"{name:<30}{m * 1e3:10.1f}ms{s * 1e3:10.1f}ms{100 * (s - m) / m:8.0f}%")
