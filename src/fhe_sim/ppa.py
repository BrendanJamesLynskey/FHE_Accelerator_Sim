"""Power, performance and area: an area model, die yield and cost, and composite PPA metrics.

The simulator measures performance (latency, throughput) and power (energy per bootstrap, peak
power); this module adds the third axis an architect trades against them.

* ``AreaModel`` / ``area_mm2``  silicon area per component from the hardware knobs, at a **stated
  node (7 nm)**. Every coefficient is commented with its source:
    - functional units: ARK's published 7 nm area breakdown (Kim et al., MICRO 2022, Table IV,
      https://arxiv.org/abs/2205.00922), divided by its unit counts;
    - SRAM: the shape of area against capacity from a CACTI 7 sweep (``calibration/cacti``, 22 nm,
      the smallest node CACTI has data for), scaled to 7 nm by one factor anchored on ARK's
      512 MB scratchpad (BTS, https://arxiv.org/abs/2112.15479, agrees within 3%);
    - HBM PHYs: ARK's and BTS's two-stack figure;
    - wiring, register files and NoC: one overhead fraction, ARK's ratio;
    - the optical engine: **speculative** round numbers (see ``FHESim 04`` for its model).
  Scaling a published design's per-unit area to other unit counts is linear and **illustrative**:
  real floorplans have wiring that grows faster than the units it connects.
* ``poisson_yield`` / ``murphy_yield`` / ``dies_per_wafer`` / ``die_cost``  why area is cost.
  D0 and the wafer price are **illustrative**.
* ``ppa_metrics``  perf/W, perf/mm², bootstraps/s per dollar, EDP and ED²P from a run's summary.

Nothing here changes a simulated time or energy: area is computed *from* an ``Accelerator`` and
never fed back into it. ``web/sim_engine.js`` has a port that matches this file exactly (only
``exp`` in the yield models is compared with a tolerance; ``sqrt`` is correctly rounded in both).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hardware import Accelerator

# CACTI 7, 22 nm, itrs-lstp cells and peripherals, scratchpad ("ram"), 4 MiB banks:
# (capacity MiB, mm² per MiB). Copied from calibration/cacti/out/results.json, which
# tests/test_ppa.py checks against this table. Low-standby-power cells, because a scratchpad of
# hundreds of MiB built from high-performance cells would leak hundreds of watts (CACTI: 180 W at
# 512 MiB in itrs-hp against 0.14 W in itrs-lstp).
CACTI_LSTP_22NM = ((64, 0.9189), (128, 0.8901), (256, 0.9020), (512, 0.8774), (1024, 0.8189), (2048, 0.7995))


@dataclass(frozen=True)
class AreaModel:
    node: str = "7 nm (ASAP7-class predictive PDK, as used by ARK and BTS)"
    # SRAM: CACTI's 22 nm curve x one 22 nm -> 7 nm factor. The factor is anchored on ARK's 512 MB
    # scratchpad, 229.2 mm² (Table IV) = 0.4477 mm²/MiB, against CACTI's 0.8774 at 512 MiB.
    # (BTS: 2,048 x 114,724 um² = 235.0 mm² for 512 MB, 2.5% more.) Outside 64 MiB-2 GiB the
    # nearest CACTI point is used (CACTI 7 cannot model 4 GiB).
    sram_curve: tuple = CACTI_LSTP_22NM
    sram_node_scale: float = 229.2 / 512 / 0.8774
    # NTT: ARK's 4 NTTUs are 57.2 mm²; each is a pipelined NTT with (1/2) sqrt(N) log2 N = 2,048
    # modular multipliers (butterflies) at N = 2^16, so 8,192 butterflies per cycle. Includes the
    # NTTUs' wiring, which ARK reports as most of their area (logic alone: 34.9 mm²).
    mm2_per_bfly: float = 57.2 / 8192
    # MAC lanes: ARK's 4 BConvUs (9.3 mm², 256 lanes x 6 MAC units each = 6,144) plus 8 MADUs
    # (8.9 mm², assumed one modular multiplier per lane on ARK's 256 lanes = 2,048): 18.2 mm² for
    # 8,192 modular multiply-adds per cycle.
    mm2_per_mac_lane: float = 18.2 / 8192
    # Automorphism: ARK's 4 AutoUs are 20.6 mm², each permuting one 256-word vector per cycle
    # (1,024 words per cycle). Mostly long wires (logic alone: 0.9 mm²).
    mm2_per_auto_word: float = 20.6 / 1024
    # HBM: ARK and BTS both give 29.6 mm² for two HBM2/HBM2e stacks' PHYs; ARK runs each at 500 GB/s.
    hbm_stack_gbps: float = 500.0
    mm2_per_hbm_stack: float = 29.6 / 2
    # Register files (42.8 mm²) and NoC (20.6 mm²) in ARK, as a fraction of its units + scratchpad
    # (325.2 mm²). Applied to every design: an illustrative simplification.
    uncore_frac: float = (42.8 + 20.6) / 325.2
    # Optical engine: SPECULATIVE. Electronic side on the main die: DAC and ADC channels (with their
    # SerDes and drivers) at converter_gsps each, enough for the engine's samples/s. Photonic side:
    # a separate die of photonic_die_mm2. Round numbers, not from a design.
    converter_gsps: float = 50.0
    mm2_per_dac: float = 0.05
    mm2_per_adc: float = 0.10
    photonic_die_mm2: float = 100.0

    def sram_mm2_per_mib(self, mib: float) -> float:
        c = self.sram_curve
        if mib <= c[0][0]:
            per = c[0][1]
        elif mib >= c[-1][0]:
            per = c[-1][1]
        else:
            i = 1
            while c[i][0] < mib:
                i += 1
            (x0, y0), (x1, y1) = c[i - 1], c[i]
            per = y0 + (y1 - y0) * (mib - x0) / (x1 - x0)
        return per * self.sram_node_scale


AREA_7NM = AreaModel()


def area_mm2(hw: Accelerator, am: AreaModel | None = None) -> dict:
    """Area per component (mm²). ``die`` is the main (7 nm) die; ``total`` adds a photonic die.
    The model is ``am``, else ``hw.area``, else ``AREA_7NM``."""
    am = am or hw.area or AREA_7NM
    ntt = hw.ntt_bfly_per_cycle * am.mm2_per_bfly
    mac = hw.mac_lanes * am.mm2_per_mac_lane
    auto = hw.auto_words_per_cycle * am.mm2_per_auto_word
    sram = hw.sram_mib * am.sram_mm2_per_mib(hw.sram_mib)
    uncore = (ntt + mac + auto + sram) * am.uncore_frac
    hbm_phy = math.ceil(hw.hbm_gbps / am.hbm_stack_gbps) * am.mm2_per_hbm_stack
    opt_e, photonic = 0.0, 0.0
    if hw.optical is not None:
        ch = math.ceil(hw.optical.samples_per_s / (am.converter_gsps * 1e9))
        opt_e = ch * (am.mm2_per_dac + am.mm2_per_adc)
        photonic = am.photonic_die_mm2
    die = ntt + mac + auto + sram + uncore + hbm_phy + opt_e
    return {"ntt": ntt, "mac": mac, "auto": auto, "sram": sram, "uncore": uncore, "hbm_phy": hbm_phy,
            "optical_electronic": opt_e, "die": die, "photonic_die": photonic, "total": die + photonic}


# ── yield and cost ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DieCost:
    """Illustrative manufacturing assumptions (not a foundry's numbers)."""
    d0_per_cm2: float = 0.1        # defect density: illustrative, of the order quoted for mature processes
    wafer_mm: float = 300.0
    wafer_usd: float = 10000.0     # illustrative: public estimates for a 7 nm wafer are of this order
    reticle_mm2: float = 858.0     # 26 mm x 33 mm, the lithography field: the largest die one exposure prints
    yield_model: str = "murphy"


DIE_COST = DieCost()


def poisson_yield(area_mm2: float, d0_per_cm2: float) -> float:
    """Y = exp(-A D0): defects independent and uniform. Pessimistic for large dies."""
    return math.exp(-(area_mm2 / 100.0) * d0_per_cm2)


def murphy_yield(area_mm2: float, d0_per_cm2: float) -> float:
    """Y = ((1 - exp(-A D0)) / (A D0))^2: defect density varying across the wafer (triangular)."""
    x = (area_mm2 / 100.0) * d0_per_cm2
    if x == 0:
        return 1.0
    t = -math.expm1(-x) / x       # not 1 - exp(-x): that cancels to 0 for tiny dies (Hypothesis found it)
    return t * t


def dies_per_wafer(area_mm2: float, wafer_mm: float = 300.0) -> int:
    """Gross dies: wafer area / die area, minus the partial dies lost around the edge."""
    r = wafer_mm / 2.0
    n = math.pi * r * r / area_mm2 - math.pi * wafer_mm / math.sqrt(2.0 * area_mm2)
    return max(0, math.floor(n))


def die_cost(area: float, dc: DieCost = DIE_COST) -> dict:
    dpw = dies_per_wafer(area, dc.wafer_mm)
    yp, ym = poisson_yield(area, dc.d0_per_cm2), murphy_yield(area, dc.d0_per_cm2)
    y = ym if dc.yield_model == "murphy" else yp
    good = dpw * y
    return {"area_mm2": area, "dies_per_wafer": dpw, "poisson": yp, "murphy": ym, "yield": y,
            "good_dies": good, "usd_per_good_die": dc.wafer_usd / good if good > 0 else math.inf,
            "fits_reticle": area <= dc.reticle_mm2}


# ── composite metrics ──────────────────────────────────────────────────

def ppa_metrics(m: dict, hw: Accelerator, am: AreaModel | None = None, dc: DieCost = DIE_COST) -> dict:
    """Composite metrics from ``summarise()`` output plus area. perf = bootstraps per second.

    perf/W = (1/t) / (E/t) = 1/E: bootstraps per joule, so perf/W ranks designs by energy alone.
    """
    am = am or hw.area or AREA_7NM
    a = area_mm2(hw, am)
    t, e = m["per_bootstrap_s"], m["energy"]["per_bootstrap_J"]
    cost = die_cost(a["die"], dc)
    usd = cost["usd_per_good_die"]
    if a["photonic_die"] > 0:
        usd = usd + die_cost(a["photonic_die"], dc)["usd_per_good_die"]
    perf = 1.0 / t
    return {"area_mm2": a, "node": am.node, "die_cost": cost, "usd_per_unit": usd,
            "perf_per_s": perf, "perf_per_W": 1.0 / e, "perf_per_mm2": perf / a["total"],
            "perf_per_usd": perf / usd, "edp_Js": e * t, "ed2p_Js2": e * t * t}


def format_area(p: dict) -> str:
    a, c = p["area_mm2"], p["die_cost"]
    parts = "  ".join(f"{k} {a[k]:.1f}" for k in ("ntt", "mac", "auto", "sram", "uncore", "hbm_phy")
                      ) + (f"  optical-electronic {a['optical_electronic']:.1f}" if a["optical_electronic"] else "")
    lines = [f"area ({p['node']}; illustrative)",
             f"  mm²     {parts}",
             f"  die {a['die']:.1f} mm²" + (f" + photonic die {a['photonic_die']:.1f} mm² (speculative)"
                                             if a["photonic_die"] else "")
             + ("" if c["fits_reticle"] else "  (larger than the 858 mm² reticle: cannot be one die)"),
             f"  yield   Poisson {100 * c['poisson']:.1f}%  Murphy {100 * c['murphy']:.1f}%  "
             f"{c['dies_per_wafer']} dies/wafer  ${p['usd_per_unit']:,.0f} of silicon per good unit "
             f"(illustrative D0 and wafer price; no HBM, packaging or test)",
             f"  PPA     {p['perf_per_s']:.1f} bootstraps/s  {p['perf_per_W']:.2f} per J  "
             f"{p['perf_per_mm2']:.3f} /s per mm²  {1e3 * p['perf_per_usd']:.2f} /s per $1000  "
             f"EDP {1e3 * p['edp_Js']:.3f} mJ·s"]
    return "\n".join(lines)
