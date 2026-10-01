"""The accelerator: functional units, memories, the optional optical transform
engine, and the power model.

Everything the simulator knows about *time and energy* comes from this file. The
event engine (``sim.py``) asks one question: how long does this kernel take on
this unit, and how many joules does it cost? Swapping these formulas for a
calibrated table or RTL-derived numbers needs no change to the engine.

**All coefficients are illustrative.** Unit throughputs are sized like the
published ASIC designs (ARK: 1 GHz, 512 MB scratchpad, 1 TB/s HBM, ~280 W peak);
energies per operation are round numbers chosen to give plausible totals. Treat
absolute results as orders of magnitude and ratios as the useful output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .params import MiB, cdiv, log2_int
from .workload import Kernel

UNITS = ["ntt", "mac", "auto", "optical"]
KIND_UNIT = {"ntt": "ntt", "intt": "ntt", "bconv": "mac", "mac": "mac", "auto": "auto"}


@dataclass(frozen=True)
class OpticalEngine:
    """A hybrid electro-optical transform engine (illustrative).

    Mapping used (one route among several; see deck 04): the first
    log2(N / block) butterfly stages of each NTT stay digital, splitting
    X^N + 1 into length-``block`` pieces. Each piece is an exact modular DFT,
    computed as a Bluestein chirp convolution (a cyclic convolution of size
    2 x block), which the optics evaluates as an analogue complex FFT convolution
    on b-bit **digit planes**. Rounding is exact only if the analogue full scale
    fits the converters' effective bits:

        2^(ENOB - 1) > m * block * (2^b - 1)^2,   m = d if grouped else 1

    with d = ceil(q_bits / b) digits per operand. The engine picks the widest b
    that satisfies this; narrower digits mean more passes and conversions.
    ``ideal=True`` is a hypothetical upper bound: an analogue engine that is exact
    at any precision (one plane in, one plane out).
    """

    name: str = "Hybrid optical NTT"
    block: int = 16              # points per optical sub-transform (Bluestein FFT size 2 x block)
    enob: int = 12               # effective bits of the DAC -> optics -> ADC chain
    samples_per_s: float = 1e12  # converter throughput (each of DAC and ADC), samples/s
    grouping: str = "grouped"    # "grouped": sum equal-weight digit products before the ADC; "pairwise"
    fom_dac_fj: float = 10.0     # Walden figure of merit, fJ per conversion step
    fom_adc_fj: float = 20.0
    laser_w: float = 10.0        # static: lasers (on whether or not work arrives)
    tuning_w: float = 10.0       # static: thermal tuning of resonant devices
    ideal: bool = False          # hypothetical: exact at any precision (no digit planes)

    def digits(self, q_bits: int) -> tuple[int, int]:
        """(b, d): widest digit width b whose analogue full scale fits ENOB, and digits d."""
        if self.ideal:
            return q_bits, 1
        for b in range(q_bits, 0, -1):
            d = cdiv(q_bits, b)
            m = d if self.grouping == "grouped" else 1
            if 2 ** (self.enob - 1) > m * self.block * (2 ** b - 1) ** 2:
                return b, d
        raise ValueError(f"ENOB {self.enob} cannot round a {self.block}-point transform exactly "
                         f"even with 1-bit digits; lower block or raise ENOB")

    def planes(self, q_bits: int) -> tuple[int, int]:
        """(DAC planes, ADC planes) per block: input digit planes in, products out."""
        _, d = self.digits(q_bits)
        if self.ideal:
            return 1, 1
        return (d, 2 * d - 1) if self.grouping == "grouped" else (d * d, d * d)

    def pj_dac(self) -> float:
        return self.fom_dac_fj * 2 ** self.enob * 1e-3

    def pj_adc(self) -> float:
        return self.fom_adc_fj * 2 ** self.enob * 1e-3

    @property
    def static_w(self) -> float:
        return self.laser_w + self.tuning_w


@dataclass(frozen=True)
class Accelerator:
    name: str = "Digital FHE accelerator (ARK-class, illustrative)"
    freq_ghz: float = 1.0
    ntt_bfly_per_cycle: int = 4096     # aggregate modular butterflies per cycle (all NTT units)
    mac_lanes: int = 8192              # modular multiply-adds per cycle (MAC + base conversion)
    auto_words_per_cycle: int = 4096   # automorphism / permutation network
    sram_mib: int = 512                # on-chip scratchpad
    sram_gbps: float = 20000.0         # scratchpad bandwidth seen by each unit, GB/s
    hbm_gbps: float = 1000.0           # off-chip bandwidth, GB/s (shared, FCFS)
    hbm_chunk_mib: int = 4             # transfers are split into chunks that interleave
    window: int = 4                    # HE operations in flight (issue window)
    # power (illustrative): static + pJ per op + pJ per byte; logic energy scales with clock^2
    tdp_w: float = 250.0
    static_w: float = 40.0
    pj_bfly: float = 10.0
    pj_mac: float = 5.0
    pj_auto_word: float = 1.0
    pj_sram_byte: float = 1.0
    pj_hbm_byte: float = 30.0
    s_min: float = 0.5                 # lowest clock as a fraction of nominal
    enforce_tdp: bool = True
    optical: OpticalEngine | None = None

    def with_(self, **kw) -> "Accelerator":
        return replace(self, **kw)

    @property
    def sram_bytes(self) -> int:
        return self.sram_mib * MiB

    def rate(self, unit: str) -> float:
        per_cycle = {"ntt": self.ntt_bfly_per_cycle, "mac": self.mac_lanes,
                     "auto": self.auto_words_per_cycle}[unit]
        return per_cycle * self.freq_ghz * 1e9

    def pj(self, unit: str) -> float:
        return {"ntt": self.pj_bfly, "mac": self.pj_mac, "auto": self.pj_auto_word}[unit]

    def peak_power(self, s: float) -> float:
        """Worst case: every unit streaming at full rate at clock fraction s, plus HBM and optics."""
        sram = self.sram_gbps * 1e9 * self.pj_sram_byte * 1e-12
        p = self.static_w + self.hbm_gbps * 1e9 * self.pj_hbm_byte * 1e-12
        for u in ("ntt", "mac", "auto"):
            p += self.rate(u) * s * self.pj(u) * 1e-12 * s * s + sram * s
        if self.optical is not None:
            o = self.optical
            p += o.static_w + o.samples_per_s * (o.pj_dac() + o.pj_adc()) * 1e-12
        return p

    def tdp_clock(self) -> float:
        """Largest clock fraction whose worst-case power fits the TDP (bisection, no transcendentals)."""
        if not self.enforce_tdp or self.peak_power(1.0) <= self.tdp_w:
            return 1.0
        if self.peak_power(self.s_min) > self.tdp_w:
            raise ValueError(f"TDP {self.tdp_w} W is below the power at the lowest clock")
        lo, hi = self.s_min, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.peak_power(mid) <= self.tdp_w:
                lo = mid
            else:
                hi = mid
        return lo


@dataclass
class Segment:
    unit: str
    time: float
    energy: float         # dynamic joules (logic + SRAM; converters for optics)
    work: float           # butterflies, multiply-adds, words, or converter samples
    dac: float = 0.0
    adc: float = 0.0


@dataclass
class CostModel:
    hw: Accelerator
    log_n: int
    q_bits: int
    s: float = 1.0                       # clock fraction actually used
    _opt: tuple | None = field(default=None, repr=False)

    def __post_init__(self):
        o = self.hw.optical
        if o is not None:
            if o.block > (1 << self.log_n):
                raise ValueError("optical block larger than the ring degree")
            b, d = o.digits(self.q_bits)
            dac, adc = o.planes(self.q_bits)
            self._opt = (b, d, dac, adc, log2_int(o.block))

    def _seg(self, unit: str, work: float, words: int) -> Segment:
        hw, s = self.hw, self.s
        t_logic = work / (hw.rate(unit) * s)
        t_sram = words * 8 / (hw.sram_gbps * 1e9 * s)
        t = t_logic if t_logic >= t_sram else t_sram
        e = work * hw.pj(unit) * 1e-12 * s * s + words * 8 * hw.pj_sram_byte * 1e-12
        return Segment(unit, t, e, work)

    def segments(self, k: Kernel) -> list[Segment]:
        unit = KIND_UNIT[k.kind]
        N = 1 << self.log_n
        if unit != "ntt":
            return [self._seg(unit, k.amount, k.words)]
        if self._opt is None:
            return [self._seg("ntt", k.amount * (N // 2) * self.log_n, k.words)]
        # hybrid: digital stages, then optical blocks, then digital correction
        b, d, dac_planes, adc_planes, opt_stages = self._opt
        o = self.hw.optical
        dig_stages = self.log_n - opt_stages
        segs = []
        if dig_stages:
            segs.append(self._seg("ntt", k.amount * (N // 2) * dig_stages, k.words))
        blocks = k.amount * (N // o.block)
        dac = blocks * dac_planes * 2 * o.block
        adc = blocks * adc_planes * 2 * o.block
        t = (dac if dac >= adc else adc) / o.samples_per_s
        segs.append(Segment("optical", t, dac * o.pj_dac() * 1e-12 + adc * o.pj_adc() * 1e-12,
                            dac + adc, dac, adc))
        corr = k.amount * N * (3 if o.ideal else d + adc_planes + 3)   # digit split, recombine, chirps, mod q
        segs.append(self._seg("mac", corr, 3 * corr))
        return segs

    @property
    def static_w(self) -> float:
        return self.hw.static_w + (self.hw.optical.static_w if self.hw.optical else 0.0)


# ── reference configurations (illustrative) ────────────────────────────
ARK_CLASS = Accelerator()
SMALL_DIGITAL = Accelerator(name="Small digital accelerator (NTT-bound, illustrative)",
                            ntt_bfly_per_cycle=512, mac_lanes=2048, auto_words_per_cycle=1024,
                            sram_mib=512, hbm_gbps=2000.0)
# One fitted parameter (0.18 modular ops per cycle per unit, i.e. 0.61 G/s at 3.4 GHz) makes the
# simulated HMult match OpenFHE's measured 352.6 ms (N=2^16, 24 limbs, dnum=4, 8 threads on an
# i7-3770). DRAM plays the scratchpad (8 GiB at 20 GB/s). See examples/calibrate_openfhe.py.
CPU_LIKE = Accelerator(name="CPU-like (fitted to OpenFHE on an i7-3770, 8 threads)",
                       freq_ghz=3.4, ntt_bfly_per_cycle=0.18, mac_lanes=0.18, auto_words_per_cycle=0.72,
                       sram_mib=8192, sram_gbps=20.0, hbm_gbps=20.0, window=1,
                       tdp_w=77.0, static_w=20.0, pj_bfly=2000.0, pj_mac=1000.0,
                       pj_auto_word=200.0, pj_sram_byte=5.0, pj_hbm_byte=100.0,
                       enforce_tdp=False)   # timing reference only; its power numbers are not calibrated
HYBRID_OPTICAL = SMALL_DIGITAL.with_(name="Small digital + hybrid optical NTT (illustrative)",
                                     tdp_w=300.0, optical=OpticalEngine(samples_per_s=5e11))
IDEAL_OPTICAL = SMALL_DIGITAL.with_(name="Small digital + ideal optical NTT (hypothetical bound)",
                                    tdp_w=300.0, optical=OpticalEngine(block=4096, enob=8,
                                                                       samples_per_s=5e11, ideal=True))

ACCELERATORS = {"ark": ARK_CLASS, "small": SMALL_DIGITAL, "cpu": CPU_LIKE, "hybrid": HYBRID_OPTICAL,
                "ideal-optical": IDEAL_OPTICAL}
