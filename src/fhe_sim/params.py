"""CKKS parameter sets and the sizes that follow from them.

Notation (as in the CKKS bootstrapping literature; Cryptography deck 08 writes
the ring degree as lower-case n):

* ``N``      ring degree; a polynomial in Z_q[X]/(X^N + 1) has N coefficients
* ``L``      top multiplicative level; a fresh ciphertext has ``L + 1`` RNS limbs
* ``level``  current level l; the ciphertext has ``l + 1`` limbs of one machine word each
* ``dnum``   number of key-switching digits (generalised "hybrid" key switching)
* ``alpha``  limbs per digit, ceil((L + 1) / dnum); also the number of special primes ``k``
* ``beta``   digits actually present at level l, ceil((l + 1) / alpha)

Everything is counted in 64-bit words (8 bytes), as in ARK, BTS and 100x.
Every formula here is a hand formula that ``tests/test_params.py`` re-derives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

WORD = 8          # bytes per stored coefficient (64-bit words)
MiB = 1 << 20


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CKKSParams:
    name: str
    log_n: int
    L: int                       # top level (L + 1 limbs)
    dnum: int
    q_bits: int = 50             # bits per RNS limb (used by the optical precision model)
    log_slots: int | None = None # default: fully packed, N/2 slots
    cts_levels: int = 3          # levels spent on CoeffToSlot (homomorphic DFT)
    stc_levels: int = 3          # levels spent on SlotToCoeff
    evalmod_degree: int = 59     # Chebyshev degree of the scaled-cosine approximation
    double_angle: int = 2        # cos(2x) = 2cos^2(x) - 1 iterations after the approximation

    # ── shape ──────────────────────────────────────────────────────────
    @property
    def N(self) -> int:
        return 1 << self.log_n

    @property
    def slots_log(self) -> int:
        return self.log_n - 1 if self.log_slots is None else self.log_slots

    @property
    def slots(self) -> int:
        return 1 << self.slots_log

    @property
    def full_slots(self) -> bool:
        return self.slots_log == self.log_n - 1

    @property
    def alpha(self) -> int:
        return cdiv(self.L + 1, self.dnum)

    @property
    def k(self) -> int:
        """Special primes P = p_0 ... p_{k-1}; k = alpha in hybrid key switching."""
        return self.alpha

    def limbs(self, level: int) -> int:
        return level + 1

    def beta(self, level: int) -> int:
        return cdiv(level + 1, self.alpha)

    def digit_sizes(self, level: int) -> list[int]:
        """Limbs in each key-switching digit at this level (the last may be short)."""
        l, a = level + 1, self.alpha
        return [min(a, l - i * a) for i in range(self.beta(level))]

    # ── sizes in bytes ─────────────────────────────────────────────────
    def ct_bytes(self, level: int) -> int:
        """A ciphertext: two polynomials of (level + 1) limbs."""
        return 2 * self.N * (level + 1) * WORD

    def pt_bytes(self, level: int) -> int:
        """A plaintext polynomial (for example one DFT diagonal) at this level."""
        return self.N * (level + 1) * WORD

    def evk_bytes(self, level: int | None = None) -> int:
        """One evaluation (key-switching) key.

        Stored: dnum digits x 2 polynomials x (L + 1 + k) limbs. An operation at a
        lower level needs only beta digits and (level + 1 + k) of the limbs, which is
        what an accelerator actually loads.
        """
        if level is None:
            return 2 * self.dnum * self.N * (self.L + 1 + self.k) * WORD
        return 2 * self.beta(level) * self.N * (level + 1 + self.k) * WORD

    def modup_bytes(self, level: int) -> int:
        """The ModUp-ed digits of one polynomial: beta x (level + 1 + k) limbs."""
        return self.beta(level) * self.N * (level + 1 + self.k) * WORD

    def ks_working_set(self) -> int:
        """Scratch space one top-level key switch needs (digits + two accumulators)."""
        lk = self.L + 1 + self.k
        return (self.beta(self.L) * lk + 2 * lk) * self.N * WORD

    def log_pq(self) -> int:
        """Approximate total modulus bits: a 60-bit base prime, L scaling primes, k 60-bit special primes."""
        return 60 + self.L * self.q_bits + 60 * self.k

    # ── EvalMod structure (baby-step giant-step Chebyshev evaluation) ──
    @property
    def evalmod_baby(self) -> int:
        """b: smallest power of two with b^2 >= degree + 1."""
        b = 1
        while b * b < self.evalmod_degree + 1:
            b *= 2
        return b

    @property
    def evalmod_giant(self) -> int:
        """g: number of degree-<b blocks."""
        return cdiv(self.evalmod_degree + 1, self.evalmod_baby)

    def with_(self, **kw) -> "CKKSParams":
        return replace(self, **kw)


def log2_int(x: int) -> int:
    """Exact log2 of a power of two."""
    if x <= 0 or x & (x - 1):
        raise ValueError(f"{x} is not a power of two")
    return x.bit_length() - 1


def ceil_log2(x: int) -> int:
    return (x - 1).bit_length() if x > 1 else 0


# Presets. ARK and Lattigo rows reproduce Table III of ARK (Kim et al., MICRO 2022,
# arXiv:2205.00922); "gpu100x" is the N = 2^16 row of Table 3 in "Over 100x faster
# bootstrapping" (Jung et al., TCHES 2021, IACR ePrint 2021/508); "openfhe-sparse" is
# the configuration measured with openfhe-python in calibration/.
PARAMS = {
    "ark": CKKSParams("ARK-like (N=2^16, L=23, dnum=4)", log_n=16, L=23, dnum=4),
    "lattigo": CKKSParams("Lattigo-like (N=2^16, L=24, dnum=5)", log_n=16, L=24, dnum=5),
    "gpu100x": CKKSParams("100x GPU (N=2^16, L=34, dnum=5)", log_n=16, L=34, dnum=5),
    "openfhe-sparse": CKKSParams("OpenFHE sparse (N=2^16, L=18, dnum=3, 8 slots)", log_n=16, L=18,
                                 dnum=3, q_bits=59, log_slots=3, cts_levels=1, stc_levels=1,
                                 evalmod_degree=119, double_angle=3),
    "small": CKKSParams("small test set (N=2^12, L=11, dnum=3)", log_n=12, L=11, dnum=3,
                        cts_levels=2, stc_levels=2, evalmod_degree=15, double_angle=1),
}
