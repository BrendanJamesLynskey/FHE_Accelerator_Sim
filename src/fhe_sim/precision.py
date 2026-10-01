"""A functional model of an exact modular NTT computed on an analogue complex-FFT engine.

This is the evidence behind the precision rule in ``hardware.OpticalEngine``:

1. A length-n modular DFT  y_k = sum_j x_j w^(jk) mod q  is rewritten with
   Bluestein's identity jk = (j^2 + k^2 - (k-j)^2) / 2 as a chirp multiply, a
   cyclic convolution of size 2n with the chirp psi^(-m^2), and another chirp
   multiply (psi^2 = w, so q must be 1 mod 2n).
2. Both convolution operands are split into b-bit digit planes. Each pair of
   planes is convolved by a complex FFT, the "analogue" engine; products of the
   same weight i + i' are summed before detection ("grouped").
3. The detector is a signed ADC with 2^ENOB codes over [-FS, FS], where
   FS = d * n * (2^b - 1)^2 is the largest value a grouped output can take. Its
   error is at most FS / 2^ENOB, so rounding recovers the exact integer whenever
   2^(ENOB - 1) > FS. Recombining the planes and reducing mod q is digital.

Pure Python, small n (the tests use n <= 64); this is a model to reason with,
not a fast implementation.
"""

from __future__ import annotations

import cmath


def find_psi(q: int, order: int) -> int:
    """A primitive ``order``-th root of unity mod prime q (order must divide q - 1)."""
    if (q - 1) % order:
        raise ValueError(f"{order} does not divide q - 1")
    factors, m, f = set(), order, 2
    while f * f <= m:
        while m % f == 0:
            factors.add(f)
            m //= f
        f += 1
    if m > 1:
        factors.add(m)
    for g in range(2, q):
        r = pow(g, (q - 1) // order, q)
        if all(pow(r, order // p, q) != 1 for p in factors):
            return r
    raise ValueError("no root found")


def ntt_reference(x: list[int], q: int, w: int) -> list[int]:
    n = len(x)
    return [sum(x[j] * pow(w, j * k, q) for j in range(n)) % q for k in range(n)]


def fft(a: list[complex], invert: bool = False) -> list[complex]:
    n = len(a)
    if n == 1:
        return list(a)
    even, odd = fft(a[0::2], invert), fft(a[1::2], invert)
    sign = 1 if invert else -1
    out = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(sign * 2j * cmath.pi * k / n) * odd[k]
        out[k], out[k + n // 2] = even[k] + t, even[k] - t
    return out


def digits(v: int, b: int, d: int) -> list[int]:
    return [(v >> (b * i)) & ((1 << b) - 1) for i in range(d)]


def optical_block_ntt(x: list[int], q: int, q_bits: int, b: int, enob: int) -> tuple[list[int], float]:
    """Exact-or-not modular NTT of length n via the analogue route above.

    Returns (result mod q, worst analogue error seen before rounding, in integer units).
    """
    n = len(x)
    psi = find_psi(q, 2 * n)
    M = 2 * n
    d = -(-q_bits // b)
    a = [x[j] * pow(psi, j * j, q) % q for j in range(n)] + [0] * n
    inv = pow(psi, -1, q)
    c = [0] * M
    for m in range(-(n - 1), n):
        c[m % M] = pow(inv, m * m, q)
    A = [fft([complex(v) for v in col]) for col in zip(*[digits(v, b, d) for v in a])]
    C = [fft([complex(v) for v in col]) for col in zip(*[digits(v, b, d) for v in c])]
    FS = d * n * ((1 << b) - 1) ** 2
    step = 2 * FS / 2 ** enob
    worst = 0.0
    conv = [0] * n
    for w in range(2 * d - 1):
        spec = [0j] * M
        for i in range(d):
            j = w - i
            if 0 <= j < d:
                spec = [s + A[i][t] * C[j][t] for t, s in enumerate(spec)]
        y = [v.real / M for v in fft(spec, invert=True)]
        for k in range(n):
            code = round((y[k] + FS) / step)                 # the ADC
            meas = code * step - FS
            exact = sum(digits(a[jj], b, d)[i] * digits(c[(k - jj) % M], b, d)[w - i]
                        for jj in range(n) for i in range(d) if 0 <= w - i < d)
            worst = max(worst, abs(meas - exact))
            conv[k] += round(meas) << (b * w)
    out = [pow(psi, k * k, q) * conv[k] % q for k in range(n)]
    return out, worst


def bits_needed(n: int, b: int, d: int) -> int:
    """Smallest ENOB with 2^(ENOB - 1) > d * n * (2^b - 1)^2."""
    fs = d * n * ((1 << b) - 1) ** 2
    e = 1
    while 2 ** (e - 1) <= fs:
        e += 1
    return e
