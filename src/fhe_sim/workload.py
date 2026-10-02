"""Workloads: CKKS bootstrapping (and single HE ops) as a trace of primitive kernels.

A trace is a list of **HE operations** in program order. Each HE op names the
ciphertexts it reads and writes, the evaluation key and plaintexts it needs, and
the **primitive kernels** it runs:

==========  ===========================================  ===========================
kind        what                                         ``amount`` is
==========  ===========================================  ===========================
``ntt``     forward number-theoretic transform           limbs (each N points)
``intt``    inverse NTT                                  limbs
``bconv``   RNS basis conversion (ModUp / ModDown)       modular multiply-adds
``mac``     element-wise modular multiply / add          modular multiply-adds
``auto``    automorphism (slot rotation permutation)     words moved
==========  ===========================================  ===========================

The trace is the contract between a *scheme model* (this file, or an FHE
compiler such as HEIR, or an instrumented library such as OpenFHE) and the
*hardware model* (``hardware.py``) driven by the event engine (``sim.py``).
``dump_trace`` / ``load_trace`` read and write it as JSON so traces produced
elsewhere can be replayed.

The bootstrapping structure follows the standard CKKS pipeline (Cheon et al.
2018; Han & Ki 2020; Bossuat et al. 2021) as used by ARK, BTS and 100x:

    ModRaise -> CoeffToSlot (BSGS homomorphic DFT) -> EvalMod (Chebyshev BSGS +
    double angle) -> SlotToCoeff (BSGS homomorphic DFT)

Operation counts are those of *this* scheme model, which is a faithful but
simplified rendering; real libraries differ in the details (see the README).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .params import CKKSParams, cdiv, ceil_log2

STAGES = ["modraise", "cts", "evalmod", "stc", "app", "app_post"]   # "app": application ops from a compiler front end


@dataclass
class Kernel:
    kind: str           # ntt | intt | bconv | mac | auto
    amount: int         # limbs (ntt/intt), multiply-adds (bconv/mac) or words (auto)
    words: int          # on-chip words read + written (SRAM bandwidth)


@dataclass
class HEOp:
    id: int
    op: str             # modraise | hmult | hrot | modup | pmac | add | cmult | rescale
    stage: str
    level: int          # level the op runs at
    inputs: list[str]
    output: str
    kernels: list[Kernel]
    key: tuple[str, int] | None = None              # (key id, bytes to load)
    pts: list[tuple[str, int]] = field(default_factory=list)   # (plaintext id, bytes)
    boot: int = 0


@dataclass
class BootOptions:
    """Algorithmic choices; each is an acceleration technique from the literature."""

    n_boot: int = 1                 # independent bootstraps, issued back to back
    hoisting: bool = True           # baby-step rotations share one ModUp (Halevi & Shoup)
    min_ks: bool = False            # one key per BSGS loop, rotations applied iteratively (ARK "Min-KS")
    seeded_keys: bool = False       # evk "a" halves regenerated on chip from a PRNG seed
    otf_plaintexts: bool = False    # DFT diagonals generated on chip instead of loaded (ARK "OF-Limb" spirit)
    lazy_moddown: bool = False      # OpenFHE's BSGS (found by replaying its trace): rotations stay in the
                                    # Q*P basis, one ModDown per DFT level, and many more (now NTT-free)
                                    # baby steps than giant steps


@dataclass
class Trace:
    params: CKKSParams
    ops: list[HEOp]
    sizes: dict[str, int]           # ciphertext / digit object -> bytes
    external: list[str]             # objects that start in HBM (bootstrap inputs)
    options: BootOptions | None = None
    levels: dict[str, int] = field(default_factory=dict)   # object -> level


# ───────────────────────────────────────────────────────── kernel recipes ──
def k_ntt(limbs: int, N: int, inverse: bool = False) -> Kernel:
    return Kernel("intt" if inverse else "ntt", limbs, 2 * limbs * N)


def k_mac(ops: int) -> Kernel:
    return Kernel("mac", ops, 3 * ops)


def ks_modup(p: CKKSParams, level: int) -> list[Kernel]:
    """INTT the input, convert each digit to the other limbs (incl. special primes), NTT them."""
    N, l, k = p.N, level + 1, p.k
    ds = p.digit_sizes(level)
    new_limbs = sum(l + k - a for a in ds)
    bconv = sum(N * a * (l + k - a) for a in ds) + N * l
    return [k_ntt(l, N, inverse=True), Kernel("bconv", bconv, N * (l + new_limbs)),
            k_ntt(new_limbs, N)]


def ks_tail(p: CKKSParams, level: int) -> list[Kernel]:
    """Inner product with the evk, then ModDown both output polynomials back to Q."""
    N, l, k, b = p.N, level + 1, p.k, p.beta(level)
    inner = 2 * b * (l + k) * N
    moddown = 2 * (N * k * l + N * k + N * l)
    return [Kernel("mac", inner, 3 * b * (l + k) * N + 2 * (l + k) * N),
            k_ntt(2 * k, N, inverse=True), Kernel("bconv", moddown, 4 * N * (k + l)),
            k_ntt(2 * l, N)]


def rescale_kernels(p: CKKSParams, level: int) -> list[Kernel]:
    """Divide by the last prime: INTT that limb, NTT it into every other limb, subtract and scale."""
    N, l = p.N, level + 1
    return [k_ntt(2, N, inverse=True), k_ntt(2 * (l - 1), N), k_mac(2 * (l - 1) * N)]


# ─────────────────────────────────────────────────────────────── builder ──
class Builder:
    def __init__(self, p: CKKSParams, opts: BootOptions):
        self.p, self.o = p, opts
        self.ops: list[HEOp] = []
        self.sizes: dict[str, int] = {}
        self.levels: dict[str, int] = {}
        self.external: list[str] = []
        self.stage = "modraise"
        self.boot = 0
        self._n = 0

    def _obj(self, level: int, nbytes: int) -> str:
        name = f"b{self.boot}.c{self._n}"
        self._n += 1
        self.sizes[name], self.levels[name] = nbytes, level
        return name

    def external_ct(self, level: int) -> str:
        name = self._obj(level, self.p.ct_bytes(level))
        self.external.append(name)
        return name

    def key(self, kid: str, level: int) -> tuple[str, int]:
        b = self.p.evk_bytes(level)
        return (kid, b // 2 if self.o.seeded_keys else b)

    def seed_kernels(self, level: int) -> list[Kernel]:
        """Regenerating the uniform half of a seeded key costs PRNG work on the MAC lanes."""
        if not self.o.seeded_keys:
            return []
        p = self.p
        return [k_mac(p.beta(level) * (level + 1 + p.k) * p.N)]

    def emit(self, op: str, level: int, inputs: list[str], out_level: int, kernels: list[Kernel],
             key=None, pts=(), out_bytes: int | None = None) -> str:
        out = self._obj(out_level, self.p.ct_bytes(out_level) if out_bytes is None else out_bytes)
        self.ops.append(HEOp(len(self.ops), op, self.stage, level, list(inputs), out, kernels,
                             key, list(pts), self.boot))
        return out

    # ── HE operations ─────────────────────────────────────────────────
    def lvl(self, *xs: str) -> int:
        return min(self.levels[x] for x in xs)

    def hmult(self, a: str, b: str, extra: list[str] = (), post_adds: int = 1) -> str:
        """Tensor product, relinearise (key switch with the relin key), add, rescale."""
        p, lev = self.p, self.lvl(a, b)
        N, l = p.N, lev + 1
        ks = ([k_mac(4 * l * N)] + ks_modup(p, lev) + self.seed_kernels(lev) + ks_tail(p, lev)
              + [k_mac(2 * l * N * post_adds)] + rescale_kernels(p, lev))
        return self.emit("hmult", lev, [a, b, *extra], lev - 1, ks, key=self.key("relin", lev))

    def hrot(self, x: str, kid: str) -> str:
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        ks = ([Kernel("auto", 2 * l * N, 4 * l * N)] + ks_modup(p, lev) + self.seed_kernels(lev)
              + ks_tail(p, lev))
        return self.emit("hrot", lev, [x], lev, ks, key=self.key(kid, lev))

    def modup(self, x: str) -> str:
        """Hoisting: decompose and ModUp once, share across several rotations of x."""
        p, lev = self.p, self.lvl(x)
        return self.emit("modup", lev, [x], lev, ks_modup(p, lev), out_bytes=p.modup_bytes(lev))

    def hrot_hoisted(self, x: str, digits: str, kid: str) -> str:
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        w = (l + p.beta(lev) * (l + p.k)) * N           # permute c0 and the ModUp-ed digits
        ks = [Kernel("auto", w, 2 * w)] + self.seed_kernels(lev) + ks_tail(p, lev)
        return self.emit("hrot", lev, [x, digits], lev, ks, key=self.key(kid, lev))

    def extend(self, x: str) -> str:
        """Lift a ciphertext into the Q*P basis (multiply by P mod each q): no transforms."""
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        return self.emit("extend", lev, [x], lev, [k_mac(2 * l * N)],
                         out_bytes=2 * (l + p.k) * N * 8)

    def hrot_hoisted_ext(self, x: str, digits: str, kid: str) -> str:
        """A hoisted rotation left in the Q*P basis: automorphism and inner product, no ModDown."""
        p, lev = self.p, self.lvl(x)
        N, l, k, b = p.N, lev + 1, p.k, p.beta(lev)
        w = (l + b * (l + k)) * N
        ks = [Kernel("auto", w, 2 * w)] + self.seed_kernels(lev) + ks_tail(p, lev)[:1]
        return self.emit("hrot", lev, [x, digits], lev, ks, key=self.key(kid, lev),
                         out_bytes=2 * (l + k) * N * 8)

    def pmac_ext(self, xs: list[str], pt_ids: list[str], moddown: bool = True) -> str:
        """BSGS inner sum on Q*P-basis inputs (plaintexts stored in Q*P too), optionally ModDown."""
        p, lev = self.p, self.lvl(*xs)
        N, l, m = p.N, lev + 1, len(xs)
        lk = l + p.k
        kern = [k_mac(2 * lk * N * m + 2 * lk * N * (m - 1))]
        if self.o.otf_plaintexts:
            kern = [k_ntt(lk * m, N), k_mac(lk * N * m)] + kern
            pts = [(pid, N * 8) for pid in pt_ids]
        else:
            pts = [(pid, lk * N * 8) for pid in pt_ids]
        down = ks_tail(p, lev)[1:] if moddown else []
        return self.emit("pmac", lev, xs, lev, kern + down, pts=pts,
                         out_bytes=None if moddown else 2 * lk * N * 8)

    def hrot_ext(self, x: str, kid: str) -> str:
        """A rotation whose result stays in the Q*P basis (ModUp, automorphism, inner product)."""
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        ks = ([Kernel("auto", 2 * l * N, 4 * l * N)] + ks_modup(p, lev) + self.seed_kernels(lev)
              + ks_tail(p, lev)[:1])
        return self.emit("hrot", lev, [x], lev, ks, key=self.key(kid, lev),
                         out_bytes=2 * (l + p.k) * N * 8)

    def add_ext(self, xs: list[str]) -> str:
        """Sum Q*P-basis ciphertexts, ModDown once, rescale."""
        p, lev = self.p, self.lvl(*xs)
        N, lk = p.N, lev + 1 + p.k
        kern = [k_mac(2 * lk * N * max(1, len(xs) - 1))] + ks_tail(p, lev)[1:] + rescale_kernels(p, lev)
        return self.emit("add", lev, xs, lev - 1, kern)

    def pmac(self, xs: list[str], pt_ids: list[str]) -> str:
        """sum_i pt_i * x_i: plaintext multiplies and accumulation (BSGS inner sum)."""
        p, lev = self.p, self.lvl(*xs)
        N, l, m = p.N, lev + 1, len(xs)
        kern = [k_mac(2 * l * N * m + 2 * l * N * (m - 1))]
        if self.o.otf_plaintexts:     # generate each diagonal: NTT it from a compact form
            kern = [k_ntt(l * m, N), k_mac(l * N * m)] + kern
            pts = [(pid, N * 8) for pid in pt_ids]       # load only a one-limb seed
        else:
            pts = [(pid, p.pt_bytes(lev)) for pid in pt_ids]
        return self.emit("pmac", lev, xs, lev, kern, pts=pts)

    def add(self, xs: list[str], rescale: bool = False) -> str:
        p, lev = self.p, self.lvl(*xs)
        N, l = p.N, lev + 1
        kern = [k_mac(2 * l * N * max(1, len(xs) - 1))]
        if rescale:
            kern += rescale_kernels(p, lev)
        return self.emit("add", lev, xs, lev - 1 if rescale else lev, kern)

    def cmult(self, xs: list[str]) -> str:
        """Scalar (constant) multiplies of each input, summed, then rescaled."""
        p, lev = self.p, self.lvl(*xs)
        N, l = p.N, lev + 1
        kern = [k_mac(2 * l * N * len(xs) + 2 * l * N)] + rescale_kernels(p, lev)
        return self.emit("cmult", lev, xs, lev - 1, kern)

    # ── bootstrapping stages ──────────────────────────────────────────
    def modraise(self, x: str) -> str:
        p = self.p
        kern = [k_ntt(2, p.N, inverse=True), k_ntt(2 * (p.L + 1), p.N)]
        y = self.emit("modraise", 0, [x], p.L, kern)
        for i in range(p.log_n - 1 - p.slots_log):            # sparse packing: SubSum
            y = self.add([y, self.hrot(y, f"sub{i}")])
        return y

    def dft(self, x: str, prefix: str, n_levels: int) -> str:
        p, o = self.p, self.o
        for j, k in enumerate(dft_split(p.slots_log, n_levels)):
            d = min((1 << (k + 1)) - 1, p.slots)              # non-zero diagonals of a radix-2^k stage
            lazy = o.lazy_moddown and o.hoisting and not o.min_ks
            if lazy and d == p.slots:                         # one dense level (OpenFHE's linear transform)
                n1 = 1 << (int(math.isqrt(p.slots - 1) + 1).bit_length() - 1 + 1) if p.slots > 1 else 1
                n1 = min(n1, d)
            elif lazy:                                        # OpenFHE GetCollapsedFFTParams
                n1 = min(1 << (k // 2 + 1 + (1 if d > 7 else 0)), d)
            else:
                n1 = min(1 << cdiv(k + 1, 2), d)              # baby steps (balanced)
            n2 = cdiv(d, n1)                                  # giant steps
            tag = f"{prefix}{j}"
            babies = [x]
            if n1 > 1:
                if o.min_ks:
                    for _ in range(1, n1):
                        babies.append(self.hrot(babies[-1], f"{tag}.b"))
                elif o.hoisting and o.lazy_moddown:
                    dig = self.modup(x)
                    babies = [self.extend(x)] + [self.hrot_hoisted_ext(x, dig, f"{tag}.b{i}")
                                                 for i in range(1, n1)]
                elif o.hoisting:
                    dig = self.modup(x)
                    babies += [self.hrot_hoisted(x, dig, f"{tag}.b{i}") for i in range(1, n1)]
                else:
                    babies += [self.hrot(x, f"{tag}.b{i}") for i in range(1, n1)]
            inners = []
            for g in range(n2):
                m = min(n1, d - g * n1)
                ids = [f"{tag}.d{g}.{i}" for i in range(m)]
                if lazy and n1 > 1:
                    inners.append(self.pmac_ext(babies[:m], ids, moddown=(g > 0)))
                else:
                    inners.append(self.pmac(babies[:m], ids))
            if o.min_ks:                                      # Horner over the giant steps
                acc = inners[-1]
                for g in range(n2 - 2, -1, -1):
                    acc = self.add([self.hrot(acc, f"{tag}.g"), inners[g]], rescale=(g == 0))
                x = acc if n2 > 1 else self.add([acc], rescale=True)
            elif lazy and n1 > 1:
                parts = [inners[0]] + [self.hrot_ext(inners[g], f"{tag}.g{g}") for g in range(1, n2)]
                x = self.add_ext(parts)
            else:
                parts = [inners[0]] + [self.hrot(inners[g], f"{tag}.g{g}") for g in range(1, n2)]
                x = self.add(parts, rescale=True)
        return x

    def evalmod(self, x: str) -> str:
        p = self.p
        b, g = p.evalmod_baby, p.evalmod_giant
        T = {1: self.cmult([x])}                              # scale into the approximation interval
        for i in range(2, b + 1):                             # Chebyshev T_i = 2 T_a T_b - T_{a-b}
            T[i] = self.hmult(T[(i + 1) // 2], T[i // 2])
        m = ceil_log2(g)
        G = [T[b]]
        for _ in range(1, m):
            G.append(self.hmult(G[-1], G[-1]))
        nodes = [self.cmult([T[i] for i in range(1, b)]) for _ in range(g)]
        t = 0
        while len(nodes) > 1:                                 # combine blocks: lo + hi * T_{b 2^t}
            nxt = []
            for s in range(0, len(nodes) - 1, 2):
                nxt.append(self.hmult(nodes[s + 1], G[t], extra=[nodes[s]]))
            if len(nodes) % 2:
                nxt.append(nodes[-1])
            nodes, t = nxt, t + 1
        y = nodes[0]
        for _ in range(p.double_angle):                       # cos(2x) = 2 cos^2(x) - 1
            y = self.hmult(y, y)
        return y

    def bootstrap(self, x: str) -> str:
        p = self.p
        self.stage = "modraise"
        x = self.modraise(x)
        self.stage = "cts"
        x = self.dft(x, "cts", p.cts_levels)
        c = self.hrot(x, "conj")                              # split real and imaginary parts
        if p.full_slots:
            parts = [self.add([x, c]), self.add([x, c])]
        else:
            parts = [self.add([x, c])]
        self.stage = "evalmod"
        parts = [self.evalmod(y) for y in parts]
        x = self.add(parts) if len(parts) > 1 else parts[0]
        self.stage = "stc"
        x = self.dft(x, "stc", p.stc_levels)
        if self.levels[x] < 0:
            raise ValueError(f"{p.name}: bootstrapping needs more than L = {p.L} levels")
        return x


def dft_split(log_slots: int, n_levels: int) -> list[int]:
    """Radix of each homomorphic-DFT level: log_slots FFT stages merged into n_levels."""
    base, rem = divmod(log_slots, n_levels)
    return [base + 1 if j < rem else base for j in range(n_levels)]


def bootstrap_trace(p: CKKSParams, opts: BootOptions | None = None) -> Trace:
    opts = opts or BootOptions()
    b = Builder(p, opts)
    for i in range(opts.n_boot):
        b.boot = i
        b.bootstrap(b.external_ct(0))
    return Trace(p, b.ops, b.sizes, b.external, opts, b.levels)


def he_op_trace(p: CKKSParams, op: str, level: int | None = None, n: int = 1) -> Trace:
    """n independent HMults or HRots at one level: the micro-benchmarks of deck 01."""
    b = Builder(p, BootOptions())
    lev = p.L if level is None else level
    b.stage = op
    for i in range(n):
        b.boot = i
        x, y = b.external_ct(lev), b.external_ct(lev)
        b.hmult(x, y) if op == "hmult" else b.hrot(x, "rot1")
    return Trace(p, b.ops, b.sizes, b.external, None, b.levels)


def output_level(trace: Trace) -> int:
    """Level of the last object produced (the bootstrapped ciphertext)."""
    return trace.levels[trace.ops[-1].output]


# ─────────────────────────────────────────────────────────── JSON replay ──
def dump_trace(trace: Trace, path: str | Path) -> None:
    doc = {"format": "fhe-sim-trace/1", "params": asdict(trace.params),
           "external": trace.external, "sizes": trace.sizes, "levels": trace.levels,
           "ops": [{"id": o.id, "op": o.op, "stage": o.stage, "level": o.level, "boot": o.boot,
                    "inputs": o.inputs, "output": o.output,
                    "key": list(o.key) if o.key else None, "pts": [list(t) for t in o.pts],
                    "kernels": [[k.kind, k.amount, k.words] for k in o.kernels]} for o in trace.ops]}
    Path(path).write_text(json.dumps(doc))


def load_trace(path: str | Path) -> Trace:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != "fhe-sim-trace/1":
        raise ValueError("not an fhe-sim trace")
    p = CKKSParams(**doc["params"])
    ops = [HEOp(o["id"], o["op"], o["stage"], o["level"], o["inputs"], o["output"],
                [Kernel(*k) for k in o["kernels"]], tuple(o["key"]) if o["key"] else None,
                [tuple(t) for t in o["pts"]], o["boot"]) for o in doc["ops"]]
    return Trace(p, ops, doc["sizes"], doc["external"], None, doc.get("levels", {}))


def summarise_trace(trace: Trace) -> dict:
    """Operation and byte counts per stage (no timing): the scheme-level workload."""
    out: dict = {}
    for o in trace.ops:
        s = out.setdefault(o.stage, {"hmult": 0, "hrot": 0, "pmult": 0, "ops": 0, "ntt_limbs": 0,
                                     "intt_limbs": 0, "bconv": 0, "mac": 0, "auto_words": 0,
                                     "keys": set(), "key_bytes": 0, "pt_bytes": 0})
        s["ops"] += 1
        if o.op in ("hmult", "hrot"):
            s[o.op] += 1
        if o.op == "pmac":
            s["pmult"] += len(o.pts)
        for k in o.kernels:
            if k.kind == "ntt":
                s["ntt_limbs"] += k.amount
            elif k.kind == "intt":
                s["intt_limbs"] += k.amount
            elif k.kind == "auto":
                s["auto_words"] += k.amount
            else:
                s[k.kind] += k.amount
        if o.key:
            s["keys"].add(o.key[0])
            s["key_bytes"] += o.key[1]
        s["pt_bytes"] += sum(b for _, b in o.pts)
    for s in out.values():
        s["distinct_keys"] = len(s.pop("keys"))
    return out
