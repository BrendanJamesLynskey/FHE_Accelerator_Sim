"""A HEIR front end: CKKS-dialect IR from the HEIR compiler -> an fhe-sim trace.

HEIR (https://heir.dev; Ali et al., arXiv:2508.11095) compiles ordinary tensor
programs to FHE. After ``--mlir-to-ckks`` / ``--torch-linalg-to-ckks`` (with
``unroll-fhe-kernel-loops=true``) the server function is straight-line code in
HEIR's ``ckks`` dialect: every value's type carries its level, and HEIR has
already chosen the packing, the rotations, the level management and the
bootstrap placement. This module reads that IR (textual MLIR) and emits one HE
operation per ``ckks`` op, expanded into kernels with the same recipes as the
scheme model (``workload.Builder``), with dependencies taken from SSA def-use.

What HEIR decides                         What the simulator adds
---------------------------------------   ------------------------------------------
parameters (N, the Q and P primes)        kernel recipes per op (NTT, BConv, MAC, ...)
packing, rotation amounts, levels         key and plaintext identities and sizes
where to rescale, relinearize, bootstrap  the CKKS bootstrap itself (Builder.bootstrap)

Supported ops: ckks.add, sub, add_plain, sub_plain, mul, mul_plain, relinearize,
rescale, rotate, negate, level_reduce, bootstrap; tensor.extract/insert/empty/
from_elements with constant indices; arith.constant; preprocessing.load.
Anything else in the server function is an error, so silent omissions cannot
happen. Validated against HEIR v2026.10.01 (the release binaries).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

from .params import CKKSParams, cdiv
from .workload import BootOptions, Builder, Kernel, Trace, k_mac, ks_modup, ks_tail, rescale_kernels

RE_PARAM = re.compile(r"ckks\.schemeParam = #ckks\.scheme_param<logN = (\d+), Q = \[([^\]]*)\], "
                      r"P = \[([^\]]*)\], logDefaultScale = (\d+)>")
RE_ALIAS = re.compile(r"^(![\w.]+) = !lwe\.lwe_ciphertext<(.*)>\s*$")
RE_LEVEL = re.compile(r"modulus_chain_L\d+_C(\d+)")
RE_OP = re.compile(r"^\s*(%[\w.#]+) = ([\w.]+)(.*)$")
RE_VAL = re.compile(r"%[\w.#]+")


@dataclass
class HeirProgram:
    params: CKKSParams
    trace: Trace
    counts: dict = field(default_factory=dict)       # ckks op -> dynamic count
    notes: list = field(default_factory=list)


def parse_scheme(text: str, slots: int | None) -> CKKSParams:
    m = RE_PARAM.search(text)
    if not m:
        raise ValueError("no ckks.schemeParam in the module (was it lowered to the ckks dialect?)")
    log_n = int(m.group(1))
    q = [x for x in m.group(2).split(",") if x.strip()]
    p = [x for x in m.group(3).split(",") if x.strip()]
    L = len(q) - 1
    dnum = cdiv(len(q), len(p))            # hybrid key switching: alpha = len(P) limbs per digit
    log_slots = (slots.bit_length() - 1) if slots else None
    if log_slots is not None and log_slots >= log_n:
        log_slots = log_n - 1
    return CKKSParams("HEIR program", log_n=log_n, L=L, dnum=dnum, q_bits=int(m.group(4)),
                      log_slots=log_slots)


def parse_types(text: str) -> dict[str, tuple[int, int]]:
    """Ciphertext type alias -> (level, polynomials)."""
    out = {}
    for line in text.splitlines():
        m = RE_ALIAS.match(line)
        if not m:
            continue
        lev = RE_LEVEL.search(m.group(2))
        if lev:
            degree = re.search(r"_D(\d+)$", m.group(1))
            out[m.group(1)] = (int(lev.group(1)), int(degree.group(1)) if degree else 2)
    return out


def server_function(text: str) -> list[str]:
    """Lines of the function whose HEIR interface role is server.evaluate."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("func.func") and "server.evaluate" in line:
            body, depth = [], 0
            for l2 in lines[i:]:
                depth += l2.count("{") - l2.count("}")
                body.append(l2)
                if depth <= 0 and len(body) > 1:
                    break
            return body
    raise ValueError("no function with role server.evaluate")


class HeirBuilder(Builder):
    """Builder with the single-step ops HEIR emits (it separates mul / relinearize / rescale)."""

    def obj_at(self, level: int, polys: int) -> str:
        return self._obj(level, polys * self.p.N * (level + 1) * 8)

    def rescale(self, x: str) -> str:
        p, lev = self.p, self.lvl(x)
        return self.emit("rescale", lev, [x], lev - 1, rescale_kernels(p, lev))

    def level_reduce(self, x: str, to: int) -> str:
        """Drop limbs: free in data, a copy at most."""
        lev = self.lvl(x)
        if to > lev:
            raise ValueError(f"level_reduce from {lev} up to {to}")
        return self.emit("level_reduce", lev, [x], to, [])

    def add_plain(self, x: str, pid: str) -> str:
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        return self.emit("add_plain", lev, [x], lev, [k_mac(l * N)], pts=[(pid, p.pt_bytes(lev))])

    def mul_plain(self, x: str, pid: str) -> str:
        return self.pmac([x], [pid])

    def negate(self, x: str) -> str:
        p, lev = self.p, self.lvl(x)
        return self.emit("negate", lev, [x], lev, [k_mac(2 * (lev + 1) * p.N)])

    def tensor(self, a: str, b: str) -> str:
        """Ciphertext x ciphertext without relinearisation: a degree-2 (three-polynomial) result."""
        p, lev = self.p, self.lvl(a, b)
        N, l = p.N, lev + 1
        return self.emit("tensor", lev, [a, b], lev, [k_mac(4 * l * N)], out_bytes=3 * l * N * 8)

    def chebyshev(self, x: str, degree: int) -> str:
        """sum c_i T_i(x) by baby-step giant-step (as in Builder.evalmod, without the scaling
        and double-angle steps): b baby powers, giant powers T_b 2^t, g blocks combined in a tree."""
        b = 1
        while b * b < degree + 1:
            b *= 2
        g = cdiv(degree + 1, b)
        T = {1: x}
        for i in range(2, b + 1):
            T[i] = self.hmult(T[(i + 1) // 2], T[i // 2])
        m = (g - 1).bit_length()
        G = [T[b]]
        for _ in range(1, m):
            G.append(self.hmult(G[-1], G[-1]))
        nodes = [self.cmult([T[i] for i in range(1, b)]) for _ in range(g)]
        t = 0
        while len(nodes) > 1:
            nxt = []
            for s in range(0, len(nodes) - 1, 2):
                nxt.append(self.hmult(nodes[s + 1], G[t], extra=[nodes[s]]))
            if len(nodes) % 2:
                nxt.append(nodes[-1])
            nodes, t = nxt, t + 1
        return nodes[0]

    def relevel(self, x: str, to: int) -> str:
        """Adopt the compiler's level for a value whose schedule the model costs differently."""
        return self.emit("relevel", self.lvl(x), [x], to, [])

    def relinearize(self, x: str) -> str:
        p, lev = self.p, self.lvl(x)
        N, l = p.N, lev + 1
        ks = ks_modup(p, lev) + self.seed_kernels(lev) + ks_tail(p, lev) + [k_mac(2 * l * N)]
        return self.emit("relinearize", lev, [x], lev, ks, key=self.key("relin", lev))


def slim_ir(text: str) -> str:
    """Keep only what the front end reads: the module line with the scheme parameters, the
    ciphertext type aliases and the server function. HEIR's full output also carries the client
    and preprocessing functions with every weight inline (19 MB for LoLa)."""
    keep = [line for line in text.splitlines()
            if RE_ALIAS.match(line) or ("ckks.schemeParam" in line and line.lstrip().startswith("module"))]
    if not keep or "ckks.schemeParam" not in keep[-1]:
        raise ValueError("no module line with ckks.schemeParam")
    return "\n".join(keep + server_function(text) + ["}"]) + "\n"


def compile_ir(path: str | Path, opts: BootOptions | None = None, slots: int | None = None,
               bootstrap_params: dict | None = None) -> HeirProgram:
    """Read HEIR ckks-dialect MLIR and build a trace of its server function.

    ``bootstrap_params`` overrides the bootstrap structure used to expand
    ``ckks.bootstrap`` (cts_levels, stc_levels, evalmod_degree, double_angle).
    """
    p_ = Path(path)
    text = gzip.open(p_, "rt").read() if p_.suffix == ".gz" else p_.read_text()
    m = re.search(r"scheme\.actual_slot_count = (\d+)", text)
    p = parse_scheme(text, slots or (int(m.group(1)) if m else None))
    if bootstrap_params:
        p = p.with_(**bootstrap_params)
    types = parse_types(text)
    b = HeirBuilder(p, opts or BootOptions())
    b.stage = "app"
    const: dict[str, int] = {}
    vals: dict[str, object] = {}          # SSA name -> builder object name, or a list (tensor)
    counts: dict[str, int] = {}
    notes: list[str] = []

    def ty_level(t: str) -> tuple[int, int]:
        t = t.strip()
        if t not in types:
            raise ValueError(f"unknown ciphertext type {t}")
        return types[t]

    for line in server_function(text)[1:]:
        s = line.strip()
        if s.startswith(("return", "call ", "func.call ")) or s == "}" or not s:
            continue
        mo = RE_OP.match(line)
        if not mo:
            raise ValueError(f"unsupported line in server function: {s[:120]}")
        res, op, rest = mo.groups()
        rest = rest.strip()
        ops = RE_VAL.findall(rest.split(":")[0])
        if op == "arith.constant":
            mc = re.match(r"(-?\d+) : index", rest)
            if mc:
                const[res] = int(mc.group(1))
            continue
        if op == "preprocessing.load":
            vals[res] = "pt.site" + re.search(r"site (\d+)", rest).group(1)
            continue
        if op in ("func.call", "call"):
            continue                                     # client-side layout helpers
        if op == "tensor.empty":
            n = int(re.search(r"tensor<(\d+)x", rest).group(1))
            vals[res] = [None] * n
            continue
        if op == "tensor.from_elements":
            vals[res] = [vals[x] for x in ops]
            continue
        if op == "tensor.extract":
            t, idx = ops[0], const[ops[1]] if len(ops) > 1 else 0
            if t not in vals:                            # a function argument: an external input
                ty = re.search(r"tensor<\d+x(![\w.]+)>", rest).group(1)
                lev, polys = ty_level(ty)
                n = int(re.search(r"tensor<(\d+)x", rest).group(1))
                vals[t] = [None] * n
                for i in range(n):
                    name = b.obj_at(lev, polys)
                    b.external.append(name)
                    vals[t][i] = name
            vals[res] = vals[t][idx]
            continue
        if op == "tensor.insert":
            v, t = ops[0], ops[1]
            idx = const[ops[2]] if len(ops) > 2 else 0
            new = list(vals[t])
            new[idx] = vals[v]
            vals[res] = new
            continue
        if not (op.startswith("ckks.") or op == "kernel.eval_chebyshev"):
            raise ValueError(f"unsupported op {op} in server function")
        kind = op.split(".", 1)[1]
        counts[kind] = counts.get(kind, 0) + 1
        out_ty = rest.rsplit("->", 1)[1] if "->" in rest else rest.rsplit(":", 1)[1]
        out_level, _ = ty_level(out_ty)
        x = vals[ops[0]]
        if kind == "rotate":
            amount = const.get(ops[1]) if len(ops) > 1 else int(re.search(r",\s*(-?\d+)", rest).group(1))
            y = b.hrot(x, f"rot{amount}")
        elif kind == "mul_plain":
            y = b.mul_plain(x, vals[ops[1]])
        elif kind in ("add_plain", "sub_plain"):
            y = b.add_plain(x, vals[ops[1]])
        elif kind in ("add", "sub"):
            y = b.add([x, vals[ops[1]]])
        elif kind == "mul":
            y = b.tensor(x, vals[ops[1]])
        elif kind == "relinearize":
            y = b.relinearize(x)
        elif kind == "rescale":
            y = b.rescale(x)
        elif kind == "negate":
            y = b.negate(x)
        elif kind == "level_reduce":
            y = b.level_reduce(x, out_level)
        elif kind == "eval_chebyshev":
            degree = len(re.search(r"coefficients = \[([^\]]*)\]", rest).group(1).split(",")) - 1
            y = b.chebyshev(x, degree)
            got = b.levels[y]
            if got != out_level:
                notes.append(f"eval_chebyshev degree {degree}: model schedule ends at level {got}, "
                             f"HEIR's at {out_level}; HEIR's level kept")
                y = b.level_reduce(y, out_level) if got > out_level else b.relevel(y, out_level)
        elif kind == "bootstrap":
            y = b.bootstrap(x)
            b.stage = "app_post"                         # application ops after a bootstrap
            got = b.levels[y]
            if got > out_level:
                notes.append(f"bootstrap: model output level {got}, HEIR's type says {out_level}; limbs dropped")
                y = b.level_reduce(y, out_level)
            elif got < out_level:
                notes.append(f"bootstrap: model output level {got} is below HEIR's {out_level}; "
                             f"continuing at the model's level")
        else:
            raise ValueError(f"unsupported ckks op {op}")
        if kind not in ("bootstrap", "eval_chebyshev") and b.levels[y] != out_level:
            raise ValueError(f"{op}: model level {b.levels[y]} != HEIR level {out_level} ({s[:100]})")
        vals[res] = y
    return HeirProgram(p, Trace(p, b.ops, b.sizes, b.external, b.o, b.levels), counts, notes)
