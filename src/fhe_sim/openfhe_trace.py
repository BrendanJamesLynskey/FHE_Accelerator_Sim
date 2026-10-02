"""Read a kernel-stream log recorded from an instrumented OpenFHE bootstrap.

The log comes from ``calibration/openfhe_trace`` (OpenFHE v1.5.1 with the
``fhetrace`` patch, run single-threaded). One event per line:

=================  =====================================================
``S <stage>``      bootstrap stage marker (modraise, cts, evalmod, stc, post)
``NTT n``          one forward NTT of one limb of n points
``INTT n``         one inverse NTT of one limb
``BCONV q p``      a base conversion from q limbs to p limbs
``MODUP l k d``    a key-switch ModUp of l limbs into d digits (k special limbs)
``KS key d l' l``  a key-switch inner product with evaluation key ``key`` (d digits, l' = l + k limbs)
``MODDOWN q p``    a ModDown of a (q + p)-limb product back to q limbs
``AUTO t``         an automorphism of one t-limb polynomial
``RESCALE t``      a rescale of one t-limb polynomial
``EW t``           an element-wise add/subtract/multiply of one t-limb polynomial
``PTMUL id t``     a multiply by precomputed plaintext ``id`` (t limbs)
=================  =====================================================

Two products:

* ``summarise_log`` gives per-stage counts directly comparable with
  ``workload.summarise_trace`` (rotations, multiplications, distinct keys, NTT limbs,
  base-conversion work, requested key and plaintext bytes).
* ``log_to_trace`` turns the stream into an ``fhe-sim`` ``Trace`` that the engine can
  replay. OpenFHE ran serially, so each operation depends on the previous one (a
  conservative assumption for an accelerator, which could overlap independent
  work). Keys and plaintexts keep their real identities, so on-chip reuse is
  modelled faithfully; ciphertext identities are not recorded.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

from .params import CKKSParams
from .workload import STAGES, HEOp, Kernel, Trace

WORD = 8


@dataclass
class OpenFHELog:
    header: dict
    relin: str
    events: list = field(default_factory=list)      # (stage, kind, args as tuple)


def read_log(path: str | Path) -> OpenFHELog:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    header, relin, stage, events = {}, "", "modraise", []
    with opener(p, "rt") as f:
        started = False
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "H":
                header = {k: int(v) for k, v in (x.split("=") for x in parts[1:])}
                started = True
                continue
            if not started:
                continue
            if tag == "RELIN":
                relin = parts[1]
            elif tag == "S":
                stage = parts[1]
            elif tag == "E":
                header["out_towers"] = int(parts[1].split("=")[1])
                break
            else:
                events.append((stage, tag, tuple(parts[1:])))
    return OpenFHELog(header, relin, events)


def summarise_log(log: OpenFHELog) -> dict:
    """Per-stage counts, with the same keys as ``workload.summarise_trace`` where they overlap."""
    N = log.header["N"]
    out: dict = {}
    for stage, kind, a in log.events:
        s = out.setdefault(stage, {"hmult": 0, "hrot": 0, "pmult": 0, "ntt_limbs": 0, "intt_limbs": 0,
                                   "bconv": 0, "modup": 0, "auto_polys": 0, "rescale_polys": 0, "ew_polys": 0,
                                   "keys": set(), "key_bytes": 0, "pts": set(), "pt_bytes": 0})
        if kind == "NTT":
            s["ntt_limbs"] += 1
        elif kind == "INTT":
            s["intt_limbs"] += 1
        elif kind == "BCONV":
            q, p = int(a[0]), int(a[1])
            s["bconv"] += N * q * p + N * q
        elif kind == "MODUP":
            s["modup"] += 1
        elif kind == "KS":
            key, d, lk = a[0], int(a[1]), int(a[2])
            if key == log.relin:
                s["hmult"] += 1
            else:
                s["hrot"] += 1
                s["keys"].add(key)
            s["key_bytes"] += 2 * d * lk * N * WORD
        elif kind == "AUTO":
            s["auto_polys"] += 1
        elif kind == "RESCALE":
            s["rescale_polys"] += 1
        elif kind == "EW":
            s["ew_polys"] += 1
        elif kind == "PTMUL":
            s["pmult"] += 1
            s["pts"].add(a[0])
            s["pt_bytes"] += int(a[1]) * N * WORD
    for s in out.values():
        s["distinct_keys"] = len(s.pop("keys"))
        s["distinct_pts"] = len(s.pop("pts"))
    return out


def params_from_header(h: dict, name: str = "OpenFHE trace") -> CKKSParams:
    """A parameter set with OpenFHE's shape (ring degree, limbs, digits), for sizing the replay."""
    log_n = h["N"].bit_length() - 1
    return CKKSParams(name, log_n=log_n, L=h["towersQ"] - 1, dnum=h["dnum"], q_bits=59,
                      log_slots=h["slots"].bit_length() - 1)


def log_to_trace(log: OpenFHELog) -> Trace:
    """Group the stream into operations and build a replayable trace.

    A new operation starts at every ModUp, key-switch inner product and plaintext
    multiply, so each operation carries at most one key or plaintext. Consecutive
    kernels of the same kind inside an operation are merged.
    """
    N = log.header["N"]
    p = params_from_header(log.header)
    ops: list[HEOp] = []
    sizes: dict[str, int] = {"in": 2 * log.header["in_towers"] * N * WORD}
    levels: dict[str, int] = {"in": log.header["in_towers"] - 1}
    cur: dict | None = None
    towers = log.header["in_towers"]

    def flush():
        nonlocal cur
        if cur is None or not cur["kernels"]:
            cur = None
            return
        i = len(ops)
        prev = ops[-1].output if ops else "in"
        out = f"o{i}"
        sizes[out] = 2 * cur["towers"] * N * WORD
        levels[out] = cur["towers"] - 1
        ops.append(HEOp(i, cur["op"], cur["stage"], cur["towers"] - 1, [prev], out, cur["kernels"],
                        cur["key"], cur["pts"]))
        cur = None

    def add(kind: str, amount: int, words: int):
        ks = cur["kernels"]
        if ks and ks[-1].kind == kind:
            ks[-1] = Kernel(kind, ks[-1].amount + amount, ks[-1].words + words)
        else:
            ks.append(Kernel(kind, amount, words))

    for stage, kind, a in log.events:
        if kind in ("MODUP", "KS", "PTMUL") or cur is None or cur["stage"] != stage:
            flush()
            cur = {"stage": stage, "kernels": [], "key": None, "pts": [], "towers": towers,
                   "op": {"MODUP": "modup", "KS": "keyswitch", "PTMUL": "pmac"}.get(kind, "misc")}
        if kind == "NTT":
            add("ntt", 1, 2 * N)
        elif kind == "INTT":
            add("intt", 1, 2 * N)
        elif kind == "BCONV":
            q, pp = int(a[0]), int(a[1])
            add("bconv", N * q * pp + N * q, N * (q + pp))
        elif kind == "MODUP":
            towers = int(a[0])
            cur["towers"] = towers
        elif kind == "KS":
            key, d, lk, l = a[0], int(a[1]), int(a[2]), int(a[3])
            towers = l
            cur["towers"] = l
            add("mac", 2 * d * lk * N, 3 * d * lk * N + 2 * lk * N)
            cur["key"] = ("relin" if key == log.relin else key, 2 * d * lk * N * WORD)
            cur["op"] = "hmult" if key == log.relin else "hrot"
        elif kind == "MODDOWN":
            q = int(a[0])
            add("mac", 2 * q * N, 6 * q * N)
        elif kind == "AUTO":
            t = int(a[0])
            add("auto", t * N, 2 * t * N)
        elif kind == "RESCALE":
            t = int(a[0])
            towers = t - 1
            cur["towers"] = towers
            add("mac", t * N, 3 * t * N)
        elif kind == "EW":
            t = int(a[0])
            add("mac", t * N, 3 * t * N)
        elif kind == "PTMUL":
            cur["pts"].append((a[0], int(a[1]) * N * WORD))
    flush()
    return Trace(p, ops, sizes, ["in"], None, levels)


def stage_order(log: OpenFHELog) -> list[str]:
    seen = []
    for stage, _, _ in log.events:
        if stage not in seen:
            seen.append(stage)
    return [s for s in STAGES + ["post"] if s in seen]
