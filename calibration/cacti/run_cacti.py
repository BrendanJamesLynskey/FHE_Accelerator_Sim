#!/usr/bin/env python3
"""CACTI 7 sweep of large banked SRAM scratchpads: the SRAM calibration of ``fhe_sim.ppa``.

CACTI (https://github.com/HewlettPackard/cacti) models an SRAM array's area, energy and timing
from ITRS technology data. Its smallest node is 22 nm, so this sweep gives the *shape* of
area against capacity (how array efficiency changes as a scratchpad grows); ``ppa.py`` scales
it to 7 nm with one factor anchored on ARK's and BTS's published 7 nm scratchpads.

The configurations are the upstream cache.cfg with only these lines changed (the full .cfg
files are written to configs/, and CACTI's full output to out/):
  -size (bytes)        64 MiB ... 2 GiB (CACTI's size field overflows at 4 GiB)
  -UCA bank count      size / 4 MiB (a multi-banked scratchpad, as in ARK and BTS)
  -technology (u)      0.022
  -associativity 1, -cache type "ram" (a scratchpad: no tag array)
  -Data array cell / peripheral type   itrs-lstp (low standby power) and itrs-hp

    CACTI=~/.local/opt/cacti python calibration/cacti/run_cacti.py   # writes out/results.json
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACTI = Path(os.environ.get("CACTI", Path.home() / ".local/opt/cacti"))
MIB = 1 << 20
BANK_MIB = 4
SIZES_MIB = [64, 128, 256, 512, 1024, 2048]   # CACTI rejects 4 GiB (a 32-bit size overflows)
RUNS = [(cell, s) for cell in ("itrs-lstp", "itrs-hp") for s in SIZES_MIB]


def config(cell: str, size: int, banks: int) -> str:
    text = (CACTI / "cache.cfg").read_text()
    subs = {r"^-size \(bytes\) .*$": f"-size (bytes) {size}",
            r"^-technology \(u\) .*$": "-technology (u) 0.022",
            r"^-associativity .*$": "-associativity 1",
            r'^-cache type .*$': '-cache type "ram"',
            r"^-UCA bank count .*$": f"-UCA bank count {banks}",
            r'^-Data array cell type - .*$': f'-Data array cell type - "{cell}"',
            r'^-Data array peripheral type - .*$': f'-Data array peripheral type - "{cell}"'}
    for pat, rep in subs.items():
        text, n = re.subn(pat, rep, text, count=1, flags=re.M)
        assert n == 1, pat
    return text


def grab(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"{pattern!r} not in CACTI output")
    return float(m[1])


def main() -> None:
    (HERE / "configs").mkdir(exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)
    version = subprocess.run(["git", "-C", str(CACTI), "log", "-1", "--format=%H"], capture_output=True,
                             text=True).stdout.strip()
    rows = []
    for cell, mib in RUNS:
        banks = mib // BANK_MIB
        name = f"sram_{mib}MiB_{cell}_{banks}bank"
        cfg = HERE / "configs" / f"{name}.cfg"
        cfg.write_text(config(cell, mib * MIB, banks))
        t = time.perf_counter()
        p = subprocess.run([str(CACTI / "cacti"), "-infile", str(cfg)], cwd=CACTI, capture_output=True,
                           text=True, timeout=3600)
        dt = time.perf_counter() - t
        (HERE / "out" / f"{name}.txt").write_text(p.stdout + p.stderr)
        (CACTI / f"{cfg}.out").unlink(missing_ok=True)
        o = p.stdout
        h, w = re.search(r"Cache height x width \(mm\): ([\d.]+) x ([\d.]+)", o).groups()
        area = float(h) * float(w)
        rows.append(dict(name=name, cell=cell, size_mib=mib, banks=banks, runtime_s=round(dt, 2),
                         area_mm2=round(area, 3), mm2_per_mib=round(area / mib, 4),
                         access_ns=grab(r"Access time \(ns\): ([\d.]+)", o),
                         read_nj=grab(r"Total dynamic read energy per access \(nJ\): ([\d.]+)", o),
                         leak_mw=grab(r"Total leakage power of a bank \(mW\): ([\d.]+)", o) * banks,
                         efficiency_pct=grab(r"Area efficiency \(Memory cell area/Total area\) - ([\d.]+)", o)))
        print(rows[-1], flush=True)
    (HERE / "out" / "results.json").write_text(json.dumps(dict(cacti_commit=version, node_nm=22, bank_mib=BANK_MIB,
                                                               rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
