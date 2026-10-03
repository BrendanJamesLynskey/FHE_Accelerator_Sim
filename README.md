# FHE_Accelerator_Sim

A [SimPy](https://simpy.readthedocs.io/) discrete-event simulator of an
**FHE accelerator running CKKS bootstrapping**, with a conventional digital NTT
datapath and an optional **hybrid electro-optical transform engine**. It is the
companion code for the
[FHE Accelerator Simulators](https://github.com/BrendanJamesLynskey/FHE_Hub_Accelerator_Simulators)
presentation series. New to a term (RNS limb, dnum, hoisting, EvalMod, T<sub>A.S.</sub>, SimPy
resource)? The series [glossary](https://brendanjameslynskey.github.io/FHE_Hub_Accelerator_Simulators/#glossary)
explains each concept briefly and links to the slides that explain it in depth.

It answers four questions about a design:

1. How long does a bootstrap take, and where does the time go?
2. Is the design bound by NTT throughput, by memory (evaluation keys), or by power?
3. How much on-chip SRAM stops key traffic dominating?
4. Does an optical NTT engine win once conversion energy and precision corrections are counted?

The goal is **learnability**: a small, readable, fully tested simulator with the
structure of a serious one. Scheme model, cost model, event engine and probes are
separate, and a test ladder runs from hand formulas to a JavaScript twin.

* **Trace generator:** CKKS bootstrapping (ModRaise → CoeffToSlot → EvalMod → SlotToCoeff)
  as HE operations broken into primitive kernels (NTT/iNTT, base conversion,
  modular multiply-add, automorphism), with hybrid key switching (`dnum`), BSGS
  homomorphic DFTs, a BSGS Chebyshev EvalMod and double-angle steps. JSON trace
  dump and replay.
* **Engine:** NTT, MAC, automorphism and optical units as `simpy.Resource`s;
  shared, chunked HBM; a capacity-limited LRU scratchpad where misses become key,
  plaintext or ciphertext traffic and spills become write-backs; dependency-aware
  issue with a window and decoupled key/plaintext prefetch.
* **Acceleration techniques:**
  - hoisted rotations;
  - Min-KS key reuse (as in ARK);
  - seeded keys;
  - on-the-fly plaintext generation;
  - OpenFHE's BSGS split (lazy ModDown);
  - **SlotToCoeff-first** ordering, which helps every design (see below).
* **Metrics:** bootstrap latency and per-stage breakdown, utilisation per unit,
  NTT-bound / memory-bound / power-bound attribution, hot-spot per stage, HBM bytes
  by class, Perfetto traces, and an analytic lower bound.
* **Power:** static power plus pJ per butterfly, multiply-add, permuted word, SRAM byte
  and HBM byte; DAC/ADC energy from a Walden figure of merit (energy ∝ 2^ENOB);
  laser and thermal-tuning static power. The **TDP is enforced by a dynamic power
  manager**: each kernel gets the highest clock, and each HBM chunk the highest
  bandwidth, that fits the headroom left by everything running at that moment, and
  waits if nothing fits. A worst-case fixed clock is kept for comparison
  (`power_mode="worst-case"`). Optional DVFS.
* **Optical engine:** a precision model (digit planes, a Bluestein convolution, ENOB)
  backed by a functional model that shows the rounding rule is exact and tight.
* **Calibration:** one throughput parameter fitted to OpenFHE measured on this
  machine, and two out-of-sample predictions.
* A **JavaScript port** (`web/sim_engine.js`) that runs live in
  [deck 03](https://brendanjameslynskey.github.io/FHESim_03_Simulating_an_FHE_Accelerator/)
  and matches the Python **bit for bit**.
* **Real OpenFHE traces:** two bootstraps recorded from OpenFHE v1.5.1, which was
  instrumented with an 82-line patch to log every NTT, base conversion, key switch,
  automorphism, rescale and plaintext multiply. The model is checked against them
  stage by stage, and the streams replay on the engine (`fhe-sim --openfhe-log`).
* **A HEIR front end:** reads the `ckks`-dialect output of the
  [HEIR](https://heir.dev) compiler (release v2026.10.01) and builds a trace of the
  server function, with HEIR's parameters, levels, rotations and bootstrap
  placement. Three of HEIR's example programs are included (`fhe-sim --heir`): LoLa,
  an MNIST MLP, and LoLa with a HEIR-placed bootstrap.
* **An optional command-level memory model:** `Accelerator(memory=...)` times each
  HBM chunk with [Memory_System_Sim](https://github.com/BrendanJamesLynskey/Memory_System_Sim)'s
  HBM model (bank timing, refresh, scheduling, address mapping) instead of peak
  bandwidth (`fhe-sim --memsim`). Off by default; default results are unchanged.
* **RTL-calibrated NTT throughput:** [RTL_CoSim_NTT](https://github.com/BrendanJamesLynskey/RTL_CoSim_NTT)
  verifies a SystemVerilog NTT core against this repo's `ntt_reference` and feeds
  the butterfly efficiency it measures back into this simulator.
* **104 tests:** parameter sizes, closed-form operation counts, invariants, analytic
  queueing checks, behaviour, power (both power modes), Hypothesis properties,
  precision, calibration, the model against recorded OpenFHE streams, the HEIR
  front end (including the same program executed by OpenFHE), the memory-model
  interface, and JS ↔ Python parity.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest                                        # 104 tests, about 20 seconds

fhe-sim                                       # ARK-like parameters on an ARK-class digital design
fhe-sim --counts                              # operation and byte counts per stage, no timing
fhe-sim --min-ks --seeded-keys --otf-pt       # cut key traffic: watch the bound move
fhe-sim --sweep-sram 128 256 512 1024 2048    # traffic against scratchpad size
fhe-sim --hw small                            # an NTT-starved design
fhe-sim --hw small --optical ideal            # hypothetical precision-free optical NTT
fhe-sim --hw hybrid --enob 16                 # realistic optical engine at 16 ENOB
fhe-sim --dvfs                                # lower the clock when memory-bound
fhe-sim --ntt 16384 --mac 32768 --tdp 150 --power-mode worst-case   # vs the default power manager
fhe-sim --trace boot.json                     # open in https://ui.perfetto.dev
fhe-sim --dump-trace t.json; fhe-sim --replay t.json
fhe-sim --openfhe-log calibration/openfhe_trace/sparse16.log.gz --hw cpu   # replay a real OpenFHE bootstrap
fhe-sim --lazy-moddown                        # OpenFHE's BSGS split: fewer NTTs, more keys
fhe-sim --stc-first                           # SlotToCoeff before ModRaise: 10 levels left instead of 7
fhe-sim --heir calibration/heir/lola.ckks.mlir.gz   # LoLa, compiled by HEIR

python examples/results.py                    # regenerate every number in this README and the decks
python examples/calibrate_openfhe.py          # the OpenFHE calibration
```

Example report (`fhe-sim`):

```
── ARK-like (N=2^16, L=23, dnum=4) on Digital FHE accelerator (ARK-class, illustrative)
bootstraps 1   latency 13.94 ms   per bootstrap 13.94 ms   clock 100% (dynamic)
utilisation  ntt 13%  mac 29%  auto 2%  hbm 89%
verdict      memory-bound
stages       modraise 0.01 ms (0%)  cts 10.00 ms (72%)  evalmod 1.45 ms (10%)  stc 2.47 ms (18%)
hot-spots    modraise->ntt  cts->hbm  evalmod->mac  stc->hbm   (dominant: cts -> hbm)
HBM traffic  keys 6.74 GB  plaintexts 3.27 GB  ct read 1.21 GB  ct write 1.22 GB  (keys 54%)
power        avg 82 W  peak 176 W (TDP 250 W)  1139.1 mJ/bootstrap  energy: static 49%  ntt 8%  mac 10%  auto 1%  hbm 33%
```

---

## What's in the box

| Module | Role |
|--------|------|
| `src/fhe_sim/params.py` | `CKKSParams`: ring degree, levels, `dnum`, digits; ciphertext, plaintext and evaluation-key sizes; presets |
| `src/fhe_sim/workload.py` | The scheme model: HE ops → primitive kernels; bootstrap and single-op traces; JSON dump/replay; per-stage counts |
| `src/fhe_sim/hardware.py` | `Accelerator`, `OpticalEngine`, the `CostModel` (the only place that knows about time and energy), worst-case TDP clocking |
| `src/fhe_sim/sim.py` | The SimPy engine: units as resources, HBM, scratchpad, dependency-aware issue, prefetch, write-backs, the dynamic power manager, DVFS |
| `src/fhe_sim/metrics.py` | Latency, stage breakdown, utilisation, bound attribution, hot-spots, traffic by class, energy and power |
| `src/fhe_sim/precision.py` | Functional model of an exact modular NTT on an analogue FFT engine (digit planes, Bluestein, ADC) |
| `src/fhe_sim/trace.py` | Chrome trace-event export for Perfetto |
| `src/fhe_sim/heir_frontend.py` | Reads HEIR `ckks`-dialect IR: parameters, levels, SSA dependencies, bootstraps, Chebyshev activations; builds a trace |
| `src/fhe_sim/openfhe_trace.py` | Reads kernel streams recorded from instrumented OpenFHE: per-stage counts, conversion to a replayable trace |
| `src/fhe_sim/search.py` | Analytic lower bound, SRAM sweep, bisection for minimum SRAM, parallel design sweep, Pareto front |
| `src/fhe_sim/cli.py` | The `fhe-sim` command |
| `web/sim_engine.js` | The browser port (with a minimal SimPy core) used in deck 03 |
| `calibration/` | OpenFHE timing measurements and scripts; `openfhe_trace/`: the OpenFHE patch, the trace driver and two recorded bootstraps; `heir/`: three HEIR-compiled programs and the OpenFHE execution of one |
| `examples/results.py` | Generates `examples/results.md`, the source of every quoted number |

### Selected results (from `examples/results.md`)

**Acceleration techniques on the ARK-class design** (ark parameter set, 512 MiB scratchpad, 1 TB/s HBM):

| Algorithm | Bootstrap | Key traffic | All HBM traffic | Energy | Verdict |
|-----------|-----------|-------------|-----------------|--------|---------|
| Baseline (hoisted BSGS) | 13.94 ms | 6.74 GB | 12.44 GB | 1139 mJ | memory-bound |
| + Min-KS | 8.51 ms | 1.15 GB | 5.24 GB | 726 mJ | memory-bound |
| + Min-KS + seeded keys | 8.86 ms | 0.58 GB | 4.62 GB | 738 mJ | MAC-bound |
| + Min-KS + seeded keys + on-the-fly plaintexts | 7.19 ms | 0.58 GB | 0.78 GB | 607 mJ | MAC-bound |

On an NTT-starved design the same techniques make things **worse** (17.46 → 26.65 ms),
because they trade compute for bandwidth.

**SlotToCoeff first** (`stc_first`, OpenFHE's `BTSlotsEncoding`) runs SlotToCoeff on the
nearly exhausted input, before ModRaise. Its keys are cheap there, EvalMod runs once
(real-valued data), and the output keeps 3 more levels. It is the one technique that
helps every design:

| Design | Order | Bootstrap | Levels left | Per useful level |
|--------|-------|-----------|-------------|------------------|
| ARK-class | conventional | 13.94 ms | 7 | 1.99 ms |
| ARK-class | SlotToCoeff first | 11.51 ms | 10 | 1.15 ms |
| ARK-class | all three + SlotToCoeff first | 5.42 ms | 10 | 0.54 ms |
| small digital | conventional | 17.46 ms | 7 | 2.49 ms |
| small digital | SlotToCoeff first | 12.74 ms | 10 | 1.27 ms |

OpenFHE's own StC-first bootstrap, recorded at N=2^14, shows the same shape: 3 more
output towers, 24 instead of 48 EvalMod multiplications, and 0.18 instead of 0.57 GB
of SlotToCoeff keys.

**SRAM against key traffic.** With the baseline algorithm every rotation key is used
once per bootstrap, so key traffic stays at 6.74 GB from 512 MiB to 4 GiB of SRAM.
More SRAM only removes ciphertext spills (12.44 → 10.02 GB). Keys are reused across
two back-to-back bootstraps only at 16 GiB (3.37 GB per bootstrap). With Min-KS,
seeded keys and on-the-fly plaintexts, **512 MiB** brings all HBM traffic down to
0.78 GB per bootstrap.

**Optical NTT engine** (ark parameter set, NTT-starved digital design):

| Design | Bootstrap | Energy | Verdict |
|--------|-----------|--------|---------|
| Digital only | 17.46 ms | 1280 mJ | NTT-bound |
| + realistic engine (16-point blocks, ENOB 12, 50-bit limbs) | 401.38 ms | 46861 mJ | optical-bound |
| + ideal engine (exact at any precision, 4096-point blocks) | 12.59 ms | 1375 mJ | MAC-bound |
| ARK-class (memory-bound) + ideal engine | 16.87 ms (vs 13.94) | 1632 mJ | memory-bound |

Exact rounding needs 2^(ENOB−1) > d · block · (2^b − 1)^2. With 50-bit limbs at
ENOB 12 that forces 1-bit digits: 298 conversions per point to offload 2 butterflies
per point. Even a precision-free engine pays off in energy only if a conversion
costs under about 30 pJ (with 5 W of laser and tuning power) or 47 pJ (with none).

**Dynamic power manager against a worst-case clock** (4× NTT and MAC, all key techniques):

| TDP | Worst-case clock | Dynamic power manager |
|-----|------------------|-----------------------|
| 250 W | 8.28 ms at 74% | 6.20 ms at 100%, peak 239 W |
| 150 W | 11.62 ms at 53% | 6.41 ms, mean clock 99%, peak 150 W |
| 100 W | cannot run | 6.68 ms, mean clock 92%, power-bound |

A worst-case clock reserves power for every unit and HBM running flat out at once,
which this workload never does. The manager spends the real headroom instead; peak
power stays at or under the TDP in both modes.

### Calibration and validation

| Quantity | OpenFHE (i7-3770, 8 threads) | Simulated | Error |
|----------|------------------------------|-----------|-------|
| HMult, N=2^16, 24 limbs, dnum 4 | 352.6 ms | 352.6 ms | fitted |
| HRotate, same parameters | 324.8 ms | 292.5 ms | −10% |
| Bootstrap, N=2^16, 8 slots, dnum 3 (scheme model) | 12,113 ms | 8,473 ms | −30% |
| Same bootstrap, OpenFHE's recorded kernel stream replayed | 12,113 ms | 10,759 ms | −11% |

One parameter was fitted (0.18 modular operations per cycle per unit at 3.4 GHz);
the other rows are predictions. The scheme model under-predicts the bootstrap mainly
because OpenFHE rescales copies of each input before every multiplication (see
below). Replaying the recorded stream removes that gap; most of the remaining −11%
is element-wise work the tracer does not log. Full-slot bootstrapping at N=2^16
could not be measured or recorded: OpenFHE's keys and precomputed plaintexts
exceeded the memory cap on this 15 GB machine.

### Real programs compiled by HEIR

`examples/results.md` §18, on the ARK-class design:

| Program | N, limbs | HE ops | SRAM | Latency | Keys / plaintexts / ciphertexts | Verdict |
|---------|----------|--------|------|---------|---------------------------------|---------|
| LoLa (MNIST CNN, square activations) | 2^15, 11 | 409 | 512 MiB | 0.61 ms | 0.35 / 0.17 / 0.00 GB | memory-bound |
| MNIST MLP (polynomial ReLU) | 2^15, 13 | 1,655 | 512 MiB | 2.25 ms | 0.73 / 0.95 / 0.27 GB | memory-bound |
| LoLa, level budget 2 (HEIR places a bootstrap) | 2^17, 47 | 625 | 512 MiB | 202.75 ms | 61.43 / 24.58 / 112.46 GB | memory-bound |
| … same program | 2^17, 47 | 625 | 2 GiB | 101.45 ms | 47.92 / 24.58 / 21.20 GB | memory-bound |

* **The MLP is weight-bound:** it moves more plaintext than key material.
* **HEIR's bootstrap parameters outgrow the chip.** To fit a bootstrap under the tiny
  budget it chose N=2^17 with 47 primes, and a single key switch then needs 354 MiB of scratch.
* **The same LoLa through HEIR's OpenFHE code generation**, run on the instrumented
  OpenFHE, performs exactly the 55 rotations, 41 rotation keys and 2 relinearisations
  the front end reads. It costs 3× more on the ARK-class model (1.84 vs 0.61 ms): the
  generated context carries 12 limbs where HEIR's level analysis needs 6, and OpenFHE's
  automatic rescaling adds 641 polynomial rescales to HEIR's 19 rescale ops.

### Checked against real OpenFHE bootstraps

`calibration/openfhe_trace/full14.log.gz` (N=2^14, 8,192 slots, level budget {3,3},
dnum 3) against the model at the same parameters (`examples/results.md`, §17):

| Stage | Rotations: OpenFHE / model / model with OpenFHE's BSGS | NTT + iNTT limbs | Key GB requested |
|-------|------|------|------|
| CoeffToSlot | 46 / 35 / 51 | 2,419 / 5,151 / 2,025 | 1.45 / 1.13 / 1.64 |
| EvalMod (HMults 48 / 56) | — | 20,411 / 11,816 / 11,816 | 1.05 / 1.28 / 1.28 |
| SlotToCoeff | 45 / 34 / 50 | 1,025 / 2,494 / 854 | 0.57 / 0.43 / 0.63 |

* **Exact where the algorithms agree.** Bootstrap depth matches (20, and 16 for the
  N=2^16 recording). The N=2^16 SubSum's 12 rotations match on keys and key bytes
  exactly, and on transforms to within 3%.
* **The DFT split.** OpenFHE keeps rotations in the Q·P basis and uses more baby
  steps (16 against 8 per radix-2^5 level). The `lazy_moddown` option models this
  and lands within 20%. On the ARK-class accelerator it is a bad trade: a third
  fewer NTTs, but 28% more key traffic and 42% more time.
* **EvalMod.** OpenFHE's FLEXIBLEAUTO scaling rescales input copies before every
  multiplication: 542 polynomial rescales for 48 HMults, against 2 per HMult in the
  model. That is a library policy an accelerator compiler would not copy; the
  difference is tested, not modelled.
* **Corrected from the trace.** OpenFHE's EvalMod is degree 88 with 6 double-angle
  steps (the `openfhe-sparse` preset had guessed 119 and 3). A DFT level has at most
  `slots` diagonals (8 for 8 slots; the model had used 15).

Published results are **order-of-magnitude references only**, with their own
parameters. The 100x GPU work (Jung et al., TCHES 2021) bootstraps N=2^16, L=34,
dnum=5 in 328 ms on a V100. ARK (MICRO 2022) reports a logistic-regression
training iteration, bootstrap included, of 7.42 ms (CraterLake 15.2 ms, BTS
28.4 ms), with 512 MB of SRAM, 1 TB/s HBM, 418 mm² and 281 W peak. This model's
ARK-class design takes 7–14 ms per bootstrap depending on the algorithm. That is
the same order, not a reproduction.

**All hardware coefficients are illustrative.** Unit throughputs are sized like
the published ASICs; energies are round numbers. Calibrate them by regressing
measured or RTL-derived power on the simulator's event counts.

### A command-level HBM model instead of peak bandwidth

`examples/results.md` §20 runs the same bootstraps with HBM at peak bandwidth (the
default), with a flat "bandwidth × efficiency" derating, and with
[Memory_System_Sim](https://github.com/BrendanJamesLynskey/Memory_System_Sim)'s
command-level HBM2E-class model plugged in as `Accelerator(memory=HBMChunkModel())`:

| ARK-class, baseline algorithm | bootstrap | verdict |
|---|---|---|
| peak bandwidth (default) | 13.94 ms | memory-bound |
| flat 0.7 × bandwidth | 19.26 ms | memory-bound |
| Memory_System_Sim, FR-FCFS, bank groups interleaved | 15.31 ms | memory-bound |
| ... FCFS scheduler | 23.70 ms | memory-bound |

FHE traffic is long sequential chunks, so a well-configured controller reaches about
0.90 of peak, refresh being most of the loss, and the detailed model then agrees with a
flat 0.9 to 0.2%. Its value is that it derives the number instead of assuming it,
and shows what breaks it (FCFS scheduling, a mapping that keeps a row's bursts in one bank
group). Near a balance point it changes the verdict. With all three techniques at 160
GB/s, peak bandwidth says MAC-bound; the HBM model says memory-bound. At 200 GB/s a
folklore flat 0.7 says memory-bound, and the HBM model agrees with peak that it is
MAC-bound.

---

## The trace format, and where traces come from

`fhe-sim --dump-trace` writes `fhe-sim-trace/1` JSON. It holds the parameters, the
external inputs, object sizes and levels, and per HE op: inputs, output, key
`[id, bytes]`, plaintexts `[[id, bytes], …]` and kernels `[[kind, amount, words], …]`.
Anything that can emit this can drive the engine:

* **This repo's scheme model** (`workload.py`), the default.
* **An FHE compiler: done for HEIR.** `heir_frontend.compile_ir` reads HEIR's output after
  `--torch-linalg-to-ckks`/`--mlir-to-ckks` with `unroll-fhe-kernel-loops=true`. At that point
  the server function is straight-line `ckks` code with the level in every type, and the front
  end emits one HE op per `ckks` op, with dependencies from SSA. `ckks.bootstrap` is expanded
  with the scheme model's bootstrap, and `kernel.eval_chebyshev` with its Chebyshev evaluation.
  Validated on HEIR v2026.10.01 only; HEIR's dialects change quickly. See `calibration/heir/README.md`.
* **An instrumented library.** Done for OpenFHE v1.5.1: `calibration/openfhe_trace`
  holds the 82-line patch, the C++ driver, two recorded bootstraps and build steps.
  `openfhe_trace.log_to_trace` turns a log into this format. Operations are
  serialised in program order (OpenFHE ran single-threaded), keys and plaintexts
  keep their real identities, and ciphertext identities are not recorded.

---

## Modelling assumptions (read before trusting a number)

* Operation counts come from this repo's own scheme model. Real libraries differ
  by tens of per cent: different DFT factorisations, EvalMod polynomials, level
  orderings and fused kernels.
* HBM delivers its peak bandwidth unless a memory model is plugged in (above).
* Each functional-unit class is one resource with aggregate throughput, and every
  kernel gets the full scratchpad port bandwidth. Bank conflicts and on-chip
  network contention are not modelled.
* Scratchpad hits and misses are decided at issue, in program order (a
  compiler-managed scratchpad). Kernel temporaries live in a reserved working set
  sized for one top-level key switch.
* The power manager is greedy and first-come: the first kernel to ask gets the highest
  clock that fits, and later ones get what is left. Clock changes are instantaneous
  (no DVFS transition latency), and a kernel keeps its clock until it finishes.
* DVFS is a clock cap for the whole run, chosen from a first pass. Energy per
  operation scales with clock², and static power is charged for the whole run.
* The optical mapping is one illustrative route among several: digital stages,
  then Bluestein convolutions on digit planes, then digital correction. Its
  converter-limited throughput and the Walden energy model are first-order.

Each of these is an exercise in deck 05.

---

## Part of

The [FHE Accelerator Simulators](https://github.com/BrendanJamesLynskey/FHE_Hub_Accelerator_Simulators)
series, the sister of [LLM Inference Simulators](https://github.com/BrendanJamesLynskey/LLM_Hub_Inference_Simulators)
(whose code repo, [Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim),
this one mirrors). For FHE fundamentals see the
[Cryptography section](https://github.com/BrendanJamesLynskey/Mathematics#cryptography) of the
Mathematics hub, especially the
[Fully Homomorphic Encryption deck](https://brendanjameslynskey.github.io/Cryptography/08-fully-homomorphic-encryption/).
The [Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit) series
builds on it: [RTL_CoSim_NTT](https://github.com/BrendanJamesLynskey/RTL_CoSim_NTT) (RTL verified against
this NTT), [Memory_System_Sim](https://github.com/BrendanJamesLynskey/Memory_System_Sim) (the HBM model) and
[SystemC_Accelerator_Model](https://github.com/BrendanJamesLynskey/SystemC_Accelerator_Model) (this
model's tile in SystemC TLM-2.0, checked against it on the same traces).
