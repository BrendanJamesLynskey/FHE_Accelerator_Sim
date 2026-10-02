# Real OpenFHE kernel traces

Two CKKS bootstraps recorded from **OpenFHE v1.5.1**. The library was
instrumented with a small tracer (`openfhe_fhetrace.patch`, 82 added lines in 6
files), and each bootstrap ran single-threaded.

| File | Configuration | Events | Result |
|------|---------------|--------|--------|
| `full14.log.gz` | N = 2^14, 8,192 slots, level budget {3, 3}, dnum 3, 30 levels (not a 128-bit set) | 27,666 | decrypts to max error 1.6e-4 |
| `sparse16.log.gz` | N = 2^16, 8 slots, level budget {1, 1}, dnum 3, 18 levels, 128-bit classic: the timing-calibration configuration | 9,035 | decrypts to max error 1.2e-5 |

`full14_stcfirst.log.gz` and `sparse16_stcfirst.log.gz` record the same configurations
with SlotToCoeff first (`EvalBootstrapSetup(…, BTSlotsEncoding=true)`), starting from an
input with SlotToCoeff budget + 2 towers (5 and 3). They decrypt to max error 7.3e-6 and
1.5e-5.

Full-slot N = 2^16 was not recorded. OpenFHE's keys and precomputed plaintexts
for it exceed the memory available on the 15 GB recording machine.

## What is logged

One line per event: per-limb NTT and iNTT, base conversions (limbs in and out),
key-switch ModUp, key-switch inner products (with the evaluation key's identity
and size), ModDown, automorphisms, rescales, element-wise polynomial operations,
and multiplies by precomputed plaintexts (with their identity). Stage markers
are placed in `FHECKKSRNS::EvalBootstrap`. The line format is documented in
`src/fhe_sim/openfhe_trace.py`.

Not logged: scalar and integer multiplies, and arithmetic inside `NativePoly`
helpers. Element-wise work is therefore a lower bound.

## Reproduce

```bash
git clone --depth 1 --branch v1.5.1 https://github.com/openfheorg/openfhe-development.git
cd openfhe-development && git apply ../openfhe_fhetrace.patch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
         -DBUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=$PWD/../install
make -j4 && make install                     # about 2 minutes on an i7-3770

# build the driver against it
mkdir bt && cd bt && cmake <this directory> -DCMAKE_PREFIX_PATH=<install> && make
FHETRACE=full14.log OMP_NUM_THREADS=1 ./boot_trace 14 13 3 3 3 10 0
FHETRACE=sparse16.log OMP_NUM_THREADS=1 ./boot_trace 16 3 1 1 3 2 1
FHETRACE=full14_stcfirst.log OMP_NUM_THREADS=1 ./boot_trace 14 13 3 3 3 10 0 1   # SlotToCoeff first
```

Run single-threaded: each shared library writes through its own unbuffered
handle in append mode, so lines arrive in program order only without OpenMP
threads. Cap memory (for example `systemd-run --user --scope -p MemoryMax=6G`)
before trying larger rings.

## Use

```bash
fhe-sim --openfhe-log calibration/openfhe_trace/sparse16.log.gz --hw cpu   # replay on the CPU-like model
pytest tests/test_openfhe_trace.py                                        # model against library
```

`tests/test_openfhe_trace.py` checks three things:
- Where the model agrees exactly: key switching in SubSum.
- Where it differs, and by how much: the DFT split, and EvalMod rescale-on-use.
- That the replayed stream predicts the measured bootstrap time better than the scheme model does.
