# Programs compiled by HEIR

[HEIR](https://heir.dev) (Ali et al., [arXiv:2508.11095](https://arxiv.org/abs/2508.11095))
compiles tensor programs to FHE. The files here are HEIR's own example programs,
compiled with the release binaries `heir-opt` and `heir-translate` (v2026.10.01).
They are lowered to the `ckks` dialect, the level at which the front end
(`src/fhe_sim/heir_frontend.py`) reads them.

| File | Source (in the HEIR repo) | Flags (`--torch-linalg-to-ckks=…`) | Server ops |
|------|---------------------------|------------------------------------|-----------|
| `lola.ckks.mlir.gz` | `tests/Examples/common/lola.mlir`: LoLa, an MNIST CNN with square activations | `min-slot-count=2048 unroll-fhe-kernel-loops=true` | 409 |
| `lola_bootstrap.ckks.mlir.gz` | the same, with `greedy-level-budget=2` so HEIR must place a bootstrap | as above plus `greedy-level-budget=2` | 625 after expanding the bootstrap |
| `mnist_mlp.ckks.mlir.gz` | `tests/Examples/common/mnist/mnist.mlir`: a 784-512-10 MLP with a polynomial ReLU | `min-slot-count=4096 greedy-level-budget=8 greedy-modulus-switch-after-mul=true first-mod-bits=30 scaling-mod-bits=24 unroll-fhe-kernel-loops=true` | 1,655 |
| `lola_openfhe.log.gz` | `lola` after `--scheme-to-openfhe` and `heir-translate --emit-openfhe-pke`, run by `heir_harness.cpp` on the instrumented OpenFHE (`../openfhe_trace`) | | 11,222 kernel events |

All were run with `--annotate-module="backend=openfhe scheme=ckks"`. The `.gz`
files are *slimmed* with `heir_frontend.slim_ir`: the module line with
`ckks.schemeParam`, the ciphertext type aliases and the server function. HEIR's
full output also carries client and preprocessing functions with every weight
inline (19 MB for LoLa). The slim files give the same trace as the full output.

## Reproduce

```bash
curl -LO https://github.com/google/heir/releases/download/v2026.10.01/heir-opt-manylinux_2_28_x86_64
curl -LO https://github.com/google/heir/releases/download/v2026.10.01/heir-v2026.10.01.tar.gz   # examples
./heir-opt-manylinux_2_28_x86_64 heir-v2026.10.01/tests/Examples/common/lola.mlir \
    --annotate-module="backend=openfhe scheme=ckks" \
    --torch-linalg-to-ckks="min-slot-count=2048 unroll-fhe-kernel-loops=true" -o lola.ckks.mlir
fhe-sim --heir lola.ckks.mlir

# the OpenFHE cross-check: generate C++, build against the patched OpenFHE, run with an unlimited stack
heir-opt … --scheme-to-openfhe -o lola.openfhe.mlir
heir-translate lola.openfhe.mlir --emit-openfhe-pke-header -o lola_lib.h
heir-translate lola.openfhe.mlir --emit-openfhe-pke -o lola_lib.cpp
g++ -std=c++17 -O1 -fopenmp <OpenFHE include dirs> -c lola_lib.cpp          # about 30 s, 1.3 GB
g++ -std=c++17 -O2 -fopenmp <includes> heir_harness.cpp lola_lib.o -lOPENFHEpke -lOPENFHEcore -o heir_lola
ulimit -s unlimited          # the generated code initialises its weights on the stack
FHETRACE=lola_openfhe.log OMP_NUM_THREADS=1 ./heir_lola
```

## Notes

* Three HEIR examples failed in this release and are not used:
  - LeNet: an unrealized conversion cast; its test target is disabled upstream.
  - MNIST at `greedy-level-budget=3`: an internal assertion.
  - batchnorm_sigmoid: ciphertext division is not supported.
* HEIR's `kernel.eval_chebyshev` (MNIST's ReLU) is expanded with the scheme
  model's baby-step giant-step Chebyshev evaluation. HEIR's IR evaluates the
  degree-5 series in 3 levels, while the model's schedule uses 4, so the front end
  keeps HEIR's level and records a note.
* In `lola_bootstrap`, HEIR's types take the bootstrap from level 20 to level 1. The
  front end expands it with the model's CKKS bootstrap at HEIR's parameters
  (output level 30), then drops limbs to HEIR's level 1, and records a note.
