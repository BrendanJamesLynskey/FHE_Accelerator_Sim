"""The HEIR front end: real programs compiled by HEIR v2026.10.01 -> fhe-sim traces.

calibration/heir holds the CKKS-dialect output of HEIR's own example programs,
slimmed to the server function: LoLa (an MNIST CNN with square activations), an
MNIST MLP with a polynomial ReLU, and LoLa compiled under a level budget that
forces HEIR to place a bootstrap. lola_openfhe.log.gz is the kernel stream of the
same LoLa after HEIR's OpenFHE code generation, run on the instrumented OpenFHE.
"""

import gzip
import re
from pathlib import Path

import pytest

from fhe_sim import ACCELERATORS, simulate, summarise, summarise_trace
from fhe_sim.heir_frontend import compile_ir, server_function, slim_ir
from fhe_sim.openfhe_trace import read_log, summarise_log

D = Path(__file__).parent.parent / "calibration" / "heir"
PROGRAMS = {"lola": (1 << 15, 10, 3), "mnist_mlp": (1 << 15, 12, 2), "lola_bootstrap": (1 << 17, 46, 12)}


def text(name):
    return gzip.open(D / f"{name}.ckks.mlir.gz", "rt").read()


@pytest.mark.parametrize("name", PROGRAMS)
def test_parameters_come_from_heir(name):
    prog = compile_ir(D / f"{name}.ckks.mlir.gz")
    N, L, k = PROGRAMS[name]
    assert (prog.params.N, prog.params.L, prog.params.k) == (N, L, k)


@pytest.mark.parametrize("name", PROGRAMS)
def test_every_ckks_op_becomes_one_he_op(name):
    """The server function is straight-line after unrolling, so the dynamic counts must equal
    the static ones in the IR."""
    body = "\n".join(server_function(text(name)))
    static = {}
    for m in re.finditer(r"= (?:ckks|kernel)\.(\w+)", body):
        static[m.group(1)] = static.get(m.group(1), 0) + 1
    prog = compile_ir(D / f"{name}.ckks.mlir.gz")
    assert prog.counts == static


@pytest.mark.parametrize("name", PROGRAMS)
def test_replay_respects_dependencies_and_power(name):
    prog = compile_ir(D / f"{name}.ckks.mlir.gz")
    hw = ACCELERATORS["ark"].with_(sram_mib=2048)
    r = simulate(prog.trace, hw)
    producer = {o.output: o.id for o in prog.trace.ops}
    for o in prog.trace.ops:
        for x in o.inputs:
            if x in producer:
                assert r.op_start[o.id] >= r.op_end[producer[x]]
    assert summarise(r)["energy"]["peak_power_W"] <= hw.tdp_w * (1 + 1e-12)


def test_same_program_in_openfhe_has_the_same_structure():
    """HEIR's LoLa, run by OpenFHE after HEIR's code generation, performs exactly the rotations,
    rotation keys and relinearisations the front end reads from the IR."""
    lg = read_log(D / "lola_openfhe.log.gz")
    o = summarise_log(lg)["app"]
    prog = compile_ir(D / "lola.ckks.mlir.gz")
    rot_keys = {op.key[0] for op in prog.trace.ops if op.key and op.key[0] != "relin"}
    relins = sum(1 for op in prog.trace.ops if op.op == "relinearize")
    assert o["hrot"] == prog.counts["rotate"] == 55
    assert o["distinct_keys"] == len(rot_keys) == 41
    assert o["hmult"] == relins == 2


def test_openfhe_codegen_runs_with_more_limbs_and_more_rescales():
    """The generated OpenFHE context starts at 12 limbs where HEIR's level analysis needs 6, and
    FLEXIBLEAUTO adds library rescales: the library execution costs more than the IR implies."""
    lg = read_log(D / "lola_openfhe.log.gz")
    assert lg.header["in_towers"] == 12
    prog = compile_ir(D / "lola.ckks.mlir.gz")
    first = prog.trace.levels[prog.trace.external[0]]
    assert first + 1 == 6
    o = summarise_log(lg)["app"]
    assert o["rescale_polys"] > 10 * 2 * prog.counts["rescale"]
    m = summarise_trace(prog.trace)["app"]
    assert o["ntt_limbs"] + o["intt_limbs"] > 3 * (m["ntt_limbs"] + m["intt_limbs"])


def test_bootstrap_placed_by_heir_is_expanded():
    prog = compile_ir(D / "lola_bootstrap.ckks.mlir.gz")
    stages = {o.stage for o in prog.trace.ops}
    assert {"app", "modraise", "cts", "evalmod", "stc", "app_post"} <= stages
    assert prog.counts["bootstrap"] == 1 and prog.notes


def test_polynomial_activation_is_expanded():
    prog = compile_ir(D / "mnist_mlp.ckks.mlir.gz")
    assert prog.counts["eval_chebyshev"] == 1
    hmults = sum(1 for o in prog.trace.ops if o.op == "hmult")
    assert hmults == (4 - 1) + (1 - 1) + (2 - 1)    # degree 5: b = 4, g = 2, m = 1: (b-1)+(m-1)+(g-1)


def test_slim_ir_is_idempotent():
    t = text("lola")
    assert slim_ir(t) == t


def test_unsupported_ops_are_errors(tmp_path):
    t = text("lola").replace("ckks.negate", "ckks.negate")
    body = t.replace("= ckks.add_plain", "= ckks.frobnicate", 1)
    (tmp_path / "x.mlir").write_text(body)
    with pytest.raises(ValueError):
        compile_ir(tmp_path / "x.mlir")
