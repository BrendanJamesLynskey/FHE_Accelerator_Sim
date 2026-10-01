"""Level 1 of the verification ladder: sizes and operation counts against hand formulas.

* parameter sizes against hand formulas and ARK's published Table III
* the trace generator's operation counts against closed-form expressions for
  each bootstrapping stage (independently derived here, not imported)
"""

import math

import pytest

from fhe_sim import PARAMS, BootOptions, bootstrap_trace, he_op_trace, summarise_trace
from fhe_sim.params import MiB, CKKSParams, cdiv
from fhe_sim.workload import dft_split, dump_trace, load_trace, output_level

ARK, LATTIGO = PARAMS["ark"], PARAMS["lattigo"]


# ── sizes ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("p,ct_mib,pt_mib,evk_mib", [
    (ARK, 24, 12, 120),          # ARK Table III: [[m]] 24 MB, Pm 12 MB, evk 120 MB (MiB)
    (LATTIGO, 25, 12.5, 150),    # Lattigo row: 25, 12.5, 150
])
def test_sizes_match_ark_table_iii(p, ct_mib, pt_mib, evk_mib):
    assert p.ct_bytes(p.L) == ct_mib * MiB
    assert p.pt_bytes(p.L) == pt_mib * MiB
    assert p.evk_bytes() == evk_mib * MiB


def test_hand_formulas():
    p = ARK                                   # N = 65536, L = 23, dnum = 4
    assert p.N == 65536 and p.slots == 32768
    assert p.alpha == p.k == 6                # ceil(24 / 4)
    assert p.beta(23) == 4 and p.beta(11) == 2 and p.beta(5) == 1
    assert p.digit_sizes(13) == [6, 6, 2]     # 14 limbs in digits of 6
    assert p.ct_bytes(10) == 2 * 65536 * 11 * 8
    # a key used at level 10 loads beta = 2 digits of (11 + 6) limbs, two polynomials each
    assert p.evk_bytes(10) == 2 * 2 * 65536 * 17 * 8
    assert p.modup_bytes(10) == 2 * 65536 * 17 * 8
    assert p.ks_working_set() == (4 * 30 + 2 * 30) * 65536 * 8


def test_evalmod_shape():
    assert (ARK.evalmod_baby, ARK.evalmod_giant) == (8, 8)                  # degree 59
    assert (PARAMS["openfhe-sparse"].evalmod_baby, PARAMS["openfhe-sparse"].evalmod_giant) == (16, 8)
    assert (PARAMS["small"].evalmod_baby, PARAMS["small"].evalmod_giant) == (4, 4)


def test_dft_split():
    assert dft_split(15, 3) == [5, 5, 5]
    assert dft_split(14, 3) == [5, 5, 4]
    assert dft_split(3, 1) == [3]


# ── single HE operations ─────────────────────────────────────────────────
def ks_ntt_limbs(p, lev):
    """Hand count of transforms in one hybrid key switch at level lev."""
    l, k, a = lev + 1, p.k, p.alpha
    digits = [min(a, l - i * a) for i in range(cdiv(l, a))]
    return l + sum(l + k - d for d in digits) + 2 * k + 2 * l


def test_hmult_and_hrot_kernel_counts():
    p, lev = ARK, ARK.L
    l = lev + 1
    hm = summarise_trace(he_op_trace(p, "hmult"))["hmult"]
    rescale = 2 + 2 * (l - 1)
    assert hm["ntt_limbs"] + hm["intt_limbs"] == ks_ntt_limbs(p, lev) + rescale == 228
    hr = summarise_trace(he_op_trace(p, "hrot"))["hrot"]
    assert hr["ntt_limbs"] + hr["intt_limbs"] == ks_ntt_limbs(p, lev)
    assert hr["auto_words"] == 2 * l * p.N
    # base conversion: ModUp each digit into the other l + k - a limbs, plus ModDown of both polys
    N, k = p.N, p.k
    modup = sum(N * 6 * (l + k - 6) for _ in range(4)) + N * l
    moddown = 2 * (N * k * l + N * k + N * l)
    assert hr["bconv"] == modup + moddown
    assert hr["key_bytes"] == p.evk_bytes(lev)


# ── bootstrapping stages ─────────────────────────────────────────────────
def dft_formulas(log_slots, levels):
    rots = pmults = keys = 0
    for k in dft_split(log_slots, levels):
        d = 2 ** (k + 1) - 1
        n1 = min(2 ** math.ceil((k + 1) / 2), d)
        n2 = math.ceil(d / n1)
        rots += (n1 - 1) + (n2 - 1)
        pmults += d
        keys += (n1 - 1) + (n2 - 1)
    return rots, pmults, keys


@pytest.mark.parametrize("name", ["ark", "lattigo", "small", "openfhe-sparse"])
def test_bootstrap_counts_match_closed_forms(name):
    p = PARAMS[name]
    s = summarise_trace(bootstrap_trace(p))
    rots, pm, keys = dft_formulas(p.slots_log, p.cts_levels)
    conj = 1
    assert s["cts"]["hrot"] == rots + conj
    assert s["cts"]["pmult"] == pm
    assert s["cts"]["distinct_keys"] == keys + conj
    rots, pm, keys = dft_formulas(p.slots_log, p.stc_levels)
    assert (s["stc"]["hrot"], s["stc"]["pmult"], s["stc"]["distinct_keys"]) == (rots, pm, keys)
    b, g, r = p.evalmod_baby, p.evalmod_giant, p.double_angle
    m = math.ceil(math.log2(g))
    copies = 2 if p.full_slots else 1
    assert s["evalmod"]["hmult"] == copies * ((b - 1) + (m - 1) + (g - 1) + r)
    subsum = p.log_n - 1 - p.slots_log
    assert s["modraise"]["hrot"] == subsum


@pytest.mark.parametrize("name", ["ark", "lattigo", "gpu100x", "small", "openfhe-sparse"])
def test_levels_consumed(name):
    p = PARAMS[name]
    b, g, r = p.evalmod_baby, p.evalmod_giant, p.double_angle
    evalmod = 2 + math.ceil(math.log2(b - 1)) + math.ceil(math.log2(g)) + r
    assert p.L - output_level(bootstrap_trace(p)) == p.cts_levels + evalmod + p.stc_levels


def test_not_enough_levels_is_an_error():
    with pytest.raises(ValueError):
        bootstrap_trace(CKKSParams("tiny", log_n=12, L=8, dnum=3))


def test_acceleration_options_change_counts_as_expected():
    p = ARK
    base = summarise_trace(bootstrap_trace(p))
    mk = summarise_trace(bootstrap_trace(p, BootOptions(min_ks=True)))
    assert mk["cts"]["distinct_keys"] == 2 * p.cts_levels + 1          # one baby and one giant key per level
    assert mk["cts"]["hrot"] == base["cts"]["hrot"]                     # same rotations, fewer keys
    sk = summarise_trace(bootstrap_trace(p, BootOptions(seeded_keys=True)))
    assert sk["cts"]["key_bytes"] * 2 == base["cts"]["key_bytes"]
    otf = summarise_trace(bootstrap_trace(p, BootOptions(otf_plaintexts=True)))
    assert otf["cts"]["pt_bytes"] == base["cts"]["pmult"] * p.N * 8
    nh = summarise_trace(bootstrap_trace(p, BootOptions(hoisting=False)))
    assert nh["cts"]["ntt_limbs"] > base["cts"]["ntt_limbs"]           # every rotation pays its own ModUp


def test_trace_json_round_trip(tmp_path):
    t = bootstrap_trace(PARAMS["small"], BootOptions(n_boot=2))
    dump_trace(t, tmp_path / "t.json")
    back = load_trace(tmp_path / "t.json")
    assert summarise_trace(back) == summarise_trace(t)
    assert [o.inputs for o in back.ops] == [o.inputs for o in t.ops]
    assert back.levels == t.levels
