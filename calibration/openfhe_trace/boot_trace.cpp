// Records the kernel stream of one OpenFHE CKKS bootstrap, using the fhetrace instrumentation
// (see openfhe_fhetrace.patch). Usage:
//   FHETRACE=out.log OMP_NUM_THREADS=1 ./boot_trace <logN> <log2 slots> <lb_enc> <lb_dec> <dnum> <levels_after> <secure 0|1>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "openfhe.h"
#include "utils/fhetrace.h"
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"

using namespace lbcrypto;

int main(int argc, char** argv) {
    if (argc != 8) { std::cerr << "usage: boot_trace logN logSlots lbEnc lbDec dnum levelsAfter secure\n"; return 2; }
    uint32_t logN = std::atoi(argv[1]), logSlots = std::atoi(argv[2]);
    std::vector<uint32_t> lb = {(uint32_t)std::atoi(argv[3]), (uint32_t)std::atoi(argv[4])};
    uint32_t dnum = std::atoi(argv[5]), after = std::atoi(argv[6]);
    bool secure = std::atoi(argv[7]) != 0;
    uint32_t slots = 1u << logSlots;

    CCParams<CryptoContextCKKSRNS> p;
    SecretKeyDist sk = UNIFORM_TERNARY;
    p.SetSecretKeyDist(sk);
    p.SetSecurityLevel(secure ? HEStd_128_classic : HEStd_NotSet);
    if (!secure) p.SetRingDim(1u << logN);
    p.SetNumLargeDigits(dnum);
    p.SetKeySwitchTechnique(HYBRID);
    p.SetScalingModSize(59);
    p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(lb, sk);
    uint32_t depth = after + bootDepth;
    p.SetMultiplicativeDepth(depth);
    p.SetBatchSize(slots);
    auto cc = GenCryptoContext(p);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    cc->EvalBootstrapSetup(lb, {0, 0}, slots);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    std::vector<double> x(slots);
    for (uint32_t i = 0; i < slots; ++i) x[i] = 0.25 * ((int)(i % 7) - 3) / 3.0;
    auto pt = cc->MakeCKKSPackedPlaintext(x, 1, depth - 1, nullptr, slots);
    auto ct = cc->Encrypt(keys.publicKey, pt);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto relin = cc->GetEvalMultKeyVector(keys.secretKey->GetKeyTag())[0];
    setenv("FHETRACE_ON", "1", 1);
    fhetrace::emit("H N=%u slots=%u depth=%u bootDepth=%u dnum=%u alpha=%u sizeP=%u towersQ=%u in_towers=%u",
                   cc->GetRingDimension(), slots, depth, bootDepth, dnum, cp->GetNumPerPartQ(),
                   (unsigned)cp->GetParamsP()->GetParams().size(),
                   (unsigned)cp->GetElementParams()->GetParams().size(), (unsigned)ct->GetElements()[0].GetNumOfElements());
    fhetrace::emit("RELIN %p", static_cast<const void*>(relin.get()));
    auto out = cc->EvalBootstrap(ct);
    fhetrace::emit("E out_towers=%u", (unsigned)out->GetElements()[0].GetNumOfElements());
    unsetenv("FHETRACE_ON");

    Plaintext dec;
    cc->Decrypt(keys.secretKey, out, &dec);
    dec->SetLength(slots);
    auto v = dec->GetRealPackedValue();
    double err = 0;
    for (uint32_t i = 0; i < slots; ++i) err = std::max(err, std::abs(v[i] - x[i]));
    std::cout << "N=" << cc->GetRingDimension() << " slots=" << slots << " depth=" << depth
              << " max_err=" << err << std::endl;
    return 0;
}
