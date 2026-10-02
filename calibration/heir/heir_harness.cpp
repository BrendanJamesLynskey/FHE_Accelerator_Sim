// Runs a HEIR-generated OpenFHE program (lola) on the fhetrace-instrumented OpenFHE and records
// the kernel stream of its server computation. Build against the patched OpenFHE (see
// ../openfhe_trace/README.md) with the generated lola_lib.{h,cpp} (see README.md here).
#include <cstdlib>
#include <iostream>
#include "lola_lib.h"
#include "utils/fhetrace.h"
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"

int main() {
    auto cc = lola__generate_crypto_context();
    auto kp = cc->KeyGen();
    cc = lola__configure_crypto_context(cc, kp.secretKey);
    std::vector<float> image(784);
    for (size_t i = 0; i < image.size(); ++i) image[i] = 0.5f * static_cast<float>((i * 37) % 11) / 10.0f;
    auto ct = lola__encrypt__arg0(cc, image, kp.publicKey);
    auto pts = lola__preprocessing(cc);
    auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto relin = cc->GetEvalMultKeyVector(kp.secretKey->GetKeyTag())[0];
    setenv("FHETRACE_ON", "1", 1);
    fhetrace::emit("H N=%u slots=%u depth=%u bootDepth=0 dnum=%u alpha=%u sizeP=%u towersQ=%u in_towers=%u",
                   cc->GetRingDimension(), cc->GetRingDimension() / 2, (unsigned)(cp->GetElementParams()->GetParams().size() - 1),
                   cp->GetNumPartQ(), cp->GetNumPerPartQ(), (unsigned)cp->GetParamsP()->GetParams().size(),
                   (unsigned)cp->GetElementParams()->GetParams().size(), (unsigned)ct[0]->GetElements()[0].GetNumOfElements());
    fhetrace::emit("RELIN %p", static_cast<const void*>(relin.get()));
    fhetrace::emit("S app");
    auto out = lola__preprocessed(cc, ct, pts);
    fhetrace::emit("E out_towers=%u", (unsigned)out[0]->GetElements()[0].GetNumOfElements());
    unsetenv("FHETRACE_ON");
    auto res = lola__decrypt__result0(cc, out, kp.secretKey);
    std::cout << "N=" << cc->GetRingDimension() << " logits:";
    for (auto v : res) std::cout << " " << v;
    std::cout << std::endl;
    return 0;
}
