import time, json, sys
from openfhe import *
def run(logN, depth, dnum):
    p = CCParamsCKKSRNS(); p.SetSecurityLevel(HEStd_NotSet); p.SetRingDim(1<<logN)
    p.SetMultiplicativeDepth(depth); p.SetNumLargeDigits(dnum); p.SetKeySwitchTechnique(HYBRID)
    p.SetScalingModSize(50); p.SetFirstModSize(60); p.SetScalingTechnique(FIXEDMANUAL)
    p.SetBatchSize(1<<(logN-1))
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE): cc.Enable(f)
    k = cc.KeyGen(); cc.EvalMultKeyGen(k.secretKey); cc.EvalRotateKeyGen(k.secretKey,[1])
    x=[0.1]*(1<<(logN-1)); ct=cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(x))
    out={"logN":logN,"limbs":depth+1,"dnum":dnum,"log2Q":__import__("math").log2(cc.GetModulus())}
    for name,fn in (("hmult_relin",lambda: cc.EvalMult(ct,ct)),("hrot",lambda: cc.EvalRotate(ct,1)),
                    ("hadd",lambda: cc.EvalAdd(ct,ct)),("rescale",lambda: cc.Rescale(cc.EvalMult(ct,ct)))):
        fn(); n=5; t=time.perf_counter()
        for _ in range(n): fn()
        out[name+"_ms"]=(time.perf_counter()-t)/n*1e3
    out["rescale_ms"]-=out["hmult_relin_ms"]
    print(json.dumps(out), flush=True)
run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
