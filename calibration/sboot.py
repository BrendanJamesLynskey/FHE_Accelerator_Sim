import time, json, sys
from openfhe import *
slots=int(sys.argv[1]); lb=[int(sys.argv[2]),int(sys.argv[3])]; dnum=int(sys.argv[4])
p=CCParamsCKKSRNS(); sk=SecretKeyDist.UNIFORM_TERNARY; p.SetSecretKeyDist(sk)
p.SetSecurityLevel(HEStd_128_classic); p.SetNumLargeDigits(dnum); p.SetKeySwitchTechnique(HYBRID)
p.SetScalingModSize(59); p.SetFirstModSize(60); p.SetScalingTechnique(FLEXIBLEAUTO)
bd=FHECKKSRNS.GetBootstrapDepth(lb, sk); depth=int(sys.argv[5])+bd; p.SetMultiplicativeDepth(depth); p.SetBatchSize(slots)
cc=GenCryptoContext(p); print("N",cc.GetRingDimension(),"depth",depth,"bd",bd,flush=True)
for f in (PKESchemeFeature.PKE,PKESchemeFeature.KEYSWITCH,PKESchemeFeature.LEVELEDSHE,PKESchemeFeature.ADVANCEDSHE,PKESchemeFeature.FHE): cc.Enable(f)
cc.EvalBootstrapSetup(lb,[0,0],slots); k=cc.KeyGen(); cc.EvalMultKeyGen(k.secretKey); cc.EvalBootstrapKeyGen(k.secretKey,slots)
x=[0.5*((i%5)-2)/2 for i in range(slots)]
ct=cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(x,1,depth-1,None,slots))
ts=[]
for _ in range(3):
    t=time.perf_counter(); o=cc.EvalBootstrap(ct); ts.append(time.perf_counter()-t)
d=cc.Decrypt(o,k.secretKey); d.SetLength(slots); v=d.GetRealPackedValue()
print(json.dumps({"N":cc.GetRingDimension(),"slots":slots,"levelBudget":lb,"bootDepth":bd,"depth":depth,"dnum":dnum,
  "boot_s":ts,"maxerr":max(abs(a-b) for a,b in zip(v,x)),"out_level":o.GetLevel()}),flush=True)
