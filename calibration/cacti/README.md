# CACTI 7 scratchpad sweep (the SRAM area calibration)

`run_cacti.py` runs [CACTI 7](https://github.com/HewlettPackard/cacti) (commit `1ffd8dfb`, built with `make`) on
banked SRAM scratchpads of 64 MiB to 2 GiB, 4 MiB per bank, at 22 nm, with low-standby-power (`itrs-lstp`) and
high-performance (`itrs-hp`) cells. `configs/` holds every configuration file, `out/` CACTI's full output and
`out/results.json` the extracted area, read energy, leakage and access time.

`src/fhe_sim/ppa.py` copies the `itrs-lstp` area per MiB into `CACTI_LSTP_22NM` (`tests/test_ppa.py` checks the
copy) and scales it to 7 nm with one factor anchored on ARK's published 512 MB scratchpad.

* CACTI's results are model outputs from ITRS-based technology data, not silicon.
* CACTI 7 rejects a 4 GiB array (its size field overflows), so the model holds the 2 GiB density above 2 GiB.
* With `itrs-hp` cells a 512 MiB scratchpad would leak about 180 W (against 0.14 W with `itrs-lstp`), which is why
  the low-standby-power curve is used.

```bash
CACTI=~/.local/opt/cacti python calibration/cacti/run_cacti.py    # about 30 s
```
