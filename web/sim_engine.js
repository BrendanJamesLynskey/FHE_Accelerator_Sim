// fhe-sim, JavaScript port of FHE_Accelerator_Sim (SimPy).
// Same parameter sets, the same bootstrap trace generator, cost model, scratchpad,
// event engine and metrics. A ~100-line SimPy core (events, processes as generators,
// Resource with SimPy's request/release ordering) stands in for SimPy, so the results
// match the Python package bit for bit (tests/test_simulator.py checks this).
(function (root) {
    'use strict';
    const MiB = 1 << 20, WORD = 8;
    const cdiv = (a, b) => Math.ceil(a / b);
    const ceilLog2 = x => (x > 1 ? (x - 1).toString(2).length : 0);
    const log2Int = x => x.toString(2).length - 1;

    // ── parameters (params.py) ───────────────────────────────────────
    function mkParams(o) {
        const p = Object.assign({ qBits: 50, logSlots: null, ctsLevels: 3, stcLevels: 3, evalmodDegree: 59, doubleAngle: 2 }, o);
        p.N = 2 ** p.logN;
        p.slotsLog = p.logSlots === null ? p.logN - 1 : p.logSlots;
        p.fullSlots = p.slotsLog === p.logN - 1;
        p.alpha = cdiv(p.L + 1, p.dnum);
        p.k = p.alpha;
        p.beta = lev => cdiv(lev + 1, p.alpha);
        p.digitSizes = lev => { const l = lev + 1, out = []; for (let i = 0; i < p.beta(lev); i++) out.push(Math.min(p.alpha, l - i * p.alpha)); return out; };
        p.ctBytes = lev => 2 * p.N * (lev + 1) * WORD;
        p.ptBytes = lev => p.N * (lev + 1) * WORD;
        p.evkBytes = lev => (lev === undefined || lev === null) ? 2 * p.dnum * p.N * (p.L + 1 + p.k) * WORD : 2 * p.beta(lev) * p.N * (lev + 1 + p.k) * WORD;
        p.modupBytes = lev => p.beta(lev) * p.N * (lev + 1 + p.k) * WORD;
        p.ksWorkingSet = () => { const lk = p.L + 1 + p.k; return (p.beta(p.L) * lk + 2 * lk) * p.N * WORD; };
        p.logPQ = () => 60 + p.L * p.qBits + 60 * p.k;
        let b = 1; while (b * b < p.evalmodDegree + 1) b *= 2;
        p.evalmodBaby = b;
        p.evalmodGiant = cdiv(p.evalmodDegree + 1, b);
        return p;
    }
    const PARAMS = {
        'ark': { name: 'ARK-like (N=2^16, L=23, dnum=4)', logN: 16, L: 23, dnum: 4 },
        'lattigo': { name: 'Lattigo-like (N=2^16, L=24, dnum=5)', logN: 16, L: 24, dnum: 5 },
        'gpu100x': { name: '100x GPU (N=2^16, L=34, dnum=5)', logN: 16, L: 34, dnum: 5 },
        'openfhe-sparse': { name: 'OpenFHE sparse (N=2^16, L=18, dnum=3, 8 slots)', logN: 16, L: 18, dnum: 3, qBits: 59,
                            logSlots: 3, ctsLevels: 1, stcLevels: 1, evalmodDegree: 88, doubleAngle: 6 },
        'openfhe-full14': { name: 'OpenFHE full slots (N=2^14, L=30, dnum=3)', logN: 14, L: 30, dnum: 3, qBits: 59,
                            ctsLevels: 3, stcLevels: 3, evalmodDegree: 88, doubleAngle: 6 },
        'small': { name: 'small test set (N=2^12, L=11, dnum=3)', logN: 12, L: 11, dnum: 3, ctsLevels: 2, stcLevels: 2,
                   evalmodDegree: 15, doubleAngle: 1 },
    };

    // ── workload (workload.py) ───────────────────────────────────────
    const STAGES = ['modraise', 'cts', 'evalmod', 'stc', 'app', 'app_post'];
    const K = (kind, amount, words) => ({ kind, amount, words });
    const kNtt = (limbs, N, inv) => K(inv ? 'intt' : 'ntt', limbs, 2 * limbs * N);
    const kMac = ops => K('mac', ops, 3 * ops);
    function ksModup(p, lev) {
        const N = p.N, l = lev + 1, k = p.k, ds = p.digitSizes(lev);
        let newLimbs = 0, bconv = 0;
        for (const a of ds) newLimbs += l + k - a;
        for (const a of ds) bconv += N * a * (l + k - a);
        bconv += N * l;
        return [kNtt(l, N, true), K('bconv', bconv, N * (l + newLimbs)), kNtt(newLimbs, N)];
    }
    function ksTail(p, lev) {
        const N = p.N, l = lev + 1, k = p.k, b = p.beta(lev);
        const inner = 2 * b * (l + k) * N, moddown = 2 * (N * k * l + N * k + N * l);
        return [K('mac', inner, 3 * b * (l + k) * N + 2 * (l + k) * N), kNtt(2 * k, N, true),
                K('bconv', moddown, 4 * N * (k + l)), kNtt(2 * l, N)];
    }
    function rescaleKernels(p, lev) {
        const N = p.N, l = lev + 1;
        return [kNtt(2, N, true), kNtt(2 * (l - 1), N), kMac(2 * (l - 1) * N)];
    }
    function dftSplit(logSlots, n) {
        const base = Math.floor(logSlots / n), rem = logSlots % n, out = [];
        for (let j = 0; j < n; j++) out.push(j < rem ? base + 1 : base);
        return out;
    }

    function Builder(p, o) {
        const B = { p, o, ops: [], sizes: {}, levels: {}, external: [], stage: 'modraise', boot: 0, n: 0 };
        B.obj = (level, nbytes) => { const name = `b${B.boot}.c${B.n}`; B.n++; B.sizes[name] = nbytes; B.levels[name] = level; return name; };
        B.externalCt = level => { const name = B.obj(level, p.ctBytes(level)); B.external.push(name); return name; };
        B.key = (kid, level) => { const b = p.evkBytes(level); return [kid, o.seededKeys ? Math.floor(b / 2) : b]; };
        B.seedKernels = level => o.seededKeys ? [kMac(p.beta(level) * (level + 1 + p.k) * p.N)] : [];
        B.emit = (op, level, inputs, outLevel, kernels, key, pts, outBytes) => {
            const out = B.obj(outLevel, outBytes === undefined ? p.ctBytes(outLevel) : outBytes);
            B.ops.push({ id: B.ops.length, op, stage: B.stage, level, inputs: inputs.slice(), output: out, kernels,
                         key: key || null, pts: pts || [], boot: B.boot });
            return out;
        };
        B.lvl = (...xs) => Math.min(...xs.map(x => B.levels[x]));
        B.hmult = (a, b, extra, postAdds) => {
            extra = extra || []; postAdds = postAdds === undefined ? 1 : postAdds;
            const lev = B.lvl(a, b), N = p.N, l = lev + 1;
            const ks = [kMac(4 * l * N), ...ksModup(p, lev), ...B.seedKernels(lev), ...ksTail(p, lev),
                        kMac(2 * l * N * postAdds), ...rescaleKernels(p, lev)];
            return B.emit('hmult', lev, [a, b, ...extra], lev - 1, ks, B.key('relin', lev));
        };
        B.hrot = (x, kid) => {
            const lev = B.lvl(x), N = p.N, l = lev + 1;
            const ks = [K('auto', 2 * l * N, 4 * l * N), ...ksModup(p, lev), ...B.seedKernels(lev), ...ksTail(p, lev)];
            return B.emit('hrot', lev, [x], lev, ks, B.key(kid, lev));
        };
        B.modup = x => { const lev = B.lvl(x); return B.emit('modup', lev, [x], lev, ksModup(p, lev), null, [], p.modupBytes(lev)); };
        B.hrotHoisted = (x, digits, kid) => {
            const lev = B.lvl(x), N = p.N, l = lev + 1;
            const w = (l + p.beta(lev) * (l + p.k)) * N;
            const ks = [K('auto', w, 2 * w), ...B.seedKernels(lev), ...ksTail(p, lev)];
            return B.emit('hrot', lev, [x, digits], lev, ks, B.key(kid, lev));
        };
        B.extend = x => {
            const lev = B.lvl(x), N = p.N, l = lev + 1;
            return B.emit('extend', lev, [x], lev, [kMac(2 * l * N)], null, [], 2 * (l + p.k) * N * 8);
        };
        B.hrotHoistedExt = (x, digits, kid) => {
            const lev = B.lvl(x), N = p.N, l = lev + 1, k = p.k, b = p.beta(lev);
            const w = (l + b * (l + k)) * N;
            const ks = [K('auto', w, 2 * w), ...B.seedKernels(lev), ...ksTail(p, lev).slice(0, 1)];
            return B.emit('hrot', lev, [x, digits], lev, ks, B.key(kid, lev), [], 2 * (l + k) * N * 8);
        };
        B.pmacExt = (xs, ptIds, moddown) => {
            const lev = B.lvl(...xs), N = p.N, l = lev + 1, m = xs.length, lk = l + p.k;
            let kern = [kMac(2 * lk * N * m + 2 * lk * N * (m - 1))], pts;
            if (o.otfPlaintexts) { kern = [kNtt(lk * m, N), kMac(lk * N * m), ...kern]; pts = ptIds.map(id => [id, N * 8]); }
            else pts = ptIds.map(id => [id, lk * N * 8]);
            const down = moddown ? ksTail(p, lev).slice(1) : [];
            return B.emit('pmac', lev, xs, lev, [...kern, ...down], null, pts, moddown ? undefined : 2 * lk * N * 8);
        };
        B.hrotExt = (x, kid) => {
            const lev = B.lvl(x), N = p.N, l = lev + 1;
            const ks = [K('auto', 2 * l * N, 4 * l * N), ...ksModup(p, lev), ...B.seedKernels(lev), ...ksTail(p, lev).slice(0, 1)];
            return B.emit('hrot', lev, [x], lev, ks, B.key(kid, lev), [], 2 * (l + p.k) * N * 8);
        };
        B.addExt = xs => {
            const lev = B.lvl(...xs), N = p.N, lk = lev + 1 + p.k;
            const kern = [kMac(2 * lk * N * Math.max(1, xs.length - 1)), ...ksTail(p, lev).slice(1), ...rescaleKernels(p, lev)];
            return B.emit('add', lev, xs, lev - 1, kern);
        };
        B.pmac = (xs, ptIds) => {
            const lev = B.lvl(...xs), N = p.N, l = lev + 1, m = xs.length;
            let kern = [kMac(2 * l * N * m + 2 * l * N * (m - 1))], pts;
            if (o.otfPlaintexts) { kern = [kNtt(l * m, N), kMac(l * N * m), ...kern]; pts = ptIds.map(id => [id, N * 8]); }
            else pts = ptIds.map(id => [id, p.ptBytes(lev)]);
            return B.emit('pmac', lev, xs, lev, kern, null, pts);
        };
        B.add = (xs, rescale) => {
            const lev = B.lvl(...xs), N = p.N, l = lev + 1;
            let kern = [kMac(2 * l * N * Math.max(1, xs.length - 1))];
            if (rescale) kern = kern.concat(rescaleKernels(p, lev));
            return B.emit('add', lev, xs, rescale ? lev - 1 : lev, kern);
        };
        B.cmult = xs => {
            const lev = B.lvl(...xs), N = p.N, l = lev + 1;
            return B.emit('cmult', lev, xs, lev - 1, [kMac(2 * l * N * xs.length + 2 * l * N), ...rescaleKernels(p, lev)]);
        };
        B.modraise = x => {
            let y = B.emit('modraise', 0, [x], p.L, [kNtt(2, p.N, true), kNtt(2 * (p.L + 1), p.N)]);
            for (let i = 0; i < p.logN - 1 - p.slotsLog; i++) y = B.add([y, B.hrot(y, `sub${i}`)]);
            return y;
        };
        B.dft = (x, prefix, nLevels) => {
            const split = dftSplit(p.slotsLog, nLevels);
            for (let j = 0; j < split.length; j++) {
                const k = split[j], d = Math.min(2 ** (k + 1) - 1, 2 ** p.slotsLog), tag = `${prefix}${j}`;
                const lazy = o.lazyModdown && o.hoisting && !o.minKs;
                let n1;
                if (lazy && d === 2 ** p.slotsLog) {
                    const sl = 2 ** p.slotsLog, c = Math.floor(Math.sqrt(sl - 1)) + 1;     // ceil(sqrt(slots)), exact for these sizes
                    n1 = sl > 1 ? Math.min(2 ** (c.toString(2).length - 1 + 1), d) : 1;
                } else if (lazy) n1 = Math.min(2 ** (Math.floor(k / 2) + 1 + (d > 7 ? 1 : 0)), d);
                else n1 = Math.min(2 ** cdiv(k + 1, 2), d);
                const n2 = cdiv(d, n1);
                const babies = [x];
                if (n1 > 1) {
                    if (o.minKs) { for (let i = 1; i < n1; i++) babies.push(B.hrot(babies[babies.length - 1], `${tag}.b`)); }
                    else if (o.hoisting && o.lazyModdown) {
                        const dig = B.modup(x); babies[0] = B.extend(x);
                        for (let i = 1; i < n1; i++) babies.push(B.hrotHoistedExt(x, dig, `${tag}.b${i}`));
                    }
                    else if (o.hoisting) { const dig = B.modup(x); for (let i = 1; i < n1; i++) babies.push(B.hrotHoisted(x, dig, `${tag}.b${i}`)); }
                    else { for (let i = 1; i < n1; i++) babies.push(B.hrot(x, `${tag}.b${i}`)); }
                }
                const inners = [];
                for (let g = 0; g < n2; g++) {
                    const m = Math.min(n1, d - g * n1), ids = [];
                    for (let i = 0; i < m; i++) ids.push(`${tag}.d${g}.${i}`);
                    inners.push(lazy && n1 > 1 ? B.pmacExt(babies.slice(0, m), ids, g > 0) : B.pmac(babies.slice(0, m), ids));
                }
                if (o.minKs) {
                    let acc = inners[inners.length - 1];
                    for (let g = n2 - 2; g >= 0; g--) acc = B.add([B.hrot(acc, `${tag}.g`), inners[g]], g === 0);
                    x = n2 > 1 ? acc : B.add([acc], true);
                } else if (lazy && n1 > 1) {
                    const parts = [inners[0]];
                    for (let g = 1; g < n2; g++) parts.push(B.hrotExt(inners[g], `${tag}.g${g}`));
                    x = B.addExt(parts);
                } else {
                    const parts = [inners[0]];
                    for (let g = 1; g < n2; g++) parts.push(B.hrot(inners[g], `${tag}.g${g}`));
                    x = B.add(parts, true);
                }
            }
            return x;
        };
        B.evalmod = x => {
            const b = p.evalmodBaby, g = p.evalmodGiant;
            const T = { 1: B.cmult([x]) };
            for (let i = 2; i <= b; i++) T[i] = B.hmult(T[Math.floor((i + 1) / 2)], T[Math.floor(i / 2)]);
            const m = ceilLog2(g), G = [T[b]];
            for (let t = 1; t < m; t++) G.push(B.hmult(G[G.length - 1], G[G.length - 1]));
            let nodes = [];
            for (let t = 0; t < g; t++) { const ts = []; for (let i = 1; i < b; i++) ts.push(T[i]); nodes.push(B.cmult(ts)); }
            let t = 0;
            while (nodes.length > 1) {
                const nxt = [];
                for (let s = 0; s < nodes.length - 1; s += 2) nxt.push(B.hmult(nodes[s + 1], G[t], [nodes[s]]));
                if (nodes.length % 2) nxt.push(nodes[nodes.length - 1]);
                nodes = nxt; t++;
            }
            let y = nodes[0];
            for (let r = 0; r < p.doubleAngle; r++) y = B.hmult(y, y);
            return y;
        };
        B.bootstrapStcFirst = x => {
            if (B.levels[x] < p.stcLevels) throw new Error(`StC-first needs the input at level >= ${p.stcLevels}`);
            B.stage = 'stc'; x = B.dft(x, 'stc', p.stcLevels);
            if (!p.fullSlots) x = B.add([x, B.hrot(x, 'stcfirst.rep')]);
            B.stage = 'modraise'; x = B.modraise(x);
            B.stage = 'cts'; x = B.dft(x, 'cts', p.ctsLevels);
            x = B.add([x, B.hrot(x, 'conj')]);
            B.stage = 'evalmod'; x = B.evalmod(x);
            if (B.levels[x] < 0) throw new Error(`${p.name}: bootstrapping needs more than L = ${p.L} levels`);
            return x;
        };
        B.bootstrap = x => {
            if (o.stcFirst) return B.bootstrapStcFirst(x);
            B.stage = 'modraise'; x = B.modraise(x);
            B.stage = 'cts'; x = B.dft(x, 'cts', p.ctsLevels);
            const c = B.hrot(x, 'conj');
            let parts = p.fullSlots ? [B.add([x, c]), B.add([x, c])] : [B.add([x, c])];
            B.stage = 'evalmod'; parts = parts.map(y => B.evalmod(y));
            x = parts.length > 1 ? B.add(parts) : parts[0];
            B.stage = 'stc'; x = B.dft(x, 'stc', p.stcLevels);
            if (B.levels[x] < 0) throw new Error(`${p.name}: bootstrapping needs more than L = ${p.L} levels`);
            return x;
        };
        return B;
    }
    function bootOptions(o) {
        return Object.assign({ nBoot: 1, hoisting: true, minKs: false, seededKeys: false, otfPlaintexts: false, lazyModdown: false, stcFirst: false }, o || {});
    }
    function bootstrapTrace(p, opts) {
        const o = bootOptions(opts), B = Builder(p, o);
        for (let i = 0; i < o.nBoot; i++) { B.boot = i; B.bootstrap(B.externalCt(o.stcFirst ? p.stcLevels : 0)); }
        return { params: p, ops: B.ops, sizes: B.sizes, external: B.external, levels: B.levels, options: o };
    }
    function heOpTrace(p, op, level, n) {
        const B = Builder(p, bootOptions({})), lev = level === undefined || level === null ? p.L : level;
        B.stage = op;
        for (let i = 0; i < (n || 1); i++) {
            B.boot = i;
            const x = B.externalCt(lev), y = B.externalCt(lev);
            if (op === 'hmult') B.hmult(x, y); else B.hrot(x, 'rot1');
        }
        return { params: p, ops: B.ops, sizes: B.sizes, external: B.external, levels: B.levels, options: B.o };
    }
    function summariseTrace(trace) {
        const out = {};
        for (const o of trace.ops) {
            const s = out[o.stage] || (out[o.stage] = { hmult: 0, hrot: 0, pmult: 0, ops: 0, ntt_limbs: 0, intt_limbs: 0, bconv: 0, mac: 0, auto_words: 0, keys: new Set(), key_bytes: 0, pt_bytes: 0 });
            s.ops++;
            if (o.op === 'hmult' || o.op === 'hrot') s[o.op]++;
            if (o.op === 'pmac') s.pmult += o.pts.length;
            for (const k of o.kernels) {
                if (k.kind === 'ntt') s.ntt_limbs += k.amount; else if (k.kind === 'intt') s.intt_limbs += k.amount;
                else if (k.kind === 'auto') s.auto_words += k.amount; else s[k.kind] += k.amount;
            }
            if (o.key) { s.keys.add(o.key[0]); s.key_bytes += o.key[1]; }
            for (const t of o.pts) s.pt_bytes += t[1];
        }
        for (const s of Object.values(out)) { s.distinct_keys = s.keys.size; delete s.keys; }
        return out;
    }

    // ── hardware (hardware.py) ───────────────────────────────────────
    const UNITS = ['ntt', 'mac', 'auto', 'optical'];
    const KIND_UNIT = { ntt: 'ntt', intt: 'ntt', bconv: 'mac', mac: 'mac', auto: 'auto' };
    function optical(o) {
        const e = Object.assign({ name: 'Hybrid optical NTT', block: 16, enob: 12, samplesPerS: 1e12, grouping: 'grouped',
                                  fomDacFj: 10, fomAdcFj: 20, laserW: 10, tuningW: 10, ideal: false }, o);
        e.digits = qBits => {
            if (e.ideal) return [qBits, 1];
            for (let b = qBits; b >= 1; b--) {
                const d = cdiv(qBits, b), m = e.grouping === 'grouped' ? d : 1;
                if (2 ** (e.enob - 1) > m * e.block * (2 ** b - 1) ** 2) return [b, d];
            }
            throw new Error(`ENOB ${e.enob} cannot round a ${e.block}-point transform exactly even with 1-bit digits; lower block or raise ENOB`);
        };
        e.planes = qBits => { const d = e.digits(qBits)[1]; if (e.ideal) return [1, 1]; return e.grouping === 'grouped' ? [d, 2 * d - 1] : [d * d, d * d]; };
        e.pjDac = () => e.fomDacFj * 2 ** e.enob * 1e-3;
        e.pjAdc = () => e.fomAdcFj * 2 ** e.enob * 1e-3;
        e.staticW = e.laserW + e.tuningW;
        return e;
    }
    function accelerator(o) {
        const hw = Object.assign({ name: 'Digital FHE accelerator (ARK-class, illustrative)', freqGhz: 1.0, nttBflyPerCycle: 4096,
            macLanes: 8192, autoWordsPerCycle: 4096, sramMib: 512, sramGbps: 20000.0, hbmGbps: 1000.0, hbmChunkMib: 4, window: 4,
            tdpW: 250.0, staticW: 40.0, pjBfly: 10.0, pjMac: 5.0, pjAutoWord: 1.0, pjSramByte: 1.0, pjHbmByte: 30.0, sMin: 0.5,
            enforceTdp: true, powerMode: 'dynamic', hbmMinFrac: 0.25, optical: null }, o);
        hw.sramBytes = hw.sramMib * MiB;
        hw.rate = u => ({ ntt: hw.nttBflyPerCycle, mac: hw.macLanes, auto: hw.autoWordsPerCycle })[u] * hw.freqGhz * 1e9;
        hw.pj = u => ({ ntt: hw.pjBfly, mac: hw.pjMac, auto: hw.pjAutoWord })[u];
        hw.peakPower = s => {
            const sram = hw.sramGbps * 1e9 * hw.pjSramByte * 1e-12;
            let p = hw.staticW + hw.hbmGbps * 1e9 * hw.pjHbmByte * 1e-12;
            for (const u of ['ntt', 'mac', 'auto']) p += hw.rate(u) * s * hw.pj(u) * 1e-12 * s * s + sram * s;
            if (hw.optical) { const op = hw.optical; p += op.staticW + op.samplesPerS * (op.pjDac() + op.pjAdc()) * 1e-12; }
            return p;
        };
        hw.tdpClock = () => {
            if (!hw.enforceTdp || hw.peakPower(1.0) <= hw.tdpW) return 1.0;
            if (hw.peakPower(hw.sMin) > hw.tdpW) throw new Error(`TDP ${hw.tdpW} W is below the power at the lowest clock`);
            let lo = hw.sMin, hi = 1.0;
            for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (hw.peakPower(mid) <= hw.tdpW) lo = mid; else hi = mid; }
            return lo;
        };
        return hw;
    }
    const withHw = (hw, o) => accelerator(Object.assign({}, stripHw(hw), o));
    const stripHw = hw => { const c = {}; for (const [k, v] of Object.entries(hw)) if (typeof v !== 'function' && k !== 'sramBytes') c[k] = v; return c; };
    const SMALL_DIGITAL = { name: 'Small digital accelerator (NTT-bound, illustrative)', nttBflyPerCycle: 512, macLanes: 2048,
                            autoWordsPerCycle: 1024, sramMib: 512, hbmGbps: 2000.0 };
    const ACCELERATORS = {
        'ark': () => accelerator({}),
        'small': () => accelerator(SMALL_DIGITAL),
        'cpu': () => accelerator({ name: 'CPU-like (fitted to OpenFHE on an i7-3770, 8 threads)', freqGhz: 3.4, nttBflyPerCycle: 0.18,
            macLanes: 0.18, autoWordsPerCycle: 0.72, sramMib: 8192, sramGbps: 20.0, hbmGbps: 20.0, window: 1, tdpW: 77.0, staticW: 20.0,
            pjBfly: 2000.0, pjMac: 1000.0, pjAutoWord: 200.0, pjSramByte: 5.0, pjHbmByte: 100.0, enforceTdp: false }),
        'hybrid': () => accelerator(Object.assign({}, SMALL_DIGITAL, { name: 'Small digital + hybrid optical NTT (illustrative)',
            tdpW: 300.0, optical: optical({ samplesPerS: 5e11 }) })),
        'ideal-optical': () => accelerator(Object.assign({}, SMALL_DIGITAL, { name: 'Small digital + ideal optical NTT (hypothetical bound)',
            tdpW: 300.0, optical: optical({ block: 4096, enob: 8, samplesPerS: 5e11, ideal: true }) })),
        'hot': () => accelerator({ nttBflyPerCycle: 16384, macLanes: 32768 }),
    };
    function costModel(hw, logN, qBits, s) {
        let opt = null;
        if (hw.optical) {
            const o = hw.optical;
            if (o.block > 2 ** logN) throw new Error('optical block larger than the ring degree');
            const [b, d] = o.digits(qBits), [dac, adc] = o.planes(qBits);
            opt = [b, d, dac, adc, log2Int(o.block)];
        }
        const N = 2 ** logN;
        const seg = (unit, work, words) => {
            const tLogic = work / (hw.rate(unit) * s), tSram = words * 8 / (hw.sramGbps * 1e9 * s);
            const t = tLogic >= tSram ? tLogic : tSram;
            const el = work * hw.pj(unit) * 1e-12 * s * s, es = words * 8 * hw.pjSramByte * 1e-12;
            return { unit, time: t, energy: el + es, work, dac: 0, adc: 0, eLogic: el, eSram: es, scalable: true };
        };
        return {
            s, staticW: hw.staticW + (hw.optical ? hw.optical.staticW : 0.0),
            segments(k) {
                const unit = KIND_UNIT[k.kind];
                if (unit !== 'ntt') return [seg(unit, k.amount, k.words)];
                if (opt === null) return [seg('ntt', k.amount * (N / 2) * logN, k.words)];
                const [, d, dacPlanes, adcPlanes, optStages] = opt, o = hw.optical, digStages = logN - optStages, segs = [];
                if (digStages) segs.push(seg('ntt', k.amount * (N / 2) * digStages, k.words));
                const blocks = k.amount * (N / o.block);
                const dac = blocks * dacPlanes * 2 * o.block, adc = blocks * adcPlanes * 2 * o.block;
                const t = (dac >= adc ? dac : adc) / o.samplesPerS;
                segs.push({ unit: 'optical', time: t, energy: dac * o.pjDac() * 1e-12 + adc * o.pjAdc() * 1e-12, work: dac + adc, dac, adc, eLogic: 0, eSram: 0, scalable: false });
                const corr = k.amount * N * (o.ideal ? 3 : d + adcPlanes + 3);
                segs.push(seg('mac', corr, 3 * corr));
                return segs;
            },
        };
    }

    // ── a minimal SimPy core ─────────────────────────────────────────
    const PENDING = Symbol('pending'), URGENT = 0, NORMAL = 1;
    class Env {
        constructor() { this.now = 0; this.q = []; this.eid = 0; }
        schedule(ev, prio, delay) {
            const e = [this.now + (delay || 0), prio === undefined ? NORMAL : prio, this.eid++, ev], q = this.q;
            q.push(e);
            let i = q.length - 1;
            while (i > 0) { const pi = (i - 1) >> 1; if (less(q[pi], e)) break; q[i] = q[pi]; i = pi; }
            q[i] = e;
        }
        pop() {
            const q = this.q, top = q[0], last = q.pop();
            if (q.length) {
                let i = 0; const n = q.length;
                for (;;) {
                    const l = 2 * i + 1, r = l + 1; let m = i;
                    if (l < n && less(q[l], m === i ? last : q[m])) m = l;
                    if (r < n && less(q[r], m === i ? last : q[m])) m = r;
                    if (m === i) break;
                    q[i] = q[m]; i = m;
                }
                q[i] = last;
            }
            return top;
        }
        run() {
            while (this.q.length) {
                const [t, , , ev] = this.pop();
                this.now = t;
                const cbs = ev.callbacks; ev.callbacks = null;
                for (const cb of cbs) cb(ev);
            }
        }
        event() { return new Ev(this); }
        timeout(d) { const ev = new Ev(this); ev._ok = true; ev._value = null; this.schedule(ev, NORMAL, d); return ev; }
        process(gen) { return new Proc(this, gen); }
    }
    const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
    class Ev {
        constructor(env) { this.env = env; this.callbacks = []; this._value = PENDING; this._ok = true; }
        get triggered() { return this._value !== PENDING; }
        get processed() { return this.callbacks === null; }
        succeed(v) {
            if (this._value !== PENDING) throw new Error('already triggered');
            this._ok = true; this._value = v === undefined ? null : v; this.env.schedule(this, NORMAL, 0); return this;
        }
    }
    class Proc extends Ev {
        constructor(env, gen) {
            super(env);
            this.gen = gen;
            this.resume = this._resume.bind(this);
            const init = new Ev(env); init.callbacks = [this.resume]; init._ok = true; init._value = null;
            env.schedule(init, URGENT, 0);
        }
        _resume(event) {
            for (;;) {
                const r = this.gen.next(event._value);
                if (r.done) { this._ok = true; this._value = r.value === undefined ? null : r.value; this.env.schedule(this, NORMAL, 0); break; }
                event = r.value;
                if (event.callbacks !== null) { event.callbacks.push(this.resume); break; }
            }
        }
    }
    class Resource {
        constructor(env, capacity) { this.env = env; this.capacity = capacity; this.users = []; this.putQ = []; this.getQ = []; }
        request() {
            const ev = new Ev(this.env); ev.resource = this;
            this.putQ.push(ev); ev.callbacks.push(() => this._triggerGet()); this._triggerPut();
            return ev;
        }
        release(req) {
            const ev = new Ev(this.env); ev.request = req;
            this.getQ.push(ev); ev.callbacks.push(() => this._triggerPut()); this._triggerGet();
            return ev;
        }
        _triggerPut() {
            let idx = 0;
            while (idx < this.putQ.length) {
                const e = this.putQ[idx];
                if (this.users.length < this.capacity) { this.users.push(e); e.succeed(); }
                if (!e.triggered) idx++; else this.putQ.splice(idx, 1);
                break;                                  // Resource._do_put returns None: stop after one
            }
        }
        _triggerGet() {
            let idx = 0;
            while (idx < this.getQ.length) {
                const e = this.getQ[idx], u = this.users.indexOf(e.request);
                if (u >= 0) this.users.splice(u, 1);
                e.succeed();
                if (!e.triggered) idx++; else this.getQ.splice(idx, 1);
                break;
            }
        }
    }

    // ── scratchpad and engine (sim.py) ───────────────────────────────
    class Scratchpad {
        constructor(cap) { this.cap = cap; this.used = 0; this.items = new Map(); }
        hit(name, size) {
            const it = this.items.get(name);
            if (it === undefined || it[0] < size) return false;
            this.items.delete(name); this.items.set(name, it); return true;
        }
        free(name) { const it = this.items.get(name); if (it !== undefined) { this.items.delete(name); this.used -= it[0]; } }
        alloc(name, size, cls, dirty, pinned, evicted) {
            this.free(name);
            if (size > this.cap) return false;
            while (this.used + size > this.cap) {
                let victim = null;
                for (const n of this.items.keys()) if (!pinned.has(n)) { victim = n; break; }
                if (victim === null) return false;
                const [vs, vc, vd] = this.items.get(victim);
                this.free(victim); evicted.push([victim, vs, vc, vd]);
            }
            this.items.set(name, [size, cls, dirty]); this.used += size; return true;
        }
    }

    // a segment costed at full clock, run at clock fraction s (mirror of Segment.at / Segment.power)
    const segAt = (seg, s) => [seg.time / s, seg.eLogic * s * s + seg.eSram];
    const segPower = (seg, s) => { const [t, e] = segAt(seg, s); return e / t; };

    function runSim(trace, hw, clock, traceOn) {
        const dynamic = hw.enforceTdp && hw.powerMode === 'dynamic';
        if (hw.powerMode !== 'dynamic' && hw.powerMode !== 'worst-case') throw new Error(`unknown powerMode ${hw.powerMode}`);
        const p = trace.params, env = new Env(), cost = costModel(hw, p.logN, p.qBits, dynamic ? 1.0 : clock);
        const cap = clock, budget = hw.tdpW - cost.staticW;
        if (dynamic && budget <= 0) throw new Error(`TDP ${hw.tdpW} W is below static power`);
        let pwait = [], nActive = 0;
        const units = {}; for (const u of UNITS) units[u] = new Resource(env, 1);
        const hbm = new Resource(env, 1);
        const st = { busy: { ntt: 0, mac: 0, auto: 0, optical: 0, hbm: 0 }, work: { ntt: 0, mac: 0, auto: 0, optical: 0 },
                     energy: { ntt: 0, mac: 0, auto: 0, optical: 0, hbm: 0 }, bytes: { key: 0, pt: 0, ct_read: 0, ct_write: 0 },
                     dac: 0, adc: 0, stageBusy: {}, stageFirst: {}, stageLast: {}, peakW: 0, powerLoss: 0, clockTime: 0, computeTime: 0 };
        const spans = [];
        const n = trace.ops.length, producer = {}, lastUse = {};
        for (const o of trace.ops) producer[o.output] = o.id;
        for (const o of trace.ops) for (const x of o.inputs) lastUse[x] = o.id;
        const reserve = p.ksWorkingSet();
        if (hw.sramBytes < reserve) throw new Error(`scratchpad ${hw.sramMib} MiB is smaller than one key switch's working set (${(reserve / MiB).toFixed(0)} MiB)`);
        const spad = new Scratchpad(hw.sramBytes - reserve);
        const done = new Array(n).fill(false), doneEv = new Array(n).fill(null), wbEv = {};
        const opStart = new Array(n).fill(0.0), opEnd = new Array(n).fill(0.0);
        let inflight = 0, slotEv = null, pNow = 0.0;
        const chunk = hw.hbmChunkMib * MiB, hbmW = hw.hbmGbps * 1e9 * hw.pjHbmByte * 1e-12;
        const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

        function plan(o) {
            const pl = { keyLoad: 0, ptLoad: 0, ctLoad: [], writebacks: [], outWrite: 0 }, sizes = trace.sizes;
            const pinned = new Set(o.inputs); pinned.add(o.output);
            if (o.key) pinned.add(o.key[0]);
            for (const [pid] of o.pts) pinned.add(pid);
            const ev = [];
            for (const x of o.inputs) if (!spad.hit(x, sizes[x])) { pl.ctLoad.push([x, sizes[x]]); spad.alloc(x, sizes[x], 'ct', false, pinned, ev); }
            if (o.key) { const [kid, kb] = o.key; if (!spad.hit(kid, kb)) { pl.keyLoad += kb; spad.alloc(kid, kb, 'key', false, pinned, ev); } }
            for (const [pid, pb] of o.pts) if (!spad.hit(pid, pb)) { pl.ptLoad += pb; spad.alloc(pid, pb, 'pt', false, pinned, ev); }
            const outSize = sizes[o.output];
            if (!has(lastUse, o.output)) pl.outWrite = outSize;
            else if (!spad.alloc(o.output, outSize, 'ct', true, pinned, ev)) pl.outWrite = outSize;
            for (const [name, size, , dirty] of ev) if (dirty && (has(lastUse, name) ? lastUse[name] : -1) > o.id) pl.writebacks.push([name, size]);
            for (const x of o.inputs) if (lastUse[x] === o.id) spad.free(x);
            return pl;
        }
        const power = dp => { pNow += dp; if (pNow > st.peakW) st.peakW = pNow; };
        // ── the dynamic power manager (mirror of sim.py) ──
        function fit(seg) {
            const head = budget - pNow;
            if (!seg.scalable) return seg.energy / seg.time <= head ? 1.0 : null;
            if (segPower(seg, cap) <= head) return cap;
            let lo = hw.sMin < cap ? hw.sMin : cap;
            if (segPower(seg, lo) > head) return null;
            let hi = cap;
            for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; if (segPower(seg, mid) <= head) lo = mid; else hi = mid; }
            return lo;
        }
        function fitHbm() {
            const head = budget - pNow;
            if (hbmW <= head) return 1.0;
            const f = head / hbmW;
            return f >= hw.hbmMinFrac ? f : null;
        }
        function* waitPower(fitFn, what) {
            const t0 = env.now; let x;
            for (;;) {
                x = fitFn();
                if (x !== null) break;
                if (nActive === 0) throw new Error(`TDP ${hw.tdpW} W cannot power ${what} even at the lowest setting`);
                const ev = env.event(); pwait.push(ev); yield ev;
            }
            st.powerLoss += env.now - t0;
            return x;
        }
        function releasePower(dp) {
            power(-dp); nActive--;
            if (pwait.length) { const waiting = pwait; pwait = []; for (const ev of waiting) ev.succeed(); }
        }
        const stageBusy = (stage, unit, dt) => { const d = st.stageBusy[stage] || (st.stageBusy[stage] = {}); d[unit] = (d[unit] || 0.0) + dt; };
        function* waitDone(i) { if (!done[i]) yield doneEv[i]; }
        function* xfer(nbytes, cls, stage) {
            const bw = hw.hbmGbps * 1e9; let left = nbytes;
            while (left > 0) {
                const sz = left < chunk ? left : chunk;
                const req = hbm.request(); yield req;
                const start = env.now;
                let dt;
                if (dynamic) {
                    const f = yield* waitPower(fitHbm, 'HBM');
                    const pw = f >= 1.0 ? hbmW : hbmW * f;
                    dt = f >= 1.0 ? sz / bw : sz / (bw * f);
                    st.powerLoss += dt - sz / bw;
                    nActive++; power(pw);
                    yield env.timeout(dt);
                    releasePower(pw); hbm.release(req);
                } else {
                    power(hbmW);
                    dt = sz / bw;
                    yield env.timeout(dt);
                    power(-hbmW); hbm.release(req);
                }
                st.busy.hbm += dt; st.energy.hbm += sz * hw.pjHbmByte * 1e-12; stageBusy(stage, 'hbm', dt);
                if (traceOn) spans.push(['hbm', stage, start, dt, cls]);
                left -= sz;
            }
            st.bytes[cls] += nbytes;
        }
        function* writeback(name, size) { const pr = producer[name]; if (pr !== undefined) yield* waitDone(pr); yield* xfer(size, 'ct_write', 'spill'); }
        function* prefetch(o, pl) { if (pl.keyLoad) yield* xfer(pl.keyLoad, 'key', o.stage); if (pl.ptLoad) yield* xfer(pl.ptLoad, 'pt', o.stage); }
        function* runOp(o, pl) {
            const pf = (pl.keyLoad || pl.ptLoad) ? env.process(prefetch(o, pl)) : null;
            for (const [name, size] of pl.writebacks) wbEv[name] = env.process(writeback(name, size));
            for (const x of o.inputs) { const pr = producer[x]; if (pr !== undefined) yield* waitDone(pr); }
            for (const [x, size] of pl.ctLoad) { const w = wbEv[x]; if (w !== undefined && !w.processed) yield w; yield* xfer(size, 'ct_read', o.stage); }
            if (pf !== null && !pf.processed) yield pf;
            opStart[o.id] = env.now;
            if (!has(st.stageFirst, o.stage)) st.stageFirst[o.stage] = env.now;
            for (const k of o.kernels) {
                for (const seg of cost.segments(k)) {
                    const req = units[seg.unit].request(); yield req;
                    let sc, t, e;
                    if (dynamic) {
                        sc = yield* waitPower(() => fit(seg), seg.unit);
                        [t, e] = seg.scalable ? segAt(seg, sc) : [seg.time, seg.energy];
                        st.powerLoss += t - seg.time;
                        nActive++;
                    } else { sc = cost.s; t = seg.time; e = seg.energy; }
                    const start = env.now, pw = t > 0 ? e / t : 0.0;
                    power(pw);
                    yield env.timeout(t);
                    if (dynamic) releasePower(pw); else power(-pw);
                    units[seg.unit].release(req);
                    st.busy[seg.unit] += t; st.work[seg.unit] += seg.work; st.energy[seg.unit] += e;
                    st.dac += seg.dac; st.adc += seg.adc;
                    if (seg.scalable) { st.clockTime += sc * t; st.computeTime += t; }
                    stageBusy(o.stage, seg.unit, t);
                    if (traceOn) spans.push([seg.unit, o.stage, start, t, `${o.op}.${k.kind} L${o.level}`]);
                }
            }
            if (pl.outWrite) yield* xfer(pl.outWrite, 'ct_write', o.stage);
            opEnd[o.id] = env.now; st.stageLast[o.stage] = env.now;
            done[o.id] = true; doneEv[o.id].succeed(); inflight--;
            if (slotEv !== null && !slotEv.triggered) slotEv.succeed();
        }
        function* issuer() {
            for (const o of trace.ops) {
                while (inflight >= hw.window) { slotEv = env.event(); yield slotEv; }
                const pl = plan(o);
                inflight++; doneEv[o.id] = env.event(); env.process(runOp(o, pl));
            }
        }
        env.process(issuer());
        env.run();
        let meanClock = cost.s;
        if (dynamic && st.computeTime > 0) meanClock = st.clockTime / st.computeTime;
        return { trace, hw, clock: meanClock, horizon: env.now, stats: st, opStart, opEnd, staticW: cost.staticW, spans };
    }

    function simulate(trace, hw, opt) {
        opt = opt || {};
        const s = opt.clock !== undefined && opt.clock !== null ? opt.clock
                : (hw.enforceTdp && hw.powerMode === 'dynamic') ? 1.0 : hw.tdpClock();
        let res = runSim(trace, hw, s, !!opt.trace);
        if (opt.dvfs && (opt.clock === undefined || opt.clock === null)) {
            const b = res.stats.busy;
            const top = Math.max(b.ntt, b.mac, b.auto);
            if (b.hbm > top) {
                let s2 = s * top / b.hbm;
                if (s2 < hw.sMin) s2 = hw.sMin;
                if (s2 < s) res = runSim(trace, hw, s2, !!opt.trace);
            }
        }
        return res;
    }

    // ── metrics (metrics.py) ─────────────────────────────────────────
    const BOUND_NAME = { hbm: 'memory-bound', ntt: 'NTT-bound', optical: 'NTT-bound (optical engine)', mac: 'MAC-bound', auto: 'permutation-bound' };
    function summarise(res) {
        const st = res.stats, H = res.horizon, hw = res.hw;
        let nBoot = 0; for (const o of res.trace.ops) if (o.boot + 1 > nBoot) nBoot = o.boot + 1;
        const resources = [...UNITS, 'hbm'], util = {};
        for (const u of resources) util[u] = st.busy[u] / H;
        let top = 'hbm';
        for (const u of resources) if (util[u] > util[top]) top = u;
        let bound = BOUND_NAME[top];
        const dynamic = hw.enforceTdp && hw.powerMode === 'dynamic';
        if (top !== 'hbm' && hw.enforceTdp && ((dynamic && st.powerLoss > 0.1 * H) || (!dynamic && res.clock < 1.0)))
            bound = 'power-bound (' + BOUND_NAME[top] + ' at a TDP-limited clock)';
        const stages = {}, hot = {};
        for (const s of STAGES) {
            if (!(s in st.stageFirst)) continue;
            const span = st.stageLast[s] - st.stageFirst[s], busy = st.stageBusy[s] || {};
            let r = null;
            for (const u of resources) if ((busy[u] || 0.0) > 0 && (r === null || busy[u] > busy[r])) r = u;
            const bs = {}; for (const u of resources) bs[u] = busy[u] || 0.0;
            stages[s] = { span_s: span, busy_s: bs }; hot[s] = r;
        }
        let totalSpan = 0; for (const v of Object.values(stages)) totalSpan += v.span_s;
        for (const v of Object.values(stages)) v.share = totalSpan > 0 ? v.span_s / totalSpan : 0.0;
        let hottest = null;
        for (const s of Object.keys(stages)) if (hottest === null || stages[s].span_s > stages[hottest].span_s) hottest = s;
        const b = st.bytes, hbmTotal = b.key + b.pt + b.ct_read + b.ct_write;
        const stat = res.staticW * H;
        let dyn = 0; for (const u of resources) dyn += st.energy[u];
        const total = stat + dyn, br = { static: stat / total };
        for (const u of resources) br[u] = st.energy[u] / total;
        let lower = 0; for (const u of resources) if (st.busy[u] > lower) lower = st.busy[u];
        return {
            params: res.trace.params.name, hardware: hw.name, nBoot, clock: res.clock, latencyS: H, perBootstrapS: H / nBoot,
            utilisation: util, bound, boundResource: top, stages,
            hotspots: { stage: hottest, resource: hot[hottest] === undefined ? null : hot[hottest], perStage: hot },
            hbmBytes: Object.assign({}, b, { total: hbmTotal, keyShare: hbmTotal ? b.key / hbmTotal : 0.0 }),
            energy: { totalJ: total, perBootstrapJ: total / nBoot, avgPowerW: total / H, peakW: res.staticW + st.peakW, tdpW: hw.tdpW,
                      breakdown: br, dacSamples: st.dac, adcSamples: st.adc },
            lowerBoundS: lower, powerMode: hw.enforceTdp ? hw.powerMode : 'none', powerLossS: st.powerLoss,
        };
    }

    // ── power, performance and area (port of ppa.py; see its docstring for every source) ──
    // Exact with Python except Math.exp/expm1 in the yield models (tested with a tolerance).
    const CACTI_LSTP_22NM = [[64, 0.9189], [128, 0.8901], [256, 0.9020], [512, 0.8774], [1024, 0.8189], [2048, 0.7995]];
    function areaModel(o) {
        const am = Object.assign({ node: '7 nm (ASAP7-class predictive PDK, as used by ARK and BTS)', sramCurve: CACTI_LSTP_22NM,
            sramNodeScale: 229.2 / 512 / 0.8774, mm2PerBfly: 57.2 / 8192, mm2PerMacLane: 18.2 / 8192, mm2PerAutoWord: 20.6 / 1024,
            hbmStackGbps: 500.0, mm2PerHbmStack: 29.6 / 2, uncoreFrac: (42.8 + 20.6) / 325.2, converterGsps: 50.0, mm2PerDac: 0.05,
            mm2PerAdc: 0.10, photonicDieMm2: 100.0 }, o);
        am.sramMm2PerMib = mib => {
            const c = am.sramCurve;
            let per;
            if (mib <= c[0][0]) per = c[0][1];
            else if (mib >= c[c.length - 1][0]) per = c[c.length - 1][1];
            else {
                let i = 1;
                while (c[i][0] < mib) i++;
                const [x0, y0] = c[i - 1], [x1, y1] = c[i];
                per = y0 + (y1 - y0) * (mib - x0) / (x1 - x0);
            }
            return per * am.sramNodeScale;
        };
        return am;
    }
    const AREA_7NM = areaModel({});
    function areaMm2(hw, am) {
        am = am || hw.area || AREA_7NM;
        const ntt = hw.nttBflyPerCycle * am.mm2PerBfly, mac = hw.macLanes * am.mm2PerMacLane, auto = hw.autoWordsPerCycle * am.mm2PerAutoWord;
        const sram = hw.sramMib * am.sramMm2PerMib(hw.sramMib);
        const uncore = (ntt + mac + auto + sram) * am.uncoreFrac;
        const hbmPhy = Math.ceil(hw.hbmGbps / am.hbmStackGbps) * am.mm2PerHbmStack;
        let optE = 0.0, photonic = 0.0;
        if (hw.optical) {
            const ch = Math.ceil(hw.optical.samplesPerS / (am.converterGsps * 1e9));
            optE = ch * (am.mm2PerDac + am.mm2PerAdc);
            photonic = am.photonicDieMm2;
        }
        const die = ntt + mac + auto + sram + uncore + hbmPhy + optE;
        return { ntt, mac, auto, sram, uncore, hbmPhy, opticalElectronic: optE, die, photonicDie: photonic, total: die + photonic };
    }
    const DIE_COST = { d0PerCm2: 0.1, waferMm: 300.0, waferUsd: 10000.0, reticleMm2: 858.0, yieldModel: 'murphy' };
    const poissonYield = (a, d0) => Math.exp(-(a / 100.0) * d0);
    function murphyYield(a, d0) {
        const x = (a / 100.0) * d0;
        if (x === 0) return 1.0;
        const t = -Math.expm1(-x) / x;   // not 1 - exp(-x): it cancels for tiny dies
        return t * t;
    }
    function diesPerWafer(a, waferMm = 300.0) {
        const r = waferMm / 2.0;
        const n = Math.PI * r * r / a - Math.PI * waferMm / Math.sqrt(2.0 * a);
        return Math.max(0, Math.floor(n));
    }
    function dieCost(area, dc) {
        dc = Object.assign({}, DIE_COST, dc);
        const dpw = diesPerWafer(area, dc.waferMm), yp = poissonYield(area, dc.d0PerCm2), ym = murphyYield(area, dc.d0PerCm2);
        const y = dc.yieldModel === 'murphy' ? ym : yp, good = dpw * y;
        return { areaMm2: area, diesPerWafer: dpw, poisson: yp, murphy: ym, yield: y, goodDies: good,
                 usdPerGoodDie: good > 0 ? dc.waferUsd / good : Infinity, fitsReticle: area <= dc.reticleMm2 };
    }
    function ppaMetrics(m, hw, am, dc) {
        am = am || hw.area || AREA_7NM;
        const a = areaMm2(hw, am), t = m.perBootstrapS, e = m.energy.perBootstrapJ;
        const cost = dieCost(a.die, dc);
        let usd = cost.usdPerGoodDie;
        if (a.photonicDie > 0) usd = usd + dieCost(a.photonicDie, dc).usdPerGoodDie;
        const perf = 1.0 / t;
        return { areaMm2: a, node: am.node, dieCost: cost, usdPerUnit: usd, perfPerS: perf, perfPerW: 1.0 / e,
                 perfPerMm2: perf / a.total, perfPerUsd: perf / usd, edpJs: e * t, ed2pJs2: e * t * t };
    }
    function dominates(a, b, keys) {
        let better = false;
        for (const k of keys) { if (a[k] > b[k]) return false; if (a[k] < b[k]) better = true; }
        return better;
    }
    const paretoNd = (rows, keys = ['latencyS', 'energyJ', 'areaMm2']) =>
        rows.filter(r => !rows.some(o => o !== r && dominates(o, r, keys)));

    // ── convenience for the parity test and the deck ─────────────────
    const camel = o => { const m = { n_boot: 'nBoot', min_ks: 'minKs', seeded_keys: 'seededKeys', otf_plaintexts: 'otfPlaintexts', lazy_moddown: 'lazyModdown', stc_first: 'stcFirst' }, out = {};
        for (const [k, v] of Object.entries(o || {})) out[m[k] || k] = v; return out; };
    function simulateNamed(paramsName, hwName, opts, dvfs, hwOver) {
        const p = mkParams(PARAMS[paramsName]);
        const hw = hwOver ? withHw(ACCELERATORS[hwName](), hwOver) : ACCELERATORS[hwName]();
        return simulate(bootstrapTrace(p, camel(opts)), hw, { dvfs: !!dvfs });
    }
    root.FheSim = { PARAMS, ACCELERATORS, STAGES, UNITS, mkParams, bootstrapTrace, heOpTrace, summariseTrace, accelerator, optical, withHw,
                    costModel, simulate, summarise, simulateNamed, dftSplit,
                    areaModel, AREA_7NM, areaMm2, DIE_COST, poissonYield, murphyYield, diesPerWafer, dieCost, ppaMetrics, dominates, paretoNd };
})(typeof window !== 'undefined' ? window : globalThis);
