//! Our scheme, under the protocol of the paper.
//!
//! ```text
//! shaped boot    [--log-n 15] [--k 16] [--levels 1] [--seeds 1,2,3,4,5] [--families unit,disk]
//!                [--reps 3] [--bench time|check|both] [--chains fresh,square] [--threads 1]
//!                [--dsize D --k-aux P] [--decode full|average] [--dry-run]
//! shaped leveled [--log-n 15] [--k 16] [--unit U_32_45] [--scales 16,18,...] [--levels 16]
//!                [--pipelines shaped,ckks] [--chains fresh,square] [--seeds ...] [--families ...]
//!                [--decode full|average]
//! shaped tail    [boot options] [--key-sets 1,2,3,4,5] [--runs 200]
//! shaped cycles  [boot options] [--seeds 1,2,3,4,5] [--cycles 100]
//! shaped convert [--log-n 15] [--k 16] [--seeds ...] [--reps 3] [--threads 1]
//! shaped circuit [--log-n 15] [--k 16] [--scales 20,24] [--variants as_is,realify,reset]
//!                [--fresh yes,no] [--seeds ...]
//! ```
//!
//! Every mode takes `--base2k` (52 by default) and `boot`, `tail` and `cycles` take `--level-bits`
//! (the output scale of the bootstrapping, 20 by default).
//!
//! `boot` bootstraps `k` values (padded to `R`) and runs `levels` products by fresh operands, or
//! squarings, after it. Times are the best of `reps` per seed; `paper.py` takes their median over
//! the seeds. `leveled` runs chains of products without bootstrapping, for our scheme and for stock
//! CKKS, decoding in quad precision, and reports the largest error one step adds on the data and on
//! the guards.
//!
//! `tail` repeats independent bootstrappings, each of fresh unit-circle values followed by `levels`
//! squarings, `runs` times under each of the key sets: the distribution of the guards. `cycles`
//! takes one ciphertext through `cycles` rounds of a bootstrapping followed by `levels` products by
//! fresh operands. `convert` times and measures the conversions between CKKS and the replicated
//! layout. `circuit` composes four degree-7 activations `tanh(2x) / tanh(2)` on real data, as the
//! activation experiment of the paper, with or without a product by a fresh operand after each.
//!
//! Precision is measured on every data slot of the ring by default (`--decode full`), as in the
//! paper: each copy of a value counts. `--decode average` decodes one period instead, which averages
//! the copies (the subring decoder of Section 6.2).

#![feature(f128)]

use std::time::Instant;

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, Quad, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSMulOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
    oep::CKKSEncapsulatedModUpImpl,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, GLWETrace, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWESecretSampling, GLWESwitchingKeyPreparedFactory, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory,
        GetTensorKey, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESwitchingKeyPrepared,
            GLWETensorKeyPrepared,
        },
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};
use poulpy_shaped::{
    backend::HostBackend,
    bootstrapping::{BootstrapKeyLayouts, BootstrapKeys, BootstrapModule, BootstrapParams, Bootstrapper},
    context::ShapedContext,
    conversion::{ConversionParams, Converter},
    encoding::ShapedEncoding,
    ops::ShapedOps,
    security::max_modulus_bits,
    unit::{ShapingUnit, catalog},
};
use poulpy_shaped_bench::{Args, Line, Quality, Rng, dispatch};

const DEFAULT_BASE2K: usize = 52;
const HAMMING_WEIGHT: usize = 1024;
const SCRATCH: usize = 1 << 30;

type Ct<BE> = CKKSCiphertextOwned<BE>;

fn main() {
    let args = Args::parse();
    let threads: usize = args.get("threads", 1);
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "boot" => dispatch!(threads, boot(&args)),
        "leveled" => dispatch!(threads, leveled(&args)),
        "tail" => dispatch!(threads, tail(&args)),
        "cycles" => dispatch!(threads, cycles(&args)),
        "convert" => dispatch!(threads, convert(&args)),
        "circuit" => dispatch!(threads, circuit(&args)),
        other => panic!("unknown mode {other:?}: boot, leveled, tail, cycles, convert or circuit"),
    }
}

fn unit_named(name: &str) -> ShapingUnit {
    let coeffs = match name {
        "U_16_20" => catalog::U_16_20,
        "U_16_40" => catalog::U_16_40,
        "U_16_60" => catalog::U_16_60,
        "U_32_20" => catalog::U_32_20,
        "U_32_40" => catalog::U_32_40,
        "U_32_41" => catalog::U_32_41,
        "U_32_45" => catalog::U_32_45,
        "U_32_50" => catalog::U_32_50,
        "U_32_60" => catalog::U_32_60,
        "U_64_20" => catalog::U_64_20,
        "U_64_40" => catalog::U_64_40,
        "U_64_60" => catalog::U_64_60,
        other => panic!("unknown unit {other}"),
    };
    ShapingUnit::new(coeffs.to_vec()).unwrap()
}

/// Everything the benchmark asks of a backend.
trait Bench<BE: Backend>:
    BootstrapModule<BE>
    + ModuleNew<BE>
    + GLWETrace<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + GLWEAutomorphismKeyEncryptSk<BE>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyEncryptSk<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + GLWESwitchingKeyEncryptSk<BE>
    + GLWESwitchingKeyPreparedFactory<BE>
    + GLWESecretSampling<BE>
    + GLWESecretPreparedFactory<BE>
    + CKKSDecryptOps<BE>
    + CKKSEncryptOps<BE>
    + CKKSMulOps<BE>
    + CKKSEncodingOps<BE, Quad>
{
}

impl<BE: Backend, M> Bench<BE> for M where
    M: BootstrapModule<BE>
        + ModuleNew<BE>
        + GLWETrace<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + CKKSDecryptOps<BE>
        + CKKSEncryptOps<BE>
        + CKKSMulOps<BE>
        + CKKSEncodingOps<BE, Quad>
{
}

/// Module, secret and scratch of one configuration.
struct Env<BE: Backend> {
    module: Module<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: ScratchOwned<BE>,
    xe: Source,
    xa: Source,
    /// Radix of the ciphertexts.
    base2k: usize,
    /// Decode every slot of the ring and measure every copy of each value (the protocol of the
    /// paper), or decode one period, which averages the copies.
    full: bool,
}

impl<BE> Env<BE>
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    /// Shaped encoding of a slot vector of one period of the layout of `ctx`.
    fn encode_slots(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64], log_delta: usize) -> CKKSPlaintextOwned<BE> {
        let mut pt = self.module.ckks_pt_vec_alloc_compact(
            ctx.layout().period(),
            self.base2k.into(),
            TorusPrecision((log_delta + 8) as u32),
        );
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        });
        self.module
            .shaped_encode_slots_into(ctx, &mut pt, re, im, &mut self.scratch.borrow())
            .unwrap();
        pt
    }

    /// Shaped encryption of a slot vector of one period of the layout of `ctx`.
    fn encrypt_slots(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64], log_delta: usize, k: usize) -> Ct<BE> {
        let pt = self.encode_slots(ctx, re, im, log_delta);
        let mut ct = self
            .module
            .ckks_ciphertext_alloc(self.base2k.into(), TorusPrecision(k as u32));
        self.module
            .shaped_encrypt_sk(
                ctx,
                &mut ct,
                &pt,
                &self.sk,
                &mut self.xe,
                &mut self.xa,
                &mut self.scratch.borrow(),
            )
            .unwrap();
        ct
    }

    fn encrypt(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64], log_delta: usize, k: usize) -> Ct<BE> {
        let layout = ctx.layout();
        self.encrypt_slots(ctx, &layout.spread(re), &layout.spread(im), log_delta, k)
    }

    /// Stock CKKS encryption of the same slot vector.
    fn encrypt_stock(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64], log_delta: usize, k: usize) -> Ct<BE> {
        let layout = ctx.layout();
        let mut pt =
            self.module
                .ckks_pt_vec_alloc_compact(layout.period(), self.base2k.into(), TorusPrecision((log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: layout.log_sparsity(),
            slots: SlotsKind::Complex,
        });
        self.module
            .ckks_encode_reim_into(&mut pt, &layout.spread(re), &layout.spread(im), &mut self.scratch.borrow())
            .unwrap();
        let mut ct = self
            .module
            .ckks_ciphertext_alloc(self.base2k.into(), TorusPrecision(k as u32));
        self.module
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.scratch.borrow())
            .unwrap();
        ct
    }

    /// The slots of `ct`, decoded in quad precision: all of them, or one period.
    fn slots(&mut self, ctx: &ShapedContext, ct: &Ct<BE>) -> (Vec<f128>, Vec<f128>) {
        let mut pt = self.module.ckks_plaintext_alloc(
            ctx.layout().n().into(),
            self.base2k.into(),
            TorusPrecision((ct.log_delta() + ct.log_budget().min(24)) as u32),
        );
        pt.set_meta(ct.meta());
        self.module
            .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let count = if self.full {
            ctx.layout().n() / 2
        } else {
            ctx.layout().period()
        };
        let (mut re, mut im) = (vec![Quad::default(); count], vec![Quad::default(); count]);
        CKKSEncodingHostOps::<BE, Quad>::ckks_decode_reim_into(&self.module, &pt, &mut re, &mut im, &mut self.scratch.borrow())
            .unwrap();
        (re.iter().map(|x| x.0).collect(), im.iter().map(|x| x.0).collect())
    }

    /// The values of the data slots, in slot order (entry `i` carries value `i mod k`), and the
    /// largest guard magnitude of `ct`.
    fn data(&mut self, ctx: &ShapedContext, ct: &Ct<BE>) -> (Vec<f128>, Vec<f128>, f64) {
        let (re, im) = self.slots(ctx, ct);
        split(ctx, &re, &im)
    }
}

/// The data slots of decoded slots `(re, im)`, in slot order, and their largest guard magnitude.
fn split(ctx: &ShapedContext, re: &[f128], im: &[f128]) -> (Vec<f128>, Vec<f128>, f64) {
    let layout = ctx.layout();
    let guard = (0..re.len())
        .filter(|&t| !layout.is_data_slot(t))
        .map(|t| (re[t] as f64).hypot(im[t] as f64))
        .fold(0.0, f64::max);
    let r = layout.replication();
    (
        re.iter().step_by(r).copied().collect(),
        im.iter().step_by(r).copied().collect(),
        guard,
    )
}

fn new_env<BE>(n: usize, k: usize, seed: u64, base2k: usize) -> Env<BE>
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let module = Module::<BE>::new(n as u64);
    let layout = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: Rank(1),
    };
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, seed)));
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&layout);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    Env {
        module,
        sk,
        scratch: ScratchOwned::<BE>::alloc(SCRATCH),
        xe: Source::new(seed32(1, seed)),
        xa: Source::new(seed32(2, seed)),
        base2k,
        full: true,
    }
}

fn seed32(tag: u8, seed: u64) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&seed.to_le_bytes());
    s
}

fn errors(have: (&[f128], &[f128]), want: (&[f128], &[f128])) -> Vec<f64> {
    have.0
        .iter()
        .zip(want.0)
        .chain(have.1.iter().zip(want.1))
        .map(|(h, w)| (*h - *w) as f64)
        .collect()
}

/// `v` repeated to length `len`, as the data slots repeat the values.
fn tile(v: &[f64], len: usize) -> Vec<f128> {
    (0..len).map(|i| v[i % v.len()] as f128).collect()
}

/// `(a, b) *= (c, d)` componentwise.
fn mul_assign(re: &mut [f128], im: &mut [f128], c: &[f128], d: &[f128]) {
    for j in 0..re.len() {
        let (a, b) = (re[j], im[j]);
        re[j] = a * c[j] - b * d[j];
        im[j] = a * d[j] + b * c[j];
    }
}

/// Bytes of a stored key of `dnum` rows of two columns of `size` limbs.
fn key_bytes<K: GGLWEInfos>(key: &K, n: usize) -> usize {
    key.dnum().as_usize() * 2 * key.size() * n * 8
}

/// A compiled bootstrapping and the environment it runs in, from the `boot` options.
struct Booted<BE: Backend> {
    env: Env<BE>,
    ctx: ShapedContext,
    params: BootstrapParams,
    boot: Bootstrapper<BE>,
    layouts: BootstrapKeyLayouts,
    setup: Line,
    n: usize,
    batch: usize,
    padded: usize,
    levels: usize,
    base2k: usize,
}

fn boot_setup<BE>(a: &Args) -> Booted<BE>
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let log_n: usize = a.get("log-n", 15);
    let batch: usize = a.get("k", 16);
    let levels: usize = a.get("levels", 1);
    let base2k: usize = a.get("base2k", DEFAULT_BASE2K);
    let n = 1usize << log_n;
    let unit = unit_named(&a.string("unit", "U_32_45"));
    // Batches smaller than R are padded with zeros to R values.
    let padded = batch.max(unit.replication());
    let ctx = ShapedContext::new(n, padded, unit).unwrap();
    let params = BootstrapParams {
        packed_eval_mod: !a.flag("unpacked"),
        level_log_delta: a.get("level-bits", BootstrapParams::default().level_log_delta),
        ..BootstrapParams::default()
    };
    let mut env = new_env::<BE>(n, 0, 0, base2k);
    env.full = a.string("decode", "full") == "full";
    let boot = Bootstrapper::new(&env.module, &ctx, params, base2k, levels, &mut env.scratch.borrow()).unwrap();
    // Two limbs per digit where every key fits the caps, one limb per digit beyond, unless given.
    let layouts = match a.get::<usize>("dsize", 0) {
        0 => boot
            .secure_key_layouts(HAMMING_WEIGHT)
            .or_else(|_| boot.key_layouts(1, base2k + 1))
            .unwrap(),
        dsize => {
            let k_aux = a.get("k-aux", if dsize == 1 { base2k + 1 } else { dsize * base2k });
            boot.key_layouts(dsize, k_aux).unwrap()
        }
    };
    let (dsize, k_aux) = (layouts.automorphism.dsize.as_usize(), layouts.automorphism.k_aux.as_usize());
    let galois = boot.galois_elements(&env.module).len();
    let log_q_key = boot.key_modulus_bits();
    // The moduli of the keys as allocated: gadget precision plus auxiliary modulus.
    let checks = boot.security(&layouts, HAMMING_WEIGHT);
    let main_qp = checks[..3].iter().map(|c| c.log_qp).max().unwrap();
    let enc = checks[3];
    let setup = Line::new("setup")
        .str("pipeline", "ours")
        .num("log_n", log_n)
        .num("base2k", base2k)
        .num("k", batch)
        .num("k_padded", padded)
        .num("levels", levels)
        .num("level_bits", params.level_log_delta)
        .str("eval_mod", if boot.packs_eval_mod() { "packed" } else { "split" })
        .num("dsize", dsize)
        .num("log2_p", k_aux)
        .num("k_in", boot.k_in())
        .num("k_boot", boot.k_boot())
        .num("log2_q_key", log_q_key)
        .num("log2_qp", main_qp)
        .num("cap", max_modulus_bits(log_n, HAMMING_WEIGHT).unwrap_or(0))
        .num("enc_log2_qp", enc.log_qp)
        .num("enc_cap", enc.cap.unwrap_or(0))
        .num("automorphism_keys", galois)
        .num(
            "key_bytes",
            galois * key_bytes(&layouts.automorphism, n)
                + key_bytes(&layouts.tensor, n)
                + key_bytes(&layouts.dense_to_sparse, n)
                + key_bytes(&layouts.sparse_to_dense, n),
        )
        .num("threads", a.get::<usize>("threads", 1));
    Booted {
        env,
        ctx,
        params,
        boot,
        layouts,
        setup,
        n,
        batch,
        padded,
        levels,
        base2k,
    }
}

/// Generates the secret and the keys of key set `key_set`; the secret becomes that of `b.env`.
fn boot_keys<BE>(b: &mut Booted<BE>, key_set: u64) -> BootstrapKeys<BE>
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: b.n.into(),
        base2k: b.base2k.into(),
        k: b.boot.k_boot().into(),
        rank: Rank(1),
    };
    let env = &mut b.env;
    let mut sk_raw = env.module.glwe_secret_alloc_from_infos(&layout);
    env.module
        .glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, key_set)));
    let mut sk = env.module.glwe_secret_prepared_alloc_from_infos(&layout);
    env.module.glwe_secret_prepare(&mut sk, &sk_raw);
    env.sk = sk;
    env.xe = Source::new(seed32(1, key_set));
    env.xa = Source::new(seed32(2, key_set));
    b.boot.generate_keys(
        &env.module,
        &sk_raw,
        &b.layouts,
        &mut Source::new(seed32(3, key_set)),
        &mut Source::new(seed32(4, key_set)),
        &mut Source::new(seed32(5, key_set)),
        &mut env.scratch.borrow(),
    )
}

/// The entries of `v` on the data slots of the values asked for, not of the padding.
fn used(v: &[f128], batch: usize, padded: usize) -> Vec<f128> {
    v.iter()
        .enumerate()
        .filter(|(i, _)| i % padded < batch)
        .map(|(_, &x)| x)
        .collect()
}

fn boot<BE>(a: &Args)
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let reps: usize = a.get("reps", 3);
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let families: Vec<String> = a.list("families", "unit");
    let chains: Vec<String> = a.list("chains", "fresh,square");
    let bench = a.string("bench", "both");

    let t0 = Instant::now();
    let mut b = boot_setup::<BE>(a);
    if a.flag("dry-run") {
        b.setup.print();
        return;
    }
    let keys = boot_keys(&mut b, 0);
    let Booted {
        env,
        ctx,
        params,
        boot,
        setup,
        batch,
        padded,
        levels,
        base2k,
        ..
    } = &mut b;
    let (batch, padded, levels, base2k) = (*batch, *padded, *levels, *base2k);
    setup.clone().float("setup_s", t0.elapsed().as_secs_f64(), 2).print();

    for family in &families {
        for &seed in &seeds {
            let mut rng = Rng::new(seed, 0);
            let (mut re, mut im) = rng.values(batch, family);
            re.resize(padded, 0.0);
            im.resize(padded, 0.0);
            let ct = env.encrypt(ctx, &re, &im, params.log_delta, boot.k_in());
            let line = Line::new("run").with(setup).str("family", family).num("seed", seed);
            let used = |v: &[f128]| used(v, batch, padded);

            if bench != "check" {
                let mut refresh = f64::INFINITY;
                let mut out = None;
                for _ in 0..reps.max(1) {
                    let t = Instant::now();
                    out = Some(boot.bootstrap(&env.module, &ct, &keys, &mut env.scratch.borrow()).unwrap());
                    refresh = refresh.min(t.elapsed().as_secs_f64());
                }
                let out = out.unwrap();
                let mut lv = f64::INFINITY;
                for _ in 0..reps.max(1) {
                    let mut cur = module_copy(env, &out);
                    let t = Instant::now();
                    for _ in 0..levels {
                        let mut next = env
                            .module
                            .ckks_ciphertext_alloc(base2k.into(), TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32));
                        env.module
                            .shaped_square_into(ctx, &mut next, &cur, &keys.tensor, &mut env.scratch.borrow())
                            .unwrap();
                        cur = next;
                    }
                    lv = lv.min(t.elapsed().as_secs_f64());
                }
                line.clone()
                    .str("stage", "time")
                    .float("refresh_s", refresh, 4)
                    .float("levels_s", lv, 4)
                    .print();
            }
            if bench == "time" {
                continue;
            }
            for chain in &chains {
                let out = boot.bootstrap(&env.module, &ct, &keys, &mut env.scratch.borrow()).unwrap();
                let (dre, dim, guard_refresh) = env.data(ctx, &out);
                let len = dre.len();
                let refreshed = Quality::of(errors(
                    (&used(&dre), &used(&dim)),
                    (&used(&tile(&re, len)), &used(&tile(&im, len))),
                ));
                let (mut want_re, mut want_im) = if chain == "square" {
                    (dre, dim)
                } else {
                    (tile(&re, len), tile(&im, len))
                };
                let mut cur = out;
                let mut guard_max = guard_refresh;
                for level in 0..levels {
                    let mut next = env
                        .module
                        .ckks_ciphertext_alloc(base2k.into(), TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32));
                    if chain == "square" {
                        env.module
                            .shaped_square_into(ctx, &mut next, &cur, &keys.tensor, &mut env.scratch.borrow())
                            .unwrap();
                        let (c, d) = (want_re.clone(), want_im.clone());
                        mul_assign(&mut want_re, &mut want_im, &c, &d);
                    } else {
                        let (mut fr, mut fi) = Rng::new(seed, 1 + level as u64).values(batch, family);
                        fr.resize(padded, 0.0);
                        fi.resize(padded, 0.0);
                        let fresh = env.encrypt(ctx, &fr, &fi, cur.log_delta(), cur.k().as_usize());
                        env.module
                            .shaped_mul_into(ctx, &mut next, &cur, &fresh, &keys.tensor, &mut env.scratch.borrow())
                            .unwrap();
                        mul_assign(&mut want_re, &mut want_im, &tile(&fr, len), &tile(&fi, len));
                    }
                    cur = next;
                    let (_, _, guard) = env.data(ctx, &cur);
                    guard_max = guard_max.max(guard);
                }
                let (hre, him, _) = env.data(ctx, &cur);
                let after = Quality::of(errors((&used(&hre), &used(&him)), (&used(&want_re), &used(&want_im))));
                line.clone()
                    .str("stage", "check")
                    .str("chain", chain)
                    .quality("refresh_", refreshed)
                    .quality("levels_", after)
                    .float("guard_refresh_log2", guard_refresh.log2(), 3)
                    .float("guard_max_log2", guard_max.log2(), 3)
                    .num("k_out", cur.k().as_usize())
                    .print();
            }
        }
    }
}

/// `levels` squarings of `ct`, and the largest guard along the way.
fn square_chain<BE>(env: &mut Env<BE>, ctx: &ShapedContext, ct: Ct<BE>, levels: usize, keys: &BootstrapKeys<BE>) -> (Ct<BE>, f64)
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let mut cur = ct;
    let mut guard_max = 0f64;
    for _ in 0..levels {
        let mut next = env.module.ckks_ciphertext_alloc(
            env.base2k.into(),
            TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32),
        );
        env.module
            .shaped_square_into(ctx, &mut next, &cur, &keys.tensor, &mut env.scratch.borrow())
            .unwrap();
        cur = next;
        let (_, _, guard) = env.data(ctx, &cur);
        guard_max = guard_max.max(guard);
    }
    (cur, guard_max)
}

/// Independent bootstrappings of fresh unit-circle values, each followed by `levels` squarings,
/// `runs` times under each key set: one line per run.
fn tail<BE>(a: &Args)
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let key_sets: Vec<u64> = a.list("key-sets", "1,2,3,4,5");
    let runs: usize = a.get("runs", 200);
    let mut b = boot_setup::<BE>(a);
    b.setup.print();
    for &key_set in &key_sets {
        let keys = boot_keys(&mut b, key_set);
        let Booted {
            env,
            ctx,
            params,
            boot,
            setup,
            batch,
            padded,
            levels,
            ..
        } = &mut b;
        let (batch, padded, levels) = (*batch, *padded, *levels);
        for run in 0..runs {
            let (mut re, mut im) = Rng::new(key_set, 1_000 + run as u64).values(batch, "unit");
            re.resize(padded, 0.0);
            im.resize(padded, 0.0);
            let ct = env.encrypt(ctx, &re, &im, params.log_delta, boot.k_in());
            let out = boot.bootstrap(&env.module, &ct, &keys, &mut env.scratch.borrow()).unwrap();
            let (dre, dim, guard_refresh) = env.data(ctx, &out);
            let len = dre.len();
            let refreshed = Quality::of(errors(
                (&used(&dre, batch, padded), &used(&dim, batch, padded)),
                (&used(&tile(&re, len), batch, padded), &used(&tile(&im, len), batch, padded)),
            ));
            let (_, guard_squarings) = square_chain(env, ctx, out, levels, &keys);
            Line::new("tail")
                .with(setup)
                .num("key_set", key_set)
                .num("run", run)
                .quality("refresh_", refreshed)
                .float("guard_refresh_log2", guard_refresh.log2(), 3)
                .float("guard_squarings_log2", guard_squarings.log2(), 3)
                .print();
        }
    }
}

/// One ciphertext through `cycles` rounds of a bootstrapping and `levels` products by fresh
/// unit-circle operands, under the key set of each seed: one line per round.
fn cycles<BE>(a: &Args)
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let rounds: usize = a.get("cycles", 100);
    let mut b = boot_setup::<BE>(a);
    b.setup.print();
    for &seed in &seeds {
        let keys = boot_keys(&mut b, seed);
        let Booted {
            env,
            ctx,
            params,
            boot,
            setup,
            batch,
            padded,
            levels,
            base2k,
            ..
        } = &mut b;
        let (batch, padded, levels, base2k) = (*batch, *padded, *levels, *base2k);
        let (mut re, mut im) = Rng::new(seed, 0).values(batch, "unit");
        re.resize(padded, 0.0);
        im.resize(padded, 0.0);
        let mut cur = env.encrypt(ctx, &re, &im, params.log_delta, boot.k_in());
        let len = env.data(ctx, &cur).0.len();
        let (mut want_re, mut want_im) = (tile(&re, len), tile(&im, len));
        let t0 = Instant::now();
        for round in 0..rounds {
            let out = boot.bootstrap(&env.module, &cur, &keys, &mut env.scratch.borrow()).unwrap();
            let (dre, dim, guard_refresh) = env.data(ctx, &out);
            let refreshed = Quality::of(errors(
                (&used(&dre, batch, padded), &used(&dim, batch, padded)),
                (&used(&want_re, batch, padded), &used(&want_im, batch, padded)),
            ));
            cur = out;
            let mut guard_max = guard_refresh;
            for level in 0..levels {
                let (mut fr, mut fi) = Rng::new(seed, 1 + (round * levels + level) as u64).values(batch, "unit");
                fr.resize(padded, 0.0);
                fi.resize(padded, 0.0);
                let fresh = env.encrypt(ctx, &fr, &fi, cur.log_delta(), cur.k().as_usize());
                let mut next = env
                    .module
                    .ckks_ciphertext_alloc(base2k.into(), TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32));
                env.module
                    .shaped_mul_into(ctx, &mut next, &cur, &fresh, &keys.tensor, &mut env.scratch.borrow())
                    .unwrap();
                mul_assign(&mut want_re, &mut want_im, &tile(&fr, len), &tile(&fi, len));
                cur = next;
                let (_, _, guard) = env.data(ctx, &cur);
                guard_max = guard_max.max(guard);
            }
            let (hre, him, _) = env.data(ctx, &cur);
            let after = Quality::of(errors(
                (&used(&hre, batch, padded), &used(&him, batch, padded)),
                (&used(&want_re, batch, padded), &used(&want_im, batch, padded)),
            ));
            Line::new("cycle")
                .with(setup)
                .num("seed", seed)
                .num("round", round)
                .quality("refresh_", refreshed)
                .quality("levels_", after)
                .float("guard_refresh_log2", guard_refresh.log2(), 3)
                .float("guard_max_log2", guard_max.log2(), 3)
                .float("elapsed_s", t0.elapsed().as_secs_f64(), 2)
                .print();
        }
    }
}

/// A copy of `ct`.
fn module_copy<BE>(env: &Env<BE>, ct: &Ct<BE>) -> Ct<BE>
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    use poulpy_ckks::api::CKKSCopyOps;
    let mut copy = env.module.ckks_ciphertext_alloc_from_infos(ct);
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 22);
    env.module.ckks_copy(&mut copy, ct, &mut scratch.borrow()).unwrap();
    copy
}

fn leveled<BE>(a: &Args)
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let log_n: usize = a.get("log-n", 15);
    let batch: usize = a.get("k", 16);
    let levels: usize = a.get("levels", 16);
    let scales: Vec<usize> = a.list("scales", "16,18,20,22,24,26,28,30,32,34,36,38,40");
    let pipelines: Vec<String> = a.list("pipelines", "shaped,ckks");
    let chains: Vec<String> = a.list("chains", "fresh");
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let families: Vec<String> = a.list("families", "unit,disk");
    let unit_name = a.string("unit", "U_32_45");
    let base2k: usize = a.get("base2k", DEFAULT_BASE2K);
    let n = 1usize << log_n;
    let unit = unit_named(&unit_name);
    let (data_log2, guard_log2) = (unit.data_log2(), unit.guard_max_log2());
    let ctx = ShapedContext::new(n, batch.max(unit.replication()), unit).unwrap();
    // The shaping depth: guard bits below a result and the largest drop of one shaped rounding.
    let ctx = ctx
        .clone()
        .with_guard_bits(a.get("guard-bits", ctx.guard_bits()))
        .with_pass_cap(a.get("pass-cap", ctx.pass_cap()))
        .unwrap();
    for &scale in &scales {
        // Room for the levels and a margin above the values; keys one limb above the ciphertexts.
        let k_in = (levels + 1) * scale + 12;
        let k_keys = (k_in + ctx.guard_bits()).next_multiple_of(base2k) + base2k;
        let k_aux = base2k + log_n;
        let mut env = new_env::<BE>(n, k_keys, 7, base2k);
        env.full = a.string("decode", "full") == "full";
        let tsk_layout = GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k_aux: k_aux.into(),
            rank: Rank(1),
            dsize: 1usize.into(),
            dnum: GGLWELayout::dnum_for_input(base2k.into(), TorusPrecision(k_keys as u32), 1usize.into()),
        };
        let tsk = {
            let layout = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_keys.into(),
                rank: Rank(1),
            };
            let mut sk_raw = env.module.glwe_secret_alloc_from_infos(&layout);
            env.module
                .glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, 7)));
            let mut raw = env.module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
            env.module.glwe_tensor_key_encrypt_sk(
                &mut raw,
                &sk_raw,
                &mut Source::new(seed32(3, 7)),
                &mut Source::new(seed32(4, 7)),
                &mut env.scratch.borrow(),
            );
            let mut tsk = env.module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
            env.module.prepare_tensor_key(&mut tsk, &raw, &mut env.scratch.borrow());
            tsk
        };
        for pipeline in &pipelines {
            let shaped = pipeline == "shaped";
            for chain in &chains {
                for family in &families {
                    for &seed in &seeds {
                        let (re, im) = Rng::new(seed, 0).values(batch, family);
                        let encrypt = |env: &mut Env<BE>, re: &[f64], im: &[f64], log_delta: usize, k: usize| {
                            let (mut re, mut im) = (re.to_vec(), im.to_vec());
                            re.resize(ctx.layout().batch(), 0.0);
                            im.resize(ctx.layout().batch(), 0.0);
                            if shaped {
                                env.encrypt(&ctx, &re, &im, log_delta, k)
                            } else {
                                env.encrypt_stock(&ctx, &re, &im, log_delta, k)
                            }
                        };
                        let mut cur = encrypt(&mut env, &re, &im, scale, k_in);
                        let (mut added_max, mut guard_max, mut guard_added_max) = (0f64, 0f64, 0f64);
                        // Every slot, for the error one step adds on the guards, and the data slots.
                        let (mut all_re, mut all_im) = env.slots(&ctx, &cur);
                        let (mut dec_re, mut dec_im, _) = split(&ctx, &all_re, &all_im);
                        let len = dec_re.len();
                        let (mut want_re, mut want_im) = (tile(&re, len), tile(&im, len));
                        for level in 0..levels {
                            let mut next = env
                                .module
                                .ckks_ciphertext_alloc(base2k.into(), TorusPrecision((cur.k().as_usize() - scale) as u32));
                            // The exact operation on the decrypted operands.
                            let (mut op_re, mut op_im) = (dec_re.clone(), dec_im.clone());
                            let (mut op_all_re, mut op_all_im) = (all_re.clone(), all_im.clone());
                            if chain == "square" {
                                if shaped {
                                    env.module
                                        .shaped_square_into(&ctx, &mut next, &cur, &tsk, &mut env.scratch.borrow())
                                        .unwrap();
                                } else {
                                    env.module
                                        .ckks_square_into(&mut next, &cur, &tsk, &mut env.scratch.borrow())
                                        .unwrap();
                                }
                                let (c, d) = (op_re.clone(), op_im.clone());
                                mul_assign(&mut op_re, &mut op_im, &c, &d);
                                let (c, d) = (op_all_re.clone(), op_all_im.clone());
                                mul_assign(&mut op_all_re, &mut op_all_im, &c, &d);
                                let (c, d) = (want_re.clone(), want_im.clone());
                                mul_assign(&mut want_re, &mut want_im, &c, &d);
                            } else {
                                let (fr, fi) = Rng::new(seed, 1 + level as u64).values(batch, family);
                                let fresh = encrypt(&mut env, &fr, &fi, scale, cur.k().as_usize());
                                let (fall_re, fall_im) = env.slots(&ctx, &fresh);
                                let (fre, fim, _) = split(&ctx, &fall_re, &fall_im);
                                if shaped {
                                    env.module
                                        .shaped_mul_into(&ctx, &mut next, &cur, &fresh, &tsk, &mut env.scratch.borrow())
                                        .unwrap();
                                } else {
                                    env.module
                                        .ckks_mul_into(&mut next, &cur, &fresh, &tsk, &mut env.scratch.borrow())
                                        .unwrap();
                                }
                                mul_assign(&mut op_re, &mut op_im, &fre, &fim);
                                mul_assign(&mut op_all_re, &mut op_all_im, &fall_re, &fall_im);
                                mul_assign(&mut want_re, &mut want_im, &tile(&fr, len), &tile(&fi, len));
                            }
                            cur = next;
                            let (nre, nim) = env.slots(&ctx, &cur);
                            let (hre, him, guard) = split(&ctx, &nre, &nim);
                            dec_re = hre;
                            dec_im = him;
                            let added = errors((&dec_re, &dec_im), (&op_re, &op_im))
                                .iter()
                                .fold(0f64, |m, e| m.max(e.abs()));
                            added_max = added_max.max(added);
                            guard_max = guard_max.max(guard);
                            if env.full {
                                let layout = ctx.layout();
                                let added = (0..nre.len())
                                    .filter(|&t| !layout.is_data_slot(t))
                                    .map(|t| ((nre[t] - op_all_re[t]) as f64).hypot((nim[t] - op_all_im[t]) as f64))
                                    .fold(0f64, f64::max);
                                guard_added_max = guard_added_max.max(added);
                            }
                            all_re = nre;
                            all_im = nim;
                        }
                        let q = Quality::of(errors((&dec_re, &dec_im), (&want_re, &want_im)));
                        Line::new("leveled")
                            .str("pipeline", pipeline)
                            .num("log_n", log_n)
                            .num("k", batch)
                            .str("unit", &unit_name)
                            .float("unit_data_log2", data_log2, 2)
                            .float("unit_guard_log2", guard_log2, 2)
                            .num("level_bits", scale)
                            .num("levels", levels)
                            .str("chain", chain)
                            .str("family", family)
                            .num("seed", seed)
                            .quality("", q)
                            .float("added_log2_max", added_max.log2(), 3)
                            .float("guard_added_log2_max", guard_added_max.log2(), 3)
                            .float("guard_max_log2", guard_max.log2(), 3)
                            .print();
                    }
                }
            }
        }
    }
}

/// Prepared automorphism keys of `layout` for the Galois elements `gal`, under the secret of seed
/// `seed` with ciphertexts of precision `k`.
fn automorphism_keys<BE>(
    env: &mut Env<BE>,
    layout: &GLWEAutomorphismKeyLayout,
    k: usize,
    seed: u64,
    gal: impl IntoIterator<Item = i64>,
) -> std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe = GLWELayout {
        n: layout.n,
        base2k: layout.base2k,
        k: k.into(),
        rank: Rank(1),
    };
    let mut sk_raw = env.module.glwe_secret_alloc_from_infos(&glwe);
    env.module
        .glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, seed)));
    let mut keys = std::collections::HashMap::new();
    for g in gal {
        let mut key = env.module.glwe_automorphism_key_alloc_from_infos(layout);
        env.module.glwe_automorphism_key_encrypt_sk(
            &mut key,
            g,
            &sk_raw,
            &mut Source::new(seed32(8, seed)),
            &mut Source::new(seed32(9, seed)),
            &mut env.scratch.borrow(),
        );
        let mut prepared = env.module.glwe_automorphism_key_prepared_alloc_from_infos(layout);
        env.module
            .glwe_automorphism_key_prepare(&mut prepared, &key, &mut env.scratch.borrow());
        keys.insert(g, prepared);
    }
    keys
}

/// Both conversions between CKKS at scale `2^40` and the replicated layout at scale `2^20`, timed
/// and measured on unit-circle values: one line per seed.
fn convert<BE>(a: &Args)
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let log_n: usize = a.get("log-n", 15);
    let batch: usize = a.get("k", 16);
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let reps: usize = a.get("reps", 3);
    let base2k: usize = a.get("base2k", DEFAULT_BASE2K);
    // Width of the incoming CKKS ciphertext.
    let ckks_k: usize = a.get("ckks-k", 150);
    let n = 1usize << log_n;
    let unit = unit_named(&a.string("unit", "U_32_45"));
    let ctx = ShapedContext::new(n, batch, unit).unwrap();
    let params = ConversionParams::default();
    let mut env = new_env::<BE>(n, 0, 0, base2k);
    let conv = Converter::new(&env.module, &ctx, params, base2k, &mut env.scratch.borrow()).unwrap();
    // Keys one limb above the widest ciphertext and the guard bits of the wide stage.
    let key_k = (ckks_k + params.guard_bits).next_multiple_of(base2k) + base2k;
    let atk = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: GGLWELayout::dnum_for_input(base2k.into(), TorusPrecision(key_k as u32), 1usize.into()),
        k_aux: TorusPrecision((base2k + 1) as u32),
        rank: Rank(1),
        dsize: 1usize.into(),
    };
    let gal = conv.galois_elements(&env.module);
    let keys = automorphism_keys(&mut env, &atk, key_k, 0, gal.iter().copied());
    env = Env {
        sk: new_env::<BE>(n, key_k, 0, base2k).sk,
        ..env
    };
    let widest = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: ckks_k.into(),
        rank: Rank(1),
    };
    let mut conv_scratch = ScratchOwned::<BE>::alloc(conv.tmp_bytes(&env.module, &widest, &atk));
    let setup = Line::new("setup")
        .str("pipeline", "convert")
        .num("log_n", log_n)
        .num("k", batch)
        .num("ckks_log_delta", params.ckks_log_delta)
        .num("log_delta", params.log_delta)
        .num("diagonal_log_delta", params.diagonal_log_delta)
        .num("ckks_k", ckks_k)
        .num("log2_qp", atk.k().as_usize())
        .num("automorphism_keys", gal.len())
        .num("threads", a.get::<usize>("threads", 1));
    setup.print();
    let all = n / 2;
    for &seed in &seeds {
        let (re, im) = Rng::new(seed, 0).values(batch, "unit");
        // CKKS: the k values with period k on every slot.
        let mut pt =
            env.module
                .ckks_pt_vec_alloc_compact(batch, base2k.into(), TorusPrecision((params.ckks_log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta: params.ckks_log_delta,
            log_sparsity: (n / (2 * batch)).trailing_zeros() as usize,
            slots: SlotsKind::Complex,
        });
        env.module
            .ckks_encode_reim_into(&mut pt, &re, &im, &mut env.scratch.borrow())
            .unwrap();
        let mut ct = env.module.ckks_ciphertext_alloc(base2k.into(), TorusPrecision(ckks_k as u32));
        env.module
            .ckks_encrypt_sk(&mut ct, &pt, &env.sk, &mut env.xe, &mut env.xa, &mut env.scratch.borrow())
            .unwrap();
        let ckks_quality = |env: &mut Env<BE>, ct: &Ct<BE>| {
            let (sre, sim) = env.slots(&ctx, ct);
            Quality::of(errors((&sre, &sim), (&tile(&re, all), &tile(&im, all))))
        };
        let incoming = ckks_quality(&mut env, &ct);
        let (mut to_shaped_s, mut to_ckks_s) = (f64::INFINITY, f64::INFINITY);
        let mut shaped = None;
        for _ in 0..reps.max(1) {
            let t = Instant::now();
            shaped = Some(conv.to_shaped(&env.module, &ct, &keys, &mut conv_scratch.borrow()).unwrap());
            to_shaped_s = to_shaped_s.min(t.elapsed().as_secs_f64());
        }
        let shaped = shaped.unwrap();
        let (dre, dim, guard) = env.data(&ctx, &shaped);
        let len = dre.len();
        let shaped_quality = Quality::of(errors((&dre, &dim), (&tile(&re, len), &tile(&im, len))));
        let mut back = None;
        for _ in 0..reps.max(1) {
            let t = Instant::now();
            back = Some(conv.to_ckks(&env.module, &shaped, &keys, &mut conv_scratch.borrow()).unwrap());
            to_ckks_s = to_ckks_s.min(t.elapsed().as_secs_f64());
        }
        let back = back.unwrap();
        let round_trip = ckks_quality(&mut env, &back);
        Line::new("convert")
            .with(&setup)
            .num("seed", seed)
            .quality("ckks_in_", incoming)
            .float("to_shaped_s", to_shaped_s, 4)
            .quality("shaped_", shaped_quality)
            .float("shaped_guard_log2", guard.log2(), 3)
            .num("shaped_k", shaped.k().as_usize())
            .float("to_ckks_s", to_ckks_s, 4)
            .quality("ckks_out_", round_trip)
            .num("ckks_out_k", back.k().as_usize())
            .print();
    }
}

/// Monomial coefficients `(c1, c3, c5, c7)` of the degree-7 Chebyshev interpolant of
/// `tanh(2x) / tanh(2)` on `[-1, 1]`, which is odd.
fn activation_coefficients() -> [f64; 4] {
    let f = |x: f64| (2.0 * x).tanh() / 2f64.tanh();
    let nodes = 8;
    let mut cheb = [0f64; 8];
    for j in 0..nodes {
        let theta = std::f64::consts::PI * (j as f64 + 0.5) / nodes as f64;
        let y = f(theta.cos());
        for (k, c) in cheb.iter_mut().enumerate() {
            *c += 2.0 / nodes as f64 * y * (k as f64 * theta).cos();
        }
    }
    let (t1, t3, t5, t7) = (cheb[1], cheb[3], cheb[5], cheb[7]);
    // T1 = x, T3 = 4x^3 - 3x, T5 = 16x^5 - 20x^3 + 5x, T7 = 64x^7 - 112x^5 + 56x^3 - 7x.
    [
        t1 - 3.0 * t3 + 5.0 * t5 - 7.0 * t7,
        4.0 * t3 - 20.0 * t5 + 56.0 * t7,
        16.0 * t5 - 112.0 * t7,
        64.0 * t7,
    ]
}

/// Four degree-7 activations `p(x) ~ tanh(2x) / tanh(2)` on real data in `[-1, 1]`, without
/// bootstrapping. The constants hold their value on every slot, guards included, so the guards are
/// mapped like the data. Variants: `as_is`; `realify`, the activation of `x + conj(x)` with the
/// factor 2 folded into the coefficients; `reset`, a product by the shaped indicator of the data
/// slots before each activation. With `--fresh yes`, a product by a fresh operand follows each
/// activation. One line per activation and per product.
fn circuit<BE>(a: &Args)
where
    BE: HostBackend,
    Module<BE>: Bench<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    const STEPS: usize = 4;
    let log_n: usize = a.get("log-n", 15);
    let batch: usize = a.get("k", 16);
    let scales: Vec<usize> = a.list("scales", "20,24");
    let variants: Vec<String> = a.list("variants", "as_is,realify,reset");
    let fresh_modes: Vec<String> = a.list("fresh", "yes,no");
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let base2k: usize = a.get("base2k", DEFAULT_BASE2K);
    let unit_name = a.string("unit", "U_32_45");
    let n = 1usize << log_n;
    let unit = unit_named(&unit_name);
    let ctx = ShapedContext::new(n, batch.max(unit.replication()), unit).unwrap();
    let padded = ctx.layout().batch();
    let layout = *ctx.layout();
    let coeffs = activation_coefficients();
    let poly = |x: f64, c: &[f64; 4]| x * (c[0] + x * x * (c[1] + x * x * (c[2] + x * x * c[3])));
    // An activation spends four levels, the fresh product one, the reset one more.
    let levels_of = |variant: &str, fresh: bool| STEPS * (4 + usize::from(fresh) + usize::from(variant == "reset"));
    for &scale in &scales {
        let k_top = (levels_of("reset", true) + 1) * scale + 12;
        let k_keys = (k_top + ctx.guard_bits()).next_multiple_of(base2k) + base2k;
        let mut env = new_env::<BE>(n, k_keys, 7, base2k);
        let tsk_layout = GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k_aux: (base2k + log_n).into(),
            rank: Rank(1),
            dsize: 1usize.into(),
            dnum: GGLWELayout::dnum_for_input(base2k.into(), TorusPrecision(k_keys as u32), 1usize.into()),
        };
        let atk_layout = GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k_aux: (base2k + log_n).into(),
            rank: Rank(1),
            dsize: 1usize.into(),
            dnum: tsk_layout.dnum,
        };
        let conj_keys = automorphism_keys(&mut env, &atk_layout, k_keys, 7, [-1]);
        let tsk = {
            let glwe = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_keys.into(),
                rank: Rank(1),
            };
            let mut sk_raw = env.module.glwe_secret_alloc_from_infos(&glwe);
            env.module
                .glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, 7)));
            let mut raw = env.module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
            env.module.glwe_tensor_key_encrypt_sk(
                &mut raw,
                &sk_raw,
                &mut Source::new(seed32(3, 7)),
                &mut Source::new(seed32(4, 7)),
                &mut env.scratch.borrow(),
            );
            let mut tsk = env.module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
            env.module.prepare_tensor_key(&mut tsk, &raw, &mut env.scratch.borrow());
            tsk
        };
        let period = layout.period();
        let indicator: Vec<f64> = (0..period).map(|t| if layout.is_data_slot(t) { 1.0 } else { 0.0 }).collect();
        let zeros = vec![0.0; period];
        for &seed in &seeds {
            for fresh_mode in &fresh_modes {
                let fresh = fresh_mode == "yes";
                for variant in &variants {
                    let (realify, reset) = (variant == "realify", variant == "reset");
                    // With realification the activation receives 2 Re(x): the factor goes into the coefficients.
                    let c = if realify {
                        [coeffs[0] / 2.0, coeffs[1] / 8.0, coeffs[2] / 32.0, coeffs[3] / 128.0]
                    } else {
                        coeffs
                    };
                    let k_in = (levels_of(variant, fresh) + 1) * scale + 12;
                    let mut rng = Rng::new(seed, 0);
                    let real = |rng: &mut Rng| -> Vec<f64> {
                        let mut v: Vec<f64> = (0..batch).map(|_| 2.0 * rng.unit() - 1.0).collect();
                        v.resize(padded, 0.0);
                        v
                    };
                    let mut values = real(&mut rng);
                    let mut x = env.encrypt(&ctx, &values, &vec![0.0; padded], scale, k_in);
                    let lower = |ct: &Ct<BE>| TorusPrecision((ct.k().as_usize() - scale) as u32);
                    let alloc = |env: &Env<BE>, k: TorusPrecision| env.module.ckks_ciphertext_alloc(base2k.into(), k);
                    let emit = |env: &mut Env<BE>, step: usize, stage: &str, ct: &Ct<BE>, values: &[f64]| {
                        let (dre, dim, guard) = env.data(&ctx, ct);
                        let len = dre.len();
                        let want: Vec<f128> = tile(values, len);
                        let q = Quality::of(errors(
                            (&used(&dre, batch, padded), &used(&dim, batch, padded)),
                            (&used(&want, batch, padded), &vec![0.0; used(&want, batch, padded).len()]),
                        ));
                        Line::new("circuit")
                            .num("log_n", log_n)
                            .num("k", batch)
                            .str("unit", &unit_name)
                            .num("level_bits", scale)
                            .str("variant", variant)
                            .str("fresh", fresh_mode)
                            .num("seed", seed)
                            .num("step", step)
                            .str("stage", stage)
                            .quality("", q)
                            .float("guard_log2", guard.log2(), 3)
                            .num("k_ct", ct.k().as_usize())
                            .num("consumed_bits", k_in - ct.k().as_usize())
                            .print();
                    };
                    emit(&mut env, 0, "input", &x, &values);
                    for step in 1..=STEPS {
                        let mut scratch = std::mem::replace(&mut env.scratch, ScratchOwned::<BE>::alloc(0));
                        if realify {
                            let mut conj = alloc(&env, x.k());
                            env.module
                                .shaped_conjugate_into(&ctx, &mut conj, &x, &conj_keys, &mut scratch.borrow())
                                .unwrap();
                            env.module.ckks_add_assign(&mut x, &conj, &mut scratch.borrow()).unwrap();
                        }
                        env.scratch = scratch;
                        if reset {
                            let mask = env.encode_slots(&ctx, &indicator, &zeros, scale);
                            let mut dst = alloc(&env, lower(&x));
                            env.module
                                .shaped_mul_pt_into(&ctx, &mut dst, &x, &mask, &mut env.scratch.borrow())
                                .unwrap();
                            x = dst;
                        }
                        let constant =
                            |env: &mut Env<BE>, value: f64| env.encode_slots(&ctx, &vec![value; period], &zeros, scale);
                        let (c1, c3, c5, c7) = (
                            constant(&mut env, c[0]),
                            constant(&mut env, c[1]),
                            constant(&mut env, c[2]),
                            constant(&mut env, c[3]),
                        );
                        let mut s = std::mem::replace(&mut env.scratch, ScratchOwned::<BE>::alloc(0));
                        let m = &env.module;
                        let mut x2 = alloc(&env, lower(&x));
                        m.shaped_square_into(&ctx, &mut x2, &x, &tsk, &mut s.borrow()).unwrap();
                        let mut x4 = alloc(&env, lower(&x2));
                        m.shaped_square_into(&ctx, &mut x4, &x2, &tsk, &mut s.borrow()).unwrap();
                        // hi = c7 x^2 + c5 and lo = c3 x^2 + c1.
                        let mut hi = alloc(&env, lower(&x2));
                        m.shaped_mul_pt_into(&ctx, &mut hi, &x2, &c7, &mut s.borrow()).unwrap();
                        m.ckks_add_pt_vec_assign(&mut hi, &c5, &mut s.borrow()).unwrap();
                        let mut lo = alloc(&env, lower(&x2));
                        m.shaped_mul_pt_into(&ctx, &mut lo, &x2, &c3, &mut s.borrow()).unwrap();
                        m.ckks_add_pt_vec_assign(&mut lo, &c1, &mut s.borrow()).unwrap();
                        // p = x (lo + x^4 hi).
                        let mut inner = alloc(&env, lower(&x4));
                        m.shaped_mul_into(&ctx, &mut inner, &x4, &hi, &tsk, &mut s.borrow()).unwrap();
                        m.ckks_add_assign(&mut inner, &lo, &mut s.borrow()).unwrap();
                        let mut p = alloc(&env, lower(&inner));
                        m.shaped_mul_into(&ctx, &mut p, &x, &inner, &tsk, &mut s.borrow()).unwrap();
                        env.scratch = s;
                        for v in values.iter_mut().take(batch) {
                            *v = poly(if realify { 2.0 * *v } else { *v }, &c);
                        }
                        emit(&mut env, step, "activation", &p, &values);
                        if !fresh {
                            x = p;
                            continue;
                        }
                        let operand_values = real(&mut rng);
                        let operand = env.encrypt(&ctx, &operand_values, &vec![0.0; padded], scale, p.k().as_usize());
                        let mut y = alloc(&env, lower(&p));
                        env.module
                            .shaped_mul_into(&ctx, &mut y, &p, &operand, &tsk, &mut env.scratch.borrow())
                            .unwrap();
                        for (v, f) in values.iter_mut().zip(&operand_values) {
                            *v *= f;
                        }
                        emit(&mut env, step, "product", &y, &values);
                        x = y;
                    }
                }
            }
        }
    }
}
