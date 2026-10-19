//! End-to-end leveled operations: shaped encryption, products, rotations and conjugation,
//! against the same computation without shaping.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, CKKSModuleInfos, SetCKKSInfos, SlotsKind,
    api::CKKSDecryptOps,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    layouts::{
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESecretSampling, GLWETensorKeyLayout,
        GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::{context::ShapedContext, encoding::ShapedEncoding, ops::ShapedOps, unit::ShapingUnit};

type BE = NTT4x30Portable;

const LOG_N: usize = 12;
const N: usize = 1 << LOG_N;
const BATCH: usize = 16;
const BASE2K: usize = 52;
const CT_K: usize = 4 * BASE2K;
const LOG_DELTA: usize = 20;
const K_AUX: usize = 53;
/// Keys one limb wider than the widest ciphertext, so that their noise, which is not shaped, is
/// suppressed by the narrower ciphertext modulus as in the paper.
const KEY_K: usize = CT_K + BASE2K;

struct Fixture {
    module: Module<BE>,
    sk: GLWESecretPrepared<AlignedBuf, BE>,
    tsk: GLWETensorKeyPrepared<AlignedBuf, BE>,
    atks: HashMap<i64, GLWEAutomorphismKeyPrepared<AlignedBuf, BE>>,
    scratch: ScratchOwned<BE>,
    ops_scratch: ScratchOwned<BE>,
    xa: Source,
    xe: Source,
}

fn glwe_layout() -> GLWELayout {
    GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    }
}

fn tsk_layout() -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k_aux: K_AUX.into(),
        rank: Rank(1),
        dsize: 1usize.into(),
        dnum: KEY_K.div_ceil(BASE2K).into(),
    }
}

fn atk_layout() -> GLWEAutomorphismKeyLayout {
    GLWEAutomorphismKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k_aux: K_AUX.into(),
        rank: Rank(1),
        dsize: 1usize.into(),
        dnum: KEY_K.div_ceil(BASE2K).into(),
    }
}

fn context() -> ShapedContext {
    ShapedContext::new(N, BATCH, ShapingUnit::default_unit()).unwrap()
}

impl Fixture {
    fn new(ctx: &ShapedContext) -> Self {
        let module = Module::<BE>::new(N as u64);
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 26);
        let mut xs = Source::new([0u8; 32]);
        let mut xa = Source::new([1u8; 32]);
        let mut xe = Source::new([2u8; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_layout());
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, 64, &mut xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
        module.glwe_secret_prepare(&mut sk, &sk_raw);

        let mut tsk_raw = module.glwe_tensor_key_alloc_from_infos(&tsk_layout());
        module.glwe_tensor_key_encrypt_sk(&mut tsk_raw, &sk_raw, &mut xa, &mut xe, &mut scratch.borrow());
        let mut tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
        module.prepare_tensor_key(&mut tsk, &tsk_raw, &mut scratch.borrow());

        let mut atks = HashMap::new();
        let rotation = ctx.layout().rotation_slots(1);
        for gal in [module.ckks_galois_element(rotation), -1] {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_layout());
            module.glwe_automorphism_key_encrypt_sk(&mut atk, gal, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout());
            module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
            atks.insert(gal, prepared);
        }

        let ops_bytes = module.shaped_ops_tmp_bytes(ctx, &glwe_layout(), &tsk_layout(), &atk_layout());
        let ops_scratch = ScratchOwned::<BE>::alloc(ops_bytes);
        Self {
            module,
            sk,
            tsk,
            atks,
            scratch,
            ops_scratch,
            xa,
            xe,
        }
    }

    fn plaintext(&self, slots: usize, log_delta: usize) -> CKKSPlaintextOwned<BE> {
        let mut pt = self
            .module
            .ckks_pt_vec_alloc_compact(slots, BASE2K.into(), TorusPrecision((log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        });
        pt
    }

    fn encrypt(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64]) -> CKKSCiphertextOwned<BE> {
        self.encrypt_at(ctx, re, im, CT_K)
    }

    fn encrypt_at(&mut self, ctx: &ShapedContext, re: &[f64], im: &[f64], k: usize) -> CKKSCiphertextOwned<BE> {
        let mut pt = self.plaintext(ctx.layout().period(), LOG_DELTA);
        self.module
            .shaped_encode_into(ctx, &mut pt, re, im, &mut self.scratch.borrow())
            .unwrap();
        let mut ct = self.module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(k as u32));
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

    /// The data values and the largest guard magnitude of `ct`.
    fn decrypt(&mut self, ctx: &ShapedContext, ct: &CKKSCiphertextOwned<BE>) -> (Vec<f64>, Vec<f64>, f64) {
        // The scale and 24 bits of headroom: enough for the values and within the host decoder.
        let mut pt = self.module.ckks_plaintext_alloc(
            N.into(),
            BASE2K.into(),
            TorusPrecision((ct.log_delta() + ct.log_budget().min(24)) as u32),
        );
        pt.set_meta(ct.meta());
        self.module
            .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let period = ctx.layout().period();
        let (mut re, mut im) = (vec![0.0; period], vec![0.0; period]);
        self.module
            .shaped_decode_slots(ctx, &pt, &mut re, &mut im, &mut self.scratch.borrow())
            .unwrap();
        let guard = (0..period)
            .filter(|t| !ctx.layout().is_data_slot(*t))
            .map(|t| re[t].hypot(im[t]))
            .fold(0.0, f64::max);
        (ctx.layout().collect(&re), ctx.layout().collect(&im), guard)
    }
}

/// `-log2` of the largest error over the real and imaginary parts.
fn precision(have: (&[f64], &[f64]), want: (&[f64], &[f64])) -> f64 {
    let err = have
        .0
        .iter()
        .zip(want.0)
        .chain(have.1.iter().zip(want.1))
        .map(|(h, w)| (h - w).abs())
        .fold(0.0, f64::max);
    -err.log2()
}

/// Values on the unit circle.
fn sample(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let angles: Vec<f64> = (0..BATCH)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU
        })
        .collect();
    (
        angles.iter().map(|a| a.cos()).collect(),
        angles.iter().map(|a| a.sin()).collect(),
    )
}

/// Precision and largest guard after `levels` products by fresh operands.
fn product_chain(fx: &mut Fixture, ctx: &ShapedContext, levels: usize) -> (f64, f64) {
    let (mut want_re, mut want_im) = sample(1);
    let mut ct = fx.encrypt(ctx, &want_re, &want_im);
    let mut guard_max: f64 = 0.0;
    for level in 0..levels {
        let (re, im) = sample(2 + level as u64);
        let fresh = fx.encrypt(ctx, &re, &im);
        fx.module
            .shaped_mul_assign(ctx, &mut ct, &fresh, &fx.tsk, &mut fx.ops_scratch.borrow())
            .unwrap();
        for j in 0..BATCH {
            let (a, b) = (want_re[j], want_im[j]);
            want_re[j] = a * re[j] - b * im[j];
            want_im[j] = a * im[j] + b * re[j];
        }
        let (_, _, guard) = fx.decrypt(ctx, &ct);
        guard_max = guard_max.max(guard);
        assert_eq!(ct.log_delta(), LOG_DELTA);
    }
    let (re, im, _) = fx.decrypt(ctx, &ct);
    (precision((&re, &im), (&want_re, &want_im)), guard_max)
}

#[test]
fn fresh_encryption_is_precise_and_guards_are_small() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(7);
    let ct = fx.encrypt(&ctx, &re, &im);
    assert_eq!((ct.log_delta(), ct.log_sparsity()), (LOG_DELTA, ctx.layout().log_sparsity()));
    let (have_re, have_im, guard) = fx.decrypt(&ctx, &ct);
    let bits = precision((&have_re, &have_im), (&re, &im));
    assert!(bits > 40.0, "fresh precision {bits}");
    assert!(guard < 0.01, "fresh guard {guard}");
}

#[test]
fn shaping_gains_precision_over_ckks_on_products() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let levels = (CT_K - LOG_DELTA) / LOG_DELTA - 1;
    let (shaped, guard) = product_chain(&mut fx, &ctx, levels);
    let (plain, _) = product_chain(&mut fx, &ctx.unshaped(), levels);
    println!(
        "{levels} products: shaped {shaped:.1} bits (guard 2^{:.2}), unshaped {plain:.1} bits",
        guard.log2()
    );
    assert!(shaped > 35.0, "shaped precision {shaped}");
    assert!(shaped > plain + 20.0, "shaped {shaped} against unshaped {plain}");
    assert!(guard < 0.5, "guard {guard}");
}

#[test]
fn rotation_and_conjugation_move_the_values() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(11);
    let ct = fx.encrypt(&ctx, &re, &im);
    let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());

    fx.module
        .shaped_rotate_into(&ctx, &mut out, &ct, 1, &fx.atks, &mut fx.ops_scratch.borrow())
        .unwrap();
    let (have_re, have_im, guard) = fx.decrypt(&ctx, &out);
    let rotate = |v: &[f64]| -> Vec<f64> { (0..BATCH).map(|j| v[(j + 1) % BATCH]).collect() };
    let bits = precision((&have_re, &have_im), (&rotate(&re), &rotate(&im)));
    assert!(bits > 40.0 && guard < 0.01, "rotation: {bits} bits, guard {guard}");

    fx.module
        .shaped_conjugate_into(&ctx, &mut out, &ct, &fx.atks, &mut fx.ops_scratch.borrow())
        .unwrap();
    let (have_re, have_im, guard) = fx.decrypt(&ctx, &out);
    let conj: Vec<f64> = im.iter().map(|x| -x).collect();
    let bits = precision((&have_re, &have_im), (&re, &conj));
    assert!(bits > 40.0 && guard < 0.01, "conjugation: {bits} bits, guard {guard}");
}

#[test]
fn plaintext_and_masked_constant_products() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(13);
    let ct = fx.encrypt(&ctx, &re, &im);
    let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());

    // A plaintext vector at a 30-bit scale.
    let (w_re, w_im) = sample(14);
    let mut pt = fx.plaintext(ctx.layout().period(), 30);
    fx.module
        .shaped_encode_into(&ctx, &mut pt, &w_re, &w_im, &mut fx.scratch.borrow())
        .unwrap();
    fx.module
        .shaped_mul_pt_into(&ctx, &mut out, &ct, &pt, &mut fx.ops_scratch.borrow())
        .unwrap();
    let want_re: Vec<f64> = (0..BATCH).map(|j| re[j] * w_re[j] - im[j] * w_im[j]).collect();
    let want_im: Vec<f64> = (0..BATCH).map(|j| re[j] * w_im[j] + im[j] * w_re[j]).collect();
    let (have_re, have_im, _) = fx.decrypt(&ctx, &out);
    let bits = precision((&have_re, &have_im), (&want_re, &want_im));
    assert!(bits > 40.0, "plaintext product: {bits} bits");

    // A masked constant keeps the data and clears the guards, which an unmasked one keeps.
    let layout = *ctx.layout();
    let guard_value = 0.75;
    let slots = |v: &[f64], guard: f64| -> Vec<f64> {
        let mut s = vec![guard; layout.period()];
        for (j, &x) in v.iter().enumerate() {
            s[j * layout.replication()] = x;
        }
        s
    };
    let mut pt = fx.plaintext(layout.period(), LOG_DELTA);
    fx.module
        .shaped_encode_slots_into(
            &ctx,
            &mut pt,
            &slots(&re, guard_value),
            &slots(&im, 0.0),
            &mut fx.scratch.borrow(),
        )
        .unwrap();
    let mut noisy = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    fx.module
        .shaped_encrypt_sk(
            &ctx,
            &mut noisy,
            &pt,
            &fx.sk,
            &mut fx.xe,
            &mut fx.xa,
            &mut fx.scratch.borrow(),
        )
        .unwrap();
    let (_, _, guard_before) = fx.decrypt(&ctx, &noisy);
    assert!((guard_before - guard_value).abs() < 0.02, "guard {guard_before}");
    let want_re: Vec<f64> = re.iter().map(|x| x / 2.0).collect();
    let want_im: Vec<f64> = im.iter().map(|x| x / 2.0).collect();
    for masked in [true, false] {
        let mut half = fx.plaintext(layout.replication(), 30);
        fx.module
            .shaped_encode_constant_into(&ctx, &mut half, 0.5, 0.0, masked, &mut fx.scratch.borrow())
            .unwrap();
        fx.module
            .shaped_mul_pt_into(&ctx, &mut out, &noisy, &half, &mut fx.ops_scratch.borrow())
            .unwrap();
        let (have_re, have_im, guard) = fx.decrypt(&ctx, &out);
        let bits = precision((&have_re, &have_im), (&want_re, &want_im));
        assert!(bits > 40.0, "constant product (masked: {masked}): {bits} bits");
        if masked {
            assert!(guard < 0.02, "masked guard {guard}");
        } else {
            assert!((guard - guard_value / 2.0).abs() < 0.02, "unmasked guard {guard}");
        }
    }
}

#[test]
fn shaped_rescale_drops_scale_bits() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(17);
    let ct = fx.encrypt(&ctx, &re, &im);
    let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    fx.module
        .shaped_rescale_into(&ctx, &mut out, &ct, 8, &mut fx.ops_scratch.borrow())
        .unwrap();
    assert_eq!((out.log_delta(), out.log_budget()), (LOG_DELTA - 8, ct.log_budget()));
    let (have_re, have_im, _) = fx.decrypt(&ctx, &out);
    let bits = precision((&have_re, &have_im), (&re, &im));
    // Shaping keeps the data far below the 12-bit scale.
    assert!(bits > 30.0, "rescaled precision {bits}");
}

/// The NEON backends of the paper's measurements provide every shaped operation.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backends_are_supported() {
    fn supported<B: poulpy_hal::layouts::Backend>()
    where
        Module<B>: ShapedOps<B> + ShapedEncoding<B>,
    {
    }
    supported::<poulpy_cpu_arm::NTT4x30Neon>();
    supported::<poulpy_cpu_arm::FFT64Neon>();
}

/// Products of ciphertexts narrower than one limb go through the tensor instead of the one-pass
/// kernel, as in poulpy-ckks.
#[test]
fn products_narrower_than_a_limb() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (mut want_re, mut want_im) = sample(21);
    let mut ct = fx.encrypt_at(&ctx, &want_re, &want_im, 3 * LOG_DELTA + 8);
    for level in 0..2 {
        let (re, im) = sample(22 + level);
        let fresh = fx.encrypt_at(&ctx, &re, &im, ct.k().as_usize());
        fx.module
            .shaped_mul_assign(&ctx, &mut ct, &fresh, &fx.tsk, &mut fx.ops_scratch.borrow())
            .unwrap();
        for j in 0..BATCH {
            let (a, b) = (want_re[j], want_im[j]);
            want_re[j] = a * re[j] - b * im[j];
            want_im[j] = a * im[j] + b * re[j];
        }
    }
    assert_eq!(ct.k().as_usize(), LOG_DELTA + 8);
    let (re, im, guard) = fx.decrypt(&ctx, &ct);
    let bits = precision((&re, &im), (&want_re, &want_im));
    assert!(bits > 35.0 && guard < 0.5, "{bits} bits, guard {guard}");
}

#[test]
fn squares_rescaled_and_unrescaled() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(31);
    let ct = fx.encrypt(&ctx, &re, &im);
    let want_re: Vec<f64> = (0..BATCH).map(|j| re[j] * re[j] - im[j] * im[j]).collect();
    let want_im: Vec<f64> = (0..BATCH).map(|j| 2.0 * re[j] * im[j]).collect();

    let mut sq = fx.module.ckks_ciphertext_alloc(BASE2K.into(), (CT_K - LOG_DELTA).into());
    fx.module
        .shaped_square_into(&ctx, &mut sq, &ct, &fx.tsk, &mut fx.ops_scratch.borrow())
        .unwrap();
    assert_eq!(sq.log_delta(), LOG_DELTA);
    let (have_re, have_im, guard) = fx.decrypt(&ctx, &sq);
    let bits = precision((&have_re, &have_im), (&want_re, &want_im));
    assert!(bits > 40.0 && guard < 0.01, "square: {bits} bits, guard {guard}");

    // Unrescaled: twice the scale at the input width, then one shaped rescale.
    let mut wide = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    fx.module
        .shaped_square_unrescaled_into(&ctx, &mut wide, &ct, &fx.tsk, &mut fx.ops_scratch.borrow())
        .unwrap();
    assert_eq!((wide.log_delta(), wide.k().as_usize()), (2 * LOG_DELTA, CT_K));
    let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), (CT_K - LOG_DELTA).into());
    fx.module
        .shaped_rescale_into(&ctx, &mut out, &wide, LOG_DELTA, &mut fx.ops_scratch.borrow())
        .unwrap();
    let (have_re, have_im, guard) = fx.decrypt(&ctx, &out);
    let bits = precision((&have_re, &have_im), (&want_re, &want_im));
    assert!(bits > 40.0 && guard < 0.01, "unrescaled square: {bits} bits, guard {guard}");

    // A destination narrower than the input cannot hold an unrescaled square.
    let mut narrow = fx.module.ckks_ciphertext_alloc(BASE2K.into(), (CT_K - BASE2K).into());
    assert!(
        fx.module
            .shaped_square_unrescaled_into(&ctx, &mut narrow, &ct, &fx.tsk, &mut fx.ops_scratch.borrow())
            .is_err()
    );
}

#[test]
fn shallow_automorphisms_round_only_the_guard_bits() {
    let deep = context();
    let shallow = context().with_deep_automorphisms(false);
    let mut fx = Fixture::new(&deep);
    let (re, im) = sample(37);
    let ct = fx.encrypt(&deep, &re, &im);
    let rotate = |v: &[f64]| -> Vec<f64> { (0..BATCH).map(|j| v[(j + 1) % BATCH]).collect() };
    let mut bits = [0.0; 2];
    for (i, ctx) in [&deep, &shallow].into_iter().enumerate() {
        let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
        fx.module
            .shaped_rotate_into(ctx, &mut out, &ct, 1, &fx.atks, &mut fx.ops_scratch.borrow())
            .unwrap();
        let (have_re, have_im, guard) = fx.decrypt(ctx, &out);
        bits[i] = precision((&have_re, &have_im), (&rotate(&re), &rotate(&im)));
        assert!(guard < 0.01, "guard {guard}");
    }
    println!("rotation: deep {:.1} bits, shallow {:.1} bits", bits[0], bits[1]);
    assert!(bits[0] > 40.0 && bits[1] > 30.0, "{bits:?}");
}

#[test]
fn encoding_helpers_at_their_exact_scratch() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let mut scratch = ScratchOwned::<BE>::alloc(fx.module.shaped_encoding_tmp_bytes(&ctx, &glwe_layout()));
    let (re, im) = sample(41);
    let mut pt = fx.module.shaped_plaintext_alloc(&ctx, BASE2K, LOG_DELTA);
    fx.module
        .shaped_encode_into(&ctx, &mut pt, &re, &im, &mut scratch.borrow())
        .unwrap();
    let mut ct = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    fx.module
        .shaped_encrypt_sk(&ctx, &mut ct, &pt, &fx.sk, &mut fx.xe, &mut fx.xa, &mut scratch.borrow())
        .unwrap();
    let (mut have_re, mut have_im) = (vec![0.0; BATCH], vec![0.0; BATCH]);
    fx.module
        .shaped_decrypt(&ctx, &ct, &fx.sk, &mut have_re, &mut have_im, &mut scratch.borrow())
        .unwrap();
    let bits = precision((&have_re, &have_im), (&re, &im));
    assert!(bits > 40.0, "decrypted {bits} bits");

    // Constants on the R slots of a constant plaintext, by the automorphism-assign path too.
    let mut half = fx.module.shaped_constant_alloc(&ctx, BASE2K, 30);
    fx.module
        .shaped_encode_constant_into(&ctx, &mut half, 0.5, 0.0, true, &mut scratch.borrow())
        .unwrap();
    fx.module
        .shaped_mul_pt_assign(&ctx, &mut ct, &half, &mut fx.ops_scratch.borrow())
        .unwrap();
    let gal = fx.module.ckks_galois_element(ctx.layout().rotation_slots(1));
    fx.module
        .shaped_automorphism_assign(&ctx, &mut ct, gal, &fx.atks, &mut fx.ops_scratch.borrow())
        .unwrap();
    fx.module
        .shaped_decrypt(&ctx, &ct, &fx.sk, &mut have_re, &mut have_im, &mut scratch.borrow())
        .unwrap();
    let want = |v: &[f64]| -> Vec<f64> { (0..BATCH).map(|j| v[(j + 1) % BATCH] / 2.0).collect() };
    let bits = precision((&have_re, &have_im), (&want(&re), &want(&im)));
    assert!(bits > 40.0, "half, rotated: {bits} bits");

    // Wrong lengths are errors, not panics.
    let mut short = vec![0.0; BATCH - 1];
    assert!(
        fx.module
            .shaped_decrypt(&ctx, &ct, &fx.sk, &mut short, &mut have_im, &mut scratch.borrow())
            .is_err()
    );
    assert!(
        fx.module
            .shaped_encode_into(&ctx, &mut pt, &re[1..], &im[1..], &mut scratch.borrow())
            .is_err()
    );
}

#[test]
fn invalid_requests_are_errors() {
    let ctx = context();
    let mut fx = Fixture::new(&ctx);
    let (re, im) = sample(43);
    let ct = fx.encrypt(&ctx, &re, &im);
    let mut out = fx.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    let cap = ctx.pass_cap();
    assert!(
        fx.module
            .shaped_rescale_into(&ctx, &mut out, &ct, cap + 1, &mut fx.ops_scratch.borrow())
            .is_err(),
        "a rescale beyond the pass cap"
    );
    assert!(context().with_pass_cap(61).is_err());
    assert!(context().with_encode_bits(130).is_err());
    assert!(context().with_encode_bits(60).is_ok());
}
