//! A leveled computation on a small batch, end to end: keys, encoding and encryption, shaped
//! products and rotations, decryption.
//!
//! ```bash
//! cargo run --release --example quickstart
//! ```
//!
//! It computes `z = x y + rot(x y, 1)` and then `z^2` on 16 complex values, and compares with the
//! same computation in the clear.

use poulpy_ckks::{CKKSInfos, CKKSModuleInfos, api::CKKSAddOps, layouts::CKKSModuleAlloc};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    layouts::{
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESecretSampling, GLWETensorKeyLayout,
        GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GLWESecretPreparedFactory,
    },
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::prelude::*;

type BE = NTT4x30Portable;

const N: usize = 1 << 12;
const BATCH: usize = 16;
/// Limbs of 52 bits: four of them hold a fresh ciphertext with room for a few products.
const BASE2K: usize = 52;
const CT_K: usize = 4 * BASE2K;
/// The leveled scale of the paper: every product consumes 20 bits.
const LOG_DELTA: usize = 20;
/// Keys one limb wider than the ciphertexts, with a 53-bit auxiliary modulus.
const KEY_K: usize = CT_K + BASE2K;
const K_AUX: usize = 53;

fn main() {
    let module = Module::<BE>::new(N as u64);
    // The public parameters: ring, batch, and the default unit U_{32,45} (R = 16).
    let ctx = ShapedContext::new(N, BATCH, ShapingUnit::default_unit()).unwrap();
    println!(
        "N = {N}, {BATCH} values, R = {}, {} copies of each value",
        ctx.layout().replication(),
        ctx.layout().copies()
    );

    // Keys: a secret, the relinearization key and the key of a rotation by one value. They are
    // ordinary poulpy keys.
    let ct_layout = GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    };
    let tsk_layout = GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k_aux: K_AUX.into(),
        rank: Rank(1),
        dsize: 1usize.into(),
        dnum: KEY_K.div_ceil(BASE2K).into(),
    };
    let atk_layout = GLWEAutomorphismKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k_aux: K_AUX.into(),
        rank: Rank(1),
        dsize: 1usize.into(),
        dnum: KEY_K.div_ceil(BASE2K).into(),
    };
    // Scratch for the shaped operations and the encoding, and a margin for key generation.
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .shaped_ops_tmp_bytes(&ctx, &ct_layout, &tsk_layout, &atk_layout)
            .max(module.shaped_encoding_tmp_bytes(&ctx, &ct_layout))
            .max(1 << 24),
    );
    let (mut xs, mut xa, mut xe) = (Source::new([1; 32]), Source::new([2; 32]), Source::new([3; 32]));
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&ct_layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, 64, &mut xs);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&ct_layout);
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    let mut tsk_raw = module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
    module.glwe_tensor_key_encrypt_sk(&mut tsk_raw, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
    let mut tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
    module.prepare_tensor_key(&mut tsk, &tsk_raw, &mut scratch.borrow());

    // A rotation by one value moves R slots.
    let gal = module.ckks_galois_element(ctx.layout().rotation_slots(1));
    let mut atk_raw = module.glwe_automorphism_key_alloc_from_infos(&atk_layout);
    module.glwe_automorphism_key_encrypt_sk(&mut atk_raw, gal, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
    let mut atk = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    module.glwe_automorphism_key_prepare(&mut atk, &atk_raw, &mut scratch.borrow());
    let atks = std::collections::HashMap::from([(gal, atk)]);

    // Two batches, encoded with shaping and encrypted with a shaped error.
    let x: Vec<(f64, f64)> = (0..BATCH)
        .map(|j| (0.9 * (0.3 + j as f64).cos(), 0.9 * (0.3 + j as f64).sin()))
        .collect();
    let y: Vec<(f64, f64)> = (0..BATCH).map(|j| (0.5 - j as f64 / 32.0, 0.25)).collect();
    let mut encrypt = |values: &[(f64, f64)]| {
        let (re, im): (Vec<f64>, Vec<f64>) = values.iter().copied().unzip();
        let mut pt = module.shaped_plaintext_alloc(&ctx, BASE2K, LOG_DELTA);
        module
            .shaped_encode_into(&ctx, &mut pt, &re, &im, &mut scratch.borrow())
            .unwrap();
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(CT_K as u32));
        module
            .shaped_encrypt_sk(&ctx, &mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();
        ct
    };
    let ct_x = encrypt(&x);
    let ct_y = encrypt(&y);

    // z = x y + rot(x y, 1), then z^2. A product returns a rescaled result, one level narrower.
    let level = |k: usize| module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision((k - LOG_DELTA) as u32));
    let mut xy = level(CT_K);
    module
        .shaped_mul_into(&ctx, &mut xy, &ct_x, &ct_y, &tsk, &mut scratch.borrow())
        .unwrap();
    let mut z = module.ckks_ciphertext_alloc_from_infos(&xy);
    module
        .shaped_rotate_into(&ctx, &mut z, &xy, 1, &atks, &mut scratch.borrow())
        .unwrap();
    module.ckks_add_assign(&mut z, &xy, &mut scratch.borrow()).unwrap();
    module
        .shaped_square_assign(&ctx, &mut z, &tsk, &mut scratch.borrow())
        .unwrap();

    // Decryption averages the copies of each value.
    let (mut re, mut im) = (vec![0.0; BATCH], vec![0.0; BATCH]);
    module
        .shaped_decrypt(&ctx, &z, &sk, &mut re, &mut im, &mut scratch.borrow())
        .unwrap();

    let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let xy: Vec<(f64, f64)> = x.iter().zip(&y).map(|(&a, &b)| mul(a, b)).collect();
    let mut err: f64 = 0.0;
    for j in 0..BATCH {
        let s = xy[j];
        let r = xy[(j + 1) % BATCH];
        let w = mul((s.0 + r.0, s.1 + r.1), (s.0 + r.0, s.1 + r.1));
        err = err.max((re[j] - w.0).abs()).max((im[j] - w.1).abs());
    }
    println!(
        "z^2: scale 2^{}, width {} bits, precision {:.1} bits",
        z.log_delta(),
        z.k().as_usize(),
        -err.log2()
    );
}
