//! End-to-end bootstrapping on the replicated layout.

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::CKKSDecryptOps,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::layouts::{
    GLWELayout, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GLWESecretPreparedFactory,
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::{
    bootstrapping::{BootstrapParams, Bootstrapper},
    context::ShapedContext,
    encoding::ShapedEncoding,
    ops::ShapedOps,
    unit::ShapingUnit,
};

type BE = NTT4x30Portable;

const N: usize = 1 << 12;
const BATCH: usize = 16;
const BASE2K: usize = 52;
const LEVELS: usize = 2;

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

#[test]
fn bootstrapping_restores_precision_and_bounded_guards() {
    let module = Module::<BE>::new(N as u64);
    let ctx = ShapedContext::new(N, BATCH, ShapingUnit::default_unit()).unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 27);
    let params = BootstrapParams::default();
    let boot = Bootstrapper::new(&module, &ctx, params, BASE2K, LEVELS, &mut scratch.borrow()).unwrap();
    println!("k_in {} k_boot {}", boot.k_in(), boot.k_boot());

    let layout = GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: boot.k_boot().into(),
        rank: Rank(1),
    };
    let (mut xs, mut xa, mut xe) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, 64, &mut xs);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&layout);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    let key_layouts = boot.key_layouts(1, 53).unwrap();
    assert!(
        boot.key_layouts(2, 67).is_err(),
        "an auxiliary modulus below one digit is refused"
    );
    // No key is within the caps at this toy ring degree.
    assert!(boot.secure_key_layouts(64).is_err());
    let keys = boot.generate_keys(
        &module,
        &sk_raw,
        &key_layouts,
        &mut xs,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );

    let encrypt =
        |re: &[f64], im: &[f64], log_delta: usize, k: usize, scratch: &mut ScratchOwned<BE>, xe: &mut Source, xa: &mut Source| {
            let mut pt =
                module.ckks_pt_vec_alloc_compact(ctx.layout().period(), BASE2K.into(), TorusPrecision((log_delta + 8) as u32));
            pt.set_meta(CKKSMeta {
                log_delta,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            });
            module
                .shaped_encode_into(&ctx, &mut pt, re, im, &mut scratch.borrow())
                .unwrap();
            let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(k as u32));
            module
                .shaped_encrypt_sk(&ctx, &mut ct, &pt, &sk, xe, xa, &mut scratch.borrow())
                .unwrap();
            ct
        };
    let decrypt = |ct: &CKKSCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        let mut pt = module.ckks_plaintext_alloc(N.into(), BASE2K.into(), TorusPrecision((ct.log_delta() + 24) as u32));
        pt.set_meta(ct.meta());
        module.ckks_decrypt(&mut pt, ct, &sk, &mut scratch.borrow()).unwrap();
        let period = ctx.layout().period();
        let (mut re, mut im) = (vec![0.0; period], vec![0.0; period]);
        module
            .shaped_decode_slots(&ctx, &pt, &mut re, &mut im, &mut scratch.borrow())
            .unwrap();
        let guard = (0..period)
            .filter(|t| !ctx.layout().is_data_slot(*t))
            .map(|t| re[t].hypot(im[t]))
            .fold(0.0, f64::max);
        (ctx.layout().collect(&re), ctx.layout().collect(&im), guard)
    };

    // Exactly the scratch the bootstrapping asks for.
    let mut boot_scratch = ScratchOwned::<BE>::alloc(boot.tmp_bytes(&module, &key_layouts));

    let (re, im) = sample(1);
    let ct = encrypt(&re, &im, params.log_delta, boot.k_in(), &mut scratch, &mut xe, &mut xa);
    let t = std::time::Instant::now();
    let out = boot.bootstrap(&module, &ct, &keys, &mut boot_scratch.borrow()).unwrap();
    println!("bootstrap {:.2} s", t.elapsed().as_secs_f64());
    let (have_re, have_im, guard) = decrypt(&out, &mut scratch);
    let bits = precision((&have_re, &have_im), (&re, &im));
    println!(
        "after bootstrapping: {bits:.1} bits, guard 2^{:.2}, log_delta {} k {}",
        guard.log2(),
        out.log_delta(),
        out.k().as_usize()
    );
    assert_eq!(out.log_delta(), params.level_log_delta);
    assert_eq!(
        out.k().as_usize(),
        boot.k_out(LEVELS),
        "the output keeps the levels it was sized for"
    );
    assert!(bits > 18.0, "precision after bootstrapping {bits}");
    assert!(guard < 0.5, "guard after bootstrapping {guard}");

    // The levels the bootstrapping was sized for, by fresh operands.
    let (mut want_re, mut want_im) = (re.clone(), im.clone());
    let mut cur = out;
    for level in 0..LEVELS {
        let (fr, fi) = sample(10 + level as u64);
        let fresh = encrypt(&fr, &fi, cur.log_delta(), cur.k().as_usize(), &mut scratch, &mut xe, &mut xa);
        module
            .shaped_mul_assign(&ctx, &mut cur, &fresh, &keys.tensor, &mut scratch.borrow())
            .unwrap();
        for j in 0..BATCH {
            let (a, b) = (want_re[j], want_im[j]);
            want_re[j] = a * fr[j] - b * fi[j];
            want_im[j] = a * fi[j] + b * fr[j];
        }
    }
    let (have_re, have_im, guard) = decrypt(&cur, &mut scratch);
    let bits = precision((&have_re, &have_im), (&want_re, &want_im));
    println!(
        "after {LEVELS} levels: {bits:.1} bits, guard 2^{:.2}, k {}",
        guard.log2(),
        cur.k().as_usize()
    );
    assert!(bits > 18.0, "precision after the levels {bits}");
    assert_eq!(cur.k().as_usize(), boot.k_in(), "the levels end at the input width");

    // EvalMod split into its real and imaginary parts, with the same keys, agrees.
    assert!(boot.packs_eval_mod());
    let split_params = BootstrapParams {
        packed_eval_mod: false,
        ..params
    };
    let split = Bootstrapper::new(&module, &ctx, split_params, BASE2K, LEVELS, &mut scratch.borrow()).unwrap();
    assert!(!split.packs_eval_mod());
    let mut split_scratch = ScratchOwned::<BE>::alloc(split.tmp_bytes(&module, &key_layouts));
    let out = split.bootstrap(&module, &ct, &keys, &mut split_scratch.borrow()).unwrap();
    let (have_re, have_im, guard) = decrypt(&out, &mut scratch);
    let bits = precision((&have_re, &have_im), (&re, &im));
    println!("split EvalMod: {bits:.1} bits, guard 2^{:.2}", guard.log2());
    assert!(bits > 18.0 && guard < 0.5, "split EvalMod: {bits} bits, guard {guard}");

    // The exhausted ciphertext, at the output scale, bootstraps again.
    let again = boot.bootstrap(&module, &cur, &keys, &mut boot_scratch.borrow()).unwrap();
    let (have_re, have_im, guard) = decrypt(&again, &mut scratch);
    let bits = precision((&have_re, &have_im), (&want_re, &want_im));
    println!("bootstrapped again: {bits:.1} bits, guard 2^{:.2}", guard.log2());
    assert_eq!(again.k().as_usize(), boot.k_out(LEVELS));
    assert!(bits > 18.0, "precision after the second bootstrapping {bits}");
    assert!(guard < 0.5, "guard after the second bootstrapping {guard}");
}

#[test]
fn invalid_parameters_are_refused() {
    let module = Module::<BE>::new(N as u64);
    let ctx = ShapedContext::new(N, BATCH, ShapingUnit::default_unit()).unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 24);
    let default = BootstrapParams::default();
    // The final rescale would drop 45 + 28 - 10 bits in one shaped rounding.
    let too_far = BootstrapParams {
        log_delta: 45,
        level_log_delta: 10,
        ..default
    };
    // An output scale above the input scale.
    let upside_down = BootstrapParams {
        level_log_delta: 40,
        defer_last_step: false,
        ..default
    };
    for params in [too_far, upside_down] {
        assert!(Bootstrapper::new(&module, &ctx, params, BASE2K, 1, &mut scratch.borrow()).is_err());
    }
    // R must divide the batch.
    let small = ShapedContext::new(N, 4, ShapingUnit::default_unit()).unwrap();
    assert!(Bootstrapper::new(&module, &small, default, BASE2K, 1, &mut scratch.borrow()).is_err());
}
