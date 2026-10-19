//! Bootstrapping a small batch at the parameters of the paper, timed.
//!
//! ```bash
//! cargo run --release --example bootstrap -- [log_n] [batch] [levels] [repetitions] [threads]
//! ```
//!
//! Defaults: `N = 2^15`, 16 values, 1 level, 3 repetitions, one thread. Any other thread count
//! runs on the Rayon NEON backend with every core (on aarch64). Keys use two limbs per digit where
//! they fit the 128-bit caps for a secret of weight 1024, and one limb per digit beyond.

use std::time::Instant;

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::CKKSDecryptOps,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::layouts::{
    GLWELayout, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GLWESecretPreparedFactory,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::{
    bootstrapping::{BootstrapParams, Bootstrapper},
    context::ShapedContext,
    encoding::ShapedEncoding,
    unit::ShapingUnit,
};

use poulpy_ckks::oep::CKKSEncapsulatedModUpImpl;
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, GLWETrace, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedFactory, GLWESwitchingKeyPreparedFactory,
        GLWETensorKeyPreparedFactory, GetTensorKey,
        prepared::{GLWESwitchingKeyPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{api::ModuleNew, layouts::ScratchArena};
use poulpy_shaped::{backend::HostBackend, bootstrapping::BootstrapModule};

const BASE2K: usize = 52;
const HAMMING_WEIGHT: usize = 1024;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer argument"))
        .collect();
    let threads = args.get(4).copied().unwrap_or(1);
    #[cfg(target_arch = "aarch64")]
    if threads == 1 {
        run::<poulpy_cpu_arm::NTT4x30Neon>(&args);
    } else {
        run::<poulpy_cpu_arm::NTT4x30NeonRayon>(&args);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = threads;
        run::<poulpy_cpu_portable::NTT4x30Portable>(&args);
    }
}

fn run<BE>(args: &[usize])
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: BootstrapModule<BE>
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
        + CKKSDecryptOps<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GetTensorKey<BE>,
{
    let log_n = args.first().copied().unwrap_or(15);
    let batch = args.get(1).copied().unwrap_or(16);
    let levels = args.get(2).copied().unwrap_or(1);
    let repetitions = args.get(3).copied().unwrap_or(3).max(1);
    let n = 1usize << log_n;
    let unit = ShapingUnit::default_unit();
    // Batches smaller than R are padded to R values.
    let padded = batch.max(unit.replication());

    let module = Module::<BE>::new(n as u64);
    let ctx = ShapedContext::new(n, padded, unit).unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 30);
    let params = BootstrapParams::default();
    let t = Instant::now();
    let boot = Bootstrapper::new(&module, &ctx, params, BASE2K, levels, &mut scratch.borrow()).unwrap();
    // Two limbs per digit where the keys fit the 128-bit caps, one limb per digit beyond. Rings
    // too small for any secure modulus run with one-limb keys, flagged below.
    let key_layouts = boot
        .secure_key_layouts(HAMMING_WEIGHT)
        .or_else(|_| boot.key_layouts(1, BASE2K + 1))
        .unwrap();
    for check in boot.security(&key_layouts, HAMMING_WEIGHT) {
        println!("{check}");
    }
    let layout = GLWELayout {
        n: n.into(),
        base2k: BASE2K.into(),
        k: boot.k_boot().into(),
        rank: Rank(1),
    };
    let (mut xs, mut xa, mut xe) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut xs);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&layout);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    let keys = boot.generate_keys(
        &module,
        &sk_raw,
        &key_layouts,
        &mut xs,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );
    println!(
        "N = 2^{log_n}, {batch} values, {levels} levels: k_in {}, k_boot {}, setup {:.1} s",
        boot.k_in(),
        boot.k_boot(),
        t.elapsed().as_secs_f64()
    );

    // Values on the unit circle, zero on the padding.
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let (re, im): (Vec<f64>, Vec<f64>) = (0..padded)
        .map(|j| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = (s >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
            if j < batch { (a.cos(), a.sin()) } else { (0.0, 0.0) }
        })
        .unzip();
    let mut pt = module.ckks_pt_vec_alloc_compact(
        ctx.layout().period(),
        BASE2K.into(),
        TorusPrecision((params.log_delta + 8) as u32),
    );
    pt.set_meta(CKKSMeta {
        log_delta: params.log_delta,
        log_sparsity: 0,
        slots: SlotsKind::Complex,
    });
    module
        .shaped_encode_into(&ctx, &mut pt, &re, &im, &mut scratch.borrow())
        .unwrap();
    let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(boot.k_in() as u32));
    module
        .shaped_encrypt_sk(&ctx, &mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())
        .unwrap();

    let mut boot_scratch = ScratchOwned::<BE>::alloc(boot.tmp_bytes(&module, &key_layouts));
    let mut best = f64::INFINITY;
    let mut out: Option<CKKSCiphertextOwned<BE>> = None;
    for _ in 0..repetitions {
        let t = Instant::now();
        out = Some(boot.bootstrap(&module, &ct, &keys, &mut boot_scratch.borrow()).unwrap());
        best = best.min(t.elapsed().as_secs_f64());
    }
    let out = out.unwrap();

    let mut dec = module.ckks_plaintext_alloc(n.into(), BASE2K.into(), TorusPrecision((out.log_delta() + 24) as u32));
    dec.set_meta(out.meta());
    module.ckks_decrypt(&mut dec, &out, &sk, &mut scratch.borrow()).unwrap();
    let period = ctx.layout().period();
    let (mut have_re, mut have_im) = (vec![0.0; period], vec![0.0; period]);
    module
        .shaped_decode_slots(&ctx, &dec, &mut have_re, &mut have_im, &mut scratch.borrow())
        .unwrap();
    let guard = ctx
        .layout()
        .guards(&have_re)
        .zip(ctx.layout().guards(&have_im))
        .map(|(a, b): (f64, f64)| a.hypot(b))
        .fold(0.0, f64::max);
    let (have_re, have_im) = (ctx.layout().collect(&have_re), ctx.layout().collect(&have_im));
    let err = (0..batch)
        .map(|j| (have_re[j] - re[j]).abs().max((have_im[j] - im[j]).abs()))
        .fold(0.0, f64::max);
    println!(
        "bootstrap {best:.3} s (best of {repetitions}), precision {:.1} bits, guard 2^{:.2}, output k {}",
        -err.log2(),
        guard.log2(),
        out.k().as_usize()
    );
}
