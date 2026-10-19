//! Conversions between CKKS and the replicated layout, against the plaintext values.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncryptOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk,
    layouts::{
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESecretSampling, LWEInfos, ModuleCoreAlloc,
        Rank, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::{
    context::ShapedContext,
    conversion::{ConversionParams, Converter},
    encoding::ShapedEncoding,
    unit::ShapingUnit,
};

type BE = NTT4x30Portable;
type Ct = CKKSCiphertextOwned<BE>;

const N: usize = 1 << 12;
const K: usize = 16;
const BASE2K: usize = 52;
const KEY_K: usize = 4 * BASE2K;

struct Fixture {
    module: Module<BE>,
    ctx: ShapedContext,
    conv: Converter<BE>,
    sk: GLWESecretPrepared<AlignedBuf, BE>,
    keys: HashMap<i64, GLWEAutomorphismKeyPrepared<AlignedBuf, BE>>,
    scratch: ScratchOwned<BE>,
    /// Exactly the scratch the converter asks for.
    conv_scratch: ScratchOwned<BE>,
    values: (Vec<f64>, Vec<f64>),
}

impl Fixture {
    fn new() -> Self {
        let module = Module::<BE>::new(N as u64);
        let ctx = ShapedContext::new(N, K, ShapingUnit::default_unit()).unwrap();
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 26);
        let conv = Converter::new(&module, &ctx, ConversionParams::default(), BASE2K, &mut scratch.borrow()).unwrap();
        let glwe = GLWELayout {
            n: N.into(),
            base2k: BASE2K.into(),
            k: KEY_K.into(),
            rank: Rank(1),
        };
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, 64, &mut Source::new([1; 32]));
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let atk = GLWEAutomorphismKeyLayout {
            n: N.into(),
            base2k: BASE2K.into(),
            k_aux: 53usize.into(),
            rank: Rank(1),
            dsize: 1usize.into(),
            dnum: (KEY_K / BASE2K).into(),
        };
        let (mut xe, mut xa) = (Source::new([2; 32]), Source::new([3; 32]));
        let mut keys = HashMap::new();
        for gal in conv.galois_elements(&module) {
            let mut key = module.glwe_automorphism_key_alloc_from_infos(&atk);
            module.glwe_automorphism_key_encrypt_sk(&mut key, gal, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk);
            module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut scratch.borrow());
            keys.insert(gal, prepared);
        }
        let widest = GLWELayout {
            k: 150usize.into(),
            ..glwe
        };
        let conv_scratch = ScratchOwned::<BE>::alloc(conv.tmp_bytes(&module, &widest, &atk));
        let values = (0..K)
            .map(|j| {
                let th = 0.37 + 1.9 * j as f64;
                let radius = 0.3 + 0.7 * (j as f64 / K as f64);
                (radius * th.cos(), radius * th.sin())
            })
            .unzip();
        Self {
            module,
            ctx,
            conv,
            sk,
            keys,
            scratch,
            conv_scratch,
            values,
        }
    }

    /// A stock CKKS encryption of the values with period `k`.
    fn ckks_encrypt(&mut self, log_delta: usize, k: usize) -> Ct {
        let mut pt = self
            .module
            .ckks_pt_vec_alloc_compact(K, BASE2K.into(), TorusPrecision((log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: (N / (2 * K)).trailing_zeros() as usize,
            slots: SlotsKind::Complex,
        });
        self.module
            .ckks_encode_reim_into(&mut pt, &self.values.0, &self.values.1, &mut self.scratch.borrow())
            .unwrap();
        let mut ct = self.module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(k as u32));
        self.module
            .ckks_encrypt_sk(
                &mut ct,
                &pt,
                &self.sk,
                &mut Source::new([4; 32]),
                &mut Source::new([5; 32]),
                &mut self.scratch.borrow(),
            )
            .unwrap();
        ct
    }

    /// A shaped encryption of the values on the replicated layout.
    fn shaped_encrypt(&mut self, log_delta: usize, k: usize) -> Ct {
        let mut pt = self.module.ckks_pt_vec_alloc_compact(
            self.ctx.layout().period(),
            BASE2K.into(),
            TorusPrecision((log_delta + 8) as u32),
        );
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        });
        self.module
            .shaped_encode_into(&self.ctx, &mut pt, &self.values.0, &self.values.1, &mut self.scratch.borrow())
            .unwrap();
        let mut ct = self.module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(k as u32));
        self.module
            .shaped_encrypt_sk(
                &self.ctx,
                &mut ct,
                &pt,
                &self.sk,
                &mut Source::new([6; 32]),
                &mut Source::new([7; 32]),
                &mut self.scratch.borrow(),
            )
            .unwrap();
        ct
    }

    /// One period of `period` slots of `ct`.
    fn slots(&mut self, ct: &Ct, period: usize) -> (Vec<f64>, Vec<f64>) {
        let mut pt = self.module.ckks_plaintext_alloc(
            N.into(),
            BASE2K.into(),
            TorusPrecision((ct.log_delta() + ct.log_budget().min(24)) as u32),
        );
        pt.set_meta(ct.meta());
        self.module
            .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let (mut re, mut im) = (vec![0.0; period], vec![0.0; period]);
        self.module
            .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.scratch.borrow())
            .unwrap();
        (re, im)
    }

    fn error(&self, re: &[f64], im: &[f64]) -> f64 {
        (0..K)
            .map(|j| (re[j] - self.values.0[j]).abs().max((im[j] - self.values.1[j]).abs()))
            .fold(0.0, f64::max)
    }

    /// Precision on the CKKS slots: every slot of the ring against the values. Decoding one
    /// period of `k` averages the `N / 2k` copies of each value, which hides the error of a single
    /// slot, and the conversions read single slots.
    fn ckks_bits(&mut self, ct: &Ct) -> f64 {
        let (re, im) = self.slots(ct, N / 2);
        let err = (0..N / 2)
            .map(|t| (re[t] - self.values.0[t % K]).abs().max((im[t] - self.values.1[t % K]).abs()))
            .fold(0.0, f64::max);
        -err.log2()
    }

    /// Precision on the data slots and largest guard of a ciphertext of the replicated layout.
    fn shaped_bits(&mut self, ct: &Ct) -> (f64, f64) {
        let layout = *self.ctx.layout();
        let (re, im) = self.slots(ct, layout.period());
        let guard = layout
            .guards(&re)
            .zip(layout.guards(&im))
            .map(|(a, b)| a.hypot(b))
            .fold(0.0, f64::max);
        (-self.error(&layout.collect(&re), &layout.collect(&im)).log2(), guard)
    }
}

#[test]
fn ckks_to_shaped_keeps_the_incoming_precision() {
    let mut f = Fixture::new();
    let p = *f.conv.params();
    let ct = f.ckks_encrypt(p.ckks_log_delta, 150);
    let incoming = f.ckks_bits(&ct);
    let out = f
        .conv
        .to_shaped(&f.module, &ct, &f.keys, &mut f.conv_scratch.borrow())
        .unwrap();
    assert_eq!(out.log_delta(), p.log_delta);
    assert_eq!(
        out.k().as_usize(),
        150 - p.diagonal_log_delta - (p.ckks_log_delta - p.log_delta)
    );
    let (bits, guard) = f.shaped_bits(&out);
    println!(
        "CKKS {incoming:.1} bits -> shaped {bits:.1} bits, guard 2^{:.2}",
        guard.log2()
    );
    assert!(incoming > 28.0, "incoming {incoming}");
    assert!(bits > incoming - 1.0, "after the conversion {bits} from {incoming}");
    assert!(guard < 0.25, "guard {guard}");
}

#[test]
fn shaped_to_ckks_repeats_the_values_on_every_slot() {
    let mut f = Fixture::new();
    let p = *f.conv.params();
    let ct = f.shaped_encrypt(p.log_delta, 100);
    let (incoming, _) = f.shaped_bits(&ct);
    let out = f.conv.to_ckks(&f.module, &ct, &f.keys, &mut f.conv_scratch.borrow()).unwrap();
    assert_eq!(out.log_delta(), p.ckks_log_delta);
    assert_eq!(
        out.k().as_usize(),
        100 + (p.ckks_log_delta - p.log_delta) - p.diagonal_log_delta
    );
    let bits = f.ckks_bits(&out);
    println!("shaped {incoming:.1} bits -> CKKS {bits:.1} bits");
    assert!(incoming > 40.0 && bits > 22.0, "{incoming} -> {bits}");
}

#[test]
fn round_trip_returns_the_values() {
    let mut f = Fixture::new();
    let p = *f.conv.params();
    let ct = f.ckks_encrypt(p.ckks_log_delta, 150);
    let there = f
        .conv
        .to_shaped(&f.module, &ct, &f.keys, &mut f.conv_scratch.borrow())
        .unwrap();
    let back = f
        .conv
        .to_ckks(&f.module, &there, &f.keys, &mut f.conv_scratch.borrow())
        .unwrap();
    let bits = f.ckks_bits(&back);
    let again = f
        .conv
        .to_shaped(&f.module, &back, &f.keys, &mut f.conv_scratch.borrow())
        .unwrap();
    let (bits2, guard) = f.shaped_bits(&again);
    println!(
        "round trip {bits:.1} bits, and back {bits2:.1} bits, guard 2^{:.2}",
        guard.log2()
    );
    assert!(bits > 22.0 && bits2 > bits - 1.0 && guard < 0.25);
}
