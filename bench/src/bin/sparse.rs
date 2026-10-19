//! The sparse CKKS baseline: poulpy's SlotsToCoeffs-first bootstrapping of `k` values packed with
//! period `k`, under the protocol of `shaped boot`.
//!
//! ```text
//! sparse [--log-n 15] [--k 16] [--levels 1] [--factors 1] [--dsize 1 --k-aux 53] [--d2s-dsize 1]
//!        [--seeds ...] [--families ...] [--reps 3] [--bench time|check|both] [--chains fresh,square]
//!        [--threads 1] [--decode full|average] [--dry-run]
//! ```
//!
//! Stock operations only, with the paper's baseline parameters: 36-bit levels, SlotsToCoeffs at
//! 38 bits, CoeffsToSlots and EvalMod constants at 48 bits, EvalMod at 60 bits, the same EvalMod
//! polynomial as our scheme, and sparse-secret encapsulation of weight 32. After ModUp a trace onto
//! the subring of period `k` removes the lifted multiples outside it, which poulpy's own entry
//! point does not do for sparse packings.

#![feature(f128)]

use std::time::Instant;

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, CKKSModuleInfos, CoeffsMeta, Quad, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSDFTOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSEvalModOps,
        CKKSImagOps, CKKSMulOps, CKKSPow2Ops,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan,
        BootstrappingTechniques, CKKSCiphertextOwned, CKKSModuleAlloc, DFTOutputFormat, DFTPlan, DFTType,
        EncapsulationKeysLayout, EvalModPlan, EvalModType, SparseSecretEncapsulation,
    },
    oep::CKKSEncapsulatedModUpImpl,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETrace,
    layouts::{
        Dsize, GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWESecretSampling, GLWESwitchingKeyLayout, GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, Rank,
        SplitStrategy, TorusPrecision, prepared::GLWESecretPreparedFactory,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use poulpy_shaped::security::max_modulus_bits;
use poulpy_shaped_bench::{Args, Line, Quality, Rng, dispatch};

const BASE2K: usize = 52;
const HAMMING_WEIGHT: usize = 1024;
const EPHEMERAL_WEIGHT: usize = 32;
const LOG_DELTA: usize = 36;
const LOG_MSG_RATIO: usize = 13;
const S2C_BITS: usize = 38;
const C2S_BITS: usize = 48;
const C2S_GUARD: usize = 6;
const EVAL_MOD_BITS: usize = 60;
const CONST_BITS: usize = 48;

type Ct<BE> = CKKSCiphertextOwned<BE>;

fn main() {
    let args = Args::parse();
    let threads: usize = args.get("threads", 1);
    dispatch!(threads, run(&args));
}

fn seed32(tag: u8, seed: u64) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&seed.to_le_bytes());
    s
}

/// The transform schedule over `2^log_slots` slots: one factor with giant step 4, or `factors`
/// factors of nearly equal depth.
fn schedule(log_slots: usize, factors: usize) -> Vec<(usize, usize)> {
    if factors <= 1 {
        return vec![(log_slots, 4)];
    }
    let f = factors.min(log_slots).max(1);
    (0..f)
        .map(|i| {
            let depth = log_slots / f + usize::from(i < log_slots % f);
            (depth, 1 << depth.div_ceil(2))
        })
        .collect()
}

fn plan(log_slots: usize, factors: usize) -> BootstrappingPlan {
    let c2s = DFTPlan::new(
        DFTType::Encode,
        schedule(log_slots, factors),
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(C2S_BITS, 2),
    )
    .unwrap();
    let s2c = DFTPlan::new(
        DFTType::Decode,
        schedule(log_slots, factors),
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(S2C_BITS, 2),
    )
    .unwrap()
    .with_scaling(0.5)
    .unwrap();
    BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: EPHEMERAL_WEIGHT,
            }),
            eval_round_plus: None,
        },
        c2s,
        EvalModPlan {
            eval_mod_type: EvalModType::CosHKEven,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: 30,
            f_mod_interval: 16,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(CONST_BITS, 4),
            f_mod_log_delta: EVAL_MOD_BITS,
        },
        s2c,
    )
    .unwrap()
    .with_c2s_guard_bits(C2S_GUARD)
    .unwrap()
}

fn key_bytes<K: GGLWEInfos>(key: &K, n: usize) -> usize {
    key.dnum().as_usize() * 2 * key.size() * n * 8
}

fn run<BE>(a: &Args)
where
    BE: Backend<ZnxWord = i64, Ring = poulpy_hal::layouts::Standard> + CKKSEncapsulatedModUpImpl,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    Module<BE>: ModuleNew<BE>
        + poulpy_core::GLWETensorKeyEncryptSk<BE>
        + poulpy_core::GLWESwitchingKeyEncryptSk<BE>
        + poulpy_hal::layouts::CyclotomicOrder
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETrace<BE>
        + CKKSModuleAlloc<BE>
        + CKKSModuleInfos
        + CKKSEncodingOps<BE, f64>
        + CKKSEncodingOps<BE, Quad>
        + CKKSEncryptOps<BE>
        + CKKSDecryptOps<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSImagOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + poulpy_ckks::api::CKKSDFTMatrixOps<BE, f64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BootstrappingContext<BE, f64>: Sized,
    poulpy_ckks::layouts::CKKSPlaintextOwned<BE>: poulpy_core::layouts::GLWEToBackendRef<BE>
        + poulpy_ckks::CKKSCtBounds
        + poulpy_core::reference::linear_transformation::DiagonalProd<BE>,
    poulpy_ckks::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>: Sized,
    poulpy_ckks::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE>: BootstrappingKeys<BE>,
    <poulpy_ckks::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE> as BootstrappingKeys<BE>>::SwitchingKey:
        GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let log_n: usize = a.get("log-n", 15);
    let k: usize = a.get("k", 16);
    let levels: usize = a.get("levels", 1);
    let factors: usize = a.get("factors", 1);
    let dsize: usize = a.get("dsize", 1);
    let k_aux: usize = a.get("k-aux", 53);
    let d2s_dsize: usize = a.get("d2s-dsize", 1);
    let reps: usize = a.get("reps", 3);
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let families: Vec<String> = a.list("families", "unit");
    let chains: Vec<String> = a.list("chains", "fresh,square");
    let bench = a.string("bench", "both");
    let full = a.string("decode", "full") == "full";
    let n = 1usize << log_n;
    let log_slots = k.trailing_zeros() as usize;
    let plan = plan(log_slots, factors);
    let log_q_b = LOG_DELTA + LOG_MSG_RATIO;
    let k_in = plan.input_k(log_q_b);
    let k_boot = k_in + levels * LOG_DELTA + plan.post_mod_up_consumed_bits();
    let k_keys = k_boot.next_multiple_of(BASE2K);
    let layout_of = |k_in: usize, dsize: usize, k_aux: usize| GLWESwitchingKeyLayout {
        n: n.into(),
        base2k: BASE2K.into(),
        dnum: GGLWELayout::dnum_for_input(BASE2K.into(), TorusPrecision(k_in as u32), Dsize(dsize as u32)),
        k_aux: TorusPrecision(k_aux as u32),
        rank_in: Rank(1),
        rank_out: Rank(1),
        dsize: Dsize(dsize as u32),
    };
    let dnum = GGLWELayout::dnum_for_input(BASE2K.into(), TorusPrecision(k_keys as u32), Dsize(dsize as u32));
    // The key under the sparse secret keeps the auxiliary modulus of its own digit size.
    let enc_aux = d2s_dsize * BASE2K + log_n;
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: BASE2K.into(),
            dnum,
            k_aux: TorusPrecision(k_aux as u32),
            rank: Rank(1),
            dsize: Dsize(dsize as u32),
        },
        tensor_key: GLWETensorKeyLayout {
            n: n.into(),
            base2k: BASE2K.into(),
            dnum,
            k_aux: TorusPrecision(k_aux as u32),
            rank: Rank(1),
            dsize: Dsize(dsize as u32),
        },
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: layout_of(log_q_b, d2s_dsize, enc_aux),
            sparse_to_dense: layout_of(k_boot, dsize, k_aux),
        }),
    };
    let enc = keys_layout.encapsulation.unwrap();
    let setup = Line::new("setup")
        .str("pipeline", "sparse")
        .num("log_n", log_n)
        .num("k", k)
        .num("levels", levels)
        .num("level_bits", LOG_DELTA)
        .num("factors", factors)
        .num("dsize", dsize)
        .num("log2_p", k_aux)
        .num("k_in", k_in)
        .num("k_boot", k_boot)
        .num("log2_q_key", k_keys)
        .num(
            "log2_qp",
            [
                keys_layout.automorphism_key.k(),
                keys_layout.tensor_key.k(),
                enc.sparse_to_dense.k(),
            ]
            .into_iter()
            .max()
            .unwrap()
            .as_usize(),
        )
        .num("cap", max_modulus_bits(log_n, HAMMING_WEIGHT).unwrap_or(0))
        .num("enc_log2_qp", enc.dense_to_sparse.k().as_usize())
        .num("enc_cap", max_modulus_bits(log_n, EPHEMERAL_WEIGHT).unwrap_or(0))
        .num("threads", a.get::<usize>("threads", 1));
    if a.flag("dry-run") {
        setup
            .num("key_bytes_each", key_bytes(&keys_layout.automorphism_key, n))
            .num(
                "key_bytes_fixed",
                key_bytes(&keys_layout.tensor_key, n) + key_bytes(&enc.dense_to_sparse, n) + key_bytes(&enc.sparse_to_dense, n),
            )
            .print();
        return;
    }

    let t0 = Instant::now();
    let module = Module::<BE>::new(n as u64);
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 31);
    let ctx = BootstrappingContext::<BE, f64>::compile(&module, BASE2K.into(), &plan, &mut scratch.borrow()).unwrap();
    let glwe = GLWELayout {
        n: n.into(),
        base2k: BASE2K.into(),
        k: k_keys.into(),
        rank: Rank(1),
    };
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut Source::new(seed32(0, 0)));
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    let raw = ctx
        .generate_keys(
            &module,
            &sk_raw,
            &keys_layout,
            &mut Source::new(seed32(3, 0)),
            &mut Source::new(seed32(4, 0)),
            &mut Source::new(seed32(5, 0)),
            &mut scratch.borrow(),
        )
        .unwrap();
    let mut keys = raw.prepare(&module, &mut scratch.borrow());
    drop(raw);
    // The trace keys that the transforms do not already have.
    let trace_skip = log_slots + 1;
    if trace_skip < log_n {
        let (mut xe, mut xa) = (Source::new(seed32(6, 0)), Source::new(seed32(7, 0)));
        for gal in module.glwe_trace_galois_elements() {
            if keys.rotation_keys.contains_key(&gal) {
                continue;
            }
            let mut key = module.glwe_automorphism_key_alloc_from_infos(&keys_layout.automorphism_key);
            module.glwe_automorphism_key_encrypt_sk(&mut key, gal, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&keys_layout.automorphism_key);
            module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut scratch.borrow());
            keys.rotation_keys.insert(gal, prepared);
        }
    }
    let rotation_keys = keys.rotation_keys.len();
    let setup = setup.num("automorphism_keys", rotation_keys).num(
        "key_bytes",
        rotation_keys * key_bytes(&keys_layout.automorphism_key, n)
            + key_bytes(&keys_layout.tensor_key, n)
            + key_bytes(&enc.dense_to_sparse, n)
            + key_bytes(&enc.sparse_to_dense, n),
    );
    setup.clone().float("setup_s", t0.elapsed().as_secs_f64(), 2).print();

    let log_sparsity = (n / (2 * k)).trailing_zeros() as usize;
    let mut xe = Source::new(seed32(8, 0));
    let mut xa = Source::new(seed32(9, 0));
    let mut encrypt = |scratch: &mut ScratchOwned<BE>, re: &[f64], im: &[f64], log_delta: usize, width: usize| -> Ct<BE> {
        let mut pt = module.ckks_pt_vec_alloc_compact(k, BASE2K.into(), TorusPrecision((log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity,
            slots: SlotsKind::Complex,
        });
        module.ckks_encode_reim_into(&mut pt, re, im, &mut scratch.borrow()).unwrap();
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(width as u32));
        module
            .ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();
        ct
    };
    let decode = |scratch: &mut ScratchOwned<BE>, ct: &Ct<BE>| -> (Vec<f128>, Vec<f128>) {
        let mut pt = module.ckks_plaintext_alloc(
            n.into(),
            BASE2K.into(),
            TorusPrecision((ct.log_delta() + ct.log_budget().min(24)) as u32),
        );
        pt.set_meta(ct.meta());
        module.ckks_decrypt(&mut pt, ct, &sk, &mut scratch.borrow()).unwrap();
        // Every slot of the ring, or one period of k, which averages the copies.
        let count = if full { n / 2 } else { k };
        let (mut re, mut im) = (vec![Quad::default(); count], vec![Quad::default(); count]);
        CKKSEncodingHostOps::<BE, Quad>::ckks_decode_reim_into(&module, &pt, &mut re, &mut im, &mut scratch.borrow()).unwrap();
        (re.iter().map(|x| x.0).collect(), im.iter().map(|x| x.0).collect())
    };
    let refresh = |scratch: &mut ScratchOwned<BE>, ct: &Ct<BE>| -> Ct<BE> {
        let alloc = |width: usize, meta: CKKSMeta| {
            let mut c = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(width as u32));
            c.set_meta(meta);
            c
        };
        let mut coeffs = alloc(k_in, ct.meta());
        module.ckks_double_into(&mut coeffs, ct, &mut scratch.borrow()).unwrap();
        module
            .ckks_dft_evaluate_assign(
                &mut coeffs,
                ctx.slots_to_coeffs(),
                keys.rotation_keys(),
                &mut scratch.borrow(),
            )
            .unwrap();
        let log_q_b = coeffs.k().as_usize();
        let mut raised = alloc(k_boot, coeffs.meta());
        let (d2s, s2d) = keys.encapsulation_keys().unwrap();
        BE::ckks_encapsulated_mod_up(
            &module,
            &mut raised,
            &mut coeffs,
            ctx.c2s_guard_bits(),
            &d2s.to_backend_ref(),
            &s2d.to_backend_ref(),
            &mut scratch.borrow(),
        )
        .unwrap();
        raised.set_meta(CKKSMeta {
            log_delta: log_q_b + ctx.c2s_guard_bits(),
            log_sparsity: ct.log_sparsity(),
            slots: ct.slots(),
        });
        if trace_skip < log_n {
            module.glwe_trace_assign(&mut raised, trace_skip, keys.rotation_keys(), &mut scratch.borrow());
        }
        let mut real = alloc(k_boot, raised.meta());
        let mut imag = alloc(k_boot, raised.meta());
        module
            .ckks_coeffs_to_slots_split(
                &mut real,
                &mut imag,
                &raised,
                ctx.coeffs_to_slots(),
                keys.rotation_keys(),
                &mut scratch.borrow(),
            )
            .unwrap();
        let mut out = alloc(k_boot, raised.meta());
        let mut out_imag = alloc(k_boot, raised.meta());
        module
            .ckks_eval_mod(&mut out, &real, ctx.eval_mod(), keys.tensor_key(), &mut scratch.borrow())
            .unwrap();
        module
            .ckks_eval_mod(&mut out_imag, &imag, ctx.eval_mod(), keys.tensor_key(), &mut scratch.borrow())
            .unwrap();
        module.ckks_mul_i_assign(&mut out_imag, &mut scratch.borrow()).unwrap();
        module.ckks_add_assign(&mut out, &out_imag, &mut scratch.borrow()).unwrap();
        out.set_log_delta(out.log_delta() - ctx.c2s_guard_bits());
        out.set_meta(CKKSMeta {
            log_delta: ct.log_delta(),
            log_sparsity: ct.log_sparsity(),
            slots: ct.slots(),
        });
        out
    };
    let square = |scratch: &mut ScratchOwned<BE>, c: &Ct<BE>| -> Ct<BE> {
        let mut next = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision((c.k().as_usize() - c.log_delta()) as u32));
        module
            .ckks_square_into(&mut next, c, keys.tensor_key(), &mut scratch.borrow())
            .unwrap();
        next
    };
    let tile = |v: &[f64], len: usize| -> Vec<f128> { (0..len).map(|i| v[i % v.len()] as f128).collect() };
    let errors = |have: (&[f128], &[f128]), want: (&[f128], &[f128])| -> Vec<f64> {
        have.0
            .iter()
            .zip(want.0)
            .chain(have.1.iter().zip(want.1))
            .map(|(h, w)| (*h - *w) as f64)
            .collect()
    };
    let mul = |re: &mut [f128], im: &mut [f128], c: &[f128], d: &[f128]| {
        for j in 0..re.len() {
            let (x, y) = (re[j], im[j]);
            re[j] = x * c[j] - y * d[j];
            im[j] = x * d[j] + y * c[j];
        }
    };

    for family in &families {
        for &seed in &seeds {
            let (re, im) = Rng::new(seed, 0).values(k, family);
            let ct = encrypt(&mut scratch, &re, &im, LOG_DELTA, k_in);
            let line = Line::new("run").with(&setup).str("family", family).num("seed", seed);
            if bench != "check" {
                let (mut best, mut out) = (f64::INFINITY, None);
                for _ in 0..reps.max(1) {
                    let t = Instant::now();
                    out = Some(refresh(&mut scratch, &ct));
                    best = best.min(t.elapsed().as_secs_f64());
                }
                let out = out.unwrap();
                let mut lv = f64::INFINITY;
                for _ in 0..reps.max(1) {
                    let t = Instant::now();
                    let mut cur = square(&mut scratch, &out);
                    for _ in 1..levels {
                        cur = square(&mut scratch, &cur);
                    }
                    lv = lv.min(t.elapsed().as_secs_f64());
                }
                line.clone()
                    .str("stage", "time")
                    .float("refresh_s", best, 4)
                    .float("levels_s", lv, 4)
                    .print();
            }
            if bench == "time" {
                continue;
            }
            for chain in &chains {
                let out = refresh(&mut scratch, &ct);
                let (dre, dim) = decode(&mut scratch, &out);
                let len = dre.len();
                let refreshed = Quality::of(errors((&dre, &dim), (&tile(&re, len), &tile(&im, len))));
                let (mut want_re, mut want_im) = if chain == "square" {
                    (dre, dim)
                } else {
                    (tile(&re, len), tile(&im, len))
                };
                let mut cur = out;
                for level in 0..levels {
                    if chain == "square" {
                        cur = square(&mut scratch, &cur);
                        let (c, d) = (want_re.clone(), want_im.clone());
                        mul(&mut want_re, &mut want_im, &c, &d);
                    } else {
                        let (fr, fi) = Rng::new(seed, 1 + level as u64).values(k, family);
                        let fresh = encrypt(&mut scratch, &fr, &fi, cur.log_delta(), cur.k().as_usize());
                        let mut next = module
                            .ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32));
                        module
                            .ckks_mul_into(&mut next, &cur, &fresh, keys.tensor_key(), &mut scratch.borrow())
                            .unwrap();
                        cur = next;
                        mul(&mut want_re, &mut want_im, &tile(&fr, len), &tile(&fi, len));
                    }
                }
                let (hre, him) = decode(&mut scratch, &cur);
                let after = Quality::of(errors((&hre, &him), (&want_re, &want_im)));
                line.clone()
                    .str("stage", "check")
                    .str("chain", chain)
                    .quality("refresh_", refreshed)
                    .quality("levels_", after)
                    .num("k_out", cur.k().as_usize())
                    .print();
            }
        }
    }
}
