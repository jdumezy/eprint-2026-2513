//! poulpy's encapsulated PaCo bootstrapping (Coron and Seuré), under the protocol of `shaped boot`.
//!
//! ```text
//! paco [--log-n 15] [--k 16] [--h 64] [--log-q 46 --log-msg 34 --bsk-scale 48 --stc-scale 42]
//!      [--c2s 3,4 --stc 5] [--levels 1] [--dsize 1] [--k-aux 0] [--seeds ...] [--families ...]
//!      [--reps 3] [--bench time|check|both] [--chains fresh,square] [--threads 1] [--decode full|average]
//!      [--dry-run]
//! ```
//!
//! The application secret is dense (weight 1024); PaCo's structured binary secret of weight `h`
//! is used at the input modulus only, through a switching key. One branch recovers `C` real
//! coefficients, so `k` complex values take `2k / C` branches.

#![feature(f128)]

use std::{collections::HashMap, time::Instant};

use poulpy_ckks::{
    CKKSInfos, CKKSMeta, Quad, SetCKKSInfos, SlotsKind,
    api::{CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSMulOps, CKKSPaCoOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, PaCoContext, PaCoDFTPlan, PaCoKeySet, PaCoPlan, PaCoSecretSpec},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, alloc_scratch, ckks_encrypt, gen_sk_with_raw},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, TransferInto,
    layouts::{GGLWEInfos, LWEInfos, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned, Standard},
    source::Source,
};
use poulpy_shaped::security::max_modulus_bits;
use poulpy_shaped_bench::{Args, Line, Quality, Rng, dispatch};

const BASE2K: usize = 52;
const HAMMING_WEIGHT: usize = 1024;

type E = poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<f64>;

fn main() {
    let args = Args::parse();
    let threads: usize = args.get("threads", 1);
    dispatch!(threads, run(&args));
}

fn key_bytes<K: GGLWEInfos>(key: &K, n: usize) -> usize {
    key.dnum().as_usize() * 2 * key.size() * n * 8
}

fn run<BE>(a: &Args)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>:
        TestContextModule<BE> + CKKSPaCoOps<BE, f64> + CKKSEncodingOps<BE, f64> + CKKSEncodingOps<BE, Quad> + CKKSMulOps<BE>,
{
    let log_n: usize = a.get("log-n", 15);
    let k: usize = a.get("k", 16);
    let h: usize = a.get("h", 64);
    let n = 1usize << log_n;
    let m = n / 2;
    // One branch recovers C coefficients; k complex values are 2k coefficients.
    let c: usize = a.get("c", (2 * k).min(n / (4 * h)));
    let (log_q, log_msg, bsk_scale): (usize, usize, usize) = (a.get("log-q", 46), a.get("log-msg", 34), a.get("bsk-scale", 48));
    let c2s: Vec<usize> = a.list("c2s", "0");
    let stc: Vec<usize> = a.list("stc", "0");
    let (c2s_scale, stc_scale): (usize, usize) = (a.get("c2s-scale", bsk_scale), a.get("stc-scale", bsk_scale - 6));
    let levels: usize = a.get("levels", 1);
    let dsize: usize = a.get("dsize", 1);
    let k_aux: usize = a.get("k-aux", 0);
    let reps: usize = a.get("reps", 3);
    let seeds: Vec<u64> = a.list("seeds", "1,2,3,4,5");
    let families: Vec<String> = a.list("families", "unit");
    let chains: Vec<String> = a.list("chains", "fresh,square");
    let bench = a.string("bench", "both");
    let full = a.string("decode", "full") == "full";
    let kappa = 2 * k / c;
    assert!(
        kappa * c == 2 * k && kappa.is_power_of_two(),
        "2k must be a power-of-two multiple of C"
    );
    let log_c = c.trailing_zeros() as usize;
    let c2s = if c2s == [0] { vec![log_c + 2] } else { c2s };
    let stc = if stc == [0] { vec![log_c] } else { stc };
    let giants = |depths: &[usize]| -> Vec<usize> { depths.iter().map(|&d| 1usize << d.div_ceil(2)).collect() };
    let plan = PaCoPlan::new(log_n, h, c, log_q as u32)
        .and_then(|p| {
            p.with_evaluation(
                bsk_scale,
                16,
                PaCoDFTPlan::new(c2s.clone(), giants(&c2s), c2s_scale, 16, 1.0)?,
                PaCoDFTPlan::new(stc.clone(), giants(&stc), stc_scale, 16, 1.0)?,
            )
        })
        .unwrap();
    let relabel = log_q as i64 - 2 - log_msg as i64 - plan.extra_scale_log2();
    let out_scale = (bsk_scale as i64 - relabel) as usize;
    let k_boot = plan.k_boot(BASE2K, log_msg + 8 + levels * out_scale).unwrap();
    let params = CKKSTestParams {
        n,
        base2k: BASE2K,
        k: k_boot,
        prec_meta: CKKSMeta {
            log_delta: bsk_scale,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 16,
        hw: HAMMING_WEIGHT,
        dsize,
        rank: 1,
    };
    let (mut atk_layout, mut tsk_layout) = (params.atk_layout(), params.tsk_layout());
    if k_aux != 0 {
        atk_layout.k_aux = TorusPrecision(k_aux as u32);
        tsk_layout.k_aux = TorusPrecision(k_aux as u32);
    }
    let d2p_layout = params.ksk_layout(log_q);
    let galois = plan.galois_elements();
    let setup = Line::new("setup")
        .str("pipeline", "paco")
        .num("log_n", log_n)
        .num("k", k)
        .num("levels", levels)
        .num("h", h)
        .num("c", c)
        .num("branches", kappa)
        .num("log_q", log_q)
        .num("log_msg", log_msg)
        .num("bsk_scale", bsk_scale)
        .str("c2s", &format!("{c2s:?}").replace(' ', ""))
        .str("stc", &format!("{stc:?}").replace(' ', ""))
        .num("stc_scale", stc_scale)
        .num("level_bits", out_scale)
        .num("dsize", dsize)
        .num("k_boot", k_boot)
        .num("log2_q_key", k_boot)
        .num("log2_p", atk_layout.k_aux.as_u32())
        .num(
            "log2_qp",
            [atk_layout.k(), tsk_layout.k(), d2p_layout.k()]
                .into_iter()
                .max()
                .unwrap()
                .as_usize(),
        )
        .num("cap", max_modulus_bits(log_n, HAMMING_WEIGHT).unwrap_or(0))
        .num("automorphism_keys", galois.len())
        .num(
            "key_bytes",
            galois.len() * key_bytes(&atk_layout, n)
                + key_bytes(&tsk_layout, n)
                + key_bytes(&d2p_layout, n)
                + 4 * 2 * k_boot.div_ceil(BASE2K) * n * 8,
        )
        .num("threads", a.get::<usize>("threads", 1));
    if a.flag("dry-run") {
        setup.print();
        return;
    }

    let t0 = Instant::now();
    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<f64>(m).unwrap();
    let mut scratch = alloc_scratch(&params, &module);
    let (sk_dense_raw, sk_dense) = gen_sk_with_raw(&params, &module, &host_module, [21u8; 32]);
    let spec = PaCoSecretSpec::sample(&plan, &mut Source::new([22u8; 32])).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_paco_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&plan, &mut sk_paco_host).unwrap();
    let mut sk_paco_raw = module.glwe_secret_alloc_from_infos(&sk_paco_host);
    sk_paco_host.transfer_into(&mut sk_paco_raw);
    let (mut xa, mut xe) = (Source::new([1u8; 32]), Source::new([2u8; 32]));
    let mut rotation_keys = HashMap::new();
    for gal in galois {
        let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_layout);
        module.glwe_automorphism_key_encrypt_sk(&mut atk, gal, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
        rotation_keys.insert(gal, atk);
    }
    let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
    module.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
    let mut dense_to_paco = module.glwe_switching_key_alloc_from_infos(&d2p_layout);
    module.glwe_switching_key_encrypt_sk(
        &mut dense_to_paco,
        &sk_dense_raw,
        &sk_paco_raw,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );
    let ctx = PaCoContext::<BE, f64>::compile(&module, BASE2K.into(), plan.clone(), &mut scratch.borrow()).unwrap();
    let bsk = std::array::from_fn(|t| {
        let (re, im) = spec.sigma_slots_reim(&module, &ctx, t, &mut scratch.borrow()).unwrap();
        ckks_encrypt(
            &params,
            &module,
            &host_module,
            &encoder_full,
            &sk_dense,
            k_boot,
            &re,
            &im,
            &mut scratch.borrow(),
        )
    });
    let keyset = PaCoKeySet::new(&plan, bsk, rotation_keys, tensor_key, Some(dense_to_paco)).unwrap();
    let prepared = keyset.into_prepare(&plan, &module, &mut scratch.borrow()).unwrap();
    let k_out = ctx.max_output_k(&prepared).unwrap();
    setup.clone().float("setup_s", t0.elapsed().as_secs_f64(), 2).print();

    let log_sparsity = (n / (kappa * c)).trailing_zeros() as usize;
    let probe_out = module.ckks_ciphertext_alloc(BASE2K.into(), k_out);
    let mut probe_in = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(log_q as u32));
    probe_in.set_meta(CKKSMeta {
        log_delta: log_msg,
        log_sparsity,
        slots: SlotsKind::Complex,
    });
    let bytes = module
        .ckks_paco_bootstrap_tmp_bytes(&probe_out, &probe_in, &ctx, &prepared)
        .unwrap()
        .max(scratch.borrow().available());
    let mut scratch = ScratchOwned::<BE>::alloc(bytes + (1 << 26));
    let tsk = prepared.tensor_key();

    let mut xe = Source::new([8u8; 32]);
    let mut xa = Source::new([9u8; 32]);
    let mut encrypt = |scratch: &mut ScratchOwned<BE>, re: &[f64], im: &[f64], log_delta: usize, width: usize| {
        let mut pt = module.ckks_pt_vec_alloc(BASE2K.into(), TorusPrecision((log_delta + 8) as u32));
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        });
        let tile = |v: &[f64]| -> Vec<f64> { (0..m).map(|t| v[t % k]).collect() };
        CKKSEncodingHostOps::<BE, f64>::ckks_encode_reim_into(&module, &mut pt, &tile(re), &tile(im), &mut scratch.borrow())
            .unwrap();
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision(width as u32));
        module
            .ckks_encrypt_sk(&mut ct, &pt, &sk_dense, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();
        ct
    };
    let decode = |scratch: &mut ScratchOwned<BE>, ct: &CKKSCiphertextOwned<BE>| -> (Vec<f128>, Vec<f128>) {
        let mut pt = module.ckks_pt_vec_alloc(
            BASE2K.into(),
            TorusPrecision((ct.log_delta() + ct.log_budget().min(24)) as u32),
        );
        let mut meta = ct.meta();
        meta.log_sparsity = 0;
        pt.set_meta(meta);
        module.ckks_decrypt(&mut pt, ct, &sk_dense, &mut scratch.borrow()).unwrap();
        let (mut re, mut im) = (vec![Quad::default(); m], vec![Quad::default(); m]);
        CKKSEncodingHostOps::<BE, Quad>::ckks_decode_reim_into(&module, &pt, &mut re, &mut im, &mut scratch.borrow()).unwrap();
        // Every slot of the ring, each against the value it repeats, or the first period.
        let count = if full { m } else { k };
        (
            re[..count].iter().map(|x| x.0).collect(),
            im[..count].iter().map(|x| x.0).collect(),
        )
    };
    let refresh = |scratch: &mut ScratchOwned<BE>, ct: &CKKSCiphertextOwned<BE>| {
        let mut out = module.ckks_ciphertext_alloc(BASE2K.into(), k_out);
        module
            .ckks_paco_bootstrap_into::<_, _>(&mut out, ct, &ctx, &prepared, &mut scratch.borrow())
            .unwrap();
        out
    };
    let square = |scratch: &mut ScratchOwned<BE>, c: &CKKSCiphertextOwned<BE>| {
        let mut next = module.ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision((c.k().as_usize() - c.log_delta()) as u32));
        module.ckks_square_into(&mut next, c, tsk, &mut scratch.borrow()).unwrap();
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
            let mut ct = encrypt(&mut scratch, &re, &im, log_msg, log_q);
            ct.set_log_sparsity(log_sparsity);
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
                        let mut fresh = encrypt(&mut scratch, &fr, &fi, cur.log_delta(), cur.k().as_usize());
                        fresh.set_log_sparsity(cur.log_sparsity());
                        let mut next = module
                            .ckks_ciphertext_alloc(BASE2K.into(), TorusPrecision((cur.k().as_usize() - cur.log_delta()) as u32));
                        module
                            .ckks_mul_into(&mut next, &cur, &fresh, tsk, &mut scratch.borrow())
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
