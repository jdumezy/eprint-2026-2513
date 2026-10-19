//! Bootstrapping on the replicated layout.
//!
//! The pipeline follows the CKKS SlotsToCoeffs-first blueprint (Section 5 of the paper):
//!
//! 1. a shaped normalized trace onto the subring of period `kR` averages the copies of each value;
//! 2. the fiber-aware SlotsToCoeffs reads the data slots only and leaves the message in the
//!    coefficients, as one wide stage ending in an ordinary rounding;
//! 3. ModUp, encapsulated by a sparse secret;
//! 4. a normalized trace onto the subring of period `k`, with an ordinary rounding;
//! 5. the fiber-aware CoeffsToSlots writes the data slots with zero guards, as one wide stage with
//!    shaped diagonals and a shaped final rounding, and a shaped conjugation splits it into its
//!    real and imaginary parts;
//! 6. EvalMod on each part, with masked shaped constants and the last squaring unrescaled;
//! 7. the parts are recombined and one shaped rescale returns to the leveled scale.
//!
//! When `2 k R` slots fit in the ring, the parts are packed instead: CoeffsToSlots writes the
//! values `z` and `-i z` side by side on a layout of `2 k` values, so that the conjugation leaves
//! the real parts on the first half and the imaginary parts on the second. One EvalMod then
//! covers both, with the constants of the second half carrying a phase that its double-angle
//! squarings raise to `i`, and one rotation by `k` values adds the two halves into `Re + i Im`
//! on every copy.
//!
//! The fiber-aware transforms need `R | k`: smaller batches are padded to `R` values.

pub mod eval_mod;
pub mod fiber;
pub mod transform;

use std::collections::{BTreeSet, HashMap};

use poulpy_ckks::{
    CKKSError, CKKSInfos, CKKSMeta, CKKSModuleInfos, CKKSResult, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSLinearTransformationOps, CKKSPow2Ops, CKKSSubOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, DFTOutputFormat, DFTPlan, DFTType, EvalModPlan, EvalModType},
    oep::CKKSEncapsulatedModUpImpl,
    reference::dft::gen_dft_matrices,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWEShift, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, GLWETrace,
    layouts::{
        BackendGLWESecret, Base2K, Dsize, GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyLayout,
        GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESecretLayout, GLWESecretSampling, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank,
        SplitStrategy, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESwitchingKeyPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    backend::HostBackend,
    context::{ContextError, ShapedContext},
    ops::{CtLayout, ShapedOps},
    rounding::MAX_PASS_BITS,
    security::ModulusCheck,
};
use eval_mod::{EvalModModule, ShapedEvalMod};
use transform::FiberTransform;

type Ct<BE> = CKKSCiphertextOwned<BE>;

/// The parameters of a bootstrapping, defaulting to the configuration of the paper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapParams {
    /// Scale of the input: `Delta_b = 2^log_delta`.
    pub log_delta: usize,
    /// `log2(q_b / Delta_b)`: headroom of the message when ModUp lifts it.
    pub log_msg_ratio: usize,
    /// Scale of the SlotsToCoeffs diagonals: the bits SlotsToCoeffs consumes.
    pub s2c_log_delta: usize,
    /// Scale of the CoeffsToSlots diagonals.
    pub c2s_log_delta: usize,
    /// Extra scale bits carried from ModUp through CoeffsToSlots.
    pub c2s_guard_bits: usize,
    /// Degree of the EvalMod base polynomial.
    pub eval_mod_degree: usize,
    /// `K`: the reduction is valid for integer parts `|I| < K`.
    pub eval_mod_interval: usize,
    /// Number of double-angle steps.
    pub eval_mod_double_angle: usize,
    /// Scale of the EvalMod arithmetic.
    pub eval_mod_log_delta: usize,
    /// Scale of the EvalMod constants.
    pub eval_mod_const_log_delta: usize,
    /// Leave the last double-angle squaring unrescaled, which folds the guard reset into EvalMod.
    pub defer_last_step: bool,
    /// Low bits kept below the result inside the bootstrapping.
    pub guard_bits: usize,
    /// Largest number of bits one shaped rounding removes inside the bootstrapping.
    pub pass_cap: usize,
    /// Scale of the output.
    pub level_log_delta: usize,
    /// Giant step of the fiber-aware transforms; the power of two next to `sqrt(k)` by default.
    pub giant_step: Option<usize>,
    /// Hamming weight of the ephemeral sparse secret of ModUp.
    pub ephemeral_hamming_weight: usize,
    /// Run EvalMod once on the real and imaginary parts packed side by side, when `2 k R` slots
    /// fit in the ring, instead of once on each part.
    pub packed_eval_mod: bool,
}

impl Default for BootstrapParams {
    fn default() -> Self {
        Self {
            log_delta: 30,
            log_msg_ratio: 13,
            s2c_log_delta: 26,
            c2s_log_delta: 34,
            c2s_guard_bits: 6,
            eval_mod_degree: 30,
            eval_mod_interval: 16,
            eval_mod_double_angle: 3,
            eval_mod_log_delta: 28,
            eval_mod_const_log_delta: 26,
            defer_last_step: true,
            guard_bits: 8,
            pass_cap: 32,
            level_log_delta: 20,
            giant_step: None,
            ephemeral_hamming_weight: 32,
            packed_eval_mod: true,
        }
    }
}

impl BootstrapParams {
    /// The EvalMod recipe of these parameters.
    pub fn eval_mod_plan(&self) -> EvalModPlan {
        EvalModPlan {
            eval_mod_type: EvalModType::CosHKEven,
            log_msg_ratio: self.log_msg_ratio,
            f_mod_degree: self.eval_mod_degree,
            f_mod_interval: self.eval_mod_interval,
            f_mod_log_interval_reduction: self.eval_mod_double_angle,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(self.eval_mod_const_log_delta, 4),
            f_mod_log_delta: self.eval_mod_log_delta,
        }
    }
}

/// Layouts of the keys of a bootstrapping.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapKeyLayouts {
    /// Every automorphism key: rotations, traces and conjugation.
    pub automorphism: GLWEAutomorphismKeyLayout,
    /// The relinearization key of EvalMod.
    pub tensor: GLWETensorKeyLayout,
    /// The switch to the ephemeral sparse secret, at the ModUp modulus.
    pub dense_to_sparse: GLWESwitchingKeyLayout,
    /// The switch back to the secret, at the bootstrapping modulus.
    pub sparse_to_dense: GLWESwitchingKeyLayout,
}

/// The prepared keys of a bootstrapping: automorphism keys for the transforms, the traces and the
/// conjugation, the relinearization key, and the two switching keys of the sparse-secret
/// encapsulation. They are ordinary keys: in the bivariate representation their noise is
/// suppressed by the width they have over the ciphertexts.
pub struct BootstrapKeys<BE: Backend> {
    /// The automorphism keys, by Galois element.
    pub automorphisms: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    /// The relinearization key, also usable for the leveled products.
    pub tensor: GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    /// The switch to the ephemeral sparse secret.
    pub dense_to_sparse: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>,
    /// The switch back from the ephemeral sparse secret.
    pub sparse_to_dense: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>,
}

/// A compiled bootstrapping for one context.
pub struct Bootstrapper<BE: Backend> {
    params: BootstrapParams,
    base2k: usize,
    /// Shaped rounding inside the bootstrapping.
    shaped: ShapedContext,
    /// Ordinary rounding, for the stages whose output is a coefficient message.
    plain: ShapedContext,
    /// Shaped rounding of the final rescale, which may drop more than the internal pass cap.
    last: ShapedContext,
    s2c: FiberTransform<BE>,
    c2s: FiberTransform<BE>,
    /// The batch of the packed EvalMod, `k`, when the parts are packed.
    packed: Option<usize>,
    eval_mod: ShapedEvalMod<BE>,
    k_in: usize,
    k_boot: usize,
}

/// The operations a bootstrapping needs from a module.
pub trait BootstrapModule<BE: Backend>:
    EvalModModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSImagOps<BE> + CKKSModuleInfos
{
}

impl<BE: Backend, M> BootstrapModule<BE> for M where
    M: EvalModModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSImagOps<BE> + CKKSModuleInfos
{
}

impl<BE> Bootstrapper<BE>
where
    BE: HostBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: BootstrapModule<BE>,
    for<'x> ScratchArena<'x, BE>: poulpy_core::ScratchArenaTakeCore<'x, BE>,
{
    /// Compiles the bootstrapping of the layout of `ctx` with ciphertexts in radix `2^base2k`,
    /// for `levels` products at the output scale before the next bootstrapping.
    pub fn new(
        module: &Module<BE>,
        ctx: &ShapedContext,
        params: BootstrapParams,
        base2k: usize,
        levels: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<Self> {
        let layout = *ctx.layout();
        let (k, r) = (layout.batch(), layout.replication());
        if !k.is_multiple_of(r) {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "bootstrapping needs R | k: pad the batch of {k} values to {r}"
            )));
        }
        // The final rescale drops the scale difference, and the unrescaled EvalMod scale with it,
        // in one shaped rounding.
        let deferred = if params.defer_last_step {
            params.eval_mod_log_delta
        } else {
            0
        };
        if params.level_log_delta > params.log_delta || params.log_delta + deferred - params.level_log_delta > MAX_PASS_BITS {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the output scale 2^{} must lie at most {MAX_PASS_BITS} bits below the input scale 2^{}{}",
                params.level_log_delta,
                params.log_delta,
                if deferred > 0 {
                    format!(" plus the deferred EvalMod scale of {deferred} bits")
                } else {
                    String::new()
                }
            )));
        }
        let internal = |guard: usize, cap: usize| -> CKKSResult<ShapedContext> {
            ctx.clone()
                .with_guard_bits(guard)
                .with_pass_cap(cap)
                .map_err(|e: ContextError| CKKSError::Internal(e.into()))
        };
        let shaped = internal(params.guard_bits, params.pass_cap)?;
        let plain = shaped.unshaped();
        let last = internal(params.guard_bits, MAX_PASS_BITS)?;

        let eval_mod_plan = params.eval_mod_plan();
        let eval_mod = ShapedEvalMod::new(module, eval_mod_plan, base2k, scratch)?;

        // The transforms over the k values, read from and written to the replicated layout.
        let log_k = k.trailing_zeros() as usize;
        let giant = params.giant_step.unwrap_or_else(|| transform::default_giant_step(k));
        let plan_err = |e: anyhow::Error| CKKSError::Internal(e);
        let s2c_plan = DFTPlan::new(
            DFTType::Decode,
            vec![(log_k, giant)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(params.s2c_log_delta, 2),
        )
        .and_then(|p| p.with_scaling(0.5))
        .map_err(plan_err)?;
        let c2s_plan = DFTPlan::new(
            DFTType::Encode,
            vec![(log_k, giant)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(params.c2s_log_delta, 2),
        )
        .and_then(|p| p.with_scaling(eval_mod_plan.input_scaling()?))
        .map_err(plan_err)?;
        let log_n = layout.n().trailing_zeros() as usize;
        let matrix = |plan: &DFTPlan| {
            let factors = gen_dft_matrices::<f64>(plan, log_n);
            assert_eq!(factors.len(), 1, "one-factor transforms");
            fiber::dense(&factors[0])
        };
        let s2c_diagonals = fiber::read_diagonals(&layout, &matrix(&s2c_plan), r as f64);
        // Packed: the rows `A` and `-i A` write `z` and `-i z` on a layout of 2k values.
        let packed = (params.packed_eval_mod && 4 * k * r <= layout.n()).then_some(k);
        let c2s_matrix = matrix(&c2s_plan);
        let c2s_diagonals = match packed {
            Some(_) => {
                let rows: Vec<_> = c2s_matrix
                    .iter()
                    .cloned()
                    .chain(c2s_matrix.iter().map(|row| row.iter().map(|&(a, b)| (b, -a)).collect()))
                    .collect();
                let double =
                    crate::layout::ReplicatedLayout::new(layout.n(), 2 * k, r).map_err(|e| CKKSError::Internal(e.into()))?;
                fiber::write_diagonals(&double, &rows)
            }
            None => fiber::write_diagonals(&layout, &c2s_matrix),
        };
        // Headroom of the diagonals: the read side carries the gain R.
        let s2c_meta = CoeffsMeta::from_delta_budget(params.s2c_log_delta, r.trailing_zeros() as usize + 3);
        let c2s_meta = CoeffsMeta::from_delta_budget(params.c2s_log_delta, 3);
        let s2c = FiberTransform::new(module, s2c_meta, &s2c_diagonals, giant, None, base2k, scratch)?;
        let c2s = FiberTransform::new(module, c2s_meta, &c2s_diagonals, giant, Some(ctx), base2k, scratch)?;

        // Moduli: the input reaches q_b after SlotsToCoeffs, and the output keeps `levels` levels
        // above the input width of the next bootstrapping.
        let k_in = params.log_delta + params.log_msg_ratio + params.s2c_log_delta;
        // An unrescaled last squaring keeps the EvalMod scale bits in the scale, and the final
        // rescale drops them with the rest.
        let post_mod_up = params.c2s_log_delta + params.c2s_guard_bits + eval_mod.consumed_bits() - deferred
            + (params.log_delta + deferred - params.level_log_delta);
        let k_boot = k_in + levels * params.level_log_delta + post_mod_up;
        Ok(Self {
            params,
            base2k,
            shaped,
            plain,
            last,
            s2c,
            c2s,
            packed,
            eval_mod,
            k_in,
            k_boot,
        })
    }

    /// The parameters the bootstrapping was compiled with.
    pub fn params(&self) -> &BootstrapParams {
        &self.params
    }

    /// Whether EvalMod runs once on the packed real and imaginary parts: requested by the
    /// parameters, and possible when `2 k R` slots fit in the ring.
    pub fn packs_eval_mod(&self) -> bool {
        self.packed.is_some()
    }

    /// Width of the input, at scale `2^log_delta`.
    pub fn k_in(&self) -> usize {
        self.k_in
    }

    /// Width of the ciphertext after ModUp: the largest modulus of the bootstrapping.
    pub fn k_boot(&self) -> usize {
        self.k_boot
    }

    /// The width the automorphism and tensor keys must cover: the bootstrapping modulus and the
    /// guard bits of the wide stages, in whole limbs. The keys round it up to whole digits and add
    /// the auxiliary modulus `P` ([`Self::key_layouts`]).
    pub fn key_modulus_bits(&self) -> usize {
        (self.k_boot + self.params.guard_bits).next_multiple_of(self.base2k)
    }

    /// Width of the output: the input width plus the levels the bootstrapping was sized for.
    pub fn k_out(&self, levels: usize) -> usize {
        self.k_in + levels * self.params.level_log_delta
    }

    /// The Galois elements of every automorphism key the bootstrapping uses.
    pub fn galois_elements(&self, module: &Module<BE>) -> BTreeSet<i64>
    where
        Module<BE>: GLWETrace<BE>,
    {
        let mut set = transform::galois_elements(module, &[&self.s2c, &self.c2s]);
        set.insert(-1);
        if let Some(k) = self.packed {
            set.insert(module.ckks_galois_element(self.shaped.layout().rotation_slots(k as i64)));
        }
        set
    }

    /// Key layouts with `dsize` limbs per digit and an auxiliary modulus of `k_aux` bits. The
    /// automorphism and tensor keys cover the bootstrapping modulus and the guard bits of the wide
    /// stages, in whole digits: their modulus is `log2(Q P) = dnum dsize base2k + k_aux`, checked
    /// by [`Self::security`].
    ///
    /// poulpy needs the auxiliary modulus to cover one digit, `k_aux >= dsize base2k`; below it
    /// the key switches lose precision. The key to the ephemeral sparse secret always has one limb
    /// per digit and `P = base2k + 1`. [`Self::secure_key_layouts`] picks `dsize` and `k_aux`.
    pub fn key_layouts(&self, dsize: usize, k_aux: usize) -> CKKSResult<BootstrapKeyLayouts> {
        if dsize == 0 || k_aux < dsize * self.base2k {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the auxiliary modulus of {k_aux} bits must cover one digit of {dsize} limbs of {} bits",
                self.base2k
            )));
        }
        let n = self.shaped.layout().n();
        let base2k = Base2K(self.base2k as u32);
        let k_keys = self.key_modulus_bits();
        let dnum = |k: usize| GGLWELayout::dnum_for_input(base2k, TorusPrecision(k as u32), Dsize(dsize as u32));
        let switching = |k: usize, dsize: usize, k_aux: usize| GLWESwitchingKeyLayout {
            n: n.into(),
            base2k,
            dnum: GGLWELayout::dnum_for_input(base2k, TorusPrecision(k as u32), Dsize(dsize as u32)),
            k_aux: TorusPrecision(k_aux as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            dsize: Dsize(dsize as u32),
        };
        let k_mod_up = self.params.log_delta + self.params.log_msg_ratio;
        Ok(BootstrapKeyLayouts {
            automorphism: GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k,
                dnum: dnum(k_keys),
                k_aux: TorusPrecision(k_aux as u32),
                rank: Rank(1),
                dsize: Dsize(dsize as u32),
            },
            tensor: GLWETensorKeyLayout {
                n: n.into(),
                base2k,
                dnum: dnum(k_keys),
                k_aux: TorusPrecision(k_aux as u32),
                rank: Rank(1),
                dsize: Dsize(dsize as u32),
            },
            // The key under the sparse secret has its own, smaller cap: one limb per digit, with
            // the auxiliary modulus of one limb.
            dense_to_sparse: switching(k_mod_up, 1, self.base2k + 1),
            sparse_to_dense: switching(self.k_boot, dsize, k_aux),
        })
    }

    /// The key layouts with the fewest digits within the 128-bit caps for a secret of Hamming
    /// weight `hamming_weight`: two limbs per digit with `P = 2 base2k` when every key fits, which
    /// halves the size of the keys, and one limb per digit with `P = base2k + 1`, the smallest
    /// modulus, otherwise. Fails when neither fits.
    pub fn secure_key_layouts(&self, hamming_weight: usize) -> CKKSResult<BootstrapKeyLayouts> {
        let mut over = None;
        let b = self.base2k;
        for (dsize, k_aux) in [(2, 2 * b), (1, b + 1)] {
            let layouts = self.key_layouts(dsize, k_aux)?;
            match self.security(&layouts, hamming_weight).into_iter().find(|c| !c.is_secure()) {
                None => return Ok(layouts),
                Some(check) => over = Some(check),
            }
        }
        Err(CKKSError::Internal(anyhow::anyhow!(
            "no key layout is within the 128-bit caps: {}",
            over.expect("two layouts were checked")
        )))
    }

    /// Every key modulus of `layouts` against the 128-bit caps of [`crate::security`], for a
    /// secret of Hamming weight `hamming_weight`. The key to the ephemeral sparse secret is
    /// checked at the weight of that secret.
    pub fn security(&self, layouts: &BootstrapKeyLayouts, hamming_weight: usize) -> [ModulusCheck; 4] {
        [
            ModulusCheck::new("automorphism keys", &layouts.automorphism, hamming_weight),
            ModulusCheck::new("tensor key", &layouts.tensor, hamming_weight),
            ModulusCheck::new("sparse-to-dense key", &layouts.sparse_to_dense, hamming_weight),
            ModulusCheck::new(
                "dense-to-sparse key",
                &layouts.dense_to_sparse,
                self.params.ephemeral_hamming_weight,
            ),
        ]
    }

    /// Scratch bytes enough for [`Self::bootstrap`] with keys of layouts `layouts`.
    pub fn tmp_bytes(&self, module: &Module<BE>, layouts: &BootstrapKeyLayouts) -> usize
    where
        Module<BE>: GLWEShift<BE>,
    {
        let p = &self.params;
        let (atk, tsk) = (&layouts.automorphism, &layouts.tensor);
        let n = self.shaped.layout().n();
        let like = GLWELayout {
            n: n.into(),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision(0),
            rank: Rank(1),
        };
        let meta = CKKSMeta {
            log_delta: p.log_delta,
            log_sparsity: self.shaped.layout().log_sparsity(),
            slots: SlotsKind::Complex,
        };
        // The ciphertexts before ModUp are at most k_in wide, after it at most k_boot.
        let ct_in = CtLayout::new(&like, self.k_in, meta);
        let ct = CtLayout::new(&like, self.k_boot, meta);
        let size = ct.size();
        module
            .shaped_ops_tmp_bytes(&self.last, &ct, tsk, atk)
            .max(module.shaped_ops_tmp_bytes(&self.shaped, &ct, tsk, atk))
            .max(self.s2c.tmp_bytes(module, &self.plain, &ct_in, atk))
            .max(self.c2s.tmp_bytes(module, &self.shaped, &ct, atk))
            .max(BE::ckks_encapsulated_mod_up_tmp_bytes(
                module,
                &ct,
                &ct_in,
                &layouts.dense_to_sparse,
                &layouts.sparse_to_dense,
            ))
            .max(self.eval_mod.tmp_bytes(module, &self.shaped, &ct, tsk))
            .max(module.glwe_shift_tmp_bytes(size))
            .max(module.ckks_copy_tmp_bytes(&ct, &ct))
            .max(module.ckks_mul_pow2_tmp_bytes(size))
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_sub_tmp_bytes(size))
            .max(module.ckks_mul_i_tmp_bytes(size))
            .max(module.ckks_div_i_tmp_bytes(size))
    }

    /// Generates and prepares the keys of the bootstrapping under the secret `sk`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_keys(
        &self,
        module: &Module<BE>,
        sk: &BackendGLWESecret<BE>,
        layouts: &BootstrapKeyLayouts,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> BootstrapKeys<BE>
    where
        Module<BE>: GLWETrace<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWESecretSampling<BE>,
    {
        let mut automorphisms = HashMap::new();
        for gal in self.galois_elements(module) {
            let mut key = module.glwe_automorphism_key_alloc_from_infos(&layouts.automorphism);
            module.glwe_automorphism_key_encrypt_sk(&mut key, gal, sk, source_xe, source_xa, scratch);
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&layouts.automorphism);
            module.glwe_automorphism_key_prepare(&mut prepared, &key, scratch);
            automorphisms.insert(gal, prepared);
        }
        let mut tensor_raw = module.glwe_tensor_key_alloc_from_infos(&layouts.tensor);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_raw, sk, source_xe, source_xa, scratch);
        let mut tensor = module.alloc_tensor_key_prepared_from_infos(&layouts.tensor);
        module.prepare_tensor_key(&mut tensor, &tensor_raw, scratch);

        // The ephemeral sparse secret of the encapsulation never leaves this function.
        let mut sparse = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: sk.n(),
            rank: Rank(1),
        });
        module.glwe_secret_fill_ternary_hw(&mut sparse, self.params.ephemeral_hamming_weight, source_xs);
        let mut switching = |layout: &GLWESwitchingKeyLayout, from: &BackendGLWESecret<BE>, to: &BackendGLWESecret<BE>| {
            let mut key = module.glwe_switching_key_alloc_from_infos(layout);
            module.glwe_switching_key_encrypt_sk(&mut key, from, to, source_xe, source_xa, scratch);
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(layout);
            module.glwe_switching_key_prepare(&mut prepared, &key, scratch);
            prepared
        };
        let dense_to_sparse = switching(&layouts.dense_to_sparse, sk, &sparse);
        let sparse_to_dense = switching(&layouts.sparse_to_dense, &sparse, sk);
        BootstrapKeys {
            automorphisms,
            tensor,
            dense_to_sparse,
            sparse_to_dense,
        }
    }

    /// Bootstraps `ct` into a ciphertext at the output scale with `levels` levels above the input
    /// width.
    ///
    /// `ct` is either at the input scale `2^log_delta` with width at least [`Self::k_in`], or at the
    /// output scale `2^level_log_delta` with width at least `k_in - (log_delta - level_log_delta)`,
    /// as a chain of `levels` products leaves it. The latter is first raised to the input scale by
    /// an exact shift, which spends `log_delta - level_log_delta` bits of its budget.
    pub fn bootstrap(
        &self,
        module: &Module<BE>,
        ct: &Ct<BE>,
        keys: &BootstrapKeys<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<Ct<BE>>
    where
        GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: poulpy_core::layouts::GetTensorKey<BE>,
    {
        let p = &self.params;
        let layout = *self.shaped.layout();
        let base2k = Base2K(self.base2k as u32);
        let alloc = |k: usize, meta: CKKSMeta| {
            let mut c = module.ckks_ciphertext_alloc(base2k, TorusPrecision(k as u32));
            c.set_meta(meta);
            c
        };

        // The input at scale 2^log_delta and width k_in: a modulus reduction, after an exact shift
        // by the scale difference for a ciphertext at the output scale.
        let raise = if ct.log_delta() == p.log_delta {
            0
        } else if ct.log_delta() == p.level_log_delta && p.level_log_delta < p.log_delta {
            p.log_delta - p.level_log_delta
        } else {
            usize::MAX
        };
        if raise == usize::MAX || ct.k().as_usize() + raise < self.k_in {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "bootstrapping takes a ciphertext at scale 2^{} and width at least {}, or at scale 2^{} and width at least {}, not 2^{} and {}",
                p.log_delta,
                self.k_in,
                p.level_log_delta,
                self.k_in.saturating_sub(p.log_delta - p.level_log_delta),
                ct.log_delta(),
                ct.k().as_usize()
            )));
        }
        let mut input = alloc(self.k_in, ct.meta());
        if raise == 0 {
            module.ckks_copy(&mut input, ct, scratch)?;
        } else {
            module.glwe_lsh(&mut input, ct, ct.k().as_usize() + raise - self.k_in, scratch);
            input.set_meta(CKKSMeta {
                log_delta: p.log_delta,
                log_sparsity: ct.log_sparsity(),
                slots: ct.slots(),
            });
        }
        let rotations = &keys.automorphisms;

        // 1. Average the copies: the input becomes periodic of period kR, guards included.
        let log_n = layout.n().trailing_zeros() as usize;
        let bottom_skip = layout.period().trailing_zeros() as usize + 1;
        if bottom_skip < log_n {
            module.shaped_trace_assign(&self.shaped, &mut input, bottom_skip, rotations, scratch)?;
        }

        // 2. SlotsToCoeffs: its half-scaled matrix runs on twice the input.
        let mut coeffs = alloc(self.k_in, ct.meta());
        module.ckks_double_into(&mut coeffs, &input, scratch)?;
        self.s2c.eval_wide(module, &self.plain, &mut coeffs, rotations, scratch)?;

        // 3. ModUp under the sparse secret: the message I q_b + Delta_b m lands at scale q_b,
        //    with the guard bits of CoeffsToSlots below.
        let log_q_b = coeffs.k().as_usize();
        let mut raised = alloc(self.k_boot, coeffs.meta());
        BE::ckks_encapsulated_mod_up(
            module,
            &mut raised,
            &mut coeffs,
            p.c2s_guard_bits,
            &keys.dense_to_sparse.to_backend_ref(),
            &keys.sparse_to_dense.to_backend_ref(),
            scratch,
        )?;
        raised.set_meta(CKKSMeta {
            log_delta: log_q_b + p.c2s_guard_bits,
            log_sparsity: ct.log_sparsity(),
            slots: ct.slots(),
        });

        // 4. Keep the coefficients the message lives on: period k.
        let top_skip = layout.batch().trailing_zeros() as usize + 1;
        if top_skip < log_n {
            module.shaped_trace_assign(&self.plain, &mut raised, top_skip, rotations, scratch)?;
        }

        // 5. CoeffsToSlots, then its real and imaginary parts.
        let mut real = alloc(self.k_boot, raised.meta());
        module.ckks_copy(&mut real, &raised, scratch)?;
        self.c2s.eval_wide(module, &self.shaped, &mut real, rotations, scratch)?;
        let mut conj = alloc(real.k().as_usize(), real.meta());
        module.shaped_conjugate_into(&self.shaped, &mut conj, &real, rotations, scratch)?;
        let mut out = alloc(real.k().as_usize(), real.meta());
        if let Some(k) = self.packed {
            // z - i z, conjugated and added: 2 Re z on the first half, 2 Im z on the second.
            module.ckks_add_assign(&mut real, &conj, scratch)?;
            real.set_slots(SlotsKind::Real);
            real.set_log_sparsity(layout.log_sparsity() - 1);

            // 6. One EvalMod for both halves, the second coming out times i.
            self.eval_mod.eval(
                module,
                &self.shaped,
                &mut out,
                &real,
                &keys.tensor,
                p.defer_last_step,
                Some(k),
                scratch,
            )?;
            let mut moved = alloc(out.k().as_usize(), out.meta());
            module.shaped_rotate_into(&self.shaped, &mut moved, &out, k as i64, rotations, scratch)?;
            module.ckks_add_assign(&mut out, &moved, scratch)?;
            out.set_log_sparsity(layout.log_sparsity());
            out.set_slots(SlotsKind::Complex);
        } else {
            let mut diff = alloc(real.k().as_usize(), real.meta());
            module.ckks_sub_into(&mut diff, &real, &conj, scratch)?;
            module.ckks_add_assign(&mut real, &conj, scratch)?;
            let mut imag = alloc(diff.k().as_usize(), diff.meta());
            module.ckks_div_i_into(&mut imag, &diff, scratch)?;
            real.set_slots(SlotsKind::Real);
            imag.set_slots(SlotsKind::Real);

            // 6. EvalMod on each part.
            let mut out_imag = alloc(imag.k().as_usize(), imag.meta());
            self.eval_mod.eval(
                module,
                &self.shaped,
                &mut out,
                &real,
                &keys.tensor,
                p.defer_last_step,
                None,
                scratch,
            )?;
            self.eval_mod.eval(
                module,
                &self.shaped,
                &mut out_imag,
                &imag,
                &keys.tensor,
                p.defer_last_step,
                None,
                scratch,
            )?;
            module.ckks_mul_i_assign(&mut out_imag, scratch)?;
            module.ckks_add_assign(&mut out, &out_imag, scratch)?;
        }

        // 7. Relabel at the input scale, and rescale to the output scale.
        let kept = if p.defer_last_step { p.eval_mod_log_delta } else { 0 };
        out.set_log_delta(out.log_delta() - p.c2s_guard_bits);
        out.set_meta(CKKSMeta {
            log_delta: p.log_delta + kept,
            log_sparsity: ct.log_sparsity(),
            slots: ct.slots(),
        });
        let drop = out.log_delta() - p.level_log_delta;
        let mut res = alloc(out.k().as_usize() - drop, out.meta());
        module.shaped_rescale_into(&self.last, &mut res, &out, drop, scratch)?;
        Ok(res)
    }
}
