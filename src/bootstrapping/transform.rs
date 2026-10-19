//! Linear maps evaluated as wide stages.
//!
//! A wide stage extends the ciphertext by `guard_bits` (an exact modulus extension), evaluates the
//! map there with the stock BSGS evaluator, whose internal roundings and key switches then sit
//! `guard_bits` lower, and returns to the input precision with one rounding: shaped when the
//! output is in the replicated slot layout, ordinary when it is a coefficient message.

use poulpy_ckks::{
    CKKSCtBounds, CKKSError, CKKSInfos, CKKSModuleInfos, CKKSResult, CoeffsMeta, SetCKKSInfos,
    api::CKKSLinearTransformationOps,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    reference::linear_transformation_diagonals::ckks_encode_linear_transformation_from_diagonals,
};
use poulpy_core::{
    GLWEShift, GLWETrace,
    layouts::{
        GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LWEInfos, LinearTransformation,
        LinearTransformationBabySteps, LinearTransformationPrepared, LinearTransformationStrategy, TorusPrecision,
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Module, ScratchArena},
};

use crate::{
    backend::HostBackend,
    context::ShapedContext,
    encoding::plaintext_integers,
    ops::{CtLayout, shaped_round_glwe, shaped_round_glwe_tmp_bytes},
    rounding::shaped_round_integers,
    shaping::ShapingPolynomial,
};

/// A linear map on slot vectors, with its diagonals prepared for evaluation.
pub struct FiberTransform<BE: poulpy_hal::layouts::Backend> {
    lt: LinearTransformationPrepared<BE>,
    baby_steps: Vec<i64>,
    rotations: Vec<i64>,
    log_delta: usize,
}

/// The default giant step of a transform over `k` values: the power of two next to `sqrt(k)`.
pub(crate) fn default_giant_step(k: usize) -> usize {
    ((k as f64).sqrt().ceil() as usize).next_power_of_two()
}

/// The Galois elements of the rotations of `transforms` and of the normalized traces.
pub(crate) fn galois_elements<BE: HostBackend>(
    module: &Module<BE>,
    transforms: &[&FiberTransform<BE>],
) -> std::collections::BTreeSet<i64>
where
    Module<BE>: CKKSModuleInfos + GLWETrace<BE>,
{
    let mut set: std::collections::BTreeSet<i64> = transforms
        .iter()
        .flat_map(|t| t.rotations())
        .map(|&r| module.ckks_galois_element(r))
        .collect();
    set.extend(module.glwe_trace_galois_elements());
    set
}

/// `5^e mod 2N`.
fn pow5(e: u64, two_n: usize) -> usize {
    let (mut acc, mut base, mut e) = (1usize, 5usize, e);
    while e != 0 {
        if e & 1 == 1 {
            acc = acc * base % two_n;
        }
        base = base * base % two_n;
        e >>= 1;
    }
    acc
}

impl<BE: HostBackend> FiberTransform<BE> {
    /// Encodes the map with diagonals `diagonals`, quantized at the scale and headroom of
    /// `coeffs_meta`, as a BSGS transform with giant step `giant_step`.
    ///
    /// With `shaped`, each diagonal is quantized `encode_bits` finer, then rounded with the
    /// shaping polynomial pre-rotated by the inverse of its giant rotation, so that after that
    /// rotation its quantization error is a multiple of `g`. This is what a map writing to the
    /// replicated layout needs; a map writing a coefficient message gains nothing from it.
    pub fn new(
        module: &Module<BE>,
        coeffs_meta: CoeffsMeta,
        diagonals: &ComplexDiagonals<f64>,
        giant_step: usize,
        shaped: Option<&ShapedContext>,
        base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<Self>
    where
        Module<BE>: CKKSLinearTransformationOps<BE>
            + CnvPVecAlloc<BE>
            + CKKSModuleAlloc<BE>
            + poulpy_ckks::api::CKKSEncodingOps<BE, f64>
            + CKKSModuleInfos,
    {
        let strategy = if diagonals.re.indexes().len() <= 2 {
            LinearTransformationStrategy::Direct
        } else {
            LinearTransformationStrategy::Bsgs { giant_step }
        };
        let encode = |meta: CoeffsMeta, scratch: &mut ScratchArena<'_, BE>| {
            ckks_encode_linear_transformation_from_diagonals::<BE, f64>(
                module,
                base2k.into(),
                meta,
                diagonals,
                strategy,
                false,
                scratch,
            )
            .map_err(CKKSError::Internal)
        };
        let mut lt = encode(coeffs_meta, scratch)?;
        if let Some(ctx) = shaped {
            let bits = ctx.encode_bits();
            let log_delta = coeffs_meta.meta.log_delta;
            let fine_meta = CoeffsMeta::from_delta_budget(log_delta + bits, coeffs_meta.k.as_usize() - log_delta);
            let fine = encode(fine_meta, scratch)?;
            shape_diagonals::<BE>(ctx, module.ckks_max_slots(), &mut lt, &fine, bits)?;
        }
        let rotations = lt
            .baby_steps
            .iter()
            .copied()
            .chain(lt.giant_steps.iter().map(|g| g.rot))
            .filter(|&r| r != 0)
            .collect();
        let first = lt.first_diagonal_plaintext().expect("a transform has a diagonal");
        let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, scratch);
        Ok(Self {
            baby_steps: lt.baby_steps.clone(),
            lt: prepared,
            rotations,
            log_delta: coeffs_meta.meta.log_delta,
        })
    }

    /// The slot rotations the transform performs, whose automorphism keys it needs.
    pub fn rotations(&self) -> &[i64] {
        &self.rotations
    }

    /// The scale of the diagonals: the bits of budget the transform consumes.
    pub fn log_delta(&self) -> usize {
        self.log_delta
    }

    /// Scratch bytes of [`Self::eval_assign`] and [`Self::eval_wide`] with the guard bits of `ctx`,
    /// on ciphertexts of layout at most `ct` with automorphism keys of layout `atk`.
    pub fn tmp_bytes<C, K>(&self, module: &Module<BE>, ctx: &ShapedContext, ct: &C, atk: &K) -> usize
    where
        Module<BE>: CKKSLinearTransformationOps<BE> + GLWEShift<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        let mut meta = ct.meta();
        meta.log_delta += ctx.guard_bits();
        let wide = CtLayout::new(ct, ct.k().as_usize() + ctx.guard_bits(), meta);
        module
            .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&wide, atk)
            .max(module.ckks_eval_linear_transformation_tmp_bytes(&wide, atk))
            .max(module.glwe_shift_tmp_bytes(wide.size()))
            .max(shaped_round_glwe_tmp_bytes(ct.n().as_usize()))
    }

    /// Evaluates the transform on `ct` in place, with the ordinary roundings of the stock
    /// evaluator at the precision of `ct`.
    pub fn eval_assign<D, H>(
        &self,
        module: &Module<BE>,
        ct: &mut D,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        let mut babies = LinearTransformationBabySteps::alloc(module, &self.baby_steps, &*ct);
        module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &*ct, keys, scratch)?;
        module.ckks_eval_linear_transformation_assign(ct, &babies, &self.lt, keys, scratch)?;
        ct.set_noise(None);
        Ok(())
    }

    /// Evaluates the transform on `ct` as one wide stage `ctx.guard_bits()` wider, ending in one
    /// rounding by the shaping polynomial of `ctx`.
    pub fn eval_wide<D, H>(
        &self,
        module: &Module<BE>,
        ctx: &ShapedContext,
        ct: &mut D,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Module<BE>: CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + GLWEShift<BE> + CnvPVecAlloc<BE>,
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        let bits = ctx.guard_bits();
        let mut wide = module.ckks_ciphertext_alloc(ct.base2k(), TorusPrecision((ct.k().as_usize() + bits) as u32));
        let mut meta = ct.meta();
        meta.log_delta += bits;
        wide.set_meta(meta);
        module.glwe_lsh(&mut wide, ct, 0, scratch);
        let mut babies = LinearTransformationBabySteps::alloc(module, &self.baby_steps, &wide);
        module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &wide, keys, scratch)?;
        module.ckks_eval_linear_transformation_assign(&mut wide, &babies, &self.lt, keys, scratch)?;
        ct.set_log_delta(wide.log_delta() - bits);
        ct.set_log_budget(wide.log_budget());
        ct.set_slots(wide.slots());
        shaped_round_glwe(ctx.shaping(), ct, &wide, scratch);
        ct.set_noise(None);
        Ok(())
    }
}

/// Replaces every diagonal of `coarse` by the shaped rounding of the same diagonal of `fine`,
/// which carries `bits` more scale bits. A diagonal inside a giant step of rotation `rot` uses
/// `g(X^(5^-rot))`, written in the subring the diagonal is stored in.
fn shape_diagonals<BE: HostBackend>(
    ctx: &ShapedContext,
    max_slots: usize,
    coarse: &mut LinearTransformation<CKKSPlaintextOwned<BE>>,
    fine: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    bits: usize,
) -> CKKSResult<()> {
    let n = ctx.layout().n();
    for (giant, fine_giant) in coarse.giant_steps.iter_mut().zip(&fine.giant_steps) {
        assert_eq!(giant.rot, fine_giant.rot, "the two encodings share their schedule");
        let gal = pow5((-giant.rot).rem_euclid(max_slots as i64) as u64, 2 * n);
        for (diag, fine_diag) in giant.diagonals.iter_mut().zip(&fine_giant.diagonals) {
            assert_eq!(diag.baby, fine_diag.baby, "the two encodings share their schedule");
            let degree = diag.plaintext.n().as_usize();
            let shaping = ShapingPolynomial::new(ctx.unit(), degree)
                .map_err(|e| CKKSError::Internal(e.into()))?
                .automorphism(gal % (2 * degree));
            let rounded = shaped_round_integers(&shaping, &plaintext_integers::<BE>(&fine_diag.plaintext), bits);
            let (base2k, k) = (diag.plaintext.base2k().as_usize(), diag.plaintext.encoded_k().as_usize());
            GLWEToBackendMut::<BE>::to_backend_mut(&mut diag.plaintext)
                .data_mut()
                .encode_vec_i128(base2k, 0, k, &rounded);
        }
    }
    Ok(())
}
