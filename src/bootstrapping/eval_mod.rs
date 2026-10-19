//! Shaped EvalMod: `x mod 1` on the replicated layout.
//!
//! The polynomial and its BSGS schedule are those of poulpy's [`EvalModPlan`], evaluated with
//! shaped products. Every constant is a shaped encoding of `c pi_S`, where `pi_S` is the
//! indicator of the data slots: the constants never write on the guards, which therefore only
//! carry the shaped rounding errors and stay small. The last double-angle squaring can be left
//! unrescaled, so that its rounding happens at the much smaller scale of the bootstrapping output
//! (the guard reset folded into EvalMod of the paper).

use poulpy_ckks::{
    CKKSCtBounds, CKKSError, CKKSInfos, CKKSMeta, CKKSResult, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCopyOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform},
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EvalModPlan, EvalModType, compile_eval_mod,
        eval_mod::EvalModPoly,
    },
};
use poulpy_core::{
    BSGSOps, GLWENormalize, GLWEPolynomialEvaluation, GLWEZero,
    layouts::{
        BSGSPolynomial, BabyStep, Basis, GGLWEInfos, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey,
        LWEInfos, Parity, PowerBasis, PowerBasisHelper, Rank, TorusPrecision, split_degree,
    },
};
use std::{cell::RefCell, collections::HashMap};

use poulpy_core::{
    ScratchArenaTakeCore, cnv_offset_to_limb_offset,
    layouts::{Degree, IntPolyInfos},
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxIdftNormalizeConsume,
        VecZnxIdftNormalizeConsumeTmpBytes,
    },
    layouts::{
        Backend, CnvPVecLOwned, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, PrepareHint,
        ScratchArena, VecZnxDftToBackendMut,
    },
};

use crate::{
    backend::HostBackend,
    context::ShapedContext,
    encoding::{EncodingModule, ShapedEncoding, plaintext_integers},
    ops::{ShapedModule, ShapedOps, glwe_bytes, layout_at, shaped_round_glwe, shaped_round_glwe_tmp_bytes},
    params::mul_pt_params,
};

type Ct<BE> = CKKSCiphertextOwned<BE>;

/// Headroom bits of an encoded constant: constants stay below 8 in magnitude.
const CONST_HEAD: usize = 4;

/// The coefficients of one baby step: a plaintext shell giving its degree and constant scale to
/// the BSGS engine, and the exact values that are shaped and encoded when they are used.
pub(crate) struct ShapedCoeffs<BE: Backend> {
    shell: CKKSPlaintextOwned<BE>,
    exact: Vec<f64>,
}

impl<BE: Backend> LWEInfos for ShapedCoeffs<BE>
where
    CKKSPlaintextOwned<BE>: LWEInfos,
{
    fn n(&self) -> poulpy_core::layouts::Degree {
        self.shell.n()
    }

    fn k(&self) -> TorusPrecision {
        self.shell.k()
    }

    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.shell.base2k()
    }

    fn max_size(&self) -> usize {
        self.shell.max_size()
    }
}

impl<BE: Backend> GLWEInfos for ShapedCoeffs<BE>
where
    CKKSPlaintextOwned<BE>: GLWEInfos,
{
    fn rank(&self) -> Rank {
        self.shell.rank()
    }
}

impl<BE: Backend> GLWEToBackendRef<BE> for ShapedCoeffs<BE>
where
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        self.shell.to_backend_ref()
    }
}

/// A compiled shaped EvalMod.
pub struct ShapedEvalMod<BE: Backend> {
    plan: EvalModPlan,
    poly: BSGSPolynomial<ShapedCoeffs<BE>>,
    range_consts: Vec<f64>,
    input_offset: Option<f64>,
    consumed_bits: usize,
}

/// The operations EvalMod needs from a module.
pub trait EvalModModule<BE: Backend>:
    ShapedOps<BE>
    + ShapedModule<BE>
    + ShapedEncoding<BE>
    + EncodingModule<BE>
    + GLWEPolynomialEvaluation<BE>
    + GLWEZero<BE>
    + CKKSCopyOps<BE>
    + CKKSPow2Ops<BE>
    + CKKSSubOps<BE>
    + Convolution<BE>
    + CnvPVecAlloc<BE>
    + VecZnxIdftNormalizeConsume<BE>
    + VecZnxIdftNormalizeConsumeTmpBytes
    + CnvPVecBytesOf
    + ModuleN
{
}

impl<BE: Backend, M> EvalModModule<BE> for M where
    M: ShapedOps<BE>
        + ShapedModule<BE>
        + ShapedEncoding<BE>
        + EncodingModule<BE>
        + GLWEPolynomialEvaluation<BE>
        + GLWEZero<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftNormalizeConsume<BE>
        + VecZnxIdftNormalizeConsumeTmpBytes
        + CnvPVecBytesOf
        + ModuleN
{
}

/// `k` of the product of `a` and `b`, as in poulpy-ckks.
fn product_k<A: CKKSInfos, B: CKKSInfos>(a: &A, b: &B) -> CKKSResult<usize> {
    let budget = a
        .log_budget()
        .min(b.log_budget())
        .checked_sub(a.log_delta().max(b.log_delta()))
        .ok_or_else(|| CKKSError::Internal(anyhow::anyhow!("EvalMod ran out of budget")))?;
    Ok(budget + a.log_delta().min(b.log_delta()))
}

/// The shaped encoding of `c pi_S` at scale `2^log_delta`, a plaintext of `R` slots.
fn masked_constant<BE>(
    module: &Module<BE>,
    ctx: &ShapedContext,
    c: f64,
    log_delta: usize,
    base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPlaintextOwned<BE>>
where
    BE: HostBackend,
    Module<BE>: EncodingModule<BE> + ShapedEncoding<BE>,
{
    let mut pt = module.ckks_pt_vec_alloc_compact(ctx.layout().replication(), base2k.into(), (log_delta + CONST_HEAD).into());
    pt.set_meta(CKKSMeta {
        log_delta,
        log_sparsity: 0,
        slots: SlotsKind::Real,
    });
    module.shaped_encode_constant_into(ctx, &mut pt, c, 0.0, true, scratch)?;
    Ok(pt)
}

/// The factor of the second half of a packed batch: of a batch of `2 half` values, values
/// `0..half` are multiplied by one and values `half..2 half` by `factor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Phase {
    pub half: usize,
    pub factor: (f64, f64),
}

impl Phase {
    /// The phase of a square.
    fn squared(self) -> Self {
        let (a, b) = self.factor;
        Self {
            half: self.half,
            factor: (a * a - b * b, 2.0 * a * b),
        }
    }
}

/// The shaped encoding of `c pi_S`, times the factor of `phase` on the second half of a packed
/// batch, at scale `2^log_delta`.
fn constant<BE>(
    module: &Module<BE>,
    ctx: &ShapedContext,
    c: f64,
    phase: Option<Phase>,
    log_delta: usize,
    base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPlaintextOwned<BE>>
where
    BE: HostBackend,
    Module<BE>: EncodingModule<BE> + ShapedEncoding<BE>,
{
    let Some(phase) = phase else {
        return masked_constant(module, ctx, c, log_delta, base2k, scratch);
    };
    let r = ctx.layout().replication();
    let period = 2 * phase.half * r;
    let (mut re, mut im) = (vec![0.0; period], vec![0.0; period]);
    for j in 0..2 * phase.half {
        (re[j * r], im[j * r]) = if j < phase.half {
            (c, 0.0)
        } else {
            (c * phase.factor.0, c * phase.factor.1)
        };
    }
    let mut pt = module.ckks_pt_vec_alloc_compact(period, base2k.into(), (log_delta + CONST_HEAD).into());
    pt.set_meta(CKKSMeta {
        log_delta,
        log_sparsity: 0,
        slots: SlotsKind::Complex,
    });
    module.shaped_encode_slots_into(ctx, &mut pt, &re, &im, scratch)?;
    Ok(pt)
}

/// `ct -= c pi_S`, at the scale of `ct`.
fn sub_masked<BE>(
    module: &Module<BE>,
    ctx: &ShapedContext,
    ct: &mut Ct<BE>,
    c: f64,
    phase: Option<Phase>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
{
    let pt = constant(module, ctx, c, phase, ct.log_delta(), ct.base2k().as_usize(), scratch)?;
    module.ckks_sub_pt_vec_assign(ct, &pt, scratch)
}

/// `ct += c pi_S`, at the scale of `ct`.
fn add_masked<BE>(
    module: &Module<BE>,
    ctx: &ShapedContext,
    ct: &mut Ct<BE>,
    c: f64,
    phase: Option<Phase>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
{
    let pt = constant(module, ctx, c, phase, ct.log_delta(), ct.base2k().as_usize(), scratch)?;
    module.ckks_add_pt_vec_assign(ct, &pt, scratch)
}

/// The arithmetic of the BSGS engine, with shaped products and masked shaped constants.
///
/// The powers of the basis recur in every baby step: their convolution form is prepared once per
/// evaluation and kept here.
struct ShapedBsgs<'a, BE: Backend> {
    ctx: &'a ShapedContext,
    /// The phase of every constant of the polynomial, for a packed evaluation.
    phase: Option<Phase>,
    prepared: RefCell<HashMap<(usize, usize), CnvPVecLOwned<BE>>>,
}

impl<BE> ShapedBsgs<'_, BE>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    /// `res = sum_i c_i pi_S term_i` in one pass: the products accumulate in the transform domain,
    /// with one inverse transform and one shaped rounding. The budget and offset rule is that of the
    /// stock fused baby step. Returns `false`, leaving `res` untouched, when a term does not fit it.
    fn fused_combination(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        terms: &[(&Ct<BE>, usize)],
        coeffs: &ShapedCoeffs<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<bool> {
        let Some(&(seed, _)) = terms.last() else {
            return Ok(false);
        };
        let kb = res.base2k().as_usize();
        let delta = seed.log_delta();
        let (c_delta, c_budget, c_k) = (
            coeffs.shell.log_delta(),
            coeffs.shell.log_budget(),
            coeffs.shell.encoded_k().as_usize(),
        );
        let mut budget = seed.log_budget();
        let mut offsets = Vec::with_capacity(terms.len());
        for (t, _) in terms {
            if t.base2k().as_usize() != kb
                || t.log_delta() != delta
                || !GLWEToBackendRef::<BE>::is_canonical(*t)
                || t.rank().as_usize() != 1
            {
                return Ok(false);
            }
            let Ok((b, d, off)) = mul_pt_params(
                "shaped_fused",
                budget + delta,
                t.log_delta(),
                t.log_budget(),
                c_delta,
                c_budget,
                c_k,
            ) else {
                return Ok(false);
            };
            if d != delta || off < c_k {
                return Ok(false);
            }
            budget = budget.min(b);
            offsets.push((b, off));
        }
        // Every constant is placed in a window one limb wider than its integers, shifted so that
        // all terms share the sub-limb part of their convolution offset.
        let vec_k = (c_delta + CONST_HEAD + 8).next_multiple_of(kb);
        let k_pt = vec_k + kb;
        let raw: Vec<usize> = offsets.iter().map(|&(b, off)| off + (b - budget) - c_k + k_pt).collect();
        let lo = raw[0] % kb;
        let mut plans = Vec::with_capacity(terms.len());
        for ((_, idx), &o) in terms.iter().zip(&raw) {
            let r = (o + kb - lo) % kb;
            if o - r < kb {
                return Ok(false);
            }
            let (hi, _) = cnv_offset_to_limb_offset(o - r, kb);
            let constant = constant(module, self.ctx, coeffs.exact[*idx], self.phase, c_delta, kb, scratch)?;
            let shifted: Vec<i128> = plaintext_integers::<BE>(&constant).iter().map(|&x| x << r).collect();
            let mut pt = module.ckks_plaintext_alloc(Degree(shifted.len() as u32), kb.into(), TorusPrecision(k_pt as u32));
            GLWEToBackendMut::<BE>::to_backend_mut(&mut pt)
                .data_mut()
                .encode_vec_i128(kb, 0, k_pt, &shifted);
            plans.push((hi, pt));
        }
        // The powers, prepared once per evaluation.
        let n = module.n();
        for (t, power) in terms {
            let key = (*power, t.size());
            if !self.prepared.borrow().contains_key(&key) {
                let mut fresh = module.cnv_pvec_left_alloc(n, 2, t.size(), PrepareHint::Reuse);
                module.cnv_prepare_left(
                    &mut fresh.to_backend_mut(),
                    GLWEToBackendRef::<BE>::to_backend_ref(*t).data(),
                    scratch,
                );
                self.prepared.borrow_mut().insert(key, fresh);
            }
        }
        let prepared = self.prepared.borrow();

        let k_res = budget + delta;
        let bits = self.ctx.pass_cap();
        let pt_size = k_pt.div_ceil(kb);
        let dft_size = terms
            .iter()
            .zip(&plans)
            .map(|((t, _), (hi, _))| (t.size() + pt_size).saturating_sub(*hi))
            .max()
            .unwrap_or(1)
            .max(1);
        let wide_layout = poulpy_core::layouts::GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision((k_res + bits) as u32),
            rank: Rank(1),
        };
        let mut sparsity = res.log_sparsity();
        let mut slots = seed.slots();
        for (t, _) in terms {
            sparsity = sparsity.min(t.log_sparsity());
            slots = slots.join(t.slots());
        }
        let (mut wide, rest) = scratch.borrow().take_glwe_scratch(&wide_layout);
        let (mut acc, mut rest) = rest.take_vec_znx_dft_scratch(n, 2, dft_size);
        for (i, ((t, power), (hi, pt))) in terms.iter().zip(&plans).enumerate() {
            let a = &prepared[&(*power, t.size())];
            let (mut b, mut inner) =
                rest.borrow()
                    .take_cnv_pvec_right_scratch(pt.n().as_usize(), 1, pt_size, PrepareHint::OneShot);
            module.cnv_prepare_right(&mut b, GLWEToBackendRef::<BE>::to_backend_ref(pt).data(), &mut inner.borrow());
            for col in 0..2 {
                let mut dst = acc.to_backend_mut();
                if i == 0 {
                    module.cnv_apply_dft(
                        *hi,
                        &mut dst,
                        col,
                        &a.to_backend_ref(),
                        col,
                        &b.to_backend_ref(),
                        0,
                        &mut inner.borrow(),
                    );
                } else {
                    module.cnv_apply_dft_add(
                        *hi,
                        &mut dst,
                        col,
                        &a.to_backend_ref(),
                        col,
                        &b.to_backend_ref(),
                        0,
                        &mut inner.borrow(),
                    );
                }
            }
        }
        for col in 0..2 {
            module.vec_znx_idft_normalize_consume(
                wide.data_mut(),
                kb,
                k_res + bits,
                lo as i64,
                col,
                &mut acc.to_backend_mut(),
                col,
                kb,
                None,
                &mut rest.borrow(),
            );
        }
        res.set_log_delta(delta);
        res.set_log_budget(budget);
        res.set_log_sparsity(sparsity);
        res.set_slots(slots);
        shaped_round_glwe(self.ctx.shaping(), res, &wide, &mut rest);
        Ok(true)
    }
}

impl<BE> BSGSOps<BE, Ct<BE>, ShapedCoeffs<BE>, Ct<BE>, Ct<BE>> for ShapedBsgs<'_, BE>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
    for<'x> ScratchArena<'x, BE>: poulpy_core::ScratchArenaTakeCore<'x, BE>,
    ShapedCoeffs<BE>: GLWEToBackendRef<BE>,
{
    type Prepared = Ct<BE>;

    fn init_accumulator(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        seed: &Ct<BE>,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        res.set_log_delta(seed.log_delta());
        res.set_log_budget(seed.log_budget());
        res.set_slots(seed.slots());
        module.glwe_zero(res);
        Ok(())
    }

    fn eval_baby_linear_combination(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        terms: &[(&Ct<BE>, usize)],
        coeffs: &ShapedCoeffs<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<bool> {
        Ok(self.fused_combination(module, res, terms, coeffs, scratch)?)
    }

    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        _res_coeff: usize,
        coeffs: &ShapedCoeffs<BE>,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        let c = coeffs.exact[idx];
        if c != 0.0 {
            add_masked(module, self.ctx, res, c, self.phase, scratch)?;
        }
        Ok(())
    }

    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        a: &Ct<BE>,
        coeffs: &ShapedCoeffs<BE>,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        let pt = constant(
            module,
            self.ctx,
            coeffs.exact[idx],
            self.phase,
            coeffs.shell.log_delta(),
            a.base2k().as_usize(),
            scratch,
        )?;
        module.shaped_mul_pt_into(self.ctx, res, a, &pt, scratch)?;
        Ok(())
    }

    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        a: &Ct<BE>,
        coeffs: &ShapedCoeffs<BE>,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        let c = coeffs.exact[idx];
        if c == 0.0 {
            return Ok(());
        }
        let pt = constant(
            module,
            self.ctx,
            c,
            self.phase,
            coeffs.shell.log_delta(),
            a.base2k().as_usize(),
            scratch,
        )?;
        let mut term = module.ckks_ciphertext_alloc_from_infos(res);
        module.shaped_mul_pt_into(self.ctx, &mut term, a, &pt, scratch)?;
        module.ckks_add_assign(res, &term, scratch)?;
        Ok(())
    }

    fn prepare_right(&self, module: &Module<BE>, a: &Ct<BE>, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<Ct<BE>> {
        let mut copy = module.ckks_ciphertext_alloc_from_infos(a);
        module.ckks_copy(&mut copy, a, scratch)?;
        Ok(copy)
    }

    fn mul_prepared_assign<H>(
        &self,
        module: &Module<BE>,
        dst: &mut Ct<BE>,
        prepared: &Ct<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()>
    where
        H: GetTensorKey<BE>,
    {
        module.shaped_mul_assign(self.ctx, dst, prepared, tsk, scratch)?;
        Ok(())
    }

    fn add_assign(
        &self,
        module: &Module<BE>,
        dst: &mut Ct<BE>,
        a: &Ct<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module.ckks_add_assign(dst, a, scratch)?;
        Ok(())
    }

    fn copy(
        &self,
        module: &Module<BE>,
        res: &mut Ct<BE>,
        src: &Ct<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module.ckks_copy(res, src, scratch)?;
        Ok(())
    }
}

/// An evaluated baby step, as the BSGS engine folds them.
struct Evaluated<BE: Backend> {
    degree: usize,
    value: Ct<BE>,
}

impl<BE: Backend> BabyStep<BE> for Evaluated<BE>
where
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::SetBSGSMeta,
{
    type Value = Ct<BE>;

    fn degree(&self) -> usize {
        self.degree
    }

    fn get(&self) -> &Ct<BE> {
        &self.value
    }

    fn get_mut(&mut self) -> &mut Ct<BE> {
        &mut self.value
    }
}

impl<BE> ShapedEvalMod<BE>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
    for<'x> ScratchArena<'x, BE>: poulpy_core::ScratchArenaTakeCore<'x, BE>,
{
    /// Compiles `plan`: the polynomial of poulpy's EvalMod, decomposed for BSGS from its exact
    /// coefficients. The real trigonometric families without an inverse are supported.
    pub fn new(module: &Module<BE>, plan: EvalModPlan, base2k: usize, scratch: &mut ScratchArena<'_, BE>) -> CKKSResult<Self> {
        if plan.eval_mod_type == EvalModType::ExpCmplx || plan.f_mod_inv_degree.is_some() {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the shaped EvalMod supports the real families without inverse"
            )));
        }
        let compiled = compile_eval_mod::<BE, f64>(base2k.into(), plan, module, scratch).map_err(CKKSError::Internal)?;
        let EvalModPoly::Real(poly) = &compiled.f_mod_poly else {
            unreachable!("a real family has a real polynomial")
        };
        let exact = |c: &[f64]| -> anyhow::Result<Vec<f64>> { Ok(c.to_vec()) };
        let decomposed = if plan.folds_even_base() {
            poly.decompose_bsgs_folded_with(plan.split_strategy, exact)
        } else {
            poly.decompose_bsgs_with(plan.split_strategy, exact)
        }
        .map_err(CKKSError::Internal)?;
        let coeffs_meta = plan.coeffs_meta;
        let poly = decomposed.map_baby_steps_ref(|values| {
            let mut shell = module.ckks_pt_coeffs_alloc(values.len(), base2k.into(), coeffs_meta.k);
            shell.set_meta(coeffs_meta.meta);
            ShapedCoeffs {
                shell,
                exact: values.clone(),
            }
        });
        let s = compiled.range_extension_scale();
        let range_consts = (0..plan.f_mod_log_interval_reduction).map(|i| s.powi(1 << (i + 1))).collect();
        Ok(Self {
            plan,
            poly,
            range_consts,
            input_offset: plan.input_offset::<f64>(),
            consumed_bits: compiled.consumed_bits(),
        })
    }

    /// The plan this EvalMod was compiled from.
    pub fn plan(&self) -> &EvalModPlan {
        &self.plan
    }

    /// Scratch bytes of [`Self::eval`] with the shaping of `ctx`, on ciphertexts of layout at most
    /// `ct` with a tensor key of layout `tsk`.
    pub fn tmp_bytes<C, T>(&self, module: &Module<BE>, ctx: &ShapedContext, ct: &C, tsk: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let n = ct.n().as_usize();
        let (kb, size) = (ct.base2k().as_usize(), ct.size());
        // The fused baby steps: a wide accumulator, the transform-domain sum and one prepared
        // constant at a time, as laid out by `fused_combination`.
        let vec_k = (self.plan.coeffs_meta.meta.log_delta + CONST_HEAD + 8).next_multiple_of(kb);
        let pt_size = (vec_k + kb) / kb;
        let dft_size = size + pt_size;
        let wide = layout_at(ct, ct.k().as_usize() + ctx.pass_cap());
        let fused = glwe_bytes(&wide)
            + BE::bytes_of_vec_znx_dft(n, 2, dft_size)
            + module.bytes_of_cnv_pvec_right(n, 1, pt_size, PrepareHint::OneShot)
            + module
                .cnv_prepare_right_tmp_bytes(pt_size, pt_size)
                .max(module.cnv_apply_dft_tmp_bytes(0, dft_size, size, pt_size))
                .max(module.vec_znx_idft_normalize_consume_tmp_bytes(wide.size(), dft_size))
                .max(shaped_round_glwe_tmp_bytes(n));
        fused
            .max(module.cnv_prepare_left_tmp_bytes(size, size))
            .max(module.shaped_ops_tmp_bytes(ctx, ct, tsk, tsk))
            .max(module.shaped_encoding_tmp_bytes(ctx, ct))
            .max(module.ckks_copy_tmp_bytes(ct, ct))
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_sub_tmp_bytes(size))
            .max(module.ckks_add_pt_const_tmp_bytes(size))
            .max(module.ckks_mul_pow2_tmp_bytes(size))
            .max(module.glwe_normalize_tmp_bytes())
    }

    /// Bits of budget the evaluation consumes at the EvalMod scale.
    pub fn consumed_bits(&self) -> usize {
        self.consumed_bits
    }

    /// Shaped power basis of `x` up to what the BSGS schedule reads.
    fn power_basis<H>(
        &self,
        module: &Module<BE>,
        ctx: &ShapedContext,
        x: Ct<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<PowerBasis<Ct<BE>>>
    where
        H: GetTensorKey<BE>,
    {
        let basis = self.poly.basis();
        let mut powers = PowerBasis::new(basis, x);
        let degree = self.poly.degree();
        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        let largest = 1usize << (log_degree - 1);
        let base = 1usize << self.poly.log_split();
        let baby_limit = base.min(degree + 1);
        if largest >= 2 {
            gen_power(module, ctx, &mut powers, largest, tsk, scratch)?;
        }
        let babies: Vec<usize> = match self.poly.parity() {
            Parity::Even => (4..baby_limit).step_by(2).collect(),
            Parity::Odd => (3..baby_limit).step_by(2).collect(),
            Parity::Full => (3..baby_limit).rev().collect(),
        };
        for i in babies {
            gen_power(module, ctx, &mut powers, i, tsk, scratch)?;
        }
        Ok(powers)
    }

    /// `res = EvalMod(ct)`.
    ///
    /// `ct` enters at its own scale, is rescaled to the EvalMod scale with shaping, and `res` comes
    /// out labelled at the input scale, as in poulpy. With `defer_last_step` the last double-angle
    /// squaring is not rescaled and `res` carries the EvalMod scale on top of the input scale.
    ///
    /// With `packed = Some(k)`, `ct` holds a batch of `2k` real values and `res` comes out with
    /// the second half multiplied by `i`: the phase is carried by the constants, at no cost.
    #[allow(clippy::too_many_arguments)]
    pub fn eval<H>(
        &self,
        module: &Module<BE>,
        ctx: &ShapedContext,
        res: &mut Ct<BE>,
        ct: &Ct<BE>,
        tsk: &H,
        defer_last_step: bool,
        packed: Option<usize>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        H: GetTensorKey<BE>,
    {
        let (s_in, s_eval) = (ct.log_delta(), self.plan.f_mod_log_delta);
        // A packed evaluation multiplies the polynomial on the second half by `exp(i pi / 2^(r+1))`,
        // which the `r` double-angle squarings raise to `i`.
        let steps = self.range_consts.len();
        let phase = packed.map(|half| {
            let angle = std::f64::consts::PI / (1u64 << (steps + 1)) as f64;
            Phase {
                half,
                factor: (angle.cos(), angle.sin()),
            }
        });
        if s_in < s_eval {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "EvalMod input scale {s_in} below the EvalMod scale {s_eval}"
            )));
        }
        // The input at the EvalMod scale.
        let mut x = module.ckks_ciphertext_alloc(ct.base2k(), TorusPrecision((ct.k().as_usize() - (s_in - s_eval)) as u32));
        module.shaped_rescale_into(ctx, &mut x, ct, s_in - s_eval, scratch)?;
        if let Some(offset) = self.input_offset {
            add_masked(module, ctx, &mut x, offset, None, scratch)?;
        }
        // Several products read every power: each is normalized once, where it is made.
        module.glwe_normalize_assign(&mut x, scratch);
        let input = match self.poly.input_transform() {
            PolynomialInputTransform::Identity => x,
            PolynomialInputTransform::ChebyshevT2 => {
                let k = product_k(&x, &x)?;
                let mut squared = module.ckks_ciphertext_alloc(x.base2k(), TorusPrecision(k as u32));
                module.shaped_square_into(ctx, &mut squared, &x, tsk, scratch)?;
                let mut t2 = module.ckks_ciphertext_alloc_from_infos(&squared);
                module.ckks_double_into(&mut t2, &squared, scratch)?;
                sub_masked(module, ctx, &mut t2, 1.0, None, scratch)?;
                module.glwe_normalize_assign(&mut t2, scratch);
                t2
            }
            other => {
                return Err(CKKSError::Internal(anyhow::anyhow!(
                    "unsupported EvalMod input transform {other:?}"
                )));
            }
        };
        let powers = self.power_basis(module, ctx, input, tsk, scratch)?;

        // BSGS, by the engine of poulpy-core over the shaped arithmetic.
        let ops = ShapedBsgs {
            ctx,
            phase,
            prepared: RefCell::new(HashMap::new()),
        };
        let n_baby = self.poly.baby_steps().len();
        let last = self.poly.baby_step(n_baby - 1);
        let fold = n_baby >= 2
            && last.shell.n().as_usize() == 1
            && PowerBasisHelper::<BE, Ct<BE>>::has_power(&powers, self.poly.degree());
        let x1 = PowerBasisHelper::<BE, Ct<BE>>::get(&powers, 1).map_err(CKKSError::Internal)?;
        let mut evaluated = Vec::with_capacity(n_baby);
        for i in 0..n_baby - usize::from(fold) {
            let coeffs = self.poly.baby_step(i);
            let mut value = module.ckks_ciphertext_alloc_from_infos(x1);
            value.set_meta(x1.meta());
            module
                .glwe_eval_baby_step(&ops, &mut value, self.poly.parity(), coeffs, &powers, scratch)
                .map_err(CKKSError::Internal)?;
            evaluated.push(Evaluated {
                degree: coeffs.shell.n().as_usize() - 1,
                value,
            });
        }
        module
            .glwe_eval_giant_steps::<_, Ct<BE>, _, Ct<BE>, ShapedCoeffs<BE>, Ct<BE>, _, H>(
                &ops,
                res,
                &mut evaluated,
                &powers,
                tsk,
                scratch,
            )
            .map_err(CKKSError::Internal)?;
        if fold {
            let xpow = PowerBasisHelper::<BE, Ct<BE>>::get(&powers, self.poly.degree()).map_err(CKKSError::Internal)?;
            BSGSOps::<BE, Ct<BE>, ShapedCoeffs<BE>, Ct<BE>, Ct<BE>>::mul_add_pt_const(&ops, module, res, xpow, last, 0, scratch)
                .map_err(CKKSError::Internal)?;
        }

        // Double-angle steps: res <- 2 res^2 - c_i pi_S, times the squared phase when packed.
        let mut phase = phase;
        for (i, &c) in self.range_consts.iter().enumerate() {
            phase = phase.map(Phase::squared);
            let squared = if defer_last_step && i + 1 == steps {
                let mut sq = module.ckks_ciphertext_alloc_from_infos(res);
                module.shaped_square_unrescaled_into(ctx, &mut sq, res, tsk, scratch)?;
                sq
            } else {
                let mut sq = module.ckks_ciphertext_alloc(res.base2k(), TorusPrecision(product_k(res, res)? as u32));
                module.shaped_square_into(ctx, &mut sq, res, tsk, scratch)?;
                sq
            };
            module.ckks_double_into(res, &squared, scratch)?;
            sub_masked(module, ctx, res, c, phase, scratch)?;
        }
        let extra = if defer_last_step && steps > 0 { s_eval } else { 0 };
        res.set_log_delta(s_in + extra);
        Ok(())
    }
}

/// Generates `T_n` (or `X^n`) into `powers` from smaller powers, with shaped products. The
/// Chebyshev recurrence `T_(a+b) = 2 T_a T_b - T_|a-b|` subtracts `pi_S` for `a = b`, not one,
/// so that the guards are not driven to `-1`.
fn gen_power<BE, H>(
    module: &Module<BE>,
    ctx: &ShapedContext,
    powers: &mut PowerBasis<Ct<BE>>,
    n: usize,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: HostBackend,
    Module<BE>: EvalModModule<BE>,
    for<'x> ScratchArena<'x, BE>: poulpy_core::ScratchArenaTakeCore<'x, BE>,
    H: GetTensorKey<BE>,
{
    if powers.contains_power(n) {
        return Ok(());
    }
    let (a, b) = split_degree(n);
    gen_power(module, ctx, powers, a, tsk, scratch)?;
    gen_power(module, ctx, powers, b, tsk, scratch)?;
    let c = a.abs_diff(b);
    let chebyshev = powers.basis() == Basis::Chebyshev;
    if chebyshev && c != 0 {
        gen_power(module, ctx, powers, c, tsk, scratch)?;
    }
    let value = {
        let a_val = powers.get_stored(a).expect("generated above");
        let b_val = powers.get_stored(b).expect("generated above");
        let k = TorusPrecision(product_k(a_val, b_val)? as u32);
        let mut product = module.ckks_ciphertext_alloc(a_val.base2k(), k);
        if a == b {
            module.shaped_square_into(ctx, &mut product, a_val, tsk, scratch)?;
        } else {
            module.shaped_mul_into(ctx, &mut product, a_val, b_val, tsk, scratch)?;
        }
        if chebyshev {
            let mut doubled = module.ckks_ciphertext_alloc(a_val.base2k(), k);
            module.ckks_double_into(&mut doubled, &product, scratch)?;
            if c == 0 {
                sub_masked(module, ctx, &mut doubled, 1.0, None, scratch)?;
            } else {
                let c_val = powers.get_stored(c).expect("generated above");
                module.ckks_sub_assign(&mut doubled, c_val, scratch)?;
            }
            module.glwe_normalize_assign(&mut doubled, scratch);
            doubled
        } else {
            product
        }
    };
    powers.set_power(n, value);
    Ok(())
}
