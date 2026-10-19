//! Conversions between CKKS and the replicated layout (Appendix A of the paper).
//!
//! Both schemes use the phase `c_0 + c_1 s` and share the secret and the evaluation keys, so a
//! conversion only changes the layout and the scale. On the CKKS side the `k` values repeat with
//! period `k` on the slots (the sparse packing of `log_sparsity = log2(N / 2k)`); on the other
//! side they sit on the data slots of the replicated layout.
//!
//! - CKKS to the replicated layout: one map with `k` diagonals writes each value on its data
//!   slots and zero on the guards, as a wide stage with shaped diagonals ending in a shaped
//!   rounding, then a shaped rescale brings the result to the leveled scale.
//! - Replicated layout to CKKS: an exact extension by `2^w` to the CKKS scale, the gather map
//!   (`k` diagonals reading the data slots only, with weight `R`), and the normalized trace onto
//!   period `k`, which averages the `h = dR` positions that share a value.

use poulpy_ckks::{
    CKKSError, CKKSInfos, CKKSMeta, CKKSModuleInfos, CKKSResult, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSLinearTransformationOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::{
    GLWEShift, GLWETrace,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    backend::HostBackend,
    bootstrapping::{
        fiber,
        transform::{self, FiberTransform},
    },
    context::ShapedContext,
    ops::{CtLayout, ShapedModule, ShapedOps},
};

type Ct<BE> = CKKSCiphertextOwned<BE>;

/// The parameters of the conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConversionParams {
    /// Scale of the CKKS side, `Delta_c`.
    pub ckks_log_delta: usize,
    /// Scale of the replicated side, `Delta`.
    pub log_delta: usize,
    /// Scale of the diagonals, `Delta_d`: the bits each conversion consumes.
    pub diagonal_log_delta: usize,
    /// Extra bits of the wide stage of the conversion to the replicated layout. They divide the
    /// errors of its key switches and cost no modulus, only key width.
    pub guard_bits: usize,
    /// Giant step of the transforms; the power of two next to `sqrt(k)` by default.
    pub giant_step: Option<usize>,
}

impl Default for ConversionParams {
    fn default() -> Self {
        Self {
            ckks_log_delta: 40,
            log_delta: 20,
            diagonal_log_delta: 30,
            guard_bits: 8,
            giant_step: None,
        }
    }
}

/// The operations the conversions need from a module.
pub trait ConversionModule<BE: Backend>:
    ShapedOps<BE>
    + ShapedModule<BE>
    + CKKSLinearTransformationOps<BE>
    + CnvPVecAlloc<BE>
    + CKKSModuleAlloc<BE>
    + CKKSCopyOps<BE>
    + CKKSModuleInfos
{
}

impl<BE: Backend, M> ConversionModule<BE> for M where
    M: ShapedOps<BE>
        + ShapedModule<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleInfos
{
}

/// The two conversions of one layout.
pub struct Converter<BE: Backend> {
    params: ConversionParams,
    shaped: ShapedContext,
    write: FiberTransform<BE>,
    read: FiberTransform<BE>,
}

impl<BE> Converter<BE>
where
    BE: HostBackend,
    Module<BE>: ConversionModule<BE> + poulpy_ckks::api::CKKSEncodingOps<BE, f64>,
    for<'x> ScratchArena<'x, BE>: poulpy_core::ScratchArenaTakeCore<'x, BE>,
{
    /// Compiles the conversions of the layout of `ctx`. The gather back to CKKS needs `R | k`.
    pub fn new(
        module: &Module<BE>,
        ctx: &ShapedContext,
        params: ConversionParams,
        base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<Self> {
        let layout = *ctx.layout();
        let (k, r) = (layout.batch(), layout.replication());
        if !k.is_multiple_of(r) {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the conversion to CKKS needs R | k: pad the batch of {k} values to {r}"
            )));
        }
        if params.ckks_log_delta < params.log_delta || params.ckks_log_delta - params.log_delta > crate::rounding::MAX_PASS_BITS {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the CKKS scale must lie within {} bits above the leveled scale",
                crate::rounding::MAX_PASS_BITS
            )));
        }
        let shaped = ctx
            .clone()
            .with_guard_bits(params.guard_bits)
            .with_pass_cap(crate::rounding::MAX_PASS_BITS)
            .map_err(|e| CKKSError::Internal(e.into()))?;
        let giant = params.giant_step.unwrap_or_else(|| transform::default_giant_step(k));
        let identity = fiber::identity(k);
        let write = FiberTransform::new(
            module,
            CoeffsMeta::from_delta_budget(params.diagonal_log_delta, 3),
            &fiber::write_diagonals(&layout, &identity),
            giant,
            Some(ctx),
            base2k,
            scratch,
        )?;
        let read = FiberTransform::new(
            module,
            CoeffsMeta::from_delta_budget(params.diagonal_log_delta, r.trailing_zeros() as usize + 3),
            &fiber::read_diagonals(&layout, &identity, r as f64),
            giant,
            None,
            base2k,
            scratch,
        )?;
        Ok(Self {
            params,
            shaped,
            write,
            read,
        })
    }

    /// Scratch bytes enough for both conversions on ciphertexts at most `k` bits wide, with
    /// automorphism keys of layout `atk`. The conversion to CKKS works `ckks_log_delta -
    /// log_delta` bits above the width of its input.
    pub fn tmp_bytes<C, K>(&self, module: &Module<BE>, ct: &C, atk: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
    {
        let p = &self.params;
        let meta = CKKSMeta {
            log_delta: p.ckks_log_delta,
            log_sparsity: self.shaped.layout().log_sparsity(),
            slots: SlotsKind::Complex,
        };
        let k = ct.k().as_usize() + p.ckks_log_delta - p.log_delta;
        let wide = CtLayout::new(ct, k, meta);
        self.write
            .tmp_bytes(module, &self.shaped, &wide, atk)
            .max(self.read.tmp_bytes(module, &self.shaped.unshaped(), &wide, atk))
            .max(module.shaped_ops_tmp_bytes(&self.shaped, &wide, atk, atk))
            .max(module.ckks_copy_tmp_bytes(&wide, &wide))
            .max(module.glwe_shift_tmp_bytes(wide.size()))
            .max(module.glwe_trace_assign_tmp_bytes(&wide, atk))
    }

    /// The parameters the conversions were compiled with.
    pub fn params(&self) -> &ConversionParams {
        &self.params
    }

    /// The Galois elements of every automorphism key the conversions use.
    pub fn galois_elements(&self, module: &Module<BE>) -> std::collections::BTreeSet<i64> {
        transform::galois_elements(module, &[&self.write, &self.read])
    }

    /// Converts a CKKS ciphertext at scale `2^ckks_log_delta`, whose slots repeat the `k` values
    /// with period `k`, to the replicated layout at scale `2^log_delta`. The result is
    /// `diagonal_log_delta` bits narrower than `ct` after the rescale.
    pub fn to_shaped<H>(
        &self,
        module: &Module<BE>,
        ct: &Ct<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<Ct<BE>>
    where
        H: GetAutomorphismKey<BE>,
    {
        let p = &self.params;
        if ct.log_delta() != p.ckks_log_delta {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the conversion takes a CKKS ciphertext at scale 2^{}, not 2^{}",
                p.ckks_log_delta,
                ct.log_delta()
            )));
        }
        let mut written = module.ckks_ciphertext_alloc_from_infos(ct);
        module.ckks_copy(&mut written, ct, scratch)?;
        self.write.eval_wide(module, &self.shaped, &mut written, keys, scratch)?;
        written.set_log_sparsity(self.shaped.layout().log_sparsity());
        let drop = p.ckks_log_delta - p.log_delta;
        let mut out = module.ckks_ciphertext_alloc(ct.base2k(), TorusPrecision((written.k().as_usize() - drop) as u32));
        module.shaped_rescale_into(&self.shaped, &mut out, &written, drop, scratch)?;
        Ok(out)
    }

    /// Converts a ciphertext of the replicated layout at scale `2^log_delta` to CKKS at scale
    /// `2^ckks_log_delta`, with the `k` values repeated with period `k` on every slot. The
    /// extension adds `ckks_log_delta - log_delta` bits and the gather consumes
    /// `diagonal_log_delta`.
    pub fn to_ckks<H>(&self, module: &Module<BE>, ct: &Ct<BE>, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> CKKSResult<Ct<BE>>
    where
        H: GetAutomorphismKey<BE>,
    {
        let p = &self.params;
        if ct.log_delta() != p.log_delta {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "the conversion takes a ciphertext at scale 2^{}, not 2^{}",
                p.log_delta,
                ct.log_delta()
            )));
        }
        // Exact extension by 2^w: the scale becomes the CKKS scale at constant budget.
        let w = p.ckks_log_delta - p.log_delta;
        let mut lifted = module.ckks_ciphertext_alloc(ct.base2k(), TorusPrecision((ct.k().as_usize() + w) as u32));
        let mut meta = ct.meta();
        meta.log_delta += w;
        lifted.set_meta(meta);
        module.glwe_lsh(&mut lifted, ct, 0, scratch);
        // Gather from the data slots, then average the h = dR positions that share a value.
        self.read.eval_assign(module, &mut lifted, keys, scratch)?;
        let layout = self.shaped.layout();
        let top_skip = layout.batch().trailing_zeros() as usize + 1;
        if top_skip < layout.n().trailing_zeros() as usize {
            module.glwe_trace_assign(&mut lifted, top_skip, keys, scratch);
        }
        lifted.set_log_sparsity((layout.n() / (2 * layout.batch())).trailing_zeros() as usize);
        GLWEToBackendMut::<BE>::set_noise(&mut lifted, None);
        Ok(lifted)
    }
}
