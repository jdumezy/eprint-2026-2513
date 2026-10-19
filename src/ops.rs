//! Shaped leveled operations.
//!
//! Every operation that rounds runs the stock poulpy kernel into a scratch ciphertext that is
//! `bits` wider than the result, then removes those bits with one shaped rounding. The kernel's
//! own ordinary rounding then sits `bits` below the result, where the shaped division
//! suppresses it by `2^-bits` (the shaping depth of the paper). Operations that do not round
//! (additions, negations, products by powers of two, modulus reduction) are the CKKS ones and
//! are taken from poulpy-ckks as they are.

use poulpy_ckks::{CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSMeta, CKKSResult, SetCKKSInfos};
use poulpy_core::{
    GLWEAutomorphism, GLWEMulPlain, GLWEShift, GLWETensoring, GLWETrace, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, GetAutomorphismKey, GetTensorKey,
        IntPolyInfos, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    backend::HostBackend,
    context::ShapedContext,
    params::{budget_sub, mul_ct_params, mul_pt_params},
    rounding::{shaped_round_column, shaped_round_tmp_bytes},
    shaping::ShapingPolynomial,
};

/// The core operations the shaped operations are built on.
pub trait ShapedModule<BE: Backend>:
    GLWETensoring<BE>
    + GLWEAutomorphism<BE>
    + GLWEShift<BE>
    + GLWEMulPlain<BE>
    + GLWETrace<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

impl<BE: Backend, M> ShapedModule<BE> for M where
    M: GLWETensoring<BE>
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + GLWEMulPlain<BE>
        + GLWETrace<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

/// The relinearized product `a * b` (or `a^2` when `b` is `None`) written into `wide` at its
/// precision: in one pass when the kernel allows it, as in poulpy-ckks, and through the tensor
/// otherwise (operands narrower than one limb, or a key in another radix).
#[allow(clippy::too_many_arguments)]
fn relinearized_product<BE, A, B, T>(
    module: &Module<BE>,
    cnv_offset: usize,
    wide: &mut GLWEViewMut<'_, BE>,
    k: TorusPrecision,
    a: &A,
    b: Option<&B>,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HostBackend,
    Module<BE>: ShapedModule<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    B: GLWEToBackendRef<BE> + GLWEInfos,
    T: GetTensorKey<BE>,
{
    let tensor_k = b.map_or(a.k(), |b| a.k().max(b.k()));
    let base2k = wide.base2k();
    let one_pass = wide.rank().as_usize() == 1
        && cnv_offset >= base2k.as_usize()
        && tsk.get_tensor_key(tensor_k).is_ok_and(|key| key.base2k() == base2k);
    if one_pass {
        match b {
            Some(b) => module.glwe_mul_relinearize(cnv_offset, wide, k, a, b, tsk, scratch),
            None => module.glwe_square_relinearize(cnv_offset, wide, k, a, tsk, scratch),
        }
    } else {
        let mut tensor = module.glwe_tensor_alloc(base2k, tensor_k, wide.rank());
        match b {
            Some(b) => module.glwe_tensor_apply(cnv_offset, &mut tensor, a, b, scratch),
            None => module.glwe_tensor_square_apply(cnv_offset, &mut tensor, a, scratch),
        }
        module.glwe_tensor_relinearize(wide, &tensor, tsk, scratch);
    }
}

/// The layout of `like` at precision `k`.
pub(crate) fn layout_at<A: GLWEInfos>(like: &A, k: usize) -> GLWELayout {
    GLWELayout {
        n: like.n(),
        base2k: like.base2k(),
        k: TorusPrecision(k as u32),
        rank: like.rank(),
    }
}

/// The layout of a CKKS ciphertext without its data, for scratch-size queries.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CtLayout {
    pub glwe: GLWELayout,
    pub meta: CKKSMeta,
}

impl CtLayout {
    pub(crate) fn new<A: GLWEInfos>(like: &A, k: usize, meta: CKKSMeta) -> Self {
        Self {
            glwe: layout_at(like, k),
            meta,
        }
    }
}

impl LWEInfos for CtLayout {
    fn n(&self) -> poulpy_core::layouts::Degree {
        self.glwe.n
    }

    fn k(&self) -> TorusPrecision {
        self.glwe.k
    }

    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.glwe.base2k
    }

    fn max_size(&self) -> usize {
        self.glwe.size()
    }
}

impl GLWEInfos for CtLayout {
    fn rank(&self) -> Rank {
        self.glwe.rank
    }
}

impl CKKSInfos for CtLayout {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

/// Bytes of a ciphertext of layout `infos` taken from a scratch arena, with alignment slack.
pub(crate) fn glwe_bytes<A: GLWEInfos>(infos: &A) -> usize {
    infos.n().as_usize() * (infos.rank().as_usize() + 1) * infos.size() * size_of::<i64>() + 64
}

/// Bytes of [`shaped_round_glwe`] at ring degree `n`.
pub fn shaped_round_glwe_tmp_bytes(n: usize) -> usize {
    shaped_round_tmp_bytes(n) + 64
}

/// Shaped rounding of every column of `wide` into `dst`, which must already carry its final
/// precision, at most [`crate::rounding::MAX_PASS_BITS`] below that of `wide`.
pub fn shaped_round_glwe<BE, D, W>(shaping: &ShapingPolynomial, dst: &mut D, wide: &W, scratch: &mut ScratchArena<'_, BE>)
where
    BE: HostBackend,
    D: GLWEToBackendMut<BE> + GLWEInfos,
    W: GLWEToBackendRef<BE> + GLWEInfos,
{
    let n = dst.n().as_usize();
    let base2k = dst.base2k().as_usize();
    assert_eq!(base2k, wide.base2k().as_usize(), "shaped rounding across radices");
    let (k_res, k_wide) = (dst.k().as_usize(), wide.k().as_usize());
    let cols = dst.rank().as_usize() + 1;
    let (tmp, _) = scratch.borrow().take_region(shaped_round_tmp_bytes(n));
    let wide = wide.to_backend_ref();
    {
        let mut res = dst.to_backend_mut();
        for col in 0..cols {
            shaped_round_column::<BE::TaskExecutor, _, _>(
                shaping,
                base2k,
                res.data_mut(),
                col,
                k_res,
                wide.data(),
                col,
                k_wide,
                tmp,
            );
        }
    }
    dst.set_canonical(true);
}

/// Metadata written to the result of an operation.
struct Stamp {
    log_budget: usize,
    log_delta: usize,
    log_sparsity: usize,
    slots: poulpy_ckks::SlotsKind,
}

impl Stamp {
    fn apply<D: SetCKKSInfos>(&self, dst: &mut D) {
        dst.set_log_delta(self.log_delta);
        dst.set_log_budget(self.log_budget);
        dst.set_log_sparsity(self.log_sparsity);
        dst.set_slots(self.slots);
    }

    fn k(&self) -> usize {
        self.log_budget + self.log_delta
    }
}

/// Runs `kernel` into a scratch ciphertext `bits` wider than the result described by `stamp`,
/// stamps `dst` and rounds into it with shaping.
fn wide_then_round<BE, D>(
    shaping: &ShapingPolynomial,
    dst: &mut D,
    stamp: Stamp,
    bits: usize,
    scratch: &mut ScratchArena<'_, BE>,
    kernel: impl FnOnce(&mut poulpy_core::layouts::GLWEViewMut<'_, BE>, TorusPrecision, &D, &mut ScratchArena<'_, BE>),
) where
    BE: HostBackend,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let k_wide = stamp.k() + bits;
    let wide_layout = layout_at(dst, k_wide);
    let (mut wide, mut rest) = scratch.borrow().take_glwe_scratch(&wide_layout);
    kernel(&mut wide, TorusPrecision(k_wide as u32), dst, &mut rest);
    stamp.apply(dst);
    shaped_round_glwe(shaping, dst, &wide, &mut rest);
}

/// Shaped leveled operations, implemented for every [`Module`] over a [`HostBackend`].
///
/// The metadata of every result is that of the corresponding poulpy-ckks operation; only the
/// rounding differs. Products return the rescaled result, as in poulpy-ckks: the destination
/// precision `dst.k()` bounds the result precision, and a narrower destination consumes budget
/// by modulus reduction.
pub trait ShapedOps<BE: Backend> {
    /// Scratch bytes enough for every shaped operation on ciphertexts of layout at most `ct`,
    /// with tensor keys of layout `tsk` and automorphism keys of layout `atk`.
    fn shaped_ops_tmp_bytes<C, T, K>(&self, ctx: &ShapedContext, ct: &C, tsk: &T, atk: &K) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `dst = a * b`, relinearized and rescaled.
    fn shaped_mul_into<D, A, B, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>;

    /// `dst = dst * a`, relinearized and rescaled.
    fn shaped_mul_assign<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>;

    /// `dst = a^2`, relinearized and rescaled.
    fn shaped_square_into<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>;

    /// `dst = dst^2`, relinearized and rescaled.
    fn shaped_square_assign<D, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<BE>;

    /// `dst = a * pt` for a plaintext, typically a shaped encoding, rescaled by its scale.
    fn shaped_mul_pt_into<D, A, P>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos + CKKSInfos;

    /// `dst = dst * pt`.
    fn shaped_mul_pt_assign<D, P>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos + CKKSInfos;

    /// `dst = a` with `bits` fewer scale bits: the shaped rescaling of the paper, dividing the
    /// scaling factor by `2^bits` at constant budget.
    fn shaped_rescale_into<D, A>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// `dst = tau_gal(a)`, key-switched back to the secret, for an odd Galois element `gal`.
    ///
    /// Only the admissible automorphisms, rotations by multiples of `R` slots and the
    /// conjugation, keep the data slots apart from the guards.
    fn shaped_automorphism_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        gal: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// `dst = tau_gal(dst)`.
    fn shaped_automorphism_assign<D, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        gal: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `dst = a` with its values rotated by `steps` positions: value `j + steps` moves to `j`.
    fn shaped_rotate_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        steps: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// `dst = a^2`, relinearized but not rescaled: `dst`, at least as wide as `a`, keeps the
    /// precision of `a` and comes out with twice its scale. Only the key-switch bits are removed, with shaping. The extra scale
    /// bits are meant to be dropped later by one [`ShapedOps::shaped_rescale_into`].
    fn shaped_square_unrescaled_into<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>;

    /// Normalized trace onto the subring selected by `skip`, as in [`GLWETrace`]: the slot vector
    /// is averaged over the slots whose index agrees modulo `2^(skip - 1)`. Its halvings and key
    /// switches round `automorphism_bits` below the result, which one shaped rounding removes.
    fn shaped_trace_assign<D, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `dst = conj(a)`.
    fn shaped_conjugate_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;
}

/// Metadata and kernel offset of a ciphertext product.
fn mul_stamp<D: CKKSInfos, A: CKKSInfos, B: CKKSInfos>(op: &'static str, dst: &D, a: &A, b: &B) -> CKKSResult<(Stamp, usize)> {
    let (log_budget, log_delta, cnv_offset) = mul_ct_params(
        op,
        dst.k().as_usize(),
        a.log_delta(),
        a.k().as_usize(),
        b.log_delta(),
        b.k().as_usize(),
    )?;
    let stamp = Stamp {
        log_budget,
        log_delta,
        log_sparsity: a.log_sparsity().min(b.log_sparsity()),
        slots: a.slots().join(b.slots()),
    };
    Ok((stamp, cnv_offset))
}

/// Metadata and kernel offset of a plaintext product.
fn mul_pt_stamp<D: CKKSInfos, A: CKKSInfos, P: CKKSInfos + IntPolyInfos>(
    op: &'static str,
    dst: &D,
    a: &A,
    pt: &P,
) -> CKKSResult<(Stamp, usize)> {
    let (log_budget, log_delta, cnv_offset) = mul_pt_params(
        op,
        dst.k().as_usize(),
        a.log_delta(),
        a.log_budget(),
        pt.log_delta(),
        pt.log_budget(),
        pt.encoded_k().as_usize(),
    )?;
    let stamp = Stamp {
        log_budget,
        log_delta,
        log_sparsity: a.log_sparsity().min(pt.log_sparsity()),
        slots: a.slots().join(pt.slots()),
    };
    Ok((stamp, cnv_offset))
}

impl<BE> ShapedOps<BE> for Module<BE>
where
    BE: HostBackend,
    Module<BE>: ShapedModule<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn shaped_ops_tmp_bytes<C, T, K>(&self, ctx: &ShapedContext, ct: &C, tsk: &T, atk: &K) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        let wide = layout_at(ct, ct.k().as_usize() + ctx.pass_cap());
        let size = ct.size();
        let kernel = self
            .glwe_mul_relinearize_tmp_bytes(&wide, size, size, ct.k(), tsk)
            .max(self.glwe_automorphism_tmp_bytes(&wide, &wide, atk))
            .max(self.glwe_shift_tmp_bytes(wide.size()))
            .max(self.glwe_tensor_apply_tmp_bytes(ct, ct, ct))
            .max(self.glwe_tensor_relinearize_tmp_bytes(&wide, ct, tsk))
            .max(self.glwe_mul_plain_tmp_bytes(&wide, ct, ct))
            .max(self.glwe_trace_assign_tmp_bytes(&wide, atk));
        glwe_bytes(&wide) + kernel.max(shaped_round_glwe_tmp_bytes(ct.n().as_usize()))
    }

    fn shaped_mul_into<D, A, B, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>,
    {
        let (stamp, cnv_offset) = mul_stamp("shaped_mul", dst, a, b)?;
        let exact = a.k().max(b.k()).as_usize().saturating_sub(stamp.k());
        let bits = ctx.product_bits(exact);
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, k, _, rest| {
            relinearized_product(self, cnv_offset, wide, k, a, Some(b), tsk, rest)
        });
        Ok(())
    }

    fn shaped_mul_assign<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>,
    {
        let (stamp, cnv_offset) = mul_stamp("shaped_mul", dst, dst, a)?;
        let exact = dst.k().max(a.k()).as_usize().saturating_sub(stamp.k());
        let bits = ctx.product_bits(exact);
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, k, d, rest| {
            relinearized_product(self, cnv_offset, wide, k, d, Some(a), tsk, rest)
        });
        Ok(())
    }

    fn shaped_square_into<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>,
    {
        let (stamp, cnv_offset) = mul_stamp("shaped_square", dst, a, a)?;
        let exact = a.k().as_usize().saturating_sub(stamp.k());
        let bits = ctx.product_bits(exact);
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, k, _, rest| {
            relinearized_product(self, cnv_offset, wide, k, a, None::<&A>, tsk, rest)
        });
        Ok(())
    }

    fn shaped_square_assign<D, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<BE>,
    {
        let (stamp, cnv_offset) = mul_stamp("shaped_square", dst, dst, dst)?;
        let exact = dst.k().as_usize().saturating_sub(stamp.k());
        let bits = ctx.product_bits(exact);
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, k, d, rest| {
            relinearized_product(self, cnv_offset, wide, k, d, None::<&D>, tsk, rest)
        });
        Ok(())
    }

    fn shaped_mul_pt_into<D, A, P>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos + CKKSInfos,
    {
        let (stamp, cnv_offset) = mul_pt_stamp("shaped_mul_pt", dst, a, pt)?;
        // No key switch: the product is exact and only its scale bits are dropped.
        let bits = a.k().as_usize().saturating_sub(stamp.k()).min(ctx.pass_cap());
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, _, rest| {
            self.glwe_mul_plain(cnv_offset, wide, a, pt, rest)
        });
        Ok(())
    }

    fn shaped_mul_pt_assign<D, P>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos + CKKSInfos,
    {
        let (stamp, cnv_offset) = mul_pt_stamp("shaped_mul_pt", dst, dst, pt)?;
        let bits = dst.k().as_usize().saturating_sub(stamp.k()).min(ctx.pass_cap());
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, d, rest| {
            self.glwe_mul_plain(cnv_offset, wide, d, pt, rest)
        });
        Ok(())
    }

    fn shaped_rescale_into<D, A>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        let log_delta = budget_sub("shaped_rescale", a.log_delta(), bits)?;
        // A narrower destination reduces the modulus first, exactly.
        let k_res = a.k().as_usize() - bits;
        let offset = k_res.saturating_sub(dst.k().as_usize());
        let log_budget = budget_sub("shaped_rescale", a.log_budget(), offset)?;
        if bits > ctx.pass_cap() {
            return Err(poulpy_ckks::CKKSError::Internal(anyhow::anyhow!(
                "a shaped rescale drops at most {} bits, not {bits}",
                ctx.pass_cap()
            )));
        }
        let stamp = Stamp {
            log_budget,
            log_delta,
            log_sparsity: a.log_sparsity(),
            slots: a.slots(),
        };
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, _, rest| {
            self.glwe_lsh(wide, a, offset, rest)
        });
        Ok(())
    }

    fn shaped_automorphism_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        gal: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        let key = keys
            .get_automorphism_key(gal, a.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "shaped_automorphism",
                rotation: gal,
                k: a.k().into(),
            })?;
        let offset = a.k().as_usize().saturating_sub(dst.k().as_usize());
        let stamp = Stamp {
            log_budget: budget_sub("shaped_automorphism", a.log_budget(), offset)?,
            log_delta: a.log_delta(),
            log_sparsity: a.log_sparsity(),
            slots: a.slots(),
        };
        let bits = ctx.automorphism_bits(a.log_delta());
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, _, rest| {
            if offset == 0 {
                self.glwe_automorphism(wide, a, &key, rest);
            } else {
                // Modulus reduction first: the top `offset` bits leave, and the key switch runs on
                // the reduced ciphertext.
                self.glwe_lsh(wide, a, offset, rest);
                self.glwe_automorphism_assign(wide, &key, rest);
            }
        });
        Ok(())
    }

    fn shaped_automorphism_assign<D, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        gal: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        let key = keys
            .get_automorphism_key(gal, dst.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "shaped_automorphism",
                rotation: gal,
                k: dst.k().into(),
            })?;
        let stamp = Stamp {
            log_budget: dst.log_budget(),
            log_delta: dst.log_delta(),
            log_sparsity: dst.log_sparsity(),
            slots: dst.slots(),
        };
        let bits = ctx.automorphism_bits(dst.log_delta());
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, d, rest| {
            self.glwe_automorphism(wide, d, &key, rest)
        });
        Ok(())
    }

    fn shaped_rotate_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        steps: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        use poulpy_ckks::CKKSModuleInfos;
        let gal = self.ckks_galois_element(ctx.layout().rotation_slots(steps));
        self.shaped_automorphism_into(ctx, dst, a, gal, keys, scratch)
    }

    fn shaped_square_unrescaled_into<D, A, T>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GetTensorKey<BE>,
    {
        if dst.k() < a.k() {
            return Err(poulpy_ckks::CKKSError::Internal(anyhow::anyhow!(
                "the destination of an unrescaled square keeps the {} bits of its input, not {}",
                a.k().as_usize(),
                dst.k().as_usize()
            )));
        }
        let (mut stamp, cnv_offset) = mul_stamp("shaped_square_unrescaled", dst, a, a)?;
        // The bits a rescaled product drops stay below the scale.
        stamp.log_delta += a.k().as_usize().saturating_sub(stamp.k());
        let bits = ctx.product_bits(0);
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, k, _, rest| {
            relinearized_product(self, cnv_offset, wide, k, a, None::<&A>, tsk, rest)
        });
        Ok(())
    }

    fn shaped_trace_assign<D, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        let stamp = Stamp {
            log_budget: dst.log_budget(),
            log_delta: dst.log_delta(),
            log_sparsity: dst.log_sparsity(),
            slots: dst.slots(),
        };
        let bits = ctx.automorphism_bits(dst.log_delta());
        wide_then_round(ctx.shaping(), dst, stamp, bits, scratch, |wide, _, src, rest| {
            self.glwe_lsh(wide, src, 0, rest);
            self.glwe_trace_assign(wide, skip, keys, rest);
        });
        Ok(())
    }

    fn shaped_conjugate_into<D, A, H>(
        &self,
        ctx: &ShapedContext,
        dst: &mut D,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        self.shaped_automorphism_into(ctx, dst, a, -1, keys, scratch)
    }
}
