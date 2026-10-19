//! Shaped encoding, encryption and decoding.
//!
//! A replicated batch is a slot vector of period `P = kR`, which poulpy stores as a compact
//! plaintext of degree `2P`: the polynomial in `X^d` of the paper's subring encoding. Encoding
//! quantizes it `encode_bits` finer with the stock encoder, then removes those bits with a shaped
//! rounding by the shaping polynomial written in the same subring, `U(X^(2P / 2R))`. Constants
//! are encoded the same way as slot vectors of period `R`, in `Z[Y]`.

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, CKKSModuleInfos, CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, CKKSResult,
    SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::{
    GLWEEncryptSk, GLWENormalize,
    layouts::{
        Degree, GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{backend::HostBackend, context::ShapedContext, rounding::shaped_round_integers, shaping::ShapingPolynomial};

/// The operations encoding needs from a module.
pub trait EncodingModule<BE: Backend>:
    CKKSEncodingOps<BE, f64> + CKKSModuleAlloc<BE> + CKKSAddOps<BE> + CKKSDecryptOps<BE> + GLWEEncryptSk<BE> + GLWENormalize<BE>
{
}

impl<BE: Backend, M> EncodingModule<BE> for M where
    M: CKKSEncodingOps<BE, f64>
        + CKKSModuleAlloc<BE>
        + CKKSAddOps<BE>
        + CKKSDecryptOps<BE>
        + GLWEEncryptSk<BE>
        + GLWENormalize<BE>
{
}

/// Headroom bits of the plaintexts allocated by [`ShapedEncoding`]: values below `2^(HEAD - 1)`.
pub const PLAINTEXT_HEAD: usize = 8;

/// Headroom bits of a decrypted plaintext: values below `2^(DECRYPT_HEAD - 1)`, within the budget
/// of the ciphertext.
const DECRYPT_HEAD: usize = 24;

/// Shaped encoding, encryption and decoding, implemented for every [`Module`] over a
/// [`HostBackend`].
pub trait ShapedEncoding<BE: Backend> {
    /// Scratch bytes enough for every encoding operation on ciphertexts of layout at most `ct`.
    fn shaped_encoding_tmp_bytes<C: GLWEInfos>(&self, ctx: &ShapedContext, ct: &C) -> usize;

    /// A plaintext for a batch of the layout of `ctx` at scale `2^log_delta`: a compact plaintext
    /// of `k R` slots with [`PLAINTEXT_HEAD`] bits of headroom, ready for
    /// [`Self::shaped_encode_into`].
    fn shaped_plaintext_alloc(&self, ctx: &ShapedContext, base2k: usize, log_delta: usize) -> CKKSPlaintextOwned<BE>;

    /// A plaintext for a constant at scale `2^log_delta`: a compact plaintext of `R` slots, ready
    /// for [`Self::shaped_encode_constant_into`].
    fn shaped_constant_alloc(&self, ctx: &ShapedContext, base2k: usize, log_delta: usize) -> CKKSPlaintextOwned<BE>;

    /// Decrypts `ct` and decodes its batch into `re` and `im`, each value averaged over its copies.
    #[allow(clippy::too_many_arguments)]
    fn shaped_decrypt<C, S>(
        &self,
        ctx: &ShapedContext,
        ct: &C,
        sk: &S,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    /// Shaped encoding of one period of slot values into `pt`, a plaintext of degree `2P` for the
    /// period `P`, which already carries its scale `log_delta` and its width.
    fn shaped_encode_slots_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: &[f64],
        im: &[f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos;

    /// Shaped encoding of the batch `(re, im)` of the context's layout into `pt`, a compact
    /// plaintext of `k R` slots.
    fn shaped_encode_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: &[f64],
        im: &[f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos;

    /// Shaped encoding of the constant `re + i im` on the data slots, and on the guard slots too
    /// unless `masked`, into `pt`, a compact plaintext of `R` slots. A masked constant clears the
    /// guards of what it multiplies.
    #[allow(clippy::too_many_arguments)]
    fn shaped_encode_constant_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: f64,
        im: f64,
        masked: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos;

    /// Secret-key encryption with error `g e`: an encryption of zero multiplied by `g`, plus `pt`.
    #[allow(clippy::too_many_arguments)]
    fn shaped_encrypt_sk<C, P, S>(
        &self,
        ctx: &ShapedContext,
        ct: &mut C,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSPlaintextToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        S: GLWESecretPreparedToBackendRef<BE>;

    /// The batch carried by a decrypted plaintext, each value averaged over its copies.
    fn shaped_decode<P>(
        &self,
        ctx: &ShapedContext,
        pt: &P,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos;

    /// Every slot of one period of a decrypted plaintext, guards included. Guard values must
    /// never be released: this is for analysis only.
    fn shaped_decode_slots<P>(
        &self,
        ctx: &ShapedContext,
        pt: &P,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos;
}

/// The integer coefficients of a plaintext, from its encoded width: plaintexts are integer
/// polynomials anchored at the bottom of their allocation.
pub(crate) fn plaintext_integers<BE: HostBackend>(pt: &CKKSPlaintextOwned<BE>) -> Vec<i128> {
    let mut c = vec![0i128; pt.n().as_usize()];
    GLWEToBackendRef::<BE>::to_backend_ref(pt).data().decode_vec_i128(
        pt.base2k().as_usize(),
        0,
        pt.encoded_k().as_usize(),
        &mut c,
    );
    c
}

/// Checks that `re` and `im` hold `len` values each.
fn check_len(re: &[f64], im: &[f64], len: usize) -> CKKSResult<()> {
    if re.len() != len || im.len() != len {
        return Err(poulpy_ckks::CKKSError::Internal(anyhow::anyhow!(
            "expected {len} real and imaginary parts, got {} and {}",
            re.len(),
            im.len()
        )));
    }
    Ok(())
}

/// Multiplies column `col` of `data`, canonical at `k`, by the polynomial with terms `terms`,
/// limb by limb and without carries: the digits grow by at most a factor `|g|_1`.
fn mul_terms_limbwise<D: poulpy_hal::layouts::HostDataMut>(
    shaping: &ShapingPolynomial,
    data: &mut poulpy_hal::layouts::VecZnx<D, i64>,
    col: usize,
    limbs: usize,
) {
    let n = data.n();
    let mut tmp = vec![0i64; n];
    for l in 0..limbs {
        tmp.fill(0);
        let src = data.at(col, l);
        for &(shift, c) in shaping.terms() {
            for (i, &x) in src.iter().enumerate() {
                if i + shift < n {
                    tmp[i + shift] += c * x;
                } else {
                    tmp[i + shift - n] -= c * x;
                }
            }
        }
        data.at_mut(col, l).copy_from_slice(&tmp);
    }
}

impl<BE> ShapedEncoding<BE> for Module<BE>
where
    BE: HostBackend,
    Module<BE>: EncodingModule<BE>,
{
    fn shaped_encoding_tmp_bytes<C: GLWEInfos>(&self, _ctx: &ShapedContext, ct: &C) -> usize {
        let decrypt = poulpy_ckks::CKKSLayout {
            glwe_layout: poulpy_core::layouts::GLWELayout {
                n: ct.n(),
                base2k: ct.base2k(),
                k: ct.k(),
                rank: ct.rank(),
            },
            meta: CKKSMeta::default(),
        };
        self.ckks_reim_tmp_bytes(self.ckks_max_slots())
            .max(self.glwe_encrypt_sk_tmp_bytes(ct))
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.ckks_add_pt_vec_tmp_bytes(ct.size()))
            .max(self.ckks_decrypt_tmp_bytes(&decrypt, &decrypt))
    }

    fn shaped_plaintext_alloc(&self, ctx: &ShapedContext, base2k: usize, log_delta: usize) -> CKKSPlaintextOwned<BE> {
        let mut pt = self.ckks_pt_vec_alloc_compact(ctx.layout().period(), base2k.into(), (log_delta + PLAINTEXT_HEAD).into());
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: ctx.layout().log_sparsity(),
            slots: SlotsKind::Complex,
        });
        pt
    }

    fn shaped_constant_alloc(&self, ctx: &ShapedContext, base2k: usize, log_delta: usize) -> CKKSPlaintextOwned<BE> {
        let mut pt =
            self.ckks_pt_vec_alloc_compact(ctx.layout().replication(), base2k.into(), (log_delta + PLAINTEXT_HEAD).into());
        pt.set_meta(CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        });
        pt
    }

    fn shaped_decrypt<C, S>(
        &self,
        ctx: &ShapedContext,
        ct: &C,
        sk: &S,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        let width = ct.log_delta() + ct.log_budget().min(DECRYPT_HEAD);
        let mut pt = self.ckks_plaintext_alloc(Degree(ctx.layout().n() as u32), ct.base2k(), TorusPrecision(width as u32));
        pt.set_meta(ct.meta());
        self.ckks_decrypt(&mut pt, ct, sk, scratch)?;
        self.shaped_decode(ctx, &pt, re, im, scratch)
    }

    fn shaped_encode_slots_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: &[f64],
        im: &[f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos,
    {
        let degree = pt.n().as_usize();
        let period = degree / 2;
        check_len(re, im, period)?;
        let shaping = ShapingPolynomial::new(ctx.unit(), degree).map_err(|e| poulpy_ckks::CKKSError::Internal(e.into()))?;
        let bits = ctx.encode_bits();
        let base2k = pt.base2k();
        let k = pt.encoded_k().as_usize();
        let log_sparsity = (ctx.layout().n() / degree).trailing_zeros() as usize;

        // Quantization `bits` finer, by the stock encoder.
        let mut wide = self.ckks_plaintext_alloc(pt.n(), base2k, (k + bits).into());
        wide.set_meta(CKKSMeta {
            log_delta: pt.log_delta() + bits,
            log_sparsity,
            slots: SlotsKind::Complex,
        });
        self.ckks_encode_reim_into(&mut wide, re, im, scratch)?;
        let c = plaintext_integers::<BE>(&wide);

        // The shaped division by 2^bits.
        let rounded = shaped_round_integers(&shaping, &c, bits);
        GLWEToBackendMut::<BE>::to_backend_mut(pt)
            .data_mut()
            .encode_vec_i128(base2k.as_usize(), 0, k, &rounded);
        pt.set_log_sparsity(log_sparsity);
        pt.set_slots(if im.iter().all(|&x| x == 0.0) {
            SlotsKind::Real
        } else {
            SlotsKind::Complex
        });
        Ok(())
    }

    fn shaped_encode_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: &[f64],
        im: &[f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos,
    {
        let layout = ctx.layout();
        let period = pt.n().as_usize() / 2;
        if !period.is_multiple_of(layout.period()) {
            return Err(poulpy_ckks::CKKSError::Internal(anyhow::anyhow!(
                "a plaintext of {period} slots cannot hold a period of {}",
                layout.period()
            )));
        }
        check_len(re, im, layout.batch())?;
        let tile = |values: &[f64]| layout.spread(values).repeat(period / layout.period());
        self.shaped_encode_slots_into(ctx, pt, &tile(re), &tile(im), scratch)
    }

    fn shaped_encode_constant_into<P>(
        &self,
        ctx: &ShapedContext,
        pt: &mut P,
        re: f64,
        im: f64,
        masked: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + IntPolyInfos + SetCKKSInfos,
    {
        let r = ctx.layout().replication();
        let period = pt.n().as_usize() / 2;
        let slots = |v: f64| -> Vec<f64> {
            (0..period)
                .map(|t| if !masked || t.is_multiple_of(r) { v } else { 0.0 })
                .collect()
        };
        self.shaped_encode_slots_into(ctx, pt, &slots(re), &slots(im), scratch)
    }

    fn shaped_encrypt_sk<C, P, S>(
        &self,
        ctx: &ShapedContext,
        ct: &mut C,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSPlaintextToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let log_budget = crate::params::budget_sub("shaped_encrypt_sk", ct.k().as_usize(), pt.log_delta())?;
        let base2k = ct.base2k().as_usize();
        if (ctx.shaping().l1() as u128) << (base2k - 1) > 1 << 62 {
            return Err(poulpy_ckks::CKKSError::Internal(anyhow::anyhow!(
                "a unit of norm {} is too large for a limbwise product in radix 2^{base2k}",
                ctx.shaping().l1()
            )));
        }
        self.glwe_encrypt_zero_sk(ct, sk, source_xe, source_xa, scratch);
        // (-a s + e, a) g = (-(g a) s + g e, g a): an encryption of zero under error g e.
        {
            let limbs = ct.k().as_usize().div_ceil(base2k);
            let cols = ct.rank().as_usize() + 1;
            let mut ct_mut = ct.to_backend_mut();
            for col in 0..cols {
                mul_terms_limbwise(ctx.shaping(), ct_mut.data_mut(), col, limbs);
            }
        }
        self.glwe_normalize_assign(ct, scratch);
        ct.set_log_budget(log_budget);
        ct.set_log_delta(pt.log_delta());
        ct.set_log_sparsity(pt.log_sparsity());
        ct.set_slots(pt.slots());
        self.ckks_add_pt_vec_assign(ct, pt, scratch)?;
        self.glwe_normalize_assign(ct, scratch);
        ct.set_noise(None);
        Ok(())
    }

    fn shaped_decode<P>(
        &self,
        ctx: &ShapedContext,
        pt: &P,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    {
        let layout = ctx.layout();
        check_len(re, im, layout.batch())?;
        let (mut slots_re, mut slots_im) = (vec![0.0; layout.period()], vec![0.0; layout.period()]);
        self.shaped_decode_slots(ctx, pt, &mut slots_re, &mut slots_im, scratch)?;
        re.copy_from_slice(&layout.collect(&slots_re));
        im.copy_from_slice(&layout.collect(&slots_im));
        Ok(())
    }

    fn shaped_decode_slots<P>(
        &self,
        ctx: &ShapedContext,
        pt: &P,
        re: &mut [f64],
        im: &mut [f64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    {
        check_len(re, im, ctx.layout().period())?;
        // Reading the slots of period kR reads the coefficients at the multiples of d: the
        // normalized trace, which averages the copies of each value.
        self.ckks_decode_reim_into(pt, re, im, scratch)
    }
}
