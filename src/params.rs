//! Metadata rules of the products, as in `poulpy_ckks::reference::mul`.
//!
//! These rules are crate-private in poulpy-ckks. They are restated here bit for bit, so that a
//! shaped product carries exactly the metadata of the stock product it replaces.

use poulpy_ckks::{CKKSCompositionError, CKKSError, CKKSResult};

/// `(log_budget, log_delta, cnv_offset)` of a ciphertext product.
pub(crate) fn mul_ct_params(
    op: &'static str,
    res_max_k: usize,
    a_log_delta: usize,
    a_k: usize,
    b_log_delta: usize,
    b_k: usize,
) -> CKKSResult<(usize, usize, usize)> {
    let a_log_budget = a_k.saturating_sub(a_log_delta);
    let b_log_budget = b_k.saturating_sub(b_log_delta);
    let res_log_budget = a_log_budget
        .min(b_log_budget)
        .checked_sub(a_log_delta.max(b_log_delta))
        .ok_or(CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op,
            lhs_log_budget: a_log_budget,
            rhs_log_budget: b_log_budget,
            lhs_log_delta: a_log_delta,
            rhs_log_delta: b_log_delta,
        })?;
    let res_log_delta = a_log_delta.min(b_log_delta);
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res_max_k);
    let cnv_offset = a_k.max(b_k) + res_offset;
    Ok((budget_sub(op, res_log_budget, res_offset)?, res_log_delta, cnv_offset))
}

/// `(log_budget, log_delta, cnv_offset)` of a product by a plaintext of width `b_max_k`.
pub(crate) fn mul_pt_params(
    op: &'static str,
    res_max_k: usize,
    a_log_delta: usize,
    a_log_budget: usize,
    b_log_delta: usize,
    b_log_budget: usize,
    b_max_k: usize,
) -> CKKSResult<(usize, usize, usize)> {
    let res_log_budget = a_log_budget
        .checked_sub(b_log_delta)
        .ok_or(CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op,
            lhs_log_budget: a_log_budget,
            rhs_log_budget: b_log_budget,
            lhs_log_delta: a_log_delta,
            rhs_log_delta: b_log_delta,
        })?;
    let res_offset = (res_log_budget + a_log_delta).saturating_sub(res_max_k);
    let cnv_offset = b_max_k + res_offset;
    Ok((budget_sub(op, res_log_budget, res_offset)?, a_log_delta, cnv_offset))
}

/// `available - required`, or the capacity error of poulpy-ckks.
pub(crate) fn budget_sub(op: &'static str, available: usize, required: usize) -> CKKSResult<usize> {
    available.checked_sub(required).ok_or_else(|| {
        CKKSError::from(CKKSCompositionError::InsufficientHomomorphicCapacity {
            op,
            available_log_budget: available,
            required_bits: required,
        })
    })
}
