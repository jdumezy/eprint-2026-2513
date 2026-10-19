//! Shaped rounding.
//!
//! Removing the low `b` bits of an integer polynomial `c` with shaping returns
//! `(c - g h) / 2^b` with `h = [g^{-1} c]_{2^b}`, an exact division whose error `g h / 2^b` is a
//! multiple of `g`. With `g = 1` it is rounding to nearest, ties toward plus infinity.
//!
//! [`shaped_round_column`] does this on one column of a [`VecZnx`] in base `2^base2k`. Only the
//! dropped bits are read: `c = 2^b c_high + c_low` gives `(c - g h) / 2^b = c_high + (c_low - g h) / 2^b`,
//! where `h` only depends on `c_low` and the carry `(c_low - g h) / 2^b` is a small integer.

use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{HostDataMut, HostDataRef, VecZnx, ZnxView, ZnxViewMut},
};

use crate::shaping::ShapingPolynomial;

/// Largest number of bits [`shaped_round_column`] removes in one call.
///
/// A product that drops more lets the stock kernel remove the bits above by ordinary rounding,
/// which the shaped division then suppresses by `2^-MAX_PASS_BITS` (the shaping depth of the
/// paper). A shaped rescale drops at most the pass cap of its context.
pub const MAX_PASS_BITS: usize = 60;

/// Bytes of the temporary buffer of [`shaped_round_column`] at ring degree `n`.
pub fn shaped_round_tmp_bytes(n: usize) -> usize {
    // tail (i64), low and high halves (u32), accumulators (u64 and u32), h and q (f32).
    n * (8 + 4 + 4 + 8 + 4 + 4 + 4) + 64
}

/// Takes `len` values of `T` from the front of `buf`, aligned for `T`.
fn carve<'a, T: bytemuck::Pod>(buf: &mut &'a mut [u8], len: usize) -> &'a mut [T] {
    let skip = buf.as_ptr().align_offset(align_of::<T>());
    let (_, rest) = std::mem::take(buf).split_at_mut(skip);
    let (head, rest) = rest.split_at_mut(len * size_of::<T>());
    *buf = rest;
    bytemuck::cast_slice_mut(head)
}

/// Runs `task(index, item)` on every item, over `E`, in runs of at most `grain` items.
fn for_each_block<E: TaskExecutor, T: Send>(items: &mut [T], base: usize, grain: usize, task: &(impl Fn(usize, &mut T) + Sync)) {
    if !E::IS_PARALLEL || items.len() <= grain {
        for (offset, item) in items.iter_mut().enumerate() {
            task(base + offset, item);
        }
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    E::join(
        || for_each_block::<E, T>(left, base, grain, task),
        || for_each_block::<E, T>(right, base + mid, grain, task),
    );
}

/// The balanced digit of `x` in base `2^base2k`, in `[-2^(base2k - 1), 2^(base2k - 1))`.
#[inline]
fn digit(x: i64, base2k: usize) -> i64 {
    (x << (64 - base2k)) >> (64 - base2k)
}

/// `2^e` for `e` in the normal range.
#[inline]
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// Source block of the product by the term `X^(t blk)` for output block `j`, and whether the
/// negacyclic wrap negates it.
#[inline]
fn source_block(j: usize, t: usize, count: usize) -> (usize, bool) {
    if j >= t { (j - t, false) } else { (j + count - t, true) }
}

/// Shaped rounding of column `wide_col` of `wide`, canonical at precision `k_wide`, into column
/// `res_col` of `res` at precision `k_res`, with `k_wide - k_res <= MAX_PASS_BITS`.
///
/// `res` is left canonical, with its limbs from `ceil(k_res / base2k)` on set to zero. The blocks
/// of the two products run over the task executor `E`. `tmp` must hold at least
/// [`shaped_round_tmp_bytes`] bytes.
///
/// The products by `g^{-1}` and `g` are blocked: every power of `g` and `g^{-1}` is a multiple of a
/// block length `blk`, so the term `c X^(t blk)` maps block `j - t` of the input to block `j` of the
/// output. The first product, `h = [g^{-1} c_low]_{2^b}`, runs in 32-bit lanes, splitting the
/// words in two halves when `b > 32`. The carry `(c_low - g h) / 2^b` is an integer of size about
/// `|g|_1 / 2`, so it is evaluated from `h / 2^b` in single precision and rounded, which is exact
/// under the bound checked by [`ShapingPolynomial`].
#[allow(clippy::too_many_arguments)]
pub fn shaped_round_column<E: TaskExecutor, DR: HostDataMut, DW: HostDataRef>(
    shaping: &ShapingPolynomial,
    base2k: usize,
    res: &mut VecZnx<DR, i64>,
    res_col: usize,
    k_res: usize,
    wide: &VecZnx<DW, i64>,
    wide_col: usize,
    k_wide: usize,
    mut tmp: &mut [u8],
) {
    let n = wide.n();
    assert_eq!(n, shaping.n(), "ring degree of the shaping polynomial");
    assert_eq!(res.n(), n, "ring degree of the result");
    assert!(0 < k_res && k_res <= k_wide, "cannot round precision {k_wide} to {k_res}");
    let bits = k_wide - k_res;
    assert!(
        bits <= MAX_PASS_BITS,
        "one shaped rounding removes at most {MAX_PASS_BITS} bits, not {bits}"
    );
    let size_w = k_wide.div_ceil(base2k);
    let size_r = k_res.div_ceil(base2k);
    assert!(size_w <= wide.size() && size_r <= res.size(), "too few limbs");
    assert!(tmp.len() >= shaped_round_tmp_bytes(n), "temporary buffer too small");

    if bits == 0 {
        for l in 0..size_r {
            res.at_mut(res_col, l).copy_from_slice(wide.at(wide_col, l));
        }
    } else {
        let pad_w = size_w * base2k - k_wide;
        let pad_r = size_r * base2k - k_res;
        let (g_terms, ginv_terms) = (shaping.terms(), shaping.inverse_terms());
        let blk = shaping.block();
        let count = n / blk;
        let grain = if E::IS_PARALLEL {
            count.div_ceil(E::max_parallelism().max(1))
        } else {
            count
        };
        let scale = pow2(-(bits as i32));

        let tail: &mut [i64] = carve(&mut tmp, n);
        let low: &mut [u32] = carve(&mut tmp, n);
        let high: &mut [u32] = carve(&mut tmp, n);
        let acc64: &mut [u64] = carve(&mut tmp, n);
        let acc32: &mut [u32] = carve(&mut tmp, n);
        let hf: &mut [f32] = carve(&mut tmp, n);
        let qf: &mut [f32] = carve(&mut tmp, n);

        // 1. The dropped part `c_low` as a signed integer below 2^(b + 1) in absolute value: the low
        //    `pad_r` bits of the last kept limb, plus the dropped limbs.
        {
            let kept = wide.at(wide_col, size_r - 1);
            let dropped: Vec<&[i64]> = (size_r..size_w).map(|l| wide.at(wide_col, l)).collect();
            let mut items: Vec<_> = tail
                .chunks_mut(blk)
                .zip(low.chunks_mut(blk))
                .zip(high.chunks_mut(blk))
                .collect();
            for_each_block::<E, _>(&mut items, 0, grain, &|j, ((tail, low), high)| {
                let range = j * blk..(j + 1) * blk;
                let pad_mask = (1i64 << pad_r) - 1;
                if pad_r == 0 {
                    tail.fill(0);
                } else if bits >= pad_r {
                    let sh = bits - pad_r;
                    tail.iter_mut()
                        .zip(&kept[range.clone()])
                        .for_each(|(t, &x)| *t = (x & pad_mask) << sh);
                } else {
                    let sh = pad_r - bits;
                    tail.iter_mut()
                        .zip(&kept[range.clone()])
                        .for_each(|(t, &x)| *t = (x & pad_mask) >> sh);
                }
                for (i, limb) in dropped.iter().enumerate() {
                    let src = &limb[range.clone()];
                    if size_r + i == size_w - 1 {
                        tail.iter_mut().zip(src).for_each(|(t, &x)| *t += x >> pad_w);
                    } else {
                        let sh = (size_w - 1 - size_r - i) * base2k - pad_w;
                        tail.iter_mut().zip(src).for_each(|(t, &x)| *t += x << sh);
                    }
                }
                low.iter_mut().zip(tail.iter()).for_each(|(l, &t)| *l = t as u32);
                if bits > 32 {
                    high.iter_mut().zip(tail.iter()).for_each(|(h, &t)| *h = (t >> 32) as u32);
                }
            });
        }

        // 2. h = [g^{-1} c_low]_{2^b}, stored as h / 2^b in single precision.
        {
            let (low, high) = (&*low, &*high);
            let mut items: Vec<_> = hf
                .chunks_mut(blk)
                .zip(acc32.chunks_mut(blk))
                .zip(acc64.chunks_mut(blk))
                .collect();
            for_each_block::<E, _>(&mut items, 0, grain, &|j, ((hf, acc32), acc64)| {
                acc32.fill(0);
                if bits <= 32 {
                    for &(shift, c) in ginv_terms {
                        let (from, negate) = source_block(j, shift / blk, count);
                        let src = &low[from * blk..(from + 1) * blk];
                        let c = c as u32;
                        if negate {
                            acc32
                                .iter_mut()
                                .zip(src)
                                .for_each(|(o, &v)| *o = o.wrapping_sub(c.wrapping_mul(v)));
                        } else {
                            acc32
                                .iter_mut()
                                .zip(src)
                                .for_each(|(o, &v)| *o = o.wrapping_add(c.wrapping_mul(v)));
                        }
                    }
                    let sh = 32 - bits as u32;
                    let sc = scale as f32;
                    hf.iter_mut()
                        .zip(acc32.iter())
                        .for_each(|(h, &a)| *h = (((a << sh) as i32) >> sh) as f32 * sc);
                } else {
                    // Modulo 2^64, (c1 2^32 + c0)(v1 2^32 + v0) = c0 v0 + 2^32 (c0 v1 + c1 v0): the
                    // first product widens to 64 bits and the cross terms only matter modulo 2^32,
                    // so every multiply is a 32-bit lane multiply.
                    acc64.fill(0);
                    for &(shift, c) in ginv_terms {
                        let (from, negate) = source_block(j, shift / blk, count);
                        let range = from * blk..(from + 1) * blk;
                        let (lo, hi) = (&low[range.clone()], &high[range]);
                        let (c0, c1) = (c as u32, (c as u64 >> 32) as u32);
                        if negate {
                            acc64
                                .iter_mut()
                                .zip(lo)
                                .for_each(|(o, &v)| *o = o.wrapping_sub(c0 as u64 * v as u64));
                            acc32.iter_mut().zip(lo.iter().zip(hi)).for_each(|(o, (&l, &h))| {
                                *o = o.wrapping_sub(c0.wrapping_mul(h).wrapping_add(c1.wrapping_mul(l)))
                            });
                        } else {
                            acc64
                                .iter_mut()
                                .zip(lo)
                                .for_each(|(o, &v)| *o = o.wrapping_add(c0 as u64 * v as u64));
                            acc32.iter_mut().zip(lo.iter().zip(hi)).for_each(|(o, (&l, &h))| {
                                *o = o.wrapping_add(c0.wrapping_mul(h).wrapping_add(c1.wrapping_mul(l)))
                            });
                        }
                    }
                    let sh = 64 - bits;
                    hf.iter_mut().zip(acc64.iter().zip(acc32.iter())).for_each(|(h, (&a, &x))| {
                        let v = a.wrapping_add((x as u64) << 32);
                        *h = ((((v << sh) as i64) >> sh) as f64 * scale) as f32
                    });
                }
            });
        }

        // 3. The carry round(c_low / 2^b - g h / 2^b), then the kept limbs: the floor of the wide
        //    value at k_res plus the carry, renormalized.
        {
            let hf = &*hf;
            let kept: Vec<&[i64]> = (0..size_r).map(|l| wide.at(wide_col, l)).collect();
            let limbs: Vec<&mut [i64]> = (0..size_r)
                .map(|l| {
                    // SAFETY: distinct limbs of one column are disjoint runs of `n` scalars.
                    unsafe { std::slice::from_raw_parts_mut(res.at_mut_ptr(res_col, l), n) }
                })
                .collect();
            let mut limb_blocks: Vec<_> = limbs.into_iter().map(|limb| limb.chunks_mut(blk)).collect();
            let mut items: Vec<_> = tail
                .chunks_mut(blk)
                .zip(qf.chunks_mut(blk))
                .map(|(carry, q)| {
                    let out: Vec<&mut [i64]> = limb_blocks
                        .iter_mut()
                        .map(|l| l.next().expect("one block per limb"))
                        .collect();
                    (carry, q, out)
                })
                .collect();
            for_each_block::<E, _>(&mut items, 0, grain, &|j, (carry, q, out)| {
                let sc = scale as f32;
                q.iter_mut().zip(carry.iter()).for_each(|(q, &t)| *q = t as f32 * sc);
                for &(shift, c) in g_terms {
                    let (from, negate) = source_block(j, shift / blk, count);
                    let src = &hf[from * blk..(from + 1) * blk];
                    let c = c as f32;
                    if negate {
                        q.iter_mut().zip(src).for_each(|(o, &v)| *o += c * v);
                    } else {
                        q.iter_mut().zip(src).for_each(|(o, &v)| *o -= c * v);
                    }
                }
                carry.iter_mut().zip(q.iter()).for_each(|(c, &q)| *c = q.round() as i64);

                let range = j * blk..(j + 1) * blk;
                {
                    let src = &kept[size_r - 1][range.clone()];
                    let dst = &mut *out[size_r - 1];
                    for i in 0..blk {
                        let x = ((src[i] >> pad_r) + carry[i]) << pad_r;
                        let d = digit(x, base2k);
                        carry[i] = (x - d) >> base2k;
                        dst[i] = d;
                    }
                }
                for l in (0..size_r - 1).rev() {
                    let src = &kept[l][range.clone()];
                    let dst = &mut *out[l];
                    for i in 0..blk {
                        let x = src[i] + carry[i];
                        let d = digit(x, base2k);
                        carry[i] = (x - d) >> base2k;
                        dst[i] = d;
                    }
                }
            });
        }
    }
    for l in size_r..res.size() {
        res.at_mut(res_col, l).fill(0);
    }
}

/// Shaped rounding of integer coefficients: `(c - g h) / 2^bits` with `h = [g^{-1} c]_{2^bits}`.
///
/// This is the host-side rounding of plaintext quantization. The coefficients must stay below
/// `2^125` in absolute value and `|g|_1 2^bits` below `2^126`.
pub fn shaped_round_integers(shaping: &ShapingPolynomial, c: &[i128], bits: usize) -> Vec<i128> {
    assert_eq!(c.len(), shaping.n());
    assert!(
        bits < 126 && (shaping.l1() as u128) << bits < 1 << 126,
        "cannot remove {bits} bits in 128-bit words"
    );
    if bits == 0 {
        return c.to_vec();
    }
    let mask = (1u128 << bits) - 1;
    let low: Vec<u128> = c.iter().map(|&x| x as u128 & mask).collect();
    let h: Vec<u128> = shaping
        .mul_ginv_wrapping(&low)
        .into_iter()
        .map(|x| ((((x & mask) << (128 - bits)) as i128) >> (128 - bits)) as u128)
        .collect();
    let gh = shaping.mul_g_wrapping(&h);
    c.iter()
        .zip(&gh)
        .map(|(&x, &p)| {
            let d = x - p as i128;
            debug_assert_eq!(d as u128 & mask, 0, "c - g h is not divisible by 2^bits");
            d >> bits
        })
        .collect()
}
