//! The shaping polynomial `g = U(X^(N / 2R))` of a ring of degree `N`.
//!
//! Both `g` and `g^{-1}` are polynomials in `Y = X^(N / 2R)` with at most `2R` terms, so every
//! product by them is a short sum of scaled negacyclic shifts, all by multiples of `N / 2R`.

use crate::unit::ShapingUnit;

/// Largest block length of the blocked products.
pub(crate) const MAX_BLOCK: usize = 2048;

/// Why a unit cannot shape a ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapingError {
    /// The ring degree is not a power of two of at least `2R`.
    Degree {
        /// The ring degree `N`.
        n: usize,
        /// `2R`, the degree of the unit's ring.
        two_r: usize,
    },
    /// The unit has too many or too large coefficients for the single-precision carry of the
    /// rounding pass: `(terms + 2) (|U|_1 + 2)` must stay below `2^20`.
    UnitTooLarge,
}

impl std::fmt::Display for ShapingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Degree { n, two_r } => write!(f, "ring degree {n} cannot hold a unit of degree {two_r}"),
            Self::UnitTooLarge => write!(f, "the unit is too large for the shaped rounding pass"),
        }
    }
}

impl std::error::Error for ShapingError {}

/// The shaping polynomial `g` of a ring of degree `n`, as its nonzero terms and those of its inverse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapingPolynomial {
    n: usize,
    replication: usize,
    /// Nonzero terms `(power of X, coefficient)` of `g`, by increasing power.
    g: Vec<(usize, i64)>,
    /// Nonzero terms of `g^{-1}`, by increasing power.
    ginv: Vec<(usize, i128)>,
    l1: u64,
    /// Length of the blocks of the blocked products: a power of two dividing every power above.
    block: usize,
}

/// Largest power of two at most [`MAX_BLOCK`] dividing `n` and every power of `terms`.
fn block_of<C>(n: usize, terms: &[(usize, C)]) -> usize {
    let mask = terms.iter().fold(n, |acc, &(s, _)| acc | s);
    (1usize << mask.trailing_zeros()).min(MAX_BLOCK)
}

impl ShapingPolynomial {
    /// `g = U(X^(n / 2R))` in the ring of degree `n`.
    pub fn new(unit: &ShapingUnit, n: usize) -> Result<Self, ShapingError> {
        let two_r = unit.two_r();
        if !n.is_power_of_two() || n < two_r {
            return Err(ShapingError::Degree { n, two_r });
        }
        let step = n / two_r;
        let g: Vec<(usize, i64)> = unit
            .coeffs()
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, &c)| (i * step, c))
            .collect();
        let ginv: Vec<(usize, i128)> = unit
            .inverse()
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, &c)| (i * step, c))
            .collect();
        Self::from_terms(n, unit.replication(), g, ginv)
    }

    /// `g = 1`: shaped rounding is then ordinary rounding, with ties toward plus infinity.
    pub fn identity(n: usize) -> Self {
        Self::from_terms(n, 1, vec![(0, 1)], vec![(0, 1)]).expect("one shapes every ring")
    }

    fn from_terms(n: usize, replication: usize, g: Vec<(usize, i64)>, ginv: Vec<(usize, i128)>) -> Result<Self, ShapingError> {
        let l1 = g.iter().map(|(_, c)| c.unsigned_abs()).sum::<u64>();
        if (g.len() as u128 + 2) * (l1 as u128 + 2) >= 1 << 20 {
            return Err(ShapingError::UnitTooLarge);
        }
        let block = block_of(n, &g).min(block_of(n, &ginv));
        Ok(Self {
            n,
            replication,
            g,
            ginv,
            l1,
            block,
        })
    }

    /// The shaping polynomial `g(X^gal)`, for an odd `gal`.
    ///
    /// It is the polynomial whose magnitudes follow the data slots through the automorphism
    /// `X -> X^gal`, as needed to quantize a diagonal that is rotated after its product. A rotation
    /// by a multiple of `R` slots fixes `g`.
    pub fn automorphism(&self, gal: usize) -> Self {
        assert!(gal % 2 == 1, "the Galois element {gal} is even");
        fn map<C: Copy + std::ops::Neg<Output = C>>(terms: &[(usize, C)], gal: usize, n: usize) -> Vec<(usize, C)> {
            let mut out: Vec<(usize, C)> = terms
                .iter()
                .map(|&(i, c)| {
                    let e = (i * gal) % (2 * n);
                    if e >= n { (e - n, -c) } else { (e, c) }
                })
                .collect();
            out.sort_by_key(|t| t.0);
            out
        }
        Self::from_terms(
            self.n,
            self.replication,
            map(&self.g, gal, self.n),
            map(&self.ginv, gal, self.n),
        )
        .expect("an automorphism keeps the number and size of the coefficients")
    }

    /// The ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The replication factor `R` of the unit: the data slots are the slots `t = 0 mod R`. It is 1
    /// for the identity, which shapes nothing.
    pub fn replication(&self) -> usize {
        self.replication
    }

    /// Whether `g = 1`.
    pub fn is_identity(&self) -> bool {
        self.g == [(0, 1)]
    }

    /// `|g|_1`.
    pub fn l1(&self) -> u64 {
        self.l1
    }

    /// The nonzero terms `(power of X, coefficient)` of `g`.
    pub fn terms(&self) -> &[(usize, i64)] {
        &self.g
    }

    /// The nonzero terms `(power of X, coefficient)` of `g^{-1}`.
    pub fn inverse_terms(&self) -> &[(usize, i128)] {
        &self.ginv
    }

    /// Length of the blocks of the blocked products: every power of `g` and `g^{-1}` is a
    /// multiple of it.
    pub(crate) fn block(&self) -> usize {
        self.block
    }

    /// `g * x` in `Z[X] / (X^N + 1)` modulo `2^128`.
    pub fn mul_g_wrapping(&self, x: &[u128]) -> Vec<u128> {
        sparse_mul_wrapping(self.g.iter().map(|&(s, c)| (s, c as i128 as u128)), x)
    }

    /// `g^{-1} * x` in `Z[X] / (X^N + 1)` modulo `2^128`.
    pub fn mul_ginv_wrapping(&self, x: &[u128]) -> Vec<u128> {
        sparse_mul_wrapping(self.ginv.iter().map(|&(s, c)| (s, c as u128)), x)
    }
}

/// Negacyclic product of `x` by the polynomial with nonzero terms `terms`, modulo `2^128`.
fn sparse_mul_wrapping(terms: impl Iterator<Item = (usize, u128)>, x: &[u128]) -> Vec<u128> {
    let n = x.len();
    let mut out = vec![0u128; n];
    for (shift, c) in terms {
        let (head, tail) = out.split_at_mut(shift);
        for (o, &v) in tail.iter_mut().zip(&x[..n - shift]) {
            *o = o.wrapping_add(c.wrapping_mul(v));
        }
        for (o, &v) in head.iter_mut().zip(&x[n - shift..]) {
            *o = o.wrapping_sub(c.wrapping_mul(v));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unit::catalog;

    fn galois(rotation: usize, n: usize) -> usize {
        (0..rotation).fold(1, |a, _| a * 5 % (2 * n))
    }

    #[test]
    fn inverse_and_support() {
        let n = 4096;
        let shaping = ShapingPolynomial::new(&ShapingUnit::default_unit(), n).unwrap();
        let mut one = vec![0u128; n];
        one[0] = 1;
        assert_eq!(shaping.mul_g_wrapping(&shaping.mul_ginv_wrapping(&one)), one);
        assert!(shaping.terms().len() <= 32 && shaping.inverse_terms().len() <= 32);
        assert!(shaping.terms().iter().all(|&(s, _)| s % (n / 32) == 0));
        assert_eq!(shaping.block(), n / 32);
        assert_eq!(shaping.l1(), ShapingUnit::default_unit().l1());
    }

    #[test]
    fn rotations_by_multiples_of_r_fix_g() {
        let n = 1024;
        let shaping = ShapingPolynomial::new(&ShapingUnit::default_unit(), n).unwrap();
        assert_eq!(shaping.automorphism(galois(16, n)), shaping);
        assert_eq!(shaping.automorphism(galois(48, n)), shaping);
        let moved = shaping.automorphism(galois(1, n));
        assert_ne!(moved, shaping);
        assert_eq!(moved.block(), shaping.block());
        // g(X^gal) g^{-1}(X^gal) = 1.
        let mut one = vec![0u128; n];
        one[0] = 1;
        assert_eq!(moved.mul_g_wrapping(&moved.mul_ginv_wrapping(&one)), one);
    }

    #[test]
    fn identity_and_degree_checks() {
        assert!(ShapingPolynomial::identity(64).is_identity());
        assert_eq!(ShapingPolynomial::identity(1 << 15).block(), MAX_BLOCK);
        let unit = ShapingUnit::new(catalog::U_64_40.to_vec()).unwrap();
        assert_eq!(
            ShapingPolynomial::new(&unit, 32),
            Err(ShapingError::Degree { n: 32, two_r: 64 })
        );
    }
}
