//! Units of `Z[Y] / (Y^(2R) + 1)` that define the shaping profile.
//!
//! The shaping polynomial is `g = U(Y)` with `Y = X^(N / 2R)` and `U` a unit of the cyclotomic
//! ring of degree `2R`. Its magnitude on slot `t` depends on `t mod R` only: class `0` holds the
//! data slots and classes `1..R` the guard slots. A unit has algebraic norm one, so the data gain
//! is the product of the guard magnitudes, inverted.

use std::f64::consts::PI;

/// Shaping units `U_{2R,T}` with `2R` coefficients and about `T` bits of suppression.
///
/// The units of the paper come from a lattice search (Babai rounding and a local search); `U_{32,41}`,
/// `U_{32,45}` and `U_{32,50}` are the optimal units of `scripts/unit_search.py` for the targets 40,
/// 45 and 50 bits: the smallest largest guard magnitude at that suppression.
pub mod catalog {
    /// `U_{16,20}`: 19.3 bits of suppression, largest guard magnitude `2^3.40`.
    pub const U_16_20: &[i64] = &[-2, 1, 2, 0, 2, -3, -1, -3, 2, 0, 2, 1, -2, 2, 0, -2];
    /// `U_{16,40}`: 39.3 bits of suppression, largest guard magnitude `2^7.04`.
    pub const U_16_40: &[i64] = &[-25, 8, -27, 8, 0, -8, 27, -8, 25, -6, -20, 2, -1, 2, -20, -6];
    /// `U_{16,60}`: 62.0 bits of suppression, largest guard magnitude `2^9.90`.
    pub const U_16_60: &[i64] = &[-30, -101, 16, 284, -122, 0, 122, -284, -16, 101, 30, -140, 91, 75, 91, -140];
    /// `U_{32,20}`: 21.8 bits of suppression, largest guard magnitude `2^4.26`.
    pub const U_32_20: &[i64] = &[
        0, 3, 3, 0, -3, -2, 0, 0, 0, 0, 1, -1, 1, -1, 1, 0, 0, 0, 0, -2, -3, 0, 3, 3, 0, -1, 1, 2, 0, -2, -1, 1,
    ];
    /// `U_{32,40}`: 39.7 bits of suppression, largest guard magnitude `2^4.70`. The unit of the paper.
    pub const U_32_40: &[i64] = &[
        -3, 0, 5, 0, -3, 1, 1, -2, 0, 0, -1, 4, -2, 0, 0, 1, -2, -3, 0, 3, 2, -1, 0, 0, 2, -4, 1, 0, 0, 2, -1, -1,
    ];
    /// `U_{32,41}`: 41.0 bits of suppression, largest guard magnitude `2^4.25`. The smallest
    /// guard magnitude of any unit with 40 bits of suppression.
    pub const U_32_41: &[i64] = &[
        0, 3, -1, 2, 0, -2, -3, -4, 3, 0, 0, 2, -1, -1, 3, 0, -3, 1, 1, -2, 0, 0, -3, 4, 3, 2, 0, -2, 1, -3, 0, -1,
    ];
    /// `U_{32,45}`: 46.0 bits of suppression, largest guard magnitude `2^4.59`. The default unit:
    /// better than `U_{32,40}` on both the data and the guards.
    pub const U_32_45: &[i64] = &[
        1, 1, 1, -5, -1, -4, 0, 2, -1, -3, 5, 1, 3, 1, 5, -3, -1, 2, 0, -4, -1, -5, 1, 1, 1, 1, -1, 1, 0, -1, 1, -1,
    ];
    /// `U_{32,50}`: 50.2 bits of suppression, largest guard magnitude `2^4.74`.
    pub const U_32_50: &[i64] = &[
        -1, 2, 3, -2, -1, 0, 1, -2, 1, -2, 1, 0, -1, -2, 3, 2, -1, 0, 6, 0, -5, -2, -5, 0, 0, 0, 5, 2, 5, 0, -6, 0,
    ];
    /// `U_{32,60}`: 67.1 bits of suppression, largest guard magnitude `2^5.69`.
    pub const U_32_60: &[i64] = &[
        -9, 0, 4, 1, 0, -2, -3, 7, 1, -3, -6, 15, -1, -7, -7, -3, 0, 3, 7, 7, 1, -15, 6, 3, -1, -7, 3, 2, 0, -1, -4, 0,
    ];
    /// `U_{64,20}`: 24.0 bits of suppression, largest guard magnitude `2^3.85`.
    pub const U_64_20: &[i64] = &[
        -1, 0, 0, 0, 0, 1, 0, 0, -1, 1, 0, -1, 0, 0, 1, -1, 1, -1, 1, -1, 1, 0, 0, -1, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0, -1, -1, 1,
        1, 0, -1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, -1, -1, 1,
    ];
    /// `U_{64,40}`: 40.7 bits of suppression, largest guard magnitude `2^4.09`.
    pub const U_64_40: &[i64] = &[
        -1, 0, -1, 0, -1, 2, 0, -1, 0, -1, 1, 0, 2, 1, -1, 0, -1, 0, 0, 1, -1, 0, -1, 0, 0, -1, 1, 0, -1, 1, 0, 0, 1, 0, 0, 0,
        -1, 0, 0, -1, 1, 0, -1, 1, 0, 0, 1, 0, 1, -1, 0, 0, 1, 0, 1, -1, -2, 0, -1, 1, 0, 1, 0, -2,
    ];
    /// `U_{64,60}`: 60.5 bits of suppression, largest guard magnitude `2^5.05`.
    pub const U_64_60: &[i64] = &[
        -1, 0, 0, -1, 2, 0, -2, 1, 0, 0, 1, 0, -1, 1, -1, 0, 2, -2, 0, 1, 0, 1, -1, 0, -3, 3, 1, -1, 0, 0, -2, 1, 1, -1, 0, 1,
        -2, 3, -2, 1, 0, -1, 1, 1, -2, 0, 0, -1, 1, 3, -3, 0, -1, 1, 0, 1, 0, -2, 2, 0, -1, 1, -1, 0,
    ];
}

/// Why a coefficient vector does not define a shaping unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitError {
    /// The number of coefficients is not a power of two of at least two.
    Length(usize),
    /// The coefficient sum is even, so the polynomial is not invertible modulo two.
    EvenSum,
    /// The polynomial is invertible modulo `2^128` but has no inverse with integer coefficients
    /// below `2^126`: it is not a unit, or not a usable one.
    NotAUnit,
}

impl std::fmt::Display for UnitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(len) => write!(f, "a unit needs a power-of-two number of coefficients, not {len}"),
            Self::EvenSum => write!(f, "the coefficient sum of a unit must be odd"),
            Self::NotAUnit => write!(f, "the polynomial has no integer inverse"),
        }
    }
}

impl std::error::Error for UnitError {}

/// A unit `U` of `Z[Y] / (Y^(2R) + 1)` together with its exact inverse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapingUnit {
    coeffs: Vec<i64>,
    inverse: Vec<i128>,
}

/// Product in `Z[Y] / (Y^m + 1)` modulo `2^128`.
fn ring_mul_wrapping(a: &[u128], b: &[u128]) -> Vec<u128> {
    let m = a.len();
    let mut out = vec![0u128; m];
    for (i, &x) in a.iter().enumerate().filter(|(_, x)| **x != 0) {
        for (j, &y) in b.iter().enumerate() {
            let p = x.wrapping_mul(y);
            if i + j < m {
                out[i + j] = out[i + j].wrapping_add(p);
            } else {
                out[i + j - m] = out[i + j - m].wrapping_sub(p);
            }
        }
    }
    out
}

/// Product in `Z[Y] / (Y^m + 1)` over the integers, or `None` on overflow.
fn ring_mul_checked(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
    let m = a.len();
    let mut out = vec![0i128; m];
    for (i, &x) in a.iter().enumerate().filter(|(_, x)| **x != 0) {
        for (j, &y) in b.iter().enumerate() {
            let p = x.checked_mul(y)?;
            let o = &mut out[(i + j) % m];
            *o = if i + j < m { o.checked_add(p)? } else { o.checked_sub(p)? };
        }
    }
    Some(out)
}

impl ShapingUnit {
    /// The unit with coefficients `coeffs`, in increasing powers of `Y`.
    ///
    /// The inverse is lifted from modulo two to modulo `2^128` by Newton iteration, then checked
    /// to be an exact inverse over the integers.
    pub fn new(coeffs: Vec<i64>) -> Result<Self, UnitError> {
        let m = coeffs.len();
        if m < 2 || !m.is_power_of_two() {
            return Err(UnitError::Length(m));
        }
        if coeffs.iter().sum::<i64>().rem_euclid(2) != 1 {
            return Err(UnitError::EvenSum);
        }
        let u: Vec<u128> = coeffs.iter().map(|&x| x as i128 as u128).collect();

        // Modulo two, Y^m + 1 = (Y + 1)^m and the unit group has exponent dividing m, so
        // U^{-1} = U^(m - 1).
        let reduce2 = |v: Vec<u128>| v.into_iter().map(|x| x & 1).collect::<Vec<_>>();
        let mut inv = vec![0u128; m];
        inv[0] = 1;
        let mut pow = reduce2(u.clone());
        let mut e = m - 1;
        while e != 0 {
            if e & 1 == 1 {
                inv = reduce2(ring_mul_wrapping(&inv, &pow));
            }
            pow = reduce2(ring_mul_wrapping(&pow, &pow));
            e >>= 1;
        }
        // Newton: inv <- inv * (2 - U * inv) doubles the number of correct bits.
        for _ in 0..7 {
            let mut t = ring_mul_wrapping(&u, &inv);
            t.iter_mut().for_each(|x| *x = x.wrapping_neg());
            t[0] = t[0].wrapping_add(2);
            inv = ring_mul_wrapping(&inv, &t);
        }

        let inverse: Vec<i128> = inv.iter().map(|&x| x as i128).collect();
        if inverse.iter().any(|x| x.unsigned_abs() >= 1 << 126) {
            return Err(UnitError::NotAUnit);
        }
        let wide: Vec<i128> = coeffs.iter().map(|&x| x as i128).collect();
        let mut one = vec![0i128; m];
        one[0] = 1;
        if ring_mul_checked(&wide, &inverse).as_deref() != Some(&one[..]) {
            return Err(UnitError::NotAUnit);
        }
        Ok(Self { coeffs, inverse })
    }

    /// The default unit `U_{32,45}`, for `R = 16`. The paper uses `U_{32,40}`.
    pub fn default_unit() -> Self {
        Self::new(catalog::U_32_45.to_vec()).expect("catalog units are units")
    }

    /// `U = 1` in the ring of degree `two_r`: no shaping.
    pub fn identity(two_r: usize) -> Self {
        let mut coeffs = vec![0; two_r];
        coeffs[0] = 1;
        Self::new(coeffs).expect("one is a unit")
    }

    /// The coefficients of `U`.
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }

    /// The coefficients of `U^{-1}`.
    pub fn inverse(&self) -> &[i128] {
        &self.inverse
    }

    /// The degree `2R` of the unit ring.
    pub fn two_r(&self) -> usize {
        self.coeffs.len()
    }

    /// The replication factor `R`: one slot in `R` carries data.
    pub fn replication(&self) -> usize {
        self.coeffs.len() / 2
    }

    /// `|U|_1`, the sum of the absolute values of the coefficients.
    pub fn l1(&self) -> u64 {
        self.coeffs.iter().map(|x| x.unsigned_abs()).sum()
    }

    /// Whether `U = 1`.
    pub fn is_identity(&self) -> bool {
        self.coeffs[0] == 1 && self.coeffs[1..].iter().all(|&x| x == 0)
    }

    /// `log2 |U(omega^(5^j))|` with `omega = exp(i pi / 2R)`: the magnitude of `g` on the slot
    /// class `j`, for `0 < j < R`. These magnitudes have no cancellation, so double precision
    /// evaluates them accurately.
    pub fn guard_class_log2(&self, j: usize) -> f64 {
        assert!(j > 0 && j < self.replication(), "class {j} is not a guard class");
        let four_r = 2 * self.two_r();
        let five_j = (0..j).fold(1usize, |acc, _| acc * 5 % four_r);
        let (mut re, mut im) = (0f64, 0f64);
        for (i, &c) in self.coeffs.iter().enumerate() {
            let angle = PI * ((i * five_j) % four_r) as f64 / self.two_r() as f64;
            re += c as f64 * angle.cos();
            im += c as f64 * angle.sin();
        }
        re.hypot(im).log2()
    }

    /// `log2 m_S = log2 |U(omega)|`, the magnitude of `g` on the data slots.
    ///
    /// Evaluating `U(omega)` directly loses most of its digits to cancellation, so it is read from
    /// the norm identity of a unit instead: the magnitudes over one root of each conjugate pair
    /// multiply to one.
    pub fn data_log2(&self) -> f64 {
        -(1..self.replication()).map(|j| self.guard_class_log2(j)).sum::<f64>()
    }

    /// `log2 M`, the largest magnitude of `g` on the guard slots.
    pub fn guard_max_log2(&self) -> f64 {
        (1..self.replication())
            .map(|j| self.guard_class_log2(j))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_profiles_match_the_paper() {
        // (unit, -log2 m_S, log2 M), from the stability table of the paper and the unit search.
        for (coeffs, gain, guard) in [
            (catalog::U_16_20, 19.3, 3.40),
            (catalog::U_16_40, 39.3, 7.04),
            (catalog::U_16_60, 62.0, 9.90),
            (catalog::U_32_20, 21.8, 4.26),
            (catalog::U_32_40, 39.7, 4.70),
            (catalog::U_32_41, 40.99, 4.25),
            (catalog::U_32_45, 45.95, 4.59),
            (catalog::U_32_50, 50.21, 4.74),
            (catalog::U_32_60, 67.1, 5.69),
            (catalog::U_64_20, 24.0, 3.85),
            (catalog::U_64_40, 40.7, 4.09),
            (catalog::U_64_60, 60.5, 5.05),
        ] {
            let unit = ShapingUnit::new(coeffs.to_vec()).unwrap();
            assert!(
                (-unit.data_log2() - gain).abs() < 0.05,
                "gain {} for {coeffs:?}",
                -unit.data_log2()
            );
            assert!(
                (unit.guard_max_log2() - guard).abs() < 0.01,
                "guard {}",
                unit.guard_max_log2()
            );
        }
        assert_eq!(ShapingUnit::default_unit().l1(), 59);
    }

    #[test]
    fn rejects_non_units() {
        assert_eq!(ShapingUnit::new(vec![1, 1, 1]), Err(UnitError::Length(3)));
        assert_eq!(ShapingUnit::new(vec![1, 1, 0, 0]), Err(UnitError::EvenSum));
        // 3 is invertible modulo two but has norm 81.
        assert_eq!(ShapingUnit::new(vec![3, 0, 0, 0]), Err(UnitError::NotAUnit));
        // 1 + 2Y has norm 2^4 ((-1/2)^4 + 1) = 17 in Z[Y] / (Y^4 + 1).
        assert_eq!(ShapingUnit::new(vec![1, 2, 0, 0]), Err(UnitError::NotAUnit));
        // 1 + Y + Y^2 is a unit there, with inverse 1 - Y^2 + Y^3.
        assert_eq!(ShapingUnit::new(vec![1, 1, 1, 0]).unwrap().inverse(), &[1, 0, -1, 1]);
    }

    #[test]
    fn identity_unit() {
        let one = ShapingUnit::identity(32);
        assert!(one.is_identity());
        assert_eq!(one.inverse()[0], 1);
        assert_eq!(one.data_log2(), 0.0);
    }
}
