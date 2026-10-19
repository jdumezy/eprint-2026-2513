//! Moduli for 128-bit security.
//!
//! The caps are those of the sparse-key-estimate tables
//! (<https://github.com/jdumezy/sparse-key-estimate>, `Precomputed-Tables/128bits_security.md`):
//! the largest `log2(Q P)` of a key under a ternary secret of Hamming weight `h` in a ring of
//! degree `N`. Every key is checked at its full modulus, auxiliary modulus `P` included.

use poulpy_core::layouts::GGLWEInfos;

/// The Hamming weights of the table.
pub const HAMMING_WEIGHTS: [usize; 8] = [32, 64, 128, 192, 256, 512, 1024, 2048];

/// `log2(Q P)` caps by `log2 N` (rows) and Hamming weight ([`HAMMING_WEIGHTS`], columns).
const CAPS: [(usize, [usize; 8]); 5] = [
    (12, [19, 57, 85, 96, 100, 104, 105, 106]),
    (13, [37, 112, 167, 186, 196, 208, 211, 213]),
    (14, [77, 227, 332, 367, 387, 415, 424, 428]),
    (15, [162, 464, 665, 737, 771, 826, 851, 861]),
    (16, [345, 954, 1344, 1475, 1545, 1653, 1707, 1730]),
];

/// The largest `log2(Q P)` with 128-bit security in a ring of degree `2^log_n` under a secret of
/// Hamming weight `hamming_weight`, or `None` outside the table. A weight between two columns
/// takes the cap of the smaller one, since security grows with the weight.
pub fn max_modulus_bits(log_n: usize, hamming_weight: usize) -> Option<usize> {
    let (_, row) = CAPS.iter().find(|(l, _)| *l == log_n)?;
    let col = HAMMING_WEIGHTS.iter().rposition(|&h| h <= hamming_weight)?;
    Some(row[col])
}

/// One key modulus against its cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModulusCheck {
    /// What the key is for.
    pub key: &'static str,
    /// `log2(Q P)` of the key: its gadget precision plus its auxiliary modulus.
    pub log_qp: usize,
    /// Hamming weight of the secret the key is encrypted under.
    pub hamming_weight: usize,
    /// The cap of [`max_modulus_bits`], `None` outside the table.
    pub cap: Option<usize>,
}

impl ModulusCheck {
    /// Checks the key of layout `layout` under a secret of Hamming weight `hamming_weight`.
    pub fn new<K: GGLWEInfos>(key: &'static str, layout: &K, hamming_weight: usize) -> Self {
        Self {
            key,
            log_qp: layout.k().as_usize(),
            hamming_weight,
            cap: max_modulus_bits(layout.n().log2(), hamming_weight),
        }
    }

    /// Whether the key is within its cap.
    pub fn is_secure(&self) -> bool {
        self.cap.is_some_and(|cap| self.log_qp <= cap)
    }
}

impl std::fmt::Display for ModulusCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cap {
            Some(cap) => write!(
                f,
                "{}: log2(QP) = {} under weight {}, cap {} ({})",
                self.key,
                self.log_qp,
                self.hamming_weight,
                cap,
                if self.log_qp <= cap { "ok" } else { "over" }
            ),
            None => write!(
                f,
                "{}: log2(QP) = {} under weight {}, outside the table",
                self.key, self.log_qp, self.hamming_weight
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_follow_the_table() {
        assert_eq!(max_modulus_bits(15, 1024), Some(851));
        assert_eq!(max_modulus_bits(15, 32), Some(162));
        assert_eq!(max_modulus_bits(16, 1024), Some(1707));
        assert_eq!(max_modulus_bits(12, 64), Some(57));
        // Between columns: the smaller weight.
        assert_eq!(max_modulus_bits(15, 1000), Some(826));
        // Dense secrets beyond the table: the last column.
        assert_eq!(max_modulus_bits(16, 1 << 15), Some(1730));
        assert_eq!(max_modulus_bits(15, 16), None);
        assert_eq!(max_modulus_bits(17, 1024), None);
    }
}
