//! The public parameters of a computation: layout, shaping unit and rounding depths.

use crate::{
    layout::{LayoutError, ReplicatedLayout},
    rounding::MAX_PASS_BITS,
    shaping::{ShapingError, ShapingPolynomial},
    unit::ShapingUnit,
};

/// Low bits kept below the result of a key switch, by default: the shaped rounding then reaches
/// `log_delta + 24` bits below the scale (the shaping depth of the paper).
pub const DEFAULT_GUARD_BITS: usize = 24;
/// Largest drop of one shaped rounding, by default. Bits above it are removed by the ordinary
/// rounding of the kernel, which the shaped division suppresses by `2^-48`.
pub const DEFAULT_PASS_CAP: usize = 48;
/// Extra scale bits of the quantization that precedes the shaped rounding of an encoding.
pub const DEFAULT_ENCODE_BITS: usize = 40;

/// Why a context cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextError {
    /// The batch does not fit the ring.
    Layout(LayoutError),
    /// The unit cannot shape the ring.
    Shaping(ShapingError),
    /// The pass cap exceeds what one shaped rounding removes.
    PassCap(usize),
    /// The encoding cannot remove this many bits in 128-bit words.
    EncodeBits(usize),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout(e) => e.fmt(f),
            Self::Shaping(e) => e.fmt(f),
            Self::PassCap(cap) => write!(f, "pass cap {cap} exceeds {MAX_PASS_BITS} bits"),
            Self::EncodeBits(bits) => write!(f, "an encoding cannot remove {bits} bits with this unit"),
        }
    }
}

impl std::error::Error for ContextError {}

impl From<LayoutError> for ContextError {
    fn from(e: LayoutError) -> Self {
        Self::Layout(e)
    }
}

impl From<ShapingError> for ContextError {
    fn from(e: ShapingError) -> Self {
        Self::Shaping(e)
    }
}

/// Everything the shaped operations need besides keys: the layout, the shaping polynomial and
/// how deep each rounding is shaped.
#[derive(Clone, Debug)]
pub struct ShapedContext {
    layout: ReplicatedLayout,
    unit: ShapingUnit,
    shaping: ShapingPolynomial,
    guard_bits: usize,
    pass_cap: usize,
    encode_bits: usize,
    deep_automorphisms: bool,
}

impl ShapedContext {
    /// `batch` values in the ring of degree `n`, shaped by `unit`, whose degree `2R` sets the
    /// replication factor.
    pub fn new(n: usize, batch: usize, unit: ShapingUnit) -> Result<Self, ContextError> {
        let layout = ReplicatedLayout::new(n, batch, unit.replication())?;
        let shaping = ShapingPolynomial::new(&unit, n)?;
        Ok(Self {
            layout,
            unit,
            shaping,
            guard_bits: DEFAULT_GUARD_BITS,
            pass_cap: DEFAULT_PASS_CAP,
            encode_bits: DEFAULT_ENCODE_BITS,
            deep_automorphisms: true,
        })
    }

    /// The same computation without shaping, `g = 1`: the operations are then those of CKKS on
    /// the replicated layout, with the same rounding positions.
    pub fn unshaped(&self) -> Self {
        Self {
            unit: ShapingUnit::identity(self.unit.two_r()),
            shaping: ShapingPolynomial::identity(self.layout.n()),
            ..self.clone()
        }
    }

    /// Sets the number of key-switch bits kept below a result before its shaped rounding.
    pub fn with_guard_bits(mut self, guard_bits: usize) -> Self {
        self.guard_bits = guard_bits;
        self
    }

    /// Sets the largest drop of one shaped rounding.
    pub fn with_pass_cap(mut self, pass_cap: usize) -> Result<Self, ContextError> {
        if pass_cap == 0 || pass_cap > MAX_PASS_BITS {
            return Err(ContextError::PassCap(pass_cap));
        }
        self.pass_cap = pass_cap;
        Ok(self)
    }

    /// Whether automorphisms and traces round as deep below the scale as a product
    /// (`log_delta + guard_bits`, the default), or only `guard_bits` below their result. Shallow
    /// rounding is cheaper and enough where the scale is far above the data error, as inside a
    /// bootstrapping.
    pub fn with_deep_automorphisms(mut self, deep: bool) -> Self {
        self.deep_automorphisms = deep;
        self
    }

    /// Sets the extra quantization bits of an encoding.
    pub fn with_encode_bits(mut self, encode_bits: usize) -> Result<Self, ContextError> {
        // The shaped division of an encoding runs in 128-bit words: `|U|_1 2^bits < 2^126`.
        if encode_bits >= 126 || (self.unit.l1() as u128) << encode_bits >= 1 << 126 {
            return Err(ContextError::EncodeBits(encode_bits));
        }
        self.encode_bits = encode_bits;
        Ok(self)
    }

    /// The replicated layout of the values.
    pub fn layout(&self) -> &ReplicatedLayout {
        &self.layout
    }

    /// The shaping unit `U`.
    pub fn unit(&self) -> &ShapingUnit {
        &self.unit
    }

    /// The shaping polynomial `g` of the full ring.
    pub fn shaping(&self) -> &ShapingPolynomial {
        &self.shaping
    }

    /// Bits a wide stage adds below the result: its ordinary roundings sit this far down.
    pub fn guard_bits(&self) -> usize {
        self.guard_bits
    }

    /// Largest number of bits one shaped rounding removes.
    pub fn pass_cap(&self) -> usize {
        self.pass_cap
    }

    /// Extra quantization bits of an encoding, removed by its shaped rounding.
    pub fn encode_bits(&self) -> usize {
        self.encode_bits
    }

    /// Bits the shaped rounding of a product removes: the `exact` bits the product drops plus the
    /// guard bits of its key switch, within the pass cap.
    pub(crate) fn product_bits(&self, exact: usize) -> usize {
        (exact + self.guard_bits).min(self.pass_cap)
    }

    /// Bits the shaped rounding of an automorphism removes, at scale `2^log_delta`: as deep below
    /// the scale as a product, within the pass cap.
    pub(crate) fn automorphism_bits(&self, log_delta: usize) -> usize {
        if self.deep_automorphisms {
            (log_delta + self.guard_bits).min(self.pass_cap)
        } else {
            self.guard_bits.min(self.pass_cap)
        }
    }
}
