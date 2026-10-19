#![warn(missing_docs)]
//! Approximate homomorphic encryption for small batches of complex numbers, on poulpy.
//!
//! The scheme generalizes CKKS: every rounding of a ciphertext (rescaling, key switching,
//! encoding) is replaced by a *shaped rounding*, whose error is a multiple of a fixed public unit
//! `g`. The unit is small on the data slots, one slot in `R`, and larger on the guard slots, which
//! are computed on but never decoded. With `g = 1` the scheme is CKKS.
//!
//! Ciphertexts, plaintexts and keys are those of poulpy-ckks and poulpy-core: the crate adds the
//! shaped encoding and encryption ([`encoding::ShapedEncoding`]), the shaped leveled operations
//! ([`ops::ShapedOps`]), the bootstrapping ([`bootstrapping::Bootstrapper`]) and the conversions
//! to and from CKKS ([`conversion::Converter`]). Additions, negations, products by powers of two
//! and modulus reductions do not round, and are the poulpy-ckks ones.
//!
//! A computation starts from a [`context::ShapedContext`], which fixes the ring degree, the batch
//! size and the unit:
//!
//! ```no_run
//! use poulpy_shaped::prelude::*;
//!
//! // 16 complex values in a ring of degree 2^12, with the default unit U_{32,45} (R = 16).
//! let ctx = ShapedContext::new(1 << 12, 16, ShapingUnit::default_unit()).unwrap();
//! assert_eq!(ctx.layout().copies(), 8);
//! ```
//!
//! `examples/quickstart.rs` runs a leveled computation end to end, and `examples/bootstrap.rs`
//! the bootstrapping at the parameters of the paper.
//!
//! - [`unit`](mod@unit): the units `U` of `Z[Y] / (Y^(2R) + 1)` that define the shaping profile.
//! - [`shaping`]: the shaping polynomial `g = U(X^(N / 2R))` of a ring of degree `N`.
//! - [`rounding`]: shaped rounding of limb columns and of plain integers.
//! - [`layout`]: the replicated layout of a batch on the slots.
//! - [`context`]: the public parameters of a computation.
//! - [`encoding`]: shaped encoding, encryption and decoding.
//! - [`ops`]: the shaped leveled operations on poulpy-ckks ciphertexts.
//! - [`bootstrapping`]: the bootstrapping on the replicated layout.
//! - [`conversion`]: the conversions between CKKS and the replicated layout.
//! - [`security`]: the key moduli of 128-bit security.
//! - [`backend`]: the backends the shaped operations run on.

pub mod backend;
pub mod bootstrapping;
pub mod context;
pub mod conversion;
pub mod encoding;
pub mod layout;
pub mod ops;
mod params;
pub mod rounding;
pub mod security;
pub mod shaping;
pub mod unit;

/// The types and traits of a typical computation.
pub mod prelude {
    pub use crate::{
        backend::HostBackend,
        bootstrapping::{BootstrapKeyLayouts, BootstrapKeys, BootstrapModule, BootstrapParams, Bootstrapper},
        context::ShapedContext,
        conversion::{ConversionModule, ConversionParams, Converter},
        encoding::ShapedEncoding,
        layout::ReplicatedLayout,
        ops::ShapedOps,
        security::ModulusCheck,
        unit::ShapingUnit,
    };
}
