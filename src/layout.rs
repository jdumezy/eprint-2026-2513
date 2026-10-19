//! The replicated layout.
//!
//! A batch of `k` complex values occupies the slots `t = 0 mod R` of a slot vector of period
//! `k R`: value `j` sits on slot `R j` and its `d = N / 2kR` copies `R j + kR i`. The other
//! slots are guards, which hold zero in exact arithmetic and absorb the larger part of the
//! shaped rounding errors. A slot vector of period `kR` is the sparse CKKS packing of
//! `log_sparsity = log2 d`, so a replicated plaintext is a polynomial in `X^d`.

/// Why a layout is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// The ring degree, batch size or replication factor is not a power of two.
    NotPowerOfTwo,
    /// `k R` exceeds the `N / 2` slots of the ring.
    TooManySlots {
        /// The number of values `k`.
        batch: usize,
        /// The replication factor `R`.
        replication: usize,
        /// The ring degree `N`.
        n: usize,
    },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPowerOfTwo => write!(f, "ring degree, batch size and replication must be powers of two"),
            Self::TooManySlots { batch, replication, n } => {
                write!(f, "{batch} values replicated {replication} times exceed the {} slots", n / 2)
            }
        }
    }
}

impl std::error::Error for LayoutError {}

/// `k` values on the slots `t = 0 mod R` of a slot vector of period `k R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicatedLayout {
    n: usize,
    batch: usize,
    replication: usize,
}

impl ReplicatedLayout {
    /// The layout of `batch` values with replication factor `replication` in the ring of degree `n`.
    pub fn new(n: usize, batch: usize, replication: usize) -> Result<Self, LayoutError> {
        if !(n.is_power_of_two() && batch.is_power_of_two() && replication.is_power_of_two()) {
            return Err(LayoutError::NotPowerOfTwo);
        }
        if batch * replication > n / 2 {
            return Err(LayoutError::TooManySlots { batch, replication, n });
        }
        Ok(Self { n, batch, replication })
    }

    /// The ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number `k` of values.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The replication factor `R`: one slot in `R` carries data.
    pub fn replication(&self) -> usize {
        self.replication
    }

    /// The period `k R` of the slot vector.
    pub fn period(&self) -> usize {
        self.batch * self.replication
    }

    /// The number `d = N / 2kR` of copies of each value.
    pub fn copies(&self) -> usize {
        self.n / (2 * self.period())
    }

    /// `log2 d`: the CKKS sparsity of the layout.
    pub fn log_sparsity(&self) -> usize {
        self.copies().trailing_zeros() as usize
    }

    /// Whether slot `t` carries data.
    pub fn is_data_slot(&self, t: usize) -> bool {
        t.is_multiple_of(self.replication)
    }

    /// The slot vector of one period carrying `values` on the data slots and zero on the guards.
    pub fn spread<T: Copy + Default>(&self, values: &[T]) -> Vec<T> {
        assert_eq!(values.len(), self.batch, "the layout holds {} values", self.batch);
        let mut slots = vec![T::default(); self.period()];
        for (j, &v) in values.iter().enumerate() {
            slots[j * self.replication] = v;
        }
        slots
    }

    /// The values on the data slots of one period of `slots`.
    pub fn collect<T: Copy>(&self, slots: &[T]) -> Vec<T> {
        assert_eq!(slots.len(), self.period(), "one period has {} slots", self.period());
        slots.iter().step_by(self.replication).copied().collect()
    }

    /// The values on the guard slots of one period of `slots`.
    pub fn guards<'a, T: Copy>(&self, slots: &'a [T]) -> impl Iterator<Item = T> + 'a {
        let r = self.replication;
        slots.iter().enumerate().filter(move |(t, _)| t % r != 0).map(|(_, &v)| v)
    }

    /// The slot rotation that moves the values by `steps` positions: value `j + steps` lands on
    /// index `j`. Only these rotations, by multiples of `R` slots, keep data and guards apart.
    pub fn rotation_slots(&self, steps: i64) -> i64 {
        steps * self.replication as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_and_collect() {
        let layout = ReplicatedLayout::new(1 << 12, 4, 16).unwrap();
        assert_eq!((layout.period(), layout.copies(), layout.log_sparsity()), (64, 32, 5));
        let slots = layout.spread(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(slots[16], 2.0);
        assert_eq!(layout.collect(&slots), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(layout.guards(&slots).all(|v| v == 0.0));
        assert_eq!(layout.guards(&slots).count(), 60);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert_eq!(ReplicatedLayout::new(1024, 3, 16), Err(LayoutError::NotPowerOfTwo));
        assert!(matches!(
            ReplicatedLayout::new(1024, 64, 16),
            Err(LayoutError::TooManySlots { .. })
        ));
        assert_eq!(ReplicatedLayout::new(1024, 32, 16).unwrap().copies(), 1);
    }
}
