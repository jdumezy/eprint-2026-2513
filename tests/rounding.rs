//! The shaped rounding pass against an exact big-integer oracle, and against the HAL
//! normalization for `g = 1`.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    execution::{SerialTaskExecutor, TaskExecutor},
    layouts::{Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut, vec_znx_backend_mut, vec_znx_backend_ref},
};
use poulpy_shaped::{
    rounding::{MAX_PASS_BITS, shaped_round_column, shaped_round_integers, shaped_round_tmp_bytes},
    shaping::ShapingPolynomial,
    unit::{ShapingUnit, catalog},
};

/// A four-thread executor, to run the block-parallel path of the pass.
struct Threads;

impl TaskExecutor for Threads {
    const IS_PARALLEL: bool = true;

    fn max_parallelism() -> usize {
        4
    }

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        std::thread::scope(|s| {
            let right = s.spawn(right);
            (left(), right.join().expect("task panicked"))
        })
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

fn alloc(n: usize, size: usize) -> VecZnx<AlignedBuf, i64> {
    VecZnx::<AlignedBuf, i64>::from_bytes(n, 1, size, vec![0u8; VecZnx::<AlignedBuf, i64>::bytes_of(n, 1, size)])
}

/// A column canonical at precision `k`, with uniform digits.
fn random_column(n: usize, base2k: usize, k: usize, rng: &mut Rng) -> VecZnx<AlignedBuf, i64> {
    let size = k.div_ceil(base2k);
    let pad = size * base2k - k;
    let mut a = alloc(n, size);
    for l in 0..size {
        for x in a.at_mut(0, l).iter_mut() {
            let d = ((rng.next() as i64) << (64 - base2k)) >> (64 - base2k);
            *x = if l == size - 1 { (d >> pad) << pad } else { d };
        }
    }
    a
}

/// The integers `2^k * value` of column 0, which must be canonical at precision `k`.
fn column_integers(a: &VecZnx<AlignedBuf, i64>, base2k: usize, k: usize) -> Vec<BigInt> {
    let size = k.div_ceil(base2k);
    let pad = size * base2k - k;
    let half = 1i64 << (base2k - 1);
    (0..a.n())
        .map(|i| {
            let mut v = BigInt::zero();
            for l in 0..size {
                let d = a.at(0, l)[i];
                assert!((-half..half).contains(&d), "digit {d} is not balanced");
                v = (v << base2k) + d;
            }
            assert!(a.at(0, size - 1)[i] & ((1 << pad) - 1) == 0, "padding bits are set");
            for l in size..a.size() {
                assert_eq!(a.at(0, l)[i], 0, "limb {l} past the precision is not zero");
            }
            v >> pad
        })
        .collect()
}

/// `terms * x` in `Z[X] / (X^n + 1)`.
fn negacyclic(terms: impl Iterator<Item = (usize, i128)>, x: &[BigInt]) -> Vec<BigInt> {
    let n = x.len();
    let mut out = vec![BigInt::zero(); n];
    for (shift, c) in terms {
        for (i, v) in x.iter().enumerate() {
            let p = v * c;
            if i + shift < n {
                out[i + shift] += p;
            } else {
                out[i + shift - n] -= p;
            }
        }
    }
    out
}

/// `(c - g h) / 2^b` with `h = [g^{-1} c]_{2^b}`, exactly.
fn oracle(shaping: &ShapingPolynomial, c: &[BigInt], bits: usize) -> Vec<BigInt> {
    let modulus = BigInt::one() << bits;
    let half = BigInt::one() << (bits - 1);
    let low: Vec<BigInt> = c.iter().map(|x| x.mod_floor(&modulus)).collect();
    let h: Vec<BigInt> = negacyclic(shaping.inverse_terms().iter().copied(), &low)
        .into_iter()
        .map(|x| {
            let r = x.mod_floor(&modulus);
            if r >= half { r - &modulus } else { r }
        })
        .collect();
    let gh = negacyclic(shaping.terms().iter().map(|&(s, c)| (s, c as i128)), &h);
    c.iter()
        .zip(gh)
        .map(|(x, p)| {
            let (q, r) = (x - p).div_mod_floor(&modulus);
            assert!(r.is_zero(), "c - g h is not divisible by 2^b");
            q
        })
        .collect()
}

fn run<E: TaskExecutor>(shaping: &ShapingPolynomial, cases: usize, seed: u64) {
    let n = shaping.n();
    let mut rng = Rng(seed);
    let mut tmp = vec![0u8; shaped_round_tmp_bytes(n)];
    for case in 0..cases {
        let base2k = [52, 30, 19][case % 3];
        let bits = 1 + rng.below(MAX_PASS_BITS);
        let k_res = 1 + rng.below(150);
        let k_wide = k_res + bits;
        let wide = random_column(n, base2k, k_wide, &mut rng);
        // One spare limb, which the pass must clear.
        let mut res = alloc(n, k_res.div_ceil(base2k) + 1);
        res.at_mut(0, 0).fill(7);
        shaped_round_column::<E, _, _>(shaping, base2k, &mut res, 0, k_res, &wide, 0, k_wide, &mut tmp);

        let modulus = BigInt::one() << k_res;
        let want = oracle(shaping, &column_integers(&wide, base2k, k_wide), bits);
        let got = column_integers(&res, base2k, k_res);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.mod_floor(&modulus),
                w.mod_floor(&modulus),
                "coefficient {i}, base2k {base2k}, k {k_wide} -> {k_res}"
            );
        }
    }
}

fn shaping(coeffs: &[i64], n: usize) -> ShapingPolynomial {
    ShapingPolynomial::new(&ShapingUnit::new(coeffs.to_vec()).unwrap(), n).unwrap()
}

#[test]
fn pass_matches_the_exact_oracle() {
    // Blocks of 128 coefficients, and blocks of 32.
    for n in [4096, 1024] {
        run::<SerialTaskExecutor>(&shaping(catalog::U_32_40, n), 24, 1 + n as u64);
    }
    // A unit with large coefficients, and one with 64 coefficients.
    run::<SerialTaskExecutor>(&shaping(catalog::U_16_60, 2048), 24, 3);
    run::<SerialTaskExecutor>(&shaping(catalog::U_64_40, 2048), 24, 4);
}

#[test]
fn pass_matches_the_exact_oracle_after_an_automorphism() {
    let n = 2048;
    let gal = (0..3).fold(1, |a, _| a * 5 % (2 * n));
    run::<SerialTaskExecutor>(&shaping(catalog::U_32_40, n).automorphism(gal), 24, 5);
    run::<SerialTaskExecutor>(&shaping(catalog::U_32_40, n).automorphism(2 * n - 1), 12, 6);
}

#[test]
fn parallel_pass_matches_the_exact_oracle() {
    run::<Threads>(&shaping(catalog::U_32_40, 4096), 24, 7);
    run::<Threads>(&ShapingPolynomial::identity(4096), 12, 8);
}

#[test]
fn identity_matches_the_hal_normalization() {
    let n = 1024;
    let module = Module::<FFT64Portable>::new(n as u64);
    let shaping = ShapingPolynomial::identity(n);
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(module.vec_znx_normalize_tmp_bytes());
    let mut tmp = vec![0u8; shaped_round_tmp_bytes(n)];
    let mut rng = Rng(9);
    for case in 0..60 {
        let base2k = [52, 30, 19][case % 3];
        let k_res = 1 + rng.below(150);
        let k_wide = k_res + 1 + rng.below(MAX_PASS_BITS);
        let wide = random_column(n, base2k, k_wide, &mut rng);
        let size_r = k_res.div_ceil(base2k);
        let (mut ours, mut stock) = (alloc(n, size_r), alloc(n, size_r));
        shaped_round_column::<SerialTaskExecutor, _, _>(&shaping, base2k, &mut ours, 0, k_res, &wide, 0, k_wide, &mut tmp);
        module.vec_znx_normalize(
            &mut vec_znx_backend_mut::<FFT64Portable>(&mut stock),
            base2k,
            k_res,
            0,
            0,
            &vec_znx_backend_ref::<FFT64Portable>(&wide),
            base2k,
            0,
            &mut scratch.arena(),
        );
        for l in 0..size_r {
            assert_eq!(ours.at(0, l), stock.at(0, l), "case {case}, limb {l}");
        }
    }
}

#[test]
fn no_drop_copies_the_column() {
    let n = 256;
    let shaping = shaping(catalog::U_32_40, n);
    let mut tmp = vec![0u8; shaped_round_tmp_bytes(n)];
    let wide = random_column(n, 52, 100, &mut Rng(10));
    let mut res = alloc(n, 3);
    res.at_mut(0, 2).fill(1);
    shaped_round_column::<SerialTaskExecutor, _, _>(&shaping, 52, &mut res, 0, 100, &wide, 0, 100, &mut tmp);
    assert_eq!(column_integers(&res, 52, 100), column_integers(&wide, 52, 100));
}

#[test]
fn integer_rounding_matches_the_oracle_and_stays_in_the_subring() {
    let n = 1024;
    let shaping = shaping(catalog::U_32_40, n);
    let mut rng = Rng(11);
    // A sparse input in Z[X^32] = Z[Y], as a quantized constant would be.
    let gap = n / 32;
    let c: Vec<i128> = (0..n)
        .map(|i| if i % gap == 0 { (rng.next() as i64 as i128) << 30 } else { 0 })
        .collect();
    for bits in [1, 24, 40, 60, 90] {
        let got = shaped_round_integers(&shaping, &c, bits);
        let big: Vec<BigInt> = c.iter().map(|&x| BigInt::from(x)).collect();
        let want = oracle(&shaping, &big, bits);
        assert!(got.iter().zip(&want).all(|(g, w)| BigInt::from(*g) == *w), "bits {bits}");
        assert!(got.iter().enumerate().all(|(i, &x)| x == 0 || i % gap == 0));
    }
}
