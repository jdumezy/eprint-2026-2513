//! Fiber-aware linear maps between the replicated layout and a slot vector of period `k`.
//!
//! The read side (SlotsToCoeffs) takes a slot vector of period `kR` holding value `j` on slot
//! `R j` and anything on the guard slots. It applies a `k x k` matrix `A` with `k` diagonals, at
//! the offsets `-t + R u` for `t < R` and `u < k / R`, each zero wherever it would read a guard
//! slot. The result equals `A z` after the normalized trace onto period `k` (the counting lemma of
//! the paper), which needs `R | k`. The write side (CoeffsToSlots) takes a slot vector of period
//! `k` and applies `C` with `k` diagonals, writing `(C x)_j` on slot `R j` and zero on the guards.
//!
//! Both follow the diagonal convention of poulpy: `y[s] = sum_o D_o[s] x[s + o]`, a diagonal of
//! one period being repeated over the slots.

use poulpy_ckks::layouts::ComplexDiagonals;
use poulpy_core::layouts::Diagonals;

use crate::layout::ReplicatedLayout;

/// A complex number as `(re, im)`.
pub type Complex = (f64, f64);

/// The dense `k x k` matrix of a diagonal map over `k` slots: `A[u][j] = D_{j - u}[u]`.
pub fn dense(diagonals: &ComplexDiagonals<f64>) -> Vec<Vec<Complex>> {
    let k = diagonals.re.slots();
    (0..k)
        .map(|u| {
            (0..k)
                .map(|j| {
                    let o = ((j + k - u) % k) as i64;
                    let at = |v: Option<&Vec<f64>>| v.map_or(0.0, |v| v[u % v.len()]);
                    (at(diagonals.re.get(o)), at(diagonals.im.get(o)))
                })
                .collect()
        })
        .collect()
}

/// The `k x k` identity.
pub fn identity(k: usize) -> Vec<Vec<Complex>> {
    (0..k)
        .map(|u| (0..k).map(|j| if u == j { (1.0, 0.0) } else { (0.0, 0.0) }).collect())
        .collect()
}

fn empty(slots: usize) -> ComplexDiagonals<f64> {
    ComplexDiagonals {
        re: Diagonals::new(slots),
        im: Diagonals::new(slots),
    }
}

/// The `k` offsets of the read side, `-t + R u` for `t < R` and `u < k / R`.
pub fn read_offsets(layout: &ReplicatedLayout) -> Vec<i64> {
    let (k, r) = (layout.batch(), layout.replication());
    (0..r)
        .flat_map(|t| (0..k / r).map(move |u| (r * u) as i64 - t as i64))
        .collect()
}

/// Read side: the diagonals that apply `gain A` to the data slots of a vector of period `kR`.
///
/// Entry `(c, j)` of `A` sits on one position of the fiber `c + k i`, chosen so that its offset is
/// one of [`read_offsets`]. With `gain = R` the normalized trace onto period `k` returns `A z`.
pub fn read_diagonals(layout: &ReplicatedLayout, a: &[Vec<Complex>], gain: f64) -> ComplexDiagonals<f64> {
    let (k, r) = (layout.batch(), layout.replication());
    assert!(k.is_multiple_of(r), "the fiber-aware transforms need R | k");
    assert_eq!(a.len(), k, "the matrix is {k} x {k}");
    let period = layout.period();
    let mut out = empty(layout.n() / 2);
    for t in 0..r {
        for u in 0..k / r {
            let o = (r * u) as i64 - t as i64;
            let (mut re, mut im) = (vec![0f64; period], vec![0f64; period]);
            // The positions s = t mod R read s + o = 0 mod R, a data slot.
            for s in (t..period).step_by(r) {
                let row = s % k;
                let col = ((s - t) / r + u) % k;
                re[s] = gain * a[row][col].0;
                im[s] = gain * a[row][col].1;
            }
            out.re.set(o, re);
            out.im.set(o, im);
        }
    }
    out
}

/// Write side: the diagonals that apply `c` to a vector of period `m` and write the result on the
/// data slots, with zero on the guard slots. `c` has one row per value of the layout and `m`
/// columns; the offsets are among `0..m`.
pub fn write_diagonals(layout: &ReplicatedLayout, c: &[Vec<Complex>]) -> ComplexDiagonals<f64> {
    let (k, r) = (layout.batch(), layout.replication());
    assert_eq!(c.len(), k, "the matrix has {k} rows");
    let m = c[0].len();
    let period = layout.period();
    assert!(period.is_multiple_of(m), "the input period {m} divides {period}");
    let mut out = empty(layout.n() / 2);
    for o in 0..m {
        let (mut re, mut im) = (vec![0f64; period], vec![0f64; period]);
        let mut any = false;
        for (j, row) in c.iter().enumerate() {
            let entry = row[(r * j + o) % m];
            re[r * j] = entry.0;
            im[r * j] = entry.1;
            any |= entry != (0.0, 0.0);
        }
        if any {
            out.re.set(o as i64, re);
            out.im.set(o as i64, im);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }

        fn complex(&mut self) -> Complex {
            (self.next(), self.next())
        }
    }

    fn mul(a: Complex, b: Complex) -> Complex {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    fn add(a: Complex, b: Complex) -> Complex {
        (a.0 + b.0, a.1 + b.1)
    }

    /// `y[s] = sum_o D_o[s] x[s + o]`.
    fn apply(diagonals: &ComplexDiagonals<f64>, x: &[Complex]) -> Vec<Complex> {
        let len = x.len() as i64;
        let mut y = vec![(0.0, 0.0); x.len()];
        for o in diagonals.re.indexes() {
            let (re, im) = (diagonals.re.get(o).unwrap(), diagonals.im.get(o).unwrap());
            for (s, out) in y.iter_mut().enumerate() {
                let entry = (re[s % re.len()], im[s % im.len()]);
                *out = add(*out, mul(entry, x[(s as i64 + o).rem_euclid(len) as usize]));
            }
        }
        y
    }

    fn close(a: Complex, b: Complex) -> bool {
        (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12
    }

    /// The counting lemma: `k` offsets reach every entry from the data slots alone, and the
    /// normalized trace onto period `k` returns `A z` whatever the guards hold.
    #[test]
    fn read_side_uses_k_offsets_and_ignores_the_guards() {
        for (k, r) in [(16usize, 16usize), (16, 4), (64, 16), (32, 8), (8, 8)] {
            let n = 8 * k * r;
            let layout = ReplicatedLayout::new(n, k, r).unwrap();
            let mut rng = Rng(0x1234_5678_9ABC_DEF1 ^ (k * 131 + r) as u64);
            let a: Vec<Vec<Complex>> = (0..k).map(|_| (0..k).map(|_| rng.complex()).collect()).collect();
            let z: Vec<Complex> = (0..k).map(|_| rng.complex()).collect();
            let period = layout.period();
            let one: Vec<Complex> = (0..period)
                .map(|t| {
                    if t % r == 0 {
                        z[t / r]
                    } else {
                        (100.0 * rng.next(), 100.0 * rng.next())
                    }
                })
                .collect();
            let x: Vec<Complex> = (0..n / 2).map(|t| one[t % period]).collect();
            let diagonals = read_diagonals(&layout, &a, r as f64);
            let mut offsets = diagonals.re.indexes();
            offsets.sort_unstable();
            let mut want: Vec<i64> = read_offsets(&layout).iter().map(|o| o.rem_euclid(n as i64 / 2)).collect();
            want.sort_unstable();
            assert_eq!(offsets, want, "k = {k}, R = {r}");
            let y = apply(&diagonals, &x);
            for c in 0..k {
                let got = (0..r).fold((0.0, 0.0), |acc, i| add(acc, y[c + k * i]));
                let got = (got.0 / r as f64, got.1 / r as f64);
                let exact = (0..k).fold((0.0, 0.0), |acc, j| add(acc, mul(a[c][j], z[j])));
                assert!(close(got, exact), "k = {k}, R = {r}, c = {c}: {got:?} against {exact:?}");
            }
        }
    }

    /// The write side puts `C x` on the data slots and zero on the guards, with at most `k`
    /// diagonals for an input of period `k`.
    #[test]
    fn write_side_fills_the_data_slots_only() {
        // `rows = k` is the plain case, `rows = 2 k` the packed CoeffsToSlots.
        for (k, r, rows) in [
            (16usize, 16usize, 16usize),
            (16, 4, 16),
            (64, 16, 64),
            (4, 16, 4),
            (16, 16, 32),
            (4, 16, 8),
        ] {
            let n = 8 * rows * r;
            let layout = ReplicatedLayout::new(n, rows, r).unwrap();
            let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D ^ (k * 131 + r + rows) as u64);
            let c: Vec<Vec<Complex>> = (0..rows).map(|_| (0..k).map(|_| rng.complex()).collect()).collect();
            let small: Vec<Complex> = (0..k).map(|_| rng.complex()).collect();
            let x: Vec<Complex> = (0..n / 2).map(|t| small[t % k]).collect();
            let diagonals = write_diagonals(&layout, &c);
            assert!(diagonals.re.indexes().len() <= k);
            let y = apply(&diagonals, &x);
            for (t, got) in y.iter().enumerate() {
                let t_in = t % layout.period();
                let exact = if layout.is_data_slot(t_in) {
                    let j = t_in / r;
                    (0..k).fold((0.0, 0.0), |acc, col| add(acc, mul(c[j][col], small[col])))
                } else {
                    (0.0, 0.0)
                };
                assert!(close(*got, exact), "k = {k}, R = {r}, {rows} rows, slot {t}");
            }
        }
    }
}
