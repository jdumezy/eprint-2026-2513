//! Shared pieces of the benchmarks: arguments, inputs, precision and JSON lines.
//!
//! Every binary prints one JSON object per line on standard output. `paper.py` runs them,
//! measures their peak memory and turns the lines into the tables of the paper.

use std::{collections::HashMap, fmt::Display, str::FromStr};

/// `--name value` pairs and bare `--flag`s.
pub struct Args {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    pub fn parse() -> Self {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let (mut values, mut flags) = (HashMap::new(), Vec::new());
        let mut i = 0;
        while i < raw.len() {
            let name = raw[i].trim_start_matches("--").to_string();
            match raw.get(i + 1) {
                Some(v) if !v.starts_with("--") => {
                    values.insert(name, v.clone());
                    i += 2;
                }
                _ => {
                    flags.push(name);
                    i += 1;
                }
            }
        }
        Self { values, flags }
    }

    pub fn get<T: FromStr>(&self, name: &str, default: T) -> T {
        self.values
            .get(name)
            .map(|v| v.parse().unwrap_or_else(|_| panic!("--{name} {v}: not a valid value")))
            .unwrap_or(default)
    }

    pub fn string(&self, name: &str, default: &str) -> String {
        self.values.get(name).cloned().unwrap_or_else(|| default.to_string())
    }

    pub fn list<T: FromStr>(&self, name: &str, default: &str) -> Vec<T> {
        self.string(name, default)
            .split(',')
            .map(|v| v.parse().unwrap_or_else(|_| panic!("--{name}: {v} is not a valid value")))
            .collect()
    }

    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }
}

/// xorshift64, seeded per run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        Self(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD129_0B2D_5C6F_2A31) ^ stream.wrapping_mul(0xA24B_AED4_963E_E407) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `k` values on the unit circle (`family = "unit"`) or uniform in the unit disk (`"disk"`).
    pub fn values(&mut self, k: usize, family: &str) -> (Vec<f64>, Vec<f64>) {
        (0..k)
            .map(|_| {
                let angle = self.unit() * std::f64::consts::TAU;
                let radius = if family == "disk" { self.unit().sqrt() } else { 1.0 };
                (radius * angle.cos(), radius * angle.sin())
            })
            .unzip()
    }
}

/// Precision in the sense of poulpy's `precision_stats`: minus `log2` of the largest and of the
/// mean absolute error over the real and imaginary parts.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    pub min_bits: f64,
    pub avg_bits: f64,
}

impl Quality {
    pub fn of(errors: impl IntoIterator<Item = f64>) -> Self {
        let (mut worst, mut sum, mut count) = (0f64, 0f64, 0usize);
        for e in errors {
            worst = worst.max(e.abs());
            sum += e.abs();
            count += 1;
        }
        Self {
            min_bits: -worst.log2(),
            avg_bits: -(sum / count.max(1) as f64).log2(),
        }
    }

    /// Against the first `k` entries of the wanted values.
    pub fn against(re: &[f64], im: &[f64], want_re: &[f64], want_im: &[f64]) -> Self {
        Self::of(re.iter().zip(want_re).chain(im.iter().zip(want_im)).map(|(h, w)| h - w))
    }
}

/// One JSON object, built field by field.
#[derive(Default, Clone)]
pub struct Line(Vec<String>);

impl Line {
    pub fn new(kind: &str) -> Self {
        Self(vec![format!("\"kind\":\"{kind}\"")])
    }

    pub fn num(mut self, name: &str, value: impl Display) -> Self {
        let text = value.to_string();
        let finite = text.parse::<f64>().map(|x| x.is_finite()).unwrap_or(false);
        self.0
            .push(format!("\"{name}\":{}", if finite { text } else { "null".into() }));
        self
    }

    /// A float rounded to `digits` decimals.
    pub fn float(self, name: &str, value: f64, digits: usize) -> Self {
        if value.is_finite() {
            self.num(name, format!("{value:.digits$}"))
        } else {
            self.num(name, "null")
        }
    }

    pub fn str(mut self, name: &str, value: &str) -> Self {
        self.0.push(format!("\"{name}\":\"{value}\""));
        self
    }

    pub fn quality(self, prefix: &str, q: Quality) -> Self {
        self.float(&format!("{prefix}min_bits"), q.min_bits, 3)
            .float(&format!("{prefix}avg_bits"), q.avg_bits, 3)
    }

    /// Appends the fields of another line, without its kind.
    pub fn with(mut self, other: &Line) -> Self {
        self.0.extend(other.0.iter().skip(1).cloned());
        self
    }

    pub fn print(&self) {
        println!("{{{}}}", self.0.join(","));
    }
}

/// Median of a non-empty slice.
pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { (v[m - 1] + v[m]) / 2.0 }
}

/// Runs `$run::<BE>($($arg),*)` on the single-thread NEON backend for `threads == 1` and on the
/// Rayon NEON backend otherwise, or on the portable backend off aarch64.
#[macro_export]
macro_rules! dispatch {
    ($threads:expr, $run:ident ( $($arg:expr),* )) => {{
        #[cfg(target_arch = "aarch64")]
        {
            if $threads == 1 {
                $run::<poulpy_cpu_arm::NTT4x30Neon>($($arg),*)
            } else {
                $run::<poulpy_cpu_arm::NTT4x30NeonRayon>($($arg),*)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = $threads;
            $run::<poulpy_cpu_portable::NTT4x30Portable>($($arg),*)
        }
    }};
}
