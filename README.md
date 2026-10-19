# poulpy-shaped

Approximate homomorphic encryption for small batches of complex numbers, with shaped rounding, on [poulpy](https://github.com/phantomzone-org/poulpy) 0.9.0.

This crate implements the scheme of *Fully Homomorphic Encryption for Small Batches of Complex Numbers* (Jules Dumezy, [IACR ePrint 2026/2513](https://eprint.iacr.org/2026/2513)): encoding and encryption, the leveled operations, bootstrapping, and the conversions to and from CKKS.
It is a separate crate on top of the published poulpy crates and does not patch them.

## The scheme in one paragraph

CKKS puts `N/2` values in the slots of a ciphertext.
When only a few values are needed, most of the slots are wasted, but every rounding (rescaling, key switching, encoding) still spreads its error over all of them.
Here, `k` values are written on one slot in `R` of a replicated layout, and every rounding is a *shaped rounding* `(a - g h) / r`: its error is a multiple of a fixed public unit `g = U(X^(N/2R))`, which is small on the data slots and larger on the guard slots in between.
The guards are computed on but never decoded.
A product at scale `2^20` then adds an error of about `2^-46` to the values, where CKKS at the same scale adds `2^-7`, and the bootstrapping is cheaper because it works at smaller moduli.
With `g = 1` the scheme is CKKS.

## Status

Everything the paper describes is implemented and tested:

| Part | Module | Paper |
|---|---|---|
| Shaping units and the shaped rounding | `unit`, `shaping`, `rounding` | Sections 3 and 4 |
| Replicated layout, encoding, encryption | `layout`, `encoding` | Section 4 |
| Products, rescaling, rotations, conjugation, traces | `ops` | Section 4 |
| Bootstrapping (fiber-aware transforms, shaped EvalMod) | `bootstrapping` | Section 5 |
| Conversions to and from CKKS | `conversion` | Appendix A |
| Key moduli against the 128-bit security caps | `security` | Parameters |

At `N = 2^15` with `k = 16` values, on one thread, a bootstrapping takes 0.41 s for `L = 1` level and 0.75 s for `L = 16`, with 23 bits of precision after the levels.
On all cores, it takes 0.12 s and 0.18 s.
Sixteen leveled products at scale `2^20` keep 48 bits, where CKKS keeps 5.
For 16 or 64 values, sparse CKKS needs `N = 2^16` beyond four levels, and is about 5 times slower on one thread at 16 levels.

Two improvements over the first version of the paper (0.55 s and 1.04 s on one thread, 45 leveled bits) account for the differences:

- **Packed EvalMod.** When `2 k R` slots fit in the ring, EvalMod runs once on the real and imaginary parts packed side by side instead of once on each (`BootstrapParams::packed_eval_mod`). Without it, the times are 0.58 s and 1.11 s on one thread.
- **An optimal unit.** The default unit `U_{32,45}` comes from an exact search ([scripts/unit_search.py](scripts/unit_search.py)): 46.0 bits of suppression with a largest guard magnitude of `2^4.59`, against 39.7 bits and `2^4.70` for `U_{32,40}` of the first version, which remains in `unit::catalog`.

See [bench/README.md](bench/README.md) for the full tables, with the sparse CKKS and PaCo baselines, and how to regenerate them.

## Quick start

```bash
cargo run --release --example quickstart
cargo run --release --example bootstrap -- 15 16 1    # log2 N, values, levels
```

The crate uses the nightly toolchain pinned in `rust-toolchain.toml`, like poulpy.

A computation starts from a `ShapedContext`, which fixes the ring degree, the batch size and the unit.
Ciphertexts, plaintexts and keys are the ordinary poulpy-ckks and poulpy-core ones.
The shaped operations are traits implemented for every poulpy `Module` over a CPU backend:

```rust
use poulpy_shaped::prelude::*;

// 16 complex values in a ring of degree 2^12, with the default unit U_{32,45} (R = 16).
let ctx = ShapedContext::new(1 << 12, 16, ShapingUnit::default_unit())?;

// Encode with shaping and encrypt with a shaped error.
let mut pt = module.shaped_plaintext_alloc(&ctx, BASE2K, 20);
module.shaped_encode_into(&ctx, &mut pt, &re, &im, scratch)?;
module.shaped_encrypt_sk(&ctx, &mut ct, &pt, &sk, &mut xe, &mut xa, scratch)?;

// Products return rescaled results: a destination one level narrower consumes 20 bits.
module.shaped_mul_into(&ctx, &mut xy, &ct_x, &ct_y, &tensor_key, scratch)?;
module.shaped_rotate_into(&ctx, &mut rotated, &xy, 1, &automorphism_keys, scratch)?;
module.ckks_add_assign(&mut rotated, &xy, scratch)?;    // exact operations are poulpy-ckks'

// Decrypt and decode, averaging the copies of each value.
module.shaped_decrypt(&ctx, &rotated, &sk, &mut re, &mut im, scratch)?;
```

[examples/quickstart.rs](examples/quickstart.rs) is the complete program, with key generation.
Its ring of degree `2^12` keeps the example fast and is not secure: see [Security](#security).

### Bootstrapping

```rust
let boot = Bootstrapper::new(&module, &ctx, BootstrapParams::default(), 52, levels, scratch)?;
// Keys within the 128-bit caps for a secret of Hamming weight 1024.
let layouts = boot.secure_key_layouts(1024)?;
let keys = boot.generate_keys(&module, &sk_raw, &layouts, &mut xs, &mut xe, &mut xa, scratch);
let mut boot_scratch = ScratchOwned::alloc(boot.tmp_bytes(&module, &layouts));

let fresh = boot.bootstrap(&module, &exhausted, &keys, &mut boot_scratch.borrow())?;
```

The bootstrapping takes a ciphertext at the leveled scale `2^20` and at least `k_in - 10` bits wide (one whose levels are spent is `k_in` wide), or at its input scale `2^30` and at least `k_in` wide.
It returns one at the leveled scale with the `levels` it was compiled for.
Its keys also serve the leveled products (`keys.tensor`).

### Conversions

`Converter::to_shaped` takes a CKKS ciphertext at scale `2^40` whose slots repeat the `k` values with period `k`, and `Converter::to_ckks` gives it back.
Both share the secret and the keys of the scheme.

## Security

The `security` module holds the 128-bit caps on `log2(Q P)` of the [sparse-key-estimate tables](https://github.com/jdumezy/sparse-key-estimate/blob/master/Precomputed-Tables/128bits_security.md), by ring degree and Hamming weight of the secret.
`Bootstrapper::security` checks every key at its full modulus as poulpy allocates it, `dnum · dsize · base2k + P`, and the key to the ephemeral sparse secret of ModUp at the weight of that secret.

`Bootstrapper::secure_key_layouts` picks keys of two limbs per digit with `P = 104` when every key fits the caps, which halves the size of the keys, and keys of one limb per digit with `P = 53`, the smallest modulus, beyond.
The key to the ephemeral secret always has one limb per digit and `P = 53`.
With a secret of Hamming weight 1024 and an ephemeral secret of weight 32, at `N = 2^15`:

| Levels `L` | Digits | `log2(Q P)` of the keys | Cap (h = 1024) | Ephemeral key | Cap (h = 32) |
|---|---|---|---|---|---|
| 1 | 2 limbs | 520 | 851 | 105 | 162 |
| 8 | 2 limbs | 624 | 851 | 105 | 162 |
| 16 | 2 limbs | 832 | 851 | 105 | 162 |
| 18 | 2 limbs | 832 | 851 | 105 | 162 |
| 19 to 21 | 1 limb | 833 | 851 | 105 | 162 |
| 22 | 1 limb | 885 | 851 | 105 | 162 |

So `N = 2^15` supports up to 21 levels between bootstrappings.
poulpy covers the key modulus in whole digits, so two-limb digits can cost one more limb of modulus than one-limb digits.
The auxiliary modulus must cover one digit (`P >= dsize · base2k`), which `key_layouts` enforces; below it, the key switches lose precision.

## Things to know

- **Batch sizes.** `k R` must fit in the `N/2` slots, and `k`, `R`, `N` are powers of two. The bootstrapping and the conversion to CKKS need `R | k`: pad smaller batches to `R` values.
- **Rotations** by `s` values are slot rotations by `s R` (`ReplicatedLayout::rotation_slots`). Only these and the conjugation keep the data slots apart from the guards.
- **Keys** are ordinary poulpy keys, one limb wider than the widest ciphertext, so that their unshaped noise is suppressed by the narrower ciphertext modulus.
- **Scratch.** `ShapedOps::shaped_ops_tmp_bytes`, `ShapedEncoding::shaped_encoding_tmp_bytes`, `Bootstrapper::tmp_bytes` and `Converter::tmp_bytes` size the scratch of each part.
- **Guards** carry the shaped errors and must never be released. `shaped_decode` returns the data slots only, and `shaped_decode_slots` (every slot) is for analysis.
- **Backends.** Every CPU backend of poulpy works (`HostBackend`). The examples use the NEON backends on aarch64 and the portable one elsewhere.

## Layout of the repository

- `src/`: the library.
- `tests/`: shaped rounding against exact integer arithmetic, the leveled operations against the same computation without shaping, bootstrapping, and the conversions.
- `examples/`: `quickstart` and `bootstrap`.
- `bench/`: the benchmarks reproducing the tables of the paper, against poulpy's sparse CKKS and PaCo bootstrappings.
- `scripts/unit_search.py`: the search for shaping units, as a mixed-integer program over the exponents of the cyclotomic units (needs numpy, scipy and mpmath).

```bash
cargo test --release
```

## Citing

The paper:

```bibtex
@misc{cryptoeprint:2026/2513,
  author       = {Jules Dumezy},
  title        = {Fully Homomorphic Encryption for Small Batches of Complex Numbers},
  howpublished = {Cryptology {ePrint} Archive, Paper 2026/2513},
  year         = {2026},
  url          = {https://eprint.iacr.org/2026/2513}
}
```

This implementation:

```bibtex
@software{poulpy-shaped,
  author = {Jules Dumezy},
  title  = {poulpy-shaped: Fully Homomorphic Encryption for Small Batches of Complex Numbers on poulpy},
  year   = {2026},
  url    = {https://github.com/jdumezy/eprint-2026-2513}
}
```

## License

Licensed under the Apache License, Version 2.0: see [LICENSE](LICENSE).

## AI disclaimer

Claude Opus 5.5 was used to assist the development of this implementation.
