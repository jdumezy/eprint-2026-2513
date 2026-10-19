# Benchmarks

Benchmarks of `poulpy-shaped` against poulpy's sparse CKKS and PaCo bootstrapping, reproducing the tables of the paper.

```bash
python3 bench/paper.py --mode fast all     # minutes: one seed, k in {16, 64}, L in {1, 16}
python3 bench/paper.py --mode paper all    # hours: the full protocol of the paper
```

The results go to `bench/results/<mode>/`: one JSON line per configuration in `main.jsonl`, `paco.jsonl` and `leveled.jsonl`, and the tables in `TABLES.md`.
Single items run with `python3 bench/paper.py --mode fast main` (or `paco`, `leveled`, `tables`), and rows already in a result file are kept, so an interrupted run resumes.
`--log-n 13` runs everything at a smaller ring degree, without the security caps, as a smoke test.

## Additional experiments

```bash
python3 bench/extra.py all    # about one hour
```

`bench/extra.py` runs experiments beyond the tables of the paper, under the same protocol, and writes them to `bench/results/extra/` with a summary in `EXTRA.md`.
It never writes to the results of `paper.py`, and it skips configurations already recorded, so an interrupted run resumes.

- `levels22`: our bootstrapping grid at 22-bit levels, the setting the failure analysis recommends.
- `n14`: bootstrapping at `N = 2^14` with 22-bit limbs, where two levels fit the caps.
- `tail`: 1000 independent bootstrappings, each followed by 16 squarings, for the distribution of the guards.
- `cycles`: one ciphertext through 100 rounds of a bootstrapping and 16 products by fresh operands.
- `edges`: the stability edges of every unit of the catalog over 16 squarings, with the model of the paper.
- `convert`: the conversions between CKKS and the replicated layout.
- `circuit`: four `tanh` activations, as is, realified or with a guard reset, with or without a product by a fresh operand.

## Binaries

Each prints one JSON object per line; `--dry-run` prints the moduli and key sizes only.

- `shaped boot`: our bootstrapping of `k` values (padded to `R = 16`) at `N = 2^15`, unit `U_32_45` (`--unit U_32_40` for the unit of the paper), followed by `L` products by fresh operands or `L` squarings. Keys use two limbs per digit and `P = 104` where every key fits the security caps, one limb per digit and `P = 53` beyond (`Bootstrapper::secure_key_layouts`), unless `--dsize` and `--k-aux` are given.
- `shaped leveled`: chains of products without bootstrapping, our scheme against stock CKKS at the same scale, decoded in quad precision, with the largest error one step adds on the data and on the guards.
- `shaped tail`, `shaped cycles`, `shaped convert` and `shaped circuit`: the modes behind `extra.py`.
- `--base2k` sets the limb size of every `shaped` mode, and `--level-bits` the output scale of the bootstrapping.
- `sparse`: poulpy's SlotsToCoeffs-first sparse CKKS bootstrapping with 36-bit levels.
- `paco`: poulpy's encapsulated PaCo bootstrapping.

`--threads 1` runs on the single-thread NEON backend, any other value on the Rayon NEON backend with every core (portable backend off aarch64).

## Protocol

Following the paper: inputs on the unit circle and in the unit disk, five seeds, times as the best of three per seed and the median over the seeds, precision as the minimum and mean of poulpy's `precision_stats` over the data values after `L` products by fresh operands.
A configuration counts when it keeps 19 bits and its largest modulus `log2(q_key p)` stays within the caps of the sparse-key-estimate tables for 128-bit security (851 and 1707 bits at `N = 2^15` and `2^16` for the weight-1024 secret, 162 and 345 bits for the weight-32 ephemeral secret).
Every binary reports the moduli of its keys as poulpy allocates them, `dnum · dsize · base2k + P`, through `poulpy_shaped::security`.
Peak memory is the `ru_maxrss` of each process, and every timed process runs alone.
