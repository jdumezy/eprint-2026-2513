#!/usr/bin/env python3
"""Additional experiments on our scheme, under the protocol of the paper mode of paper.py.

Usage: python3 bench/extra.py [--out DIR] ITEM...

Items, in the order `all` runs them:
  levels22  the bootstrapping grid of our scheme at 22-bit levels, the setting the failure analysis
            recommends: k in {16, 64, 256, 1024}, L in {1, 2, 4, 8, 16, L_max}
  n14       bootstrapping at N = 2^14 with 22-bit limbs: k in {16, 64, 256}, L in {1, 2}, 20- and
            22-bit levels
  tail      1000 independent bootstrappings (5 key sets x 200), each followed by 16 squarings, at
            20- and 22-bit levels: the distribution of the guards
  cycles    one ciphertext through 100 rounds of a bootstrapping and 16 products by fresh operands,
            five seeds, at 20- and 22-bit levels
  edges     the stability edges of Table 6: 16 squarings per scale, every unit of the catalog at
            N = 2^15, and U_32_40 and U_32_45 at N = 2^16
  convert   the conversions between CKKS and the replicated layout: k in {16, 64, 256}, one thread
            and every core
  circuit   four tanh activations: as is, realified or reset, with or without a fresh product,
            at 20- and 24-bit scales
  baseline15  every sparse baseline candidate at N = 2^15 within the caps, one-limb digits included:
            k in {16, 64, 256, 1024}, L in {1, 2, 4, 8}
  paco56    PaCo at 5 and 6 levels for 16 and 64 values (written to DIR/paco.jsonl)
  tables    the markdown tables, from the result files: DIR/EXTRA.md

The results go to bench/results/extra/ by default, one JSON file per item; the results of paper.py
are never touched. Configurations already in a result file are skipped, so an interrupted run
resumes. Every process that reports a time runs alone.
"""
import json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper  # noqa: E402

CPUS = os.cpu_count()
SEEDS = "1,2,3,4,5"
UNITS_15 = ["U_16_20", "U_16_40", "U_16_60", "U_32_20", "U_32_40", "U_32_41", "U_32_45", "U_32_50", "U_32_60",
            "U_64_20", "U_64_40", "U_64_60"]
UNITS_16 = ["U_32_40", "U_32_45"]
EDGE_SCALES = list(range(15, 26)) + [28]


def path(name):
    return os.path.join(paper.OUT, f"{name}.jsonl")


def append(name, rows):
    os.makedirs(paper.OUT, exist_ok=True)
    with open(path(name), "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def done(name, key):
    return {key(r) for r in paper.read(name)}


def l_max(flags):
    best = 0
    for lv in range(1, 40):
        d = paper.dry("shaped", ["boot"] + flags + ["--k", 16, "--levels", lv])
        if d is None or not paper.within_caps(d):
            break
        best = lv
    return best


def grid(name, flags, ks, levels, extra=None):
    """Our bootstrapping on a grid, under the paper protocol, skipping what is recorded."""
    seen = done(name, lambda r: (r["k"], r["levels"], r.get("level_bits"), r.get("log_n")))
    for k in ks:
        for lv in levels:
            f = ["boot"] + flags + ["--k", k, "--levels", lv]
            d = paper.dry("shaped", f)
            if d is None:
                continue
            if (k, lv, d["level_bits"], d["log_n"]) in seen:
                continue
            row = {"pipeline": "ours", "k": k, "levels": lv, **(extra or {})}
            if not paper.within_caps(d):
                row.update({"status": "over_cap", "log2_qp": d["log2_qp"], "cap": d["cap"], "level_bits": d["level_bits"],
                            "log_n": d["log_n"]})
            else:
                row.update(paper.measure("shaped", f, lv))
            append(name, [row])
            paper.report(row)


def levels22_item():
    lm = l_max(["--log-n", 15, "--level-bits", 22])
    print(f"L_max at 22-bit levels, logN = 15: {lm}", flush=True)
    grid("levels22", ["--log-n", 15, "--level-bits", 22], [16, 64, 256, 1024], [1, 2, 4, 8, 16, lm])


def n14_item():
    for bits in (20, 22):
        grid("n14", ["--log-n", 14, "--base2k", 22, "--level-bits", bits], [16, 64, 256], [1, 2])


def raw_item(name, configs, key):
    """Runs `shaped` once per configuration and keeps every line, tagged with the configuration."""
    seen = done(name, lambda r: r.get("config"))
    for tag, args in configs:
        if tag in seen:
            continue
        t0 = time.time()
        r = paper.run("shaped", args)
        if not r["ok"]:
            print(f"{name} {tag} failed: {r['stderr']}", flush=True)
            continue
        lines = [dict(l, config=tag) for l in r["lines"] if l["kind"] == key]
        append(name, lines)
        print(f"{name} {tag}: {len(lines)} lines in {round(time.time() - t0)} s", flush=True)


def tail_item():
    raw_item("tail", [(f"bits{b}", ["tail", "--log-n", 15, "--k", 16, "--levels", 16, "--level-bits", b, "--key-sets", SEEDS,
                                    "--runs", 200, "--threads", CPUS]) for b in (20, 22)], "tail")


def cycles_item():
    raw_item("cycles", [(f"bits{b}", ["cycles", "--log-n", 15, "--k", 16, "--levels", 16, "--level-bits", b, "--seeds", SEEDS,
                                      "--cycles", 100, "--threads", CPUS]) for b in (20, 22)], "cycle")


def edges_item():
    scales = ",".join(map(str, EDGE_SCALES))
    configs = [(f"{u}-n{n}", ["leveled", "--log-n", n, "--k", 16, "--levels", 16, "--unit", u, "--scales", scales,
                              "--pipelines", "shaped", "--chains", "square", "--seeds", SEEDS, "--families", "unit,disk",
                              "--threads", CPUS])
               for n, units in ((15, UNITS_15), (16, UNITS_16)) for u in units]
    raw_item("edges", configs, "leveled")


def convert_item():
    raw_item("convert", [(f"k{k}-t{t}", ["convert", "--log-n", 15, "--k", k, "--seeds", SEEDS, "--reps", 3, "--threads", t])
                         for k in (16, 64, 256) for t in (1, CPUS)], "convert")


def circuit_item():
    raw_item("circuit", [("all", ["circuit", "--log-n", 15, "--k", 16, "--scales", "20,24", "--variants", "as_is,realify,reset",
                                  "--fresh", "yes,no", "--seeds", SEEDS, "--threads", CPUS])], "circuit")


def baseline15_item():
    """Every sparse baseline candidate at N = 2^15 within the caps, with one-limb digits included:
    one row per candidate, the fastest that keeps 19 bits being the baseline of its cell."""
    seen = done("baseline15", lambda r: (r["k"], r["levels"], r.get("flags")))
    for k in (16, 64, 256, 1024):
        for lv in (1, 2, 4, 8):
            for factors, dsize, p in paper.sparse_candidates(15, k, lv):
                # Unfactored transforms at k = 1024 hold 1024 prepared diagonals and exceed the memory of the
                # machine: their timings would measure swapping, and the factored transforms are measured.
                if k == 1024 and factors == 1:
                    continue
                flags = ["--log-n", 15, "--k", k, "--levels", lv, "--factors", factors, "--dsize", dsize, "--k-aux", p, "--d2s-dsize", 1]
                tag = " ".join(map(str, flags))
                if (k, lv, tag) in seen:
                    continue
                d = paper.dry("sparse", flags)
                if d is None or not paper.within_caps(d):
                    continue
                row = {"pipeline": "sparse", "k": k, "levels": lv, "flags": tag}
                row.update(paper.measure("sparse", flags, lv))
                append("baseline15", [row])
                paper.report(row)


def paco56_item():
    """PaCo at 5 and 6 levels for 16 and 64 values, under the protocol of paper.py."""
    paper.cfg = dict(paper.cfg, paco_rows=[(16, 5), (16, 6), (64, 5), (64, 6)])
    paper.paco_item()


def fmt(x, digits=2):
    return "--" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def quantile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def tables_item():
    out = ["# Additional experiments (paper protocol)", ""]
    for name, title in (("levels22", "Bootstrapping at 22-bit levels, N = 2^15"),
                        ("n14", "Bootstrapping at N = 2^14 with 22-bit limbs")):
        rows = paper.read(name)
        if not rows:
            continue
        out += [f"## {title}", "", "| log2 N | level bits | k | L | log2(q_key p) | cap | Min. bits | Avg. bits | 1 thread | all cores | GiB |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r.get("log_n", 0), r.get("level_bits", 0), r["k"], r["levels"])):
            out.append(f"| {r.get('log_n')} | {r.get('level_bits')} | {r['k']} | {r['levels']} | {r.get('log2_qp')} | {r.get('cap')} | "
                       f"{fmt(r.get('min_bits'), 1)} | {fmt(r.get('avg_bits'), 1)} | {fmt(r.get('refresh_s_st'), 3)} | "
                       f"{fmt(r.get('refresh_s_mt'), 3)} | {fmt(r.get('peak_gib_st'), 1)} |" + ("" if r.get("status") == "ok" else f" {r.get('status')}"))
        out.append("")
    tail = paper.read("tail")
    if tail:
        out += ["## Guard tail: bootstrappings followed by 16 squarings, k = 16, L = 16", "",
                "| level bits | runs | min. bits after bootstrapping | guard after bootstrapping, median / 99% / max (log2) | guard over the squarings, median / 99% / max (log2) |",
                "|---|---|---|---|---|"]
        for bits in sorted({l["level_bits"] for l in tail}):
            ls = [l for l in tail if l["level_bits"] == bits]
            g = [l["guard_refresh_log2"] for l in ls]
            s = [l["guard_squarings_log2"] for l in ls]
            out.append(f"| {bits} | {len(ls)} | {fmt(min(l['refresh_min_bits'] for l in ls))} | "
                       f"{fmt(statistics.median(g))} / {fmt(quantile(g, 0.99))} / {fmt(max(g))} | "
                       f"{fmt(statistics.median(s))} / {fmt(quantile(s, 0.99))} / {fmt(max(s))} |")
        out.append("")
    cycles = paper.read("cycles")
    if cycles:
        out += ["## Repeated bootstrapping: 100 rounds of a bootstrapping and 16 products, 5 seeds", "",
                "| level bits | round | min. bits after bootstrapping | min. bits after the products | largest guard (log2) |", "|---|---|---|---|---|"]
        for bits in sorted({l["level_bits"] for l in cycles}):
            for rnd in (0, 1, 9, 49, 99):
                ls = [l for l in cycles if l["level_bits"] == bits and l["round"] == rnd]
                if ls:
                    out.append(f"| {bits} | {rnd + 1} | {fmt(min(l['refresh_min_bits'] for l in ls))} | {fmt(min(l['levels_min_bits'] for l in ls))} | "
                               f"{fmt(max(l['guard_max_log2'] for l in ls))} |")
        out.append("")
    edges = paper.read("edges")
    if edges:
        out += ["## Stability edges (Table 6): 16 squarings, 5 seeds, both input families", "",
                "Largest guard (log2) per scale; the edge is the smallest scale whose largest guard, and that of every larger scale, stays below one. "
                "The model is log2(4 (E_guard + E_data)) with the one-step guard error at 28 bits and the data error at the edge, in phase units.", "",
                "| log2 N | unit | -log2 m_S | log2 M | model | edge | guard at 19 / 20 / 21 / 22 |", "|---|---|---|---|---|---|---|"]
        for cfg_tag in dict.fromkeys(l["config"] for l in edges):
            ls = [l for l in edges if l["config"] == cfg_tag]
            by = {}
            for l in ls:
                by.setdefault(l["level_bits"], []).append(l)
            gmax = {s: max((l["guard_max_log2"] if l["guard_max_log2"] is not None else 99) for l in v) for s, v in by.items()}
            scales = sorted(gmax)
            edge = next((s for s in scales if all(gmax[t] < 0 for t in scales if t >= s)), None)
            model = None
            if edge is not None and 28 in by:
                eg = max(l["guard_added_log2_max"] for l in by[28]) + 28
                ed = max(l["added_log2_max"] for l in by[edge]) + edge
                model = math.log2(4 * (2 ** eg + 2 ** ed))
            l0 = ls[0]
            out.append(f"| {l0['log_n']} | {l0['unit']} | {fmt(-l0['unit_data_log2'], 1)} | {fmt(l0['unit_guard_log2'])} | {fmt(model)} | {edge} | "
                       + " / ".join(fmt(gmax.get(s)) for s in (19, 20, 21, 22)) + " |")
        out.append("")
    conv = paper.read("convert")
    if conv:
        out += ["## Conversions, CKKS at 2^40 to the replicated layout at 2^20 and back, N = 2^15", "",
                "| k | threads | CKKS min. bits in | to shaped (s) | shaped min. bits | to CKKS (s) | CKKS min. bits out |", "|---|---|---|---|---|---|---|"]
        for tag in dict.fromkeys(l["config"] for l in conv):
            ls = [l for l in conv if l["config"] == tag]
            l0 = ls[0]
            out.append(f"| {l0['k']} | {l0['threads']} | {fmt(min(l['ckks_in_min_bits'] for l in ls), 1)} | {fmt(statistics.median(l['to_shaped_s'] for l in ls), 4)} | "
                       f"{fmt(min(l['shaped_min_bits'] for l in ls), 1)} | {fmt(statistics.median(l['to_ckks_s'] for l in ls), 4)} | "
                       f"{fmt(min(l['ckks_out_min_bits'] for l in ls), 1)} |")
        out.append("")
    circ = paper.read("circuit")
    if circ:
        out += ["## Activations: four tanh(2x) / tanh(2), real data, 5 seeds", "",
                "Largest guard (log2) after each activation, and minimum data precision after the last.", "",
                "| scale | variant | fresh product | guard after 1 / 2 / 3 / 4 | min. bits |", "|---|---|---|---|---|"]
        keys = sorted({(l["level_bits"], l["variant"], l["fresh"]) for l in circ})
        for bits, variant, fresh in keys:
            ls = [l for l in circ if (l["level_bits"], l["variant"], l["fresh"]) == (bits, variant, fresh) and l["stage"] == "activation"]
            g = [fmt(max(l["guard_log2"] for l in ls if l["step"] == s)) for s in (1, 2, 3, 4)]
            last = [l for l in ls if l["step"] == 4]
            out.append(f"| {bits} | {variant} | {fresh} | {' / '.join(g)} | {fmt(min(l['min_bits'] for l in last), 1)} |")
        out.append("")
    os.makedirs(paper.OUT, exist_ok=True)
    with open(os.path.join(paper.OUT, "EXTRA.md"), "w") as f:
        f.write("\n".join(out))
    print(f"wrote {os.path.join(paper.OUT, 'EXTRA.md')}", flush=True)


ITEMS = {"baseline15": baseline15_item, "paco56": paco56_item, "levels22": levels22_item, "n14": n14_item, "tail": tail_item, "cycles": cycles_item, "edges": edges_item,
         "convert": convert_item, "circuit": circuit_item, "tables": tables_item}

if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if args[:1] == ["--out"]:
        out, args = args[1], args[2:]
    if not args or any(a not in ITEMS and a != "all" for a in args):
        sys.exit(__doc__)
    paper.MODE = "paper"
    paper.cfg = paper.MODES["paper"]
    paper.OUT = out or os.path.join(paper.ROOT, "bench", "results", "extra")
    assert os.path.abspath(paper.OUT) != os.path.abspath(os.path.join(paper.ROOT, "bench", "results", "paper")), \
        "extra.py never writes to the results of paper.py"
    import subprocess
    subprocess.run(["cargo", "build", "--release", "-p", "poulpy-shaped-bench"], cwd=paper.ROOT, check=True)
    for item in (list(ITEMS) if args == ["all"] else args):
        t0 = time.time()
        ITEMS[item]()
        print(f"== {item} finished in {round(time.time() - t0)} s", flush=True)
