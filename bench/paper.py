#!/usr/bin/env python3
"""Runs the benchmarks of the paper and writes its tables.

Usage: python3 bench/paper.py [--mode fast|paper] [--decode full|average] [--out DIR] [--log-n N] ITEM...

Items, in the order `all` runs them:
  main     bootstrapping grid: our scheme and the sparse CKKS baseline (Tables 3, 4 and 5)
  paco     the PaCo comparison (Table 7)
  leveled  precision of 16 products by fresh operands against the scale (Figure 3)
  tables   the markdown tables, from the result files: DIR/TABLES.md

Modes:
  fast     one seed, unit-circle inputs, one repetition, one thread; k in {16, 64}, L in {1, 16};
           the first candidate of each baseline; leveled scales every 4 bits. Minutes.
  paper    the protocol of the paper: five seeds, unit-circle and unit-disk inputs, best of three,
           one thread and every core; k in {16, 64, 256, 1024}, L in {1, 2, 4, 8, 16, L_max}; every
           baseline candidate within the modulus caps, the fastest that keeps 19 bits; leveled scales
           every 2 bits. Hours, and up to about 50 GiB of memory for the largest baseline rows.

Every process that reports a time runs alone. Peak memory is the `ru_maxrss` of the process.
`--log-n` runs our scheme and the baselines at another ring degree, without the caps (a smoke test).
`--decode full` (the default, as in the paper) measures every copy of every value on the slots of the
ring; `--decode average` decodes one period, which averages the copies.
"""
import json, os, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target", "release")
GIB = float(1 << 30)

MODES = {
    "fast": {
        "seeds": "1", "families": "unit", "reps": 1, "threads": [1],
        "ks": [16, 64], "levels": [1, 16], "all_candidates": False,
        "paco_rows": [(16, 1)], "leveled_scales": list(range(16, 41, 4)), "leveled_seeds": "1", "leveled_families": "unit",
    },
    "paper": {
        "seeds": "1,2,3,4,5", "families": "unit,disk", "reps": 3, "threads": [1, os.cpu_count()],
        "ks": [16, 64, 256, 1024], "levels": [1, 2, 4, 8, 16, "max"], "all_candidates": True,
        "paco_rows": [(k, lv) for k in (16, 64) for lv in (1, 2, 4, 8, 16, 21)] + [(256, 1), (1024, 1)],
        "leveled_scales": list(range(16, 41, 2)), "leveled_seeds": "1,2,3,4,5", "leveled_families": "unit,disk",
    },
}

# Best (digit size, P) per (logN, L) of the sweep behind the paper, one-factor baseline transforms.
SPARSE_BEST = {
    15: {1: (1, 30), 2: (1, 53), 4: (2, 40)},
    16: {1: (1, 30), 2: (2, 120), 4: (1, 40), 8: (2, 120), 16: (2, 52), 21: (4, 224)},
}
# PaCo: (log q, log msg, bootstrapping-key scale, SlotsToCoeffs scale) and transform schedules.
PACO_SCALES = {"A": (46, 34, 48, 42), "B": (48, 36, 50, 44), "C": (52, 38, 54, 48)}
PACO_SCHEDULES = {16: (("3,4", "5"), ("7", "5")), 64: (("4,5", "7"), ("4,5", "3,4")), 256: (("4,5", "7"),), 1024: (("4,5", "7"),)}

cfg = {}
OUT = os.path.join(ROOT, "bench", "results")
LOG_N = None
DECODE = "full"


def run(binary, args):
    """Runs one benchmark process: its JSON lines, peak resident memory in bytes and status."""
    import threading
    cmd = [os.path.join(BIN, binary)] + [str(a) for a in args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = [], []
    readers = [threading.Thread(target=lambda: out.append(p.stdout.read())), threading.Thread(target=lambda: err.append(p.stderr.read()))]
    for t in readers:
        t.start()
    # wait4 reaps the child and returns its own resource usage; ru_maxrss is in bytes on macOS.
    _, status, usage = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    for t in readers:
        t.join()
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    lines = [json.loads(l) for l in out[0].splitlines() if l.startswith("{")]
    return {"lines": lines, "peak_rss_bytes": rss, "ok": p.returncode == 0, "stderr": err[0][-500:], "cmd": " ".join(cmd)}


def dry(binary, args):
    r = run(binary, list(args) + ["--dry-run"])
    return r["lines"][0] if r["ok"] and r["lines"] else None


def within_caps(d):
    if LOG_N is not None:
        return True
    ok = d["cap"] == 0 or d["log2_qp"] <= d["cap"]
    if "enc_cap" in d and d["enc_cap"]:
        ok = ok and d["enc_log2_qp"] <= d["enc_cap"]
    return ok


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def read(name):
    path = os.path.join(OUT, f"{name}.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def summarize(lines_by_threads, levels):
    """One table row from the lines of the runs, keyed by thread count."""
    res = {}
    for threads, lines in lines_by_threads.items():
        tag = "st" if threads == 1 else "mt"
        timed = [l for l in lines if l.get("stage") == "time" and l.get("family") == "unit"]
        if timed:
            res[f"refresh_s_{tag}"] = round(statistics.median(l["refresh_s"] for l in timed), 4)
            res[f"levels_s_{tag}"] = round(statistics.median(l["levels_s"] for l in timed), 4)
    checks = [l for lines in lines_by_threads.values() for l in lines if l.get("stage") == "check"]
    fresh = [l for l in checks if l["chain"] == "fresh"]
    square = [l for l in checks if l["chain"] == "square"]
    if fresh:
        res.update({
            "runs": len(fresh),
            "refresh_min_bits": round(min(l["refresh_min_bits"] for l in fresh), 2),
            "refresh_avg_bits": round(statistics.mean(l["refresh_avg_bits"] for l in fresh), 2),
            "min_bits": round(min(l["levels_min_bits"] for l in fresh), 2),
            "avg_bits": round(statistics.mean(l["levels_avg_bits"] for l in fresh), 2),
        })
        guards = [l["guard_max_log2"] for l in fresh if l.get("guard_max_log2") is not None]
        if guards:
            res["guard_log2_max"] = round(max(guards), 2)
    if square:
        guards = [l["guard_max_log2"] for l in square if l.get("guard_max_log2") is not None]
        if guards:
            res["guard_squarings_log2_max"] = round(max(guards), 2)
    return res


def measure(binary, flags, levels):
    """Runs a configuration under the protocol of the mode: a timed single-thread process, then, in
    paper mode, an all-core process that also checks the precision."""
    seeds, fams, reps = cfg["seeds"], cfg["families"], cfg["reps"]
    lines, row = {}, {}
    for threads in cfg["threads"]:
        single = threads == 1
        bench = "both" if (single and len(cfg["threads"]) == 1) or not single else "time"
        families = "unit" if bench == "time" else fams
        r = run(binary, flags + ["--decode", DECODE, "--threads", threads, "--seeds", seeds, "--families", families, "--reps", reps, "--bench", bench])
        if not r["ok"]:
            return {"status": "failed", "stderr": r["stderr"], "cmd": r["cmd"]}
        lines[threads] = r["lines"]
        row[f"peak_gib_{'st' if single else 'mt'}"] = round(r["peak_rss_bytes"] / GIB, 2)
        row[f"cmd_{'st' if single else 'mt'}"] = r["cmd"]
        setup = [l for l in r["lines"] if l["kind"] == "setup"][0]
        row.update({k: setup[k] for k in setup if k not in ("kind", "threads", "setup_s")})
    row.update(summarize(lines, levels))
    row["status"] = "ok" if row.get("min_bits", 0) >= 19 else "below_19_bits"
    return row


def ours_l_max(log_n):
    best = 0
    for lv in range(1, 40):
        d = dry("shaped", ["boot", "--log-n", log_n, "--k", 16, "--levels", lv])
        if d is None or not within_caps(d):
            break
        best = lv
    return best


def main_item():
    log_n = LOG_N or 15
    l_max = ours_l_max(log_n)
    levels = [l_max if lv == "max" else lv for lv in cfg["levels"]]
    print(f"L_max of our scheme at logN = {log_n}: {l_max}", flush=True)
    rows = read("main")
    done = {(r["pipeline"], r["k"], r["levels"]) for r in rows}
    for k in cfg["ks"]:
        for lv in levels:
            if ("ours", k, lv) in done:
                continue
            flags = ["boot", "--log-n", log_n, "--k", k, "--levels", lv]
            d = dry("shaped", flags)
            row = {"pipeline": "ours", "k": k, "levels": lv}
            if d is None or not within_caps(d):
                row["status"] = "over_cap"
            else:
                row.update(measure("shaped", flags, lv))
            rows.append(row)
            write("main", rows)
            report(row)
        for lv in levels:
            if ("sparse", k, lv) in done:
                continue
            row = sparse_row(k, lv)
            rows.append(row)
            write("main", rows)
            report(row)


def sparse_candidates(log_n, k, lv):
    """(transform factors, digit size, log2 P) of the baseline configurations tried at `log_n`, in order of preference."""
    factored = k >= 256
    if log_n == 15:
        # One-limb digits keep the key modulus smallest: they fit the cap with the most levels.
        cands = [(1, 1, 53), (1, 1, 30)]
        if lv in SPARSE_BEST[15]:
            cands.append((1,) + SPARSE_BEST[15][lv])
        if factored:
            cands = [(2, 1, 53), (2, 1, 30), (2, 2, 53)] + cands
    else:
        factors = 3 if factored else 1
        if not factored:
            cands = [(1,) + SPARSE_BEST.get(16, {}).get(lv, (2, 53))]
        elif lv == 1:
            cands = [(factors, 2, 120)]
        else:
            cands = [(factors, 4, 224), (factors, 3, 172), (factors, 2, 120)]
    return list(dict.fromkeys(cands))


def sparse_row(k, lv):
    """The fastest sparse configuration within the caps that keeps 19 bits, at the smallest ring
    degree that has one."""
    degrees = [LOG_N] if LOG_N else [15, 16]
    best = None
    for log_n in degrees:
        cands = sparse_candidates(log_n, k, lv)
        if not cfg["all_candidates"]:
            cands = cands[:1]
        for factors, dsize, p in cands:
            flags = ["--log-n", log_n, "--k", k, "--levels", lv, "--factors", factors, "--dsize", dsize, "--k-aux", p,
                     "--d2s-dsize", 1]
            d = dry("sparse", flags)
            if d is None or not within_caps(d):
                continue
            row = {"pipeline": "sparse", "k": k, "levels": lv}
            row.update(measure("sparse", flags, lv))
            if row["status"] == "ok" and (best is None or row.get("refresh_s_st", 1e9) < best.get("refresh_s_st", 1e9)):
                best = row
        if best:
            return best
    return {"pipeline": "sparse", "k": k, "levels": lv, "status": "no configuration within the caps keeps 19 bits"}


def paco_item():
    rows = read("paco")
    done = {(r["k"], r["levels"]) for r in rows}
    for k, lv in cfg["paco_rows"]:
        if (k, lv) in done:
            continue
        best = None
        for log_n in ([LOG_N] if LOG_N else [15, 16]):
            pool = []
            for scale in (("A", "B") if k <= 64 else ("B", "C")):
                for c2s, stc in PACO_SCHEDULES[k][: (2 if log_n == 15 else 1)]:
                    log_q, log_msg, bsk, stc_scale = PACO_SCALES[scale]
                    flags = ["--log-n", log_n, "--h", 64, "--k", k, "--log-q", log_q, "--log-msg", log_msg, "--bsk-scale", bsk,
                             "--stc-scale", stc_scale, "--c2s", c2s, "--stc", stc, "--levels", lv, "--dsize", 1 if lv < 16 else 2]
                    d = dry("paco", flags)
                    if d is not None and within_caps(d):
                        pool.append(flags)
            if not cfg["all_candidates"]:
                pool = pool[:1]
            for flags in pool:
                row = {"pipeline": "paco", "k": k, "levels": lv}
                row.update(measure("paco", flags, lv))
                if row["status"] == "ok" and (best is None or row.get("refresh_s_st", 1e9) < best.get("refresh_s_st", 1e9)):
                    best = row
            if best:
                break
        row = best or {"pipeline": "paco", "k": k, "levels": lv, "status": "no configuration within the caps keeps 19 bits"}
        rows.append(row)
        write("paco", rows)
        report(row)


def leveled_item():
    scales = ",".join(str(s) for s in cfg["leveled_scales"])
    r = run("shaped", ["leveled", "--log-n", LOG_N or 15, "--k", 16, "--levels", 16, "--scales", scales, "--pipelines", "shaped,ckks",
                       "--chains", "fresh", "--seeds", cfg["leveled_seeds"], "--families", cfg["leveled_families"], "--decode", DECODE])
    if not r["ok"]:
        print("leveled failed:", r["stderr"], flush=True)
        return
    rows = []
    groups = {}
    for l in r["lines"]:
        groups.setdefault((l["pipeline"], l["level_bits"]), []).append(l)
    for (pipeline, scale), ls in sorted(groups.items()):
        rows.append({"pipeline": pipeline, "level_bits": scale, "runs": len(ls),
                     "min_bits": round(min(l["min_bits"] for l in ls), 2), "avg_bits": round(statistics.mean(l["avg_bits"] for l in ls), 2),
                     "added_log2_max": round(max(l["added_log2_max"] for l in ls), 2), "guard_log2_max": round(max(l["guard_max_log2"] for l in ls), 2)})
        report(rows[-1])
    write("leveled", rows)


def report(row):
    keys = ("pipeline", "k", "levels", "level_bits", "log_n", "log2_qp", "status", "min_bits", "avg_bits", "refresh_s_st", "refresh_s_mt", "peak_gib_st",
            "added_log2_max", "guard_log2_max")
    print({k: row[k] for k in keys if k in row}, flush=True)


def fmt(x, digits=3):
    return "--" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def tables_item():
    lines = [f"# Benchmarks ({MODE} mode, {DECODE} decoding)", ""]
    main = read("main")
    if main:
        lines += ["## Bootstrapping (Tables 3 and 4)", "",
                  "Precision after L products by fresh operands; times in seconds; memory is the peak resident GiB of the single-thread process.", "",
                  "| k | L | Scheme | log2 N | log2(q_key p) | Min. bits | Avg. bits | 1 thread | all cores | GiB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(main, key=lambda r: (r["k"], r["levels"], r["pipeline"] != "ours")):
            if r.get("status") not in ("ok", "below_19_bits"):
                lines.append(f"| {r['k']} | {r['levels']} | {r['pipeline']} | | | | | {r.get('status')} | | |")
                continue
            lines.append(f"| {r['k']} | {r['levels']} | {r['pipeline']} | {r.get('log_n')} | {r.get('log2_qp')} | {fmt(r.get('min_bits'), 1)} | "
                         f"{fmt(r.get('avg_bits'), 1)} | {fmt(r.get('refresh_s_st'))} | {fmt(r.get('refresh_s_mt'))} | {fmt(r.get('peak_gib_st'), 1)} |")
        lines += ["", "## Speedups (Table 5)", "", "| k | L | Baseline log2 N | 1 thread | all cores |", "|---|---|---|---|---|"]
        by = {(r["pipeline"], r["k"], r["levels"]): r for r in main if r.get("status") == "ok"}
        for (p, k, lv), ours in sorted(by.items()):
            base = by.get(("sparse", k, lv))
            if p != "ours" or not base:
                continue
            ratio = lambda key: (base.get(key) / ours.get(key)) if base.get(key) and ours.get(key) else None
            lines.append(f"| {k} | {lv} | {base.get('log_n')} | {fmt(ratio('refresh_s_st'), 2)} | {fmt(ratio('refresh_s_mt'), 2)} |")
        lines.append("")
    paco = read("paco")
    if paco:
        ours = {(r["k"], r["levels"]): r for r in main if r["pipeline"] == "ours"}
        lines += ["## PaCo (Table 7)", "", "Each entry is PaCo / ours.", "",
                  "| k | L | log2 N | log2(q_key p) | Min. bits | 1 thread | all cores |", "|---|---|---|---|---|---|---|"]
        for r in sorted(paco, key=lambda r: (r["k"], r["levels"])):
            o = ours.get((r["k"], r["levels"]), {})
            lines.append(f"| {r['k']} | {r['levels']} | {r.get('log_n')} | {r.get('log2_qp')} | {fmt(r.get('min_bits'), 1)} / {fmt(o.get('min_bits'), 1)} | "
                         f"{fmt(r.get('refresh_s_st'))} / {fmt(o.get('refresh_s_st'))} | {fmt(r.get('refresh_s_mt'))} / {fmt(o.get('refresh_s_mt'))} |")
        lines.append("")
    leveled = read("leveled")
    if leveled:
        lines += ["## Leveled precision (Figure 3)", "",
                  "16 products by fresh operands, k = R = 16, U_32_45. Precision is the minimum over runs; the added error is the largest error one product adds.", "",
                  "| log2 Delta | Ours, min. bits | CKKS, min. bits | Ours, -log2 added | CKKS, -log2 added | Ours, guard log2 |", "|---|---|---|---|---|---|"]
        by = {(r["pipeline"], r["level_bits"]): r for r in leveled}
        for s in sorted({r["level_bits"] for r in leveled}):
            o, c = by.get(("shaped", s), {}), by.get(("ckks", s), {})
            neg = lambda x: -x if x is not None else None
            lines.append(f"| {s} | {fmt(o.get('min_bits'), 1)} | {fmt(c.get('min_bits'), 1)} | {fmt(neg(o.get('added_log2_max')), 1)} | "
                         f"{fmt(neg(c.get('added_log2_max')), 1)} | {fmt(o.get('guard_log2_max'), 2)} |")
        lines.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "TABLES.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.join(OUT, 'TABLES.md')}", flush=True)


ITEMS = {"main": main_item, "paco": paco_item, "leveled": leveled_item, "tables": tables_item}

if __name__ == "__main__":
    args = sys.argv[1:]
    MODE, out = "fast", None
    while args and args[0].startswith("--"):
        if args[0] == "--mode":
            MODE = args[1]
        elif args[0] == "--out":
            out = args[1]
        elif args[0] == "--log-n":
            LOG_N = int(args[1])
        elif args[0] == "--decode":
            DECODE = args[1]
        else:
            sys.exit(__doc__)
        args = args[2:]
    if MODE not in MODES or not args:
        sys.exit(__doc__)
    cfg = MODES[MODE]
    name = MODE + ("" if LOG_N is None else f"-n{LOG_N}") + ("" if DECODE == "full" else f"-{DECODE}")
    OUT = out or os.path.join(ROOT, "bench", "results", name)
    subprocess.run(["cargo", "build", "--release", "-p", "poulpy-shaped-bench"], cwd=ROOT, check=True)
    todo = list(ITEMS) if args == ["all"] else args
    for item in todo:
        t0 = time.time()
        ITEMS[item]()
        print(f"== {item} finished in {round(time.time() - t0)} s", flush=True)
