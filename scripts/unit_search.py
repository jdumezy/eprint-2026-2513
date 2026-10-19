"""Shaping units of Z[Y]/(Y^(2R) + 1): T bits of suppression on the data class and the smallest
largest guard magnitude.

The logarithmic profile of a unit U is (log2 |U(w^(5^j))|) for j < R, with w = exp(i pi / 2R):
class 0 holds the data slots, the others the guards. The cyclotomic units (1 - Y^a) / (1 - Y),
a odd, generate the units up to a finite index, and the profile of a product is the sum of the
profiles. For each target T the script solves, as a mixed-integer program over the exponents,

    minimize  max_{j >= 1} profile_j(U)   subject to   profile_0(U) <= -T,

with the exponents in [-bound, bound], expands U exactly and checks its profile in 400-bit
precision. The norm identity sum_j profile_j = 0 makes the flat profile T / (R - 1) a lower
bound on the largest guard.

    python3 scripts/unit_search.py TWO_R T1,T2,... [--bound B] [--time-limit S]

Needs numpy, scipy (for HiGHS) and mpmath. Prints one JSON object per target, with the
coefficients in `U`, ready for `poulpy_shaped::unit::catalog`.
"""

import argparse
import json
import math

import mpmath as mp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("two_r", type=int, help="degree 2R of the unit's ring, a power of two")
parser.add_argument("targets", help="comma-separated suppressions T, in bits")
parser.add_argument("--bound", type=int, default=40, help="largest absolute exponent")
parser.add_argument("--time-limit", type=float, default=300.0, help="seconds per target")
args = parser.parse_args()

two_r = args.two_r
R = two_r // 2
m = 2 * two_r  # Y is a primitive m-th root of unity
places = list(range(1, two_r, 2))  # one odd exponent b per conjugate pair; b = 1 is the data class
gens = list(range(3, two_r, 2))  # a = 3, 5, ..., 2R - 1


def log_profile(a):
    """Profile of (1 - Y^a) / (1 - Y): |sin(pi a b / m) / sin(pi b / m)| at Y = exp(i pi b / 2R)."""
    return [math.log2(abs(math.sin(math.pi * a * b / m) / math.sin(math.pi * b / m))) for b in places]


A = np.array([log_profile(a) for a in gens]).T  # R places x (R - 1) generators


def mul(p, q):
    """Product in Z[Y] / (Y^(2R) + 1)."""
    out = [0] * two_r
    for i, pi in enumerate(p):
        if pi:
            for j, qj in enumerate(q):
                if qj:
                    k, s = i + j, 1
                    if k >= two_r:
                        k, s = k - two_r, -1
                    out[k] += s * pi * qj
    return out


def generator(a, inverse):
    """(1 - Y^a) / (1 - Y) = sum_{j < a} Y^j, and its inverse sum_{j < a'} Y^(a j) with a a' = 1 mod m."""
    p = [0] * two_r
    if not inverse:
        for j in range(a):
            p[j] += 1
        return p
    for j in range(pow(a, -1, m)):
        k, s = (a * j) % m, 1
        if k >= two_r:
            k, s = k - two_r, -1
        p[k] += s
    return p


def expand(exps):
    u = [1] + [0] * (two_r - 1)
    for a, e in zip(gens, exps):
        f = generator(a, e < 0)
        for _ in range(abs(e)):
            u = mul(u, f)
    return u


mp.mp.prec = 400


def exact_profile(u):
    out = []
    for b in places:
        y = mp.e ** (1j * mp.pi * b / two_r)
        out.append(float(mp.log(abs(sum(c * y**i for i, c in enumerate(u))), 2)))
    return out


n = R - 1
for T in (float(t) for t in args.targets.split(",")):
    # Variables: the n exponents, then z, the largest guard. Minimize z.
    cost = np.zeros(n + 1)
    cost[-1] = 1.0
    rows = [np.append(A[0], 0.0)] + [np.append(A[j], -1.0) for j in range(1, R)]
    upper = [-T] + [0.0] * (R - 1)
    constraints = LinearConstraint(np.array(rows), np.full(R, -np.inf), upper)
    integrality = np.append(np.ones(n), 0)
    bounds = Bounds(np.append(np.full(n, -args.bound), -1e3), np.append(np.full(n, args.bound), 1e3))
    res = milp(cost, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": args.time_limit})
    if res.x is None:
        print(json.dumps({"two_r": two_r, "target": T, "status": res.message}), flush=True)
        continue
    exps = [int(round(v)) for v in res.x[:n]]
    u = expand(exps)
    profile = exact_profile(u)
    print(
        json.dumps(
            {
                "two_r": two_r,
                "target": T,
                "optimal": bool(res.status == 0),
                "data_log2": round(profile[0], 3),
                "guard_max_log2": round(max(profile[1:]), 3),
                "guard_min_log2": round(min(profile[1:]), 3),
                "flat_bound_log2": round(-profile[0] / (R - 1), 3),
                "l1": sum(abs(c) for c in u),
                "terms": sum(1 for c in u if c),
                "exponents": dict(zip(map(str, gens), exps)),
                "U": u,
            }
        ),
        flush=True,
    )
