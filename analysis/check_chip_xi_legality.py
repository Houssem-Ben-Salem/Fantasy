#!/usr/bin/env python3
"""
GUARD — optimize.evaluate_chips picks a formation-ILLEGAL XI.

WHAT IS AND IS NOT AFFECTED
---------------------------
initial_squad() and transfer_plan() choose the formation correctly. The XI is a
free binary per player under XI_MIN/XI_MAX (DEF 3-5, MID 2-5, FWD 1-3, exactly
1 GK, 11 total), so all eight legal formations are reachable and the solver will
happily field a second forward in place of a defender if the points say so:

    3-4-3  3-5-2  4-3-3  4-4-2  4-5-1  5-2-3  5-3-2  5-4-1

evaluate_chips() does NOT. It picks

    ranked = in_squad.sort_values("exp_points", ascending=False)
    xi     = ranked.head(11)
    bench  = ranked.tail(4)

with no position constraints at all. That is the top 11 by points, which is not
generally a fieldable team.

MEASURED on a real 15 (2026-09-03, GW3-7 projections):

    GW3  {MID 5, DEF 3, FWD 2, GK 1}   legal
    GW4  {MID 5, DEF 3, FWD 2, GK 1}   legal
    GW5  {MID 5, DEF 4, FWD 2}         ILLEGAL — no goalkeeper
    GW6  {MID 5, DEF 4, FWD 2}         ILLEGAL — no goalkeeper
    GW7  {MID 5, DEF 4, FWD 2}         ILLEGAL — no goalkeeper

Three of five gameweeks fielded an XI with zero goalkeepers, because the
outfield eleven simply outscored the keeper. The keeper is then counted on the
BENCH, so both halves of the chip arithmetic are wrong at once:

    xi_projected / triple_captain context   overstated
    bench_projected / bench_boost_gain      contains a player who should start

WHY IT HAS NOT BITTEN YET
-------------------------
The error is small on a balanced squad — 0.02 to 0.56 xP here — because the
11th and 12th best players are close together. It is not bounded, though: a
premium goalkeeper, or a squad where XI_MIN binds (three playable defenders),
moves it. And bench_boost_gain is exactly the number the agent compares against
a ~15-point threshold, so a systematic error in it is an error in a chip
decision, not a rounding difference.

NOT FIXED HERE. Per the session ground rule fplbrain/ is unmodified. The fix is
to reuse the same legal-XI selection the MILP already implements rather than
head(11); it is a small change but it alters chip_advice output, and
LIMITATIONS.md is a record of changes that looked obviously right and failed
out-of-sample. This guard fails loudly until then.

SEPARATELY, AND MEASURED: the solver commits to ONE XI for the whole horizon
rather than re-picking weekly. Re-selecting the best legal XI each gameweek was
worth only +0.48 xP over GW3-7 on this squad, because 3-5-2 was optimal in all
five. Real limitation, currently negligible; re-run if the squad becomes more
fixture-diverse.

EXIT CODES
  0  every gameweek's chip XI is formation-legal
  1  at least one is illegal -> chip_advice numbers for that GW are unusable
  2  could not check
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
FORMS = [(d, m, f) for d in range(3, 6) for m in range(2, 6) for f in range(1, 4)
         if d + m + f == 10]


def legal(counts: dict) -> bool:
    return all(XI_MIN[p] <= counts.get(p, 0) <= XI_MAX[p] for p in XI_MIN) \
        and sum(counts.values()) == 11


def best_legal_xi(sub: pd.DataFrame):
    by = {p: g.sort_values("exp_points", ascending=False) for p, g in sub.groupby("position")}
    best, bp = None, -1.0
    if len(by.get("GK", [])) < 1:
        return None, 0.0
    for d, m, f in FORMS:
        if any(len(by.get(p, [])) < n for p, n in (("DEF", d), ("MID", m), ("FWD", f))):
            continue
        xi = pd.concat([by["GK"].head(1), by["DEF"].head(d), by["MID"].head(m), by["FWD"].head(f)])
        s = float(xi.exp_points.sum())
        if s > bp:
            best, bp = xi, s
    return best, bp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", help="comma-separated element ids (default: saved state)")
    ap.add_argument("--gws", nargs="*", type=int)
    a = ap.parse_args()

    import fplbrain.mcp_server as mcp
    if a.squad:
        squad = [int(x) for x in a.squad.split(",")]
    else:
        import sqlite3
        with sqlite3.connect(mcp.DB) as c:
            row = pd.read_sql("SELECT squad FROM squad_state ORDER BY gw DESC LIMIT 1", c)
        if row.empty:
            print("no saved squad_state and no --squad given", file=sys.stderr)
            return 2
        squad = [int(p["element"]) for p in json.loads(row.iloc[0]["squad"])]

    start = mcp._next_gw()
    gws = a.gws or list(range(start, start + 5))
    proj = mcp._projections(min(gws), len(gws))

    bad = 0
    print(f"squad of {len(squad)}   gameweeks {gws}\n")
    for gw in gws:
        sub = proj[(proj.gw == gw) & (proj.element.isin(squad))]
        if sub.empty:
            continue
        # Audit the PACKAGE, not a replica of its old logic: evaluate_chips must now
        # report the value of a fieldable team. A guard that re-implements the
        # thing it guards keeps passing after the real code regresses.
        from fplbrain import optimize as opt
        ev = opt.evaluate_chips(proj, squad, gw)
        pkg_xi = opt.best_legal_xi(sub)
        counts = pkg_xi.position.value_counts().to_dict()
        ok = legal(counts) and abs(ev["xi_projected"] - pkg_xi.exp_points.sum()) < 0.01
        naive = sub.nlargest(11, "exp_points")
        _, best = best_legal_xi(sub)
        flag = "legal" if ok else "ILLEGAL"
        why = "" if ok else "  <- " + ", ".join(
            f"{p}={counts.get(p,0)} (need {XI_MIN[p]}-{XI_MAX[p]})"
            for p in XI_MIN if not XI_MIN[p] <= counts.get(p, 0) <= XI_MAX[p])
        print(f"  GW{gw}: {counts}  {flag}{why}")
        print(f"        chip_advice XI {ev['xi_projected']:.2f} xP (package)  best legal {best:.2f}  "
              f"naive head(11) would have said {naive.exp_points.sum():.2f}")
        bad += (not ok)

    if bad:
        print(f"\n!! {bad} gameweek(s) with an unfieldable chip XI.")
        print("   xi_projected, bench_projected and bench_boost_gain are all affected,")
        print("   and bench_boost_gain is compared against a ~15-point threshold.")
        return 1
    print("\nevery chip XI is formation-legal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
