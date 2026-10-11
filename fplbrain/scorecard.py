"""
PROSPECTIVE SCORECARD — what the system said before each deadline, against
what happened. Run it every week; it is the only test this project has that no
choice of ours could have been tuned to.

Every backtest and validation harness here re-fits the model "as of" a past
date. That is careful, but it is still us choosing what to re-fit. This scores
the projections that were actually PERSISTED before each deadline — by the
local refreshes (`projections` table) and by CI's Friday exports (git history
of exports/projections_latest.csv) — so the record cannot be improved after
the fact.

WHAT IT ANSWERS, AND WHAT IT CANNOT YET
--------------------------------------
GW5 2026-27, the first week both the old and the fixed model committed before
kickoff, gave a split verdict: the fixed model beat the naive baseline overall
(Spearman 0.738 vs 0.689), but among players who actually started it was no
better (+0.009, [-0.088, +0.097]). The whole edge was predicting who plays.
Whether the POINTS model beats naive needs many weeks pooled; this script pools
them and says how far the interval still is from deciding.

For each finished gameweek and each source it takes the LAST run created
before that gameweek's deadline (first kickoff - 90 min) that covers it, and
scores, on the same players:

  spearman_all     rank correlation with actual points, every listed player
  spearman_start   the same among players who actually started — the points
                   model alone, with the minutes question removed
  rmse, top50      error, and mean actual points of the 50 highest projected
  brier            p_start against whether the player started

against a naive baseline: season-to-date points per gameweek and start rate
(previous season's for GW1). Pooled differences use a bootstrap over
(team, gameweek) clusters: teammates share outcomes, so player-rows are not
independent.

MODEL LABELS. Runs record code_version and constants since 11 Oct 2026; older
ones are labelled by date against when each change landed (UTC author times):
minutes fix 2026-09-15T18:33, ridge 18:42, fixture-scaled attack
2026-10-11T13:27. CI's exports came from the remote, which had none of these
fixes until 11 Oct 2026, so CI runs before then are "pre-fix" whatever their
date.

    python -m fplbrain.scorecard [--no-ci]      (also the MCP tool model_scorecard)
"""
from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "fpl.db"
LEAD = timedelta(minutes=90)
FIXED_MINUTES = datetime(2026, 9, 15, 18, 33, tzinfo=timezone.utc)
FIXTURE_ATTACK = datetime(2026, 10, 11, 13, 27, tzinfo=timezone.utc)
CI_FIXED_FROM = datetime(2026, 10, 11, 14, 30, tzinfo=timezone.utc)   # first push of the fixes
BOUNDARY = timedelta(minutes=15)
N_BOOT = 4000


def _ts(s: str) -> datetime:
    return pd.Timestamp(s).to_pydatetime().astimezone(timezone.utc)


def label(created: datetime, prov: dict, source: str) -> str:
    if prov.get("code_version"):
        alpha = prov.get("constants", {}).get("ATTACK_FIXTURE_ALPHA", 0)
        return f"fixed{' + fixture attack' if alpha else ''} @{prov['code_version'][:7]}"
    if source == "ci" and created < CI_FIXED_FROM:
        return "pre-fix (CI)"
    near = any(abs(created - b) < BOUNDARY for b in (FIXED_MINUTES, FIXTURE_ATTACK))
    name = ("pre-fix" if created < FIXED_MINUTES else
            "fixed, fixture-blind" if created < FIXTURE_ATTACK else "fixed + fixture attack")
    return name + (" (near a change: uncertain)" if near else "")


def deadlines(conn, season: str) -> dict[int, datetime]:
    k = pd.read_sql("SELECT gw, MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw IS "
                    "NOT NULL GROUP BY gw", conn, params=(season,))
    return {int(g): _ts(t) - LEAD for g, t in zip(k.gw, k.k)}


def finished_gws(conn, season: str) -> list[int]:
    f = pd.read_sql("SELECT gw, MIN(finished) done FROM fixtures WHERE season=? AND gw IS NOT "
                    "NULL GROUP BY gw", conn, params=(season,))
    have = set(pd.read_sql("SELECT DISTINCT gw FROM player_gw WHERE season=?", conn,
                           params=(season,)).gw)
    return sorted(int(g) for g, d in zip(f.gw, f.done) if d == 1 and g in have)


def local_runs(conn, season: str) -> pd.DataFrame:
    r = pd.read_sql("SELECT run_id, created_at, horizon_start, horizon_end, provenance FROM runs "
                    "WHERE run_id IN (SELECT DISTINCT run_id FROM projections) AND season=?",
                    conn, params=(season,))
    r["created"] = r.created_at.map(_ts)
    r["prov"] = r.provenance.map(lambda p: json.loads(p) if p else {})
    r["source"] = "local"
    return r


def ci_runs() -> pd.DataFrame:
    """Every CI export in git history, as (commit, metadata)."""
    try:
        shas = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "origin/main",
                               "--", "exports/projections_latest.csv"],
                              capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return pd.DataFrame()
    rows = []
    for sha in shas:
        meta = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:exports/run_metadata.json"],
                              capture_output=True, text=True).stdout
        if not meta:
            continue
        m = json.loads(meta)
        rows.append(dict(run_id=m["run_id"], created=_ts(m["generated_at_utc"]),
                         horizon_start=m["horizon"]["start"], horizon_end=m["horizon"]["end"],
                         prov={}, source="ci", sha=sha))
    return pd.DataFrame(rows)


def projections(conn, run: pd.Series, gw: int) -> pd.DataFrame:
    if run.source == "local":
        return pd.read_sql("SELECT element, exp_points xp, p_start ps FROM projections "
                           "WHERE run_id=? AND gw=?", conn, params=(run.run_id, gw))
    csv = subprocess.run(["git", "-C", str(ROOT), "show", f"{run.sha}:exports/projections_latest.csv"],
                         capture_output=True, text=True).stdout
    d = pd.read_csv(io.StringIO(csv))
    return d[d.gw == gw].rename(columns={"exp_points": "xp", "p_start": "ps"})[["element", "xp", "ps"]]


def actuals_and_naive(conn, season: str, gw: int) -> pd.DataFrame:
    a = pd.read_sql("""SELECT g.element, g.code, SUM(g.total_points) pts, MAX(g.starts) st,
                              MIN(CASE WHEN g.was_home=1 THEN f.home_team ELSE f.away_team END) team
                       FROM player_gw g JOIN fixtures f ON f.season=g.season AND f.fixture_id=g.fixture_id
                       WHERE g.season=? AND g.gw=? GROUP BY g.element, g.code""",
                    conn, params=(season, gw))
    if gw > 1:
        n = pd.read_sql("SELECT element, SUM(total_points)*1.0/? naive_xp, SUM(starts)*1.0/? naive_ps "
                        "FROM player_gw WHERE season=? AND gw<? GROUP BY element",
                        conn, params=(gw - 1, gw - 1, season, gw))
        a = a.merge(n, on="element", how="left")
    else:
        prev = conn.execute("SELECT MAX(season) FROM player_gw WHERE season < ?",
                            (season,)).fetchone()[0]
        n = pd.read_sql("SELECT code, AVG(total_points) naive_xp, AVG(starts) naive_ps FROM "
                        "player_gw WHERE season=? GROUP BY code", conn, params=(prev,))
        a = a.merge(n, on="code", how="left")
    return a.fillna({"naive_xp": 0.0, "naive_ps": 0.0})


def metrics(d: pd.DataFrame, xp: str, ps: str) -> dict:
    s = d[d.st >= 1]
    return dict(spearman_all=spearmanr(d[xp], d.pts).statistic,
                spearman_start=spearmanr(s[xp], s.pts).statistic if len(s) > 2 else np.nan,
                rmse=float(np.sqrt(((d[xp] - d.pts) ** 2).mean())),
                top50=d.nlargest(50, xp).pts.mean(),
                brier=float(((d[ps].clip(0, 1) - (d.st >= 1)) ** 2).mean()))


def score(conn, season: str, include_ci: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-gameweek table, pooled model-minus-naive table). Only runs created
    BEFORE a gameweek's deadline are eligible to be scored on it."""
    runs = local_runs(conn, season)
    if include_ci:
        runs = pd.concat([runs, ci_runs()], ignore_index=True)
    dl = deadlines(conn, season)
    rows, frames = [], []
    for gw in finished_gws(conn, season):
        act = actuals_and_naive(conn, season, gw)
        for source, pool in runs.groupby("source"):
            ok = pool[(pool.created < dl[gw]) & (pool.horizon_start <= gw) & (pool.horizon_end >= gw)]
            if ok.empty:
                continue
            run = ok.sort_values("created").iloc[-1]
            d = projections(conn, run, gw).merge(act, on="element")
            if d.empty:
                continue
            m, nv = metrics(d, "xp", "ps"), metrics(d, "naive_xp", "naive_ps")
            lab = label(run.created, run.prov, source)
            rows.append(dict(gw=gw, source=source, run=run.run_id, model=lab,
                             weeks_ahead=gw - int(run.horizon_start), n=len(d),
                             **m, **{f"naive_{k}": v for k, v in nv.items()}))
            frames.append(d.assign(gw=gw, source=source, model=lab))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    allf["family"] = allf.model.str.extract(r"^([a-z\- ,+]+)")[0].str.strip()
    rng = np.random.default_rng(17)
    pooled = []
    for (source, family), f in allf.groupby(["source", "family"]):
        cl = {k: g for k, g in f.groupby(["team", "gw"])}
        keys = list(cl)
        base_m, base_n = metrics(f, "xp", "ps"), metrics(f, "naive_xp", "naive_ps")
        boot = {k: [] for k in ("spearman_all", "spearman_start")}
        for _ in range(N_BOOT if len(keys) > 10 else 0):
            b = pd.concat([cl[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            m, nv = metrics(b, "xp", "ps"), metrics(b, "naive_xp", "naive_ps")
            for k in boot:
                boot[k].append(m[k] - nv[k])
        for k, v in boot.items():
            v = np.array(v)
            pooled.append(dict(source=source, model=family,
                               gws=",".join(str(int(g)) for g in sorted(f.gw.unique())),
                               metric=k, model_minus_naive=base_m[k] - base_n[k],
                               lo=np.percentile(v, 2.5) if len(v) else np.nan,
                               hi=np.percentile(v, 97.5) if len(v) else np.nan,
                               p_model_better=float(np.mean(v > 0)) if len(v) else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def main() -> None:
    import os
    season = os.environ.get("FPLBRAIN_SEASON", "2026-27")
    t, pooled = score(sqlite3.connect(os.environ.get("FPLBRAIN_DB", DB)), season,
                      include_ci="--no-ci" not in sys.argv)
    if t.empty:
        print("no finished gameweek has a pre-deadline run yet")
        return
    pd.set_option("display.width", 220)
    print(f"PROSPECTIVE SCORECARD {season} — last run before each deadline, scored on what happened\n")
    print(t[["gw", "source", "model", "weeks_ahead", "n", "spearman_all", "naive_spearman_all",
             "spearman_start", "naive_spearman_start", "brier", "naive_brier", "top50",
             "naive_top50"]].round(3).to_string(index=False))
    print("\nPOOLED, model minus naive — bootstrap over (team, gameweek) clusters:")
    print(pooled.round(3).to_string(index=False))
    print("\nRead spearman_start as the open question: until its interval excludes zero,\n"
          "the points model has not been shown to beat 'points so far'. Minutes carries\n"
          "spearman_all. One gameweek of starters is mostly noise — let the weeks pool.")


if __name__ == "__main__":
    main()
