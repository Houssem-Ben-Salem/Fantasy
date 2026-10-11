"""
Team news as evidence: claims about who will play, each with its source and the
moment it was recorded, so the news scout can be scored on what it said before
a deadline — and earn, or not, a place in the model.

WHY. LIMITATIONS s5: three remedies that transformed what the model already
knows all failed; information it cannot see — injuries, press conferences,
line-ups — is the input class that can still add value. Until October 2026 that
news was searched for at the deadline and discarded: never logged, so never
testable.

CLAIMS, and what each says about the gameweek it names:
  out        will not play                  scored on minutes == 0
  benched    will not start                 scored on started == 0
  starts     will start                     scored on started == 1
  doubtful   may miss (chance if stated)    not a start call; Brier on played, when a chance is given
  available  fit, nothing said on starting  not scored

THE BAR. Persistence ("starts iff he started last week") is free and already
right about most players. A source is worth something only where it DISAGREES
with persistence, so that is the headline score: on claims whose start call
differs from persistence's, how often was the claim right?

DECISION RULE, fixed before any data: the log may feed the model — zero p_start
on an 'out', raise it on a 'starts' — only once claims that disagree with
persistence are right with a Wilson 95% lower bound above 0.5, over at least
MIN_DISAGREEMENTS of them. Dropped if the upper bound falls below 0.5. Until
then claims inform the human-facing plan (injury scenarios) and nothing else.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CLAIMS = {"out": False, "benched": False, "starts": True, "doubtful": None, "available": None}
CONFIDENCE = ("confirmed", "rumoured")
SOURCE_TYPES = ("presser", "club", "journalist", "predicted_lineup", "fpl_flag", "newsletter",
                "other")
MIN_DISAGREEMENTS = 30
_COLS = ("season", "gw", "element", "player", "claim", "chance", "confidence", "source_type",
         "source_name", "source_url", "published_at", "quote", "recorded_at", "recorded_by")


class ClaimError(ValueError):
    """One or more claims failed validation; nothing was recorded."""


def _ddl() -> str:
    """The news_claims DDL, read from schema.sql so it is defined once. Never call
    ingest.init_schema() for this: it drops and rebuilds the derived tables."""
    sql = (Path(__file__).parent / "schema.sql").read_text()
    start = sql.index("CREATE TABLE IF NOT EXISTS news_claims")
    end = sql.index(";", sql.index("CREATE INDEX IF NOT EXISTS idx_news_claims_gw")) + 1
    return sql[start:end]


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(_ddl())


def validate(c: dict, i: int) -> list[str]:
    tag = f"claim {i + 1} ({c.get('player') or c.get('element') or '?'})"
    errs = []
    if c.get("claim") not in CLAIMS:
        errs.append(f"{tag}: claim must be one of {', '.join(CLAIMS)}, got {c.get('claim')!r}")
    if c.get("confidence") not in CONFIDENCE:
        errs.append(f"{tag}: confidence must be confirmed or rumoured, got {c.get('confidence')!r}")
    if c.get("source_type") not in SOURCE_TYPES:
        errs.append(f"{tag}: source_type must be one of {', '.join(SOURCE_TYPES)}")
    url = str(c.get("source_url") or "").strip()
    if c.get("source_type") != "fpl_flag" and not url.startswith(("http://", "https://")):
        errs.append(f"{tag}: a source URL is required (only an fpl_flag may omit it)")
    if c.get("chance") is not None:
        try:
            if not 0 <= float(c["chance"]) <= 100:
                raise ValueError
        except (TypeError, ValueError):
            errs.append(f"{tag}: chance must be a number from 0 to 100")
    if not isinstance(c.get("gw"), int) or c["gw"] < 1:
        errs.append(f"{tag}: gw must be a positive gameweek number")
    if c.get("published_at"):
        try:
            datetime.fromisoformat(str(c["published_at"]).replace("Z", "+00:00"))
        except ValueError:
            errs.append(f"{tag}: published_at must be an ISO date, got {c['published_at']!r}")
    if not isinstance(c.get("element"), int):
        errs.append(f"{tag}: player not resolved to an element id")
    return errs


def record(conn: sqlite3.Connection, season: str, claims: list[dict],
           recorded_by: str = "news-scout", now: datetime | None = None) -> list[int]:
    """All or nothing: any invalid claim raises ClaimError and records none."""
    errors = [e for i, c in enumerate(claims) for e in validate(c, i)]
    if errors:
        raise ClaimError(errors)
    ensure(conn)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    ids = []
    with conn:  # one transaction
        for c in claims:
            row = {**c, "season": season, "recorded_at": stamp, "recorded_by": recorded_by,
                   "chance": None if c.get("chance") is None else float(c["chance"])}
            cur = conn.execute(
                f"INSERT INTO news_claims ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
                [row.get(k) for k in _COLS])
            ids.append(cur.lastrowid)
    return ids


def _pick(d: pd.DataFrame) -> pd.DataFrame:
    """The claim that counts per player: the latest; among claims logged at the
    same moment, the most authoritative — confirmed over rumoured, then one with
    a stated chance (the only scoreable part of a 'doubtful'). Found on the first
    real run: a batch put Mykolenko's 75% FPL flag and a journalist's report on
    one timestamp, and the tie went to the rumour."""
    if d.empty:
        return d
    return (d.assign(_conf=(d.confidence == "confirmed").astype(int),
                     _chance=d.chance.notna().astype(int))
             .sort_values(["recorded_at", "_conf", "_chance", "claim_id"])
             .groupby("element").tail(1).drop(columns=["_conf", "_chance"]))


def latest(conn: sqlite3.Connection, season: str, gw: int) -> pd.DataFrame:
    """The claim that counts per player for a gameweek (see _pick)."""
    ensure(conn)
    d = pd.read_sql("SELECT * FROM news_claims WHERE season=? AND gw=?", conn, params=(season, gw))
    return _pick(d).sort_values("player")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, den = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(conn: sqlite3.Connection, season: str, deadlines: dict[int, datetime]) -> dict:
    """Score the latest claim per player recorded before each finished deadline."""
    ensure(conn)
    claims = pd.read_sql("SELECT * FROM news_claims WHERE season=?", conn, params=(season,))
    acts = pd.read_sql("SELECT gw, element, SUM(minutes) mins, MAX(starts) st FROM player_gw "
                       "WHERE season=? GROUP BY gw, element", conn, params=(season,))
    done = set(pd.read_sql("SELECT gw FROM fixtures WHERE season=? AND gw IS NOT NULL GROUP BY gw "
                           "HAVING MIN(finished)=1", conn, params=(season,)).gw) & set(acts.gw)
    rows = []
    for gw, dl in deadlines.items():
        if gw not in done:
            continue
        c = claims[(claims.gw == gw) & (claims.recorded_at < dl.isoformat(timespec="seconds"))]
        if c.empty:
            continue
        now_ = acts[acts.gw == gw].set_index("element")
        prev = acts[acts.gw == gw - 1].set_index("element")
        for r in _pick(c).itertuples():
            mins = float(now_.mins.get(r.element, 0) or 0)
            started = int(now_.st.get(r.element, 0) or 0) == 1
            persist = int(prev.st.get(r.element, 0) or 0) == 1
            pred = CLAIMS[r.claim]
            rows.append(dict(gw=gw, element=r.element, player=r.player, claim=r.claim,
                             chance=r.chance, pred=pred, persist=persist, started=started,
                             played=mins > 0,
                             correct=(mins == 0) if r.claim == "out"
                             else (None if pred is None else pred == started),
                             start_correct=None if pred is None else pred == started))
    d = pd.DataFrame(rows)
    out = {"rows": d, "by_claim": pd.DataFrame(), "disagree": dict(n=0, correct=0, lo=0.0, hi=1.0),
           "doubtful": dict(n=0, brier=None)}
    if d.empty:
        out["verdict"] = f"WAIT — 0 of {MIN_DISAGREEMENTS} disagreements scored."
        return out
    scored = d[d.correct.notna()]
    by = scored.groupby("claim").correct.agg(n="size", correct="sum").reset_index()
    # Built as a frame so an empty `by` works: a gameweek whose claims are all
    # doubtful/available makes no start call, and list-assignment crashed on it.
    ci = pd.DataFrame([wilson(int(k), int(n)) for k, n in zip(by.correct, by.n)],
                      columns=["lo", "hi"], index=by.index)
    by = pd.concat([by, ci], axis=1)
    by["hit"] = (by.correct / by.n) if len(by) else pd.Series(dtype=float)
    out["by_claim"] = by[["claim", "n", "correct", "hit", "lo", "hi"]]
    dis = d[d.pred.notna() & (d.pred != d.persist)]
    k, n = int(dis.start_correct.sum()), len(dis)
    lo, hi = wilson(k, n)
    out["disagree"] = dict(n=n, correct=k, lo=lo, hi=hi)
    db = d[(d.claim == "doubtful") & d.chance.notna()]
    if len(db):
        out["doubtful"] = dict(n=len(db), brier=float(((db.chance / 100 - db.played) ** 2).mean()))
    if n >= MIN_DISAGREEMENTS and lo > 0.5:
        out["verdict"] = "FEED — claims beat persistence where they disagree; they may now inform p_start."
    elif n >= MIN_DISAGREEMENTS and hi < 0.5:
        out["verdict"] = "DROP — claims are worse than persistence where they disagree."
    else:
        out["verdict"] = f"WAIT — {n} of {MIN_DISAGREEMENTS} disagreements scored."
    return out
