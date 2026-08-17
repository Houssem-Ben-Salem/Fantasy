#!/usr/bin/env python3
"""
One command to bring everything up to date. Run before each deadline.

    python scripts/refresh.py --db data/fpl.db --horizon 6

Re-pulls all sources, rebuilds the DB, refits the models, writes fresh
projections, and prints a validation summary. Exits non-zero on validation
failure so cron/CI can alert you rather than silently serving stale numbers.
"""
import argparse, json, logging, sqlite3, sys
from datetime import datetime, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from fplbrain import ingest, models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fpl.db")
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--history", nargs="*", default=["2024-25", "2025-26"])
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--start-gw", type=int, default=0, help="0 = next unfinished GW")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"[{datetime.now(timezone.utc).isoformat()}] refreshing {a.db}")

    result = ingest.run(a.db, a.season, a.history)

    failures = []
    for season, r in result["validation"].items():
        status = "OK" if r["rows"] > 0 else "EMPTY"
        if r["rows"] == 0:
            failures.append(f"{season}: no rows")
        print(f"  {season}: {status} rows={r['rows']} gws={r['n_gameweeks']}/38 "
              f"dupes_removed={r.get('source_duplicate_rows_removed', 0)}")

    conn = sqlite3.connect(a.db)
    start = a.start_gw or int(
        conn.execute("SELECT MIN(gw) FROM fixtures WHERE season=? AND finished=0 AND gw IS NOT NULL",
                     (a.season,)).fetchone()[0] or 1)
    print(f"  projecting GW{start}-{start + a.horizon - 1}")
    proj, diag = models.project_horizon(conn, a.season, start, a.horizon, a.history)
    run_id = models.persist_projections(conn, proj, a.season, diag)
    print(f"  run_id={run_id}  players={proj.element.nunique()}  rows={len(proj)}")
    print("  diagnostics:", json.dumps(diag))
    conn.close()

    if failures:
        print("VALIDATION FAILED:", "; ".join(failures), file=sys.stderr)
        return 1
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
