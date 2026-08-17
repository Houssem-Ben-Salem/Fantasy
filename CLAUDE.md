# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous Fantasy Premier League analyst: data pipeline → SQLite → expected-points
model → MILP squad optimiser → MCP server exposing 12 tools to Claude. Python 3.10+
(CI pins 3.12). No packaging config — run everything as `python -m fplbrain.<module>`
from the repo root.

## Commands

```bash
make install                    # pip install -r requirements.txt
make ingest                     # ETL 3 seasons -> data/fpl.db (~30s). REQUIRED before anything else.
make refresh                    # ingest + fit models + persist projections; exits non-zero on validation failure
make squad                      # print an optimal 15 (inline python one-liner)
make serve                      # run the MCP server on stdio
make test                       # pytest tests/ -q
make backtest                   # walk-forward eval of 2025-26
make clean                      # rm -rf data/cache data/fpl.db
```

Single test: `python -m pytest tests/test_constraints.py::test_budget -q`

**`tests/test_constraints.py` is not hermetic** — it connects to a literal `data/fpl.db`
and refits the full model in a module-scoped fixture, so run `make ingest` first or
every test in it errors. `tests/test_sources_season_gating.py` is self-contained
(monkeypatched HTTP, temp dirs) and runs anywhere. 21 tests total.

Underlying CLIs take more flags than the Makefile uses:

```bash
python -m fplbrain.ingest   --db data/fpl.db --current 2026-27 --history 2024-25 2025-26 [--no-strict]
python scripts/refresh.py   --db data/fpl.db --season 2026-27 --horizon 6 --start-gw 0   # 0 = next unfinished GW
python -m fplbrain.backtest --db data/fpl.db --season 2025-26 --history 2024-25 --start-gw 8 --end-gw 38 --out report.json
python -m fplbrain.sources  # connectivity probe — run this first in any new environment
```

Environment variables: `FPLBRAIN_DB` (default `data/fpl.db`), `FPLBRAIN_SEASON`
(`2026-27`), `FPLBRAIN_CACHE` (`./data/cache`), `FPLBRAIN_ALLOW_LIVE` (`1`; set to `0`
to skip live-API attempts entirely — faster in a sandbox where they will 403).
`FPLBRAIN_SEASON` also drives `sources.CURRENT_SEASON`, which decides which season is
allowed to use the live API; if it disagrees with `ingest --current`, the live path is
skipped and everything falls back to the mirror, which is safe but staler.

## Architecture

```
sources.py     fetch with a provenance-tracked fallback chain
ingest.py      ETL + validation + cross-season crosswalk -> SQLite (schema.sql)
models.py      Dixon-Coles goals | minutes | DefCon | attacking rates -> xP
optimize.py    PuLP/CBC MILP: initial squad, transfer plan, chip valuation
backtest.py    walk-forward eval with automatic leakage detection
mcp_server.py  12 tools; imports models + optimize, never sources directly
```

`scripts/refresh.py` is the orchestrator (ingest → project → persist → exit code).

`.github/workflows/refresh.yml` runs it daily plus Friday deadline day. **CI is the only
place the live path routinely executes** — GitHub runners can reach
`fantasy.premierleague.com` while most agent sandboxes get a 403, so a number produced
locally on the mirror is not automatically the number CI produces (current-season player
counts and the crosswalk rate both differ; see the README table). The workflow commits
`exports/` only — the SQLite file goes to an artifact, never to git, because it is
12 MB of binary that git cannot delta-compress and it rebuilds in ~30s from `make
ingest`. Don't reintroduce a `git add data/fpl.db`. The commit step is gated to the
Friday deadline run and manual dispatches; daily runs still refresh, test, and upload
the artifact, they just don't version output nobody acts on.

**A mirror fallback in CI is annotated, never fatal.** The probe step warns and writes
the resolved path to `$GITHUB_STEP_SUMMARY` when `fpl_api.ok` is false, but only fails
when the *mirror* is also unreachable — at that point there is no source at all. Failing
on a 403 would cost fresh projections on a deadline Friday over a provenance question.
Because the path moves the headline numbers, every export carries
`exports/run_metadata.json` with `resolved_path`, player count, crosswalk rate, and
source path per season, and `status()` reports the same to the agent — so a number that
moves between runs explains itself from the artefact rather than from a CI log that may
have aged out.

### The fallback chain (`sources.py`)

`fantasy.premierleague.com` is 403 from most agent sandboxes and CI egress allowlists;
`raw.githubusercontent.com` is not. Every fetch tries **live API → vaastav GitHub mirror
→ alt mirror → local cache** and returns a `Fetched` dataclass carrying `.path`
("live"/"github"/"cache") so downstream code can report staleness. Keep this shape when
adding a source — don't add a fetch that can only succeed on the live path.

**The live API is current-season-only, and every live call is gated on
`_is_current(season)`.** `bootstrap-static` and `/fixtures/` take no season parameter
and have no historical mode; calling them while loading 2024-25 returns current-season
rows that then get written stamped with the wrong season. History must always come from
the mirror, which is season-addressable. This was a real bug that failed *upward* — the
crosswalk match rate rose to a perfect 1.0 because the current player list was matching
itself, which looks better than the correct value. `tests/test_sources_season_gating.py`
monkeypatches a reachable API and asserts `bootstrap-static` is never requested for a
historical season; keep that test passing.

Cache writes record their true origin in a `.provenance` sidecar next to the CSV, so a
mirror-sourced file can never be reported as `live` on a later read. Use `_write_cache`
with its `path_label` rather than writing cache files directly.

### Database contract (`schema.sql`)

Designed so the agent can answer most questions with one readable SQL query: one wide
fact table (`player_gw`), small dimensions (`players`, `teams`, `fixtures`), plus four
views (`v_season_totals`, `v_defcon_rate`, `v_form6`, `v_latest_projections`).

- **`element` is season-scoped and reassigned every season. `code` is stable — join
  cross-season on `code`, never `element`.** `merged_gw.csv` lacks `code`, so ingest
  backfills it from that season's `players_raw.csv`.
- `schema.sql` has two halves, and the split is load-bearing. `teams`, `players`,
  `fixtures`, and `player_gw` are `DROP TABLE IF EXISTS` — derived data that ingest
  fully rebuilds. `projections`, `runs`, and `squad_state` are `CREATE TABLE IF NOT
  EXISTS` and must never be dropped: `squad_state` is user data, and `projections`
  accumulates across the season so you can audit the model for drift. `init_schema()`
  runs on every ingest, so moving a table across that line silently destroys data.
- `v_latest_projections` resolves its run by joining **through `projections`**, so it can
  only ever land on a run that actually has projection rows. Do not "simplify" it back to
  `(SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1)`: `runs` holds both
  `ingest-*` and `proj-*` rows, and an ingest run has no projections, so that form
  silently empties the view whenever an ingest lands last — which was every call to
  `refresh_data()`. Measured 3,540 rows → 0. The join is also deliberately not a filter
  on `notes='projection'`, because that label is a string set in one place and can drift.
  Costs ~285 ms against 1.29M projection rows (a simulated season of daily runs), so the
  subquery is not worth optimising. `tests/test_projection_freshness.py` guards all of it.

## Data traps encoded in the code — don't "fix" them

These are documented in module docstrings; the same three recur across `models.py`,
`backtest.py`, and `ingest.py`:

1. **No defensive stats before 2025-26.** `tackles`, `recoveries`, and
   `clearances_blocks_interceptions` are 100% NULL for all of 2024-25 (the DefCon rule
   didn't exist). Pooling that season with `fillna(0)` halves every DefCon estimate.
   `defcon_model()` filters on `defensive_contribution IS NOT NULL` rather than trusting
   the caller's season list — keep that filter. `validate()` reports this absence under
   `data_availability`, **not** `anomalies`: a column that is 100% null is absent by
   design, and filing it as a fault made the agent open every session reporting corrupt
   data on correct data. Partial nulls (`0 < nulls < total`) *are* an anomaly — that
   means a truncated load. The case is derived from the null count against the row count,
   never from a hardcoded season, so a new stat in a future season classifies itself.
2. **`defensive_contribution` is an action count, not points.** It is already the
   position-correct metric (CBIT for DEF, CBIRT for MID/FWD). Threshold it at
   `models.DEFCON_THRESHOLD` (10 DEF / 12 MID / 12 FWD); a non-zero value does not mean
   the player scored the +2.
3. **`merged_gw.csv` contains genuine duplicate rows.** Ingest dedupes on the PK and
   records the count in `_DUPES`, surfaced by `validate()`. Don't switch to a plain
   `to_sql` — it fails on the primary key.

**`ingest.check_season_distinctness()` is an assertion, not a report.** It fails if two
seasons have identical or >97%-overlapping player sets, or if a season with `player_gw`
rows has zero finished fixtures. `run(strict=True)` — the default — raises rather than
returning a corrupted database; `--no-strict` downgrades it to a log line. It keys off
`player_gw` deliberately, because a season that hasn't kicked off legitimately has no
finished fixtures and must not be flagged.

**FPL's own projections leak the future.** `players.ep_next` and `player_gw.fpl_xp` are
benchmark-only, never model inputs. `backtest.detect_leakage()` rejects any predictor
that is either sparsely populated (<50% of gameweeks) or assigns exact zeros to
non-playing players at a rate far above playing ones, and `report()` marks it EXCLUDED.
New benchmarks get the same scrutiny automatically.

**The backtest must stay strictly no-look-ahead.** `_*_as_of()` helpers rebuild every
feature from rows before the target gameweek. They deliberately skip `players.status`
and the `players` per-90 columns — those are end-of-season snapshots. That makes the
backtest slightly pessimistic vs the live model, which is the correct direction.

## Modelling conventions (`models.py`)

Four components feed one `exp_points`: goals (Dixon-Coles with time decay,
ξ=0.0018/day), minutes, DefCon, attacking returns. Everything is shrunk toward
positional priors — per-90 rates on small samples are the classic path to a bad
transfer. `PRIOR_STRENGTH = 450.0` minutes for attacking rates; a 12-start beta prior
for DefCon, keyed on the player's **current** position (11 players were reclassified for
2026/27, so old raw numbers mislead).

Scoring constants (`GOAL_POINTS`, `CS_POINTS`, `DEFCON_POINTS`, `DEFCON_THRESHOLD`) live
at the top of `models.py` and are the single source of truth — `backtest.py` imports
`DEFCON_THRESHOLD` from there rather than redefining it.

Minutes is the weakest component and everything downstream inherits its error. Players
with no history get `p_start = 0.35`, which is a guess.

## Optimiser conventions (`optimize.py`)

Constants at the top encode the FPL rules (`SQUAD_LIMITS`, `XI_MIN`/`XI_MAX`,
`MAX_PER_CLUB`, `HIT_COST`). Both entry points share `_prep()`, which time-decays the
horizon at `0.86^n` so the solver doesn't over-commit to GW+5 projections.

Two deliberate non-obvious defaults: `bench_weight=0.12` (bench counts at 12% of xP —
setting it to 0 produces the fragile 4.0m non-playing bench) and `min_bench_start=0.30`
(every squad member must be a usable substitute). `transfer_plan()` prices each transfer
beyond the free ones at −4 inside the objective and lets the solver decide, so "is this
hit worth it" is answered against the true alternative of rolling.

## MCP server conventions (`mcp_server.py`)

Rules any new tool must follow:

1. **Never return a raw API payload.** bootstrap-static is 1–3 MB and will eat the
   context window. Aggregate first, return a small markdown table via `_md()`.
2. Every tool caps its own row count; `_md()` is honest about truncation.
3. `run_sql` is SELECT/WITH-only with a keyword blocklist and an injected LIMIT.
4. Return provenance where relevant so the agent can state how stale an answer is.

Projections are cached in-process in `_PROJ`, keyed on `(start_gw, n_gw)`;
`refresh_data()` invalidates it. **`refresh_data()` must ingest *and* project *and*
persist** — ingest alone leaves persisted projections stale while `_PROJ` is fresh, so
the view-backed tools (`run_sql`) and the in-process tools (`top_players`) disagree, and
only the former looks broken. It takes ~23s, which is acceptable interactively. The server imports `FastMCP`/`MCPServer` under both SDK
1.x and 2.0 names — preserve that try/except when touching the import.

## Agent contract

`prompts/AGENT.md` is the operating instruction pasted into Claude Projects alongside
the MCP server: call `status()` first, never state a number that wasn't just retrieved,
separate Data / Model / Judgement, and the chip and transfer thresholds (e.g. recommend
a −4 hit only above +6 xP horizon gain, not +4). If you change tool names, signatures,
or the numbers those policies key on, update that file in the same change.
