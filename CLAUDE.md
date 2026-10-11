# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous Fantasy Premier League analyst: data pipeline → SQLite → expected-points
model → MILP squad optimiser → MCP server exposing 13 tools to Claude. Python 3.10+
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
(monkeypatched HTTP, temp dirs) and runs anywhere, as do the other newer test files
(`test_minutes_model.py`, `test_goal_ridge.py`, `test_chip_xi.py`, `test_player_lookup.py`,
`test_projection_freshness.py`, `test_validation_reporting.py`, `test_solver.py`, `test_attack_fixture.py`, `test_transfer_pairs.py`, `test_entry.py`,
`test_mcp_state.py`, `test_run_provenance.py`, `test_scorecard.py`). 100 tests total.

Underlying CLIs take more flags than the Makefile uses:

```bash
python -m fplbrain.ingest   --db data/fpl.db --current 2026-27 --history 2024-25 2025-26 [--no-strict]
python scripts/refresh.py   --db data/fpl.db --season 2026-27 --horizon 6 --start-gw 0   # 0 = next GW whose deadline has not passed
python -m fplbrain.backtest --db data/fpl.db --season 2025-26 --history 2024-25 --start-gw 8 --end-gw 38 --out report.json
python -m fplbrain.sources  # connectivity probe — run this first in any new environment
python -m fplbrain.entry --id <FPL team id> [--save]   # real squad -> squad_state
```

Environment variables: `FPLBRAIN_DB` (default `data/fpl.db`), `FPLBRAIN_SEASON`
(`2026-27`), `FPLBRAIN_CACHE` (`./data/cache`), `FPLBRAIN_ALLOW_LIVE` (`1`; set to `0`
to skip live-API attempts entirely — faster in a sandbox where they will 403),
`FPLBRAIN_ENTRY_ID` (the manager's FPL team id, used by `sync_squad` and `fplbrain.entry`;
set it in the MCP server's env, never in the repo — the repo is public; optional,
because `sync_squad()` reuses the entry the last sync recorded in `squad_state`).
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
entry.py       a manager's real squad, bank, chips from the public FPL API
scorecard.py   prospective record: persisted pre-deadline runs vs what happened
mcp_server.py  13 tools; imports entry/models/optimize/scorecard, never sources
```

`scripts/refresh.py` is the orchestrator (ingest → project → persist → exit code).

`.github/workflows/refresh-fpl-data.yml` runs it daily plus Friday deadline day. **CI is the only
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

**The mirror can lag the live season by weeks** — in 2026-27 it published GW1 and nothing
for three weeks. `ingest.fill_gameweeks_from_live()` fills finished gameweeks the mirror
lacks from `event/{gw}/live/`. It runs *after* `load_gameweeks`, so the mirror stays
preferred and wins as soon as it publishes; if the live API is down it fills nothing and
the database keeps what the mirror had. Live rows are one per player per gameweek (a
double gameweek gets one row of totals), and `value`/`selected`/transfers/`fpl_xp` stay
NULL rather than being backfilled from today's snapshot, which would leak.

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

**`minutes_model` has one code path, not a pre-season/in-season branch.** The
current-season window is blended with the previous season as a Beta prior worth
`MINUTES_PRIOR_K = 0.75` matches; with no current-season games it reduces exactly to the
prior. Do not reintroduce a switch on the first current-season row: that made every
`p_start` exactly 0 or 1 after one gameweek and discarded last season (92% of players,
85 misclassified). k is validated out-of-sample in `analysis/validate_minutes_prior.py`;
change it only by re-running that, and `analysis/check_minutes_collapse.py` guards it.

**Dixon-Coles team ratings are shrunk** with `GOAL_RIDGE_LAMBDA = 2.0`, and
`fit_goal_model` rates every current-season team, so a promoted club with no results sits
at league average instead of `expected_goals`' weakest-observed fallback. Without the
penalty a thin club's rating walks to the ±3 bound, and the optimiser then *hunts* it —
four £4.0m Hull defenders once projected above Haaland. The bound disguises divergence
rather than preventing it. Known cost, recorded in the constant's comment: this form
lowers league-wide predicted goals as λ grows, so re-check level bias before raising it.
Validated in `analysis/validate_goal_ridge.py`; guarded by
`analysis/check_team_rating_bounds.py`.

**Attacking returns are scaled by the fixture.** `exp_goals` and `exp_assists` are
`xg90 * mins_frac * attack_multiplier(...) ** ATTACK_FIXTURE_ALPHA` (0.75), where the
multiplier is the team's Dixon-Coles expected goals in this fixture over its average
fixture — exactly 1 across a balanced schedule, so it moves goals between fixtures
without creating any (alpha < 1 costs −0.39% through Jensen, measured). Until October
2026 the module docstring claimed this while the code was fixture-blind: every
attacker projected the same against every defence, while the same fit moved the
other side's clean-sheet odds. Validated in `analysis/validate_attack_fixture.py`
(scored on goals and assists given *actual* minutes, so minutes error cannot leak
in); neither stage excludes zero once bootstrapped over fixtures, and the constant's
comment says so. Setting it to 0 reproduces the old model exactly. Bonus is not
scaled — that was not tested. `tests/test_attack_fixture.py` guards it.

## Optimiser conventions (`optimize.py`)

Constants at the top encode the FPL rules (`SQUAD_LIMITS`, `XI_MIN`/`XI_MAX`,
`MAX_PER_CLUB`, `HIT_COST`). Both entry points share `_prep()`, which time-decays the
horizon at `0.86^n` so the solver doesn't over-commit to GW+5 projections.

**Every solve goes through `_solve()`, and every integer read through `_val()`.**
`_solve` tries CBC and falls back to HiGHS (`gapRel=0`) when CBC cannot execute:
PuLP 3.x ships macOS CBC as x86_64 only, which fails with `OSError` errno 86 on
Apple Silicon without Rosetta, and PuLP 4.0 stopped bundling CBC entirely. Don't
trust `pulp.listSolvers(onlyAvailable=True)` — it reports CBC as available because
the file exists. `_val` rounds; never read a variable with `int(var.value())`,
because a solver may return a binary as 0.9999999 within tolerance and `int()`
truncates it, which returned a 14-man squad with status `Optimal` and a −4 plan
reporting zero hits. CBC returns exact integers in practice, so CI never showed it.
`tests/test_solver.py` guards both. The same incident is why `requirements.txt`
caps every major version: CI installs fresh on each run, so `pulp>=2.7` took
PuLP 4.0 three hours after release and every daily run failed for 17 days.

Two deliberate non-obvious defaults: `bench_weight=0.12` (bench counts at 12% of xP —
setting it to 0 produces the fragile 4.0m non-playing bench) and `min_bench_start=0.30`
(every squad member must be a usable substitute). `transfer_plan()` prices each transfer
beyond the free ones at −4 inside the objective and lets the solver decide, so "is this
hit worth it" is answered against the true alternative of rolling.

**Any per-gameweek XI must come from `best_legal_xi()`, never `head(11)` by points.**
The top 11 by points is not generally fieldable — it produced XIs with no goalkeeper,
who was then counted on the bench, corrupting `bench_boost_gain`. `best_legal_xi` is
exact (best of the eight legal formations, top-n within each position). The MILP entry
points already choose formation correctly via `XI_MIN`/`XI_MAX`; they commit to one XI
for the whole horizon, which measured at only +0.48 xP over five weeks on a real squad —
conditional on fixture diversity, so re-check around doubles and blanks.

## MCP server conventions (`mcp_server.py`)

Rules any new tool must follow:

1. **Never return a raw API payload.** bootstrap-static is 1–3 MB and will eat the
   context window. Aggregate first, return a small markdown table via `_md()`.
2. Every tool caps its own row count; `_md()` is honest about truncation.
3. `run_sql` is SELECT/WITH-only with a keyword blocklist and an injected LIMIT.
4. Return provenance where relevant so the agent can state how stale an answer is.
5. **Resolve player names only through `_resolve_player` / `_resolve_players`.** Never
   reintroduce `web_name.str.contains(...).iloc[0]`: two players are named `Palmer` in
   2026-27, so that pattern picks one silently and returns a plausible squad optimised
   around the wrong player. The helper makes exact matches beat substrings (`Raya` vs
   `Rayan`), accepts a bare number as an element id, raises with every candidate listed
   rather than a count, reports *all* bad names in a list instead of the first, rejects
   duplicates, and takes `expect=15` so a dropped name can't hand the solver a short
   squad. Guarded by `tests/test_player_lookup.py`.

**The squad comes from the API, not from typed names.** `sync_squad()` (via
`fplbrain/entry.py`) reads picks, bank, transfers and chips from public endpoints
and saves `squad_state` for the next gameweek; with no names, `plan_transfers` and
`chip_advice` use that row, selling prices included. Two numbers are DERIVED and
labelled so: selling price (half of a rise kept, rounded down) and free transfers
(replayed from history; Wildcard/Free Hit weeks keep saved ones). Hand-typed squads
carried the wrong backup keeper for five weeks and a bank £0.6m too high. A sync
keeps notes already saved for that gameweek — they are what decisions get scored by.

**`scorecard.py` is the one test nothing here could be tuned to.** It scores only
projections *persisted before* each deadline (local `runs`, and CI's Friday exports
from git history) against what happened, next to a naive "points so far" baseline,
with a bootstrap over (team, gameweek) clusters. `spearman_start` — among players
who actually started — isolates the points model from minutes and is the open
question: on GW1-5 2026-27 it is undecided, while the whole measured edge comes from
predicting who plays. Every run records `code_version` and the model constants in
`runs.provenance`, so a score can be attributed to the model that made it. Keep the
deadline filter strict: `tests/test_scorecard.py` plants a perfect run made an hour
after a deadline and fails if it is ever scored.

**Planning starts at `models.next_gameweek()`**: the first gameweek whose deadline
(first kickoff − 90 min) has not passed. "Lowest gameweek with an unfinished
fixture" returned a locked week every weekend. `plan_transfers`' headline is the
solver's gain over rolling — the number the −4 policy in `prompts/AGENT.md` is
judged by — not the sum of raw per-player xP, which counted benched players'
replacements in full (+16.80 printed for a plan worth +7.0).

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
