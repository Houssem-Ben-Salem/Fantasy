# fplbrain

An autonomous Fantasy Premier League analyst: data pipeline, expected-points model,
MILP squad optimiser, and an MCP server that hands the whole thing to Claude as tools.

Built and verified against real 2024-25, 2025-26, and 2026-27 data on 16 August 2026.
**GW1 kicks off Friday 21 August 2026, 19:00 UTC** (Arsenal v Coventry).

---

## The problem this solves

Most "AI for FPL" setups fail one of three ways: the model hallucinates prices and
injuries, the agent drowns in a 3 MB JSON blob, or the advice is unfalsifiable vibes.
This fixes all three — data lives in SQLite, the agent queries it through narrow tools
that return small markdown tables, and every recommendation is a number you can check.

## Architecture

```
sources.py    live FPL API -> GitHub mirror -> local cache   (provenance tracked)
     |
ingest.py     ETL + validation + cross-season crosswalk -> SQLite
     |
models.py     Dixon-Coles goals | minutes | DefCon | attacking rates -> xP
     |
optimize.py   MILP: initial squad, transfer plan, chip valuation
     |
mcp_server.py 12 tools -> Claude
```

### Historical seasons never use the live API

`bootstrap-static` and `/fixtures/` are **current-season-only**. They take no season
parameter and have no historical mode. Every live call is therefore gated on the
season actually being the current one; history always comes from the mirror, which
is season-addressable.

This is not theoretical. An earlier version threaded a `season` argument into
functions that called those endpoints regardless, so on any machine where the FPL
API was reachable, loading 2024-25 wrote **current-season rows stamped with a
historical season label**. It failed upward — the cross-season crosswalk rose to a
perfect 1.0 (the current player list matching itself) against a correct 0.78-0.81
depending on path, so the corruption made a quality metric look *better*.

`ingest.check_season_distinctness()` now asserts that different seasons contain
different data, and `run(strict=True)` refuses to return a corrupted database.
`tests/test_sources_season_gating.py` simulates a reachable API and fails against
the old code.

A general lesson worth keeping: any figure in this README produced on one path is
not automatically true on the other. Where a number differs, both paths are stated.

### The other design decision that matters

`fantasy.premierleague.com` is **blocked from most agent sandboxes and CI runners**.
Verified in this environment:

| Host | Reachable |
|---|---|
| `fantasy.premierleague.com` | ❌ 403 |
| `raw.githubusercontent.com` | ✅ 200 |
| `understat.com` | ❌ 403 |
| `api.clubelo.com` | ❌ 403 |

The vaastav repo mirrors bootstrap-static to `players_raw.csv` and the fixture list to
`fixtures.csv` for the **current** season. So `sources.py` falls back to GitHub and the
entire pipeline runs with zero access to the FPL API. On your own machine the live path
works and takes priority automatically. You get the same code path either way, and
`status()` tells the agent which source it actually used.

## Setup

```bash
git clone <your-repo> && cd fplbrain
pip install -r requirements.txt
make ingest          # ~30s: pulls 3 seasons, builds SQLite
make refresh         # fits models, writes projections
make squad           # prints an optimal 15
```

### Connect to Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "fplbrain": {
      "command": "python",
      "args": ["-m", "fplbrain.mcp_server"],
      "cwd": "/absolute/path/to/fplbrain",
      "env": {
        "FPLBRAIN_DB": "data/fpl.db",
        "FPLBRAIN_SEASON": "2026-27"
      }
    }
  }
}
```

Restart Claude Desktop, then paste `prompts/AGENT.md` as a Project instruction.
Say "weekly check-in" and it runs the whole loop.

## What the models actually do

**Goals — Dixon-Coles with time decay.** Fitted on 760 finished matches across
2024-25 and 2025-26. Bivariate Poisson with the low-score correlation term, exponential
decay (ξ=0.0018/day, roughly a one-year half-life). Fitted home advantage came out at
0.161 (≈17.5% more goals at home) and ρ = −0.109 — both squarely in the range the
literature reports, which is a good sign the fit isn't broken. Produces per-fixture
expected goals, hence P(clean sheet) and the goals-conceded distribution.

**Minutes.** The biggest error source in FPL, and the component to distrust most.
Blends recent start rate with season-long usage, then multiplies by an availability
factor from FPL's `status` and `chance_of_playing_next_round`. Pre-season it falls back
to last season's usage; players with no history (new signings, promoted clubs) get
`p_start = 0.35`, which is frankly a guess. **Override it with team news.**

**DefCon.** The 2025/26+ rule: +2 for 10 CBIT (DEF) or 12 CBIRT (MID/FWD), capped once
per match, GKs excluded. Computed as an empirical-Bayes hit rate per player, shrunk
toward the prior for their **current** position with a 12-start prior. The position
part matters — 11 players were reclassified for 2026/27, and a converted defender
inherits the harder midfield threshold, so raw 2025-26 numbers mislead.

Real output from the 2025-26 data:

| Player | Pos | 60+ starts | DefCon hits | Rate |
|---|---|---|---|---|
| Senesi | DEF | 37 | 26 | 0.70 |
| Anderson | MID | 37 | 26 | 0.70 |
| Mavropanos | DEF | 27 | 18 | 0.67 |
| Andersen | DEF | 31 | 20 | 0.65 |

A 70% hit rate is ~1.4 pts/start before anything else happens. That is where cheap
defensive value lives this season.

**Attacking returns.** xG90/xA90 shrunk toward positional priors with weight
`minutes / (minutes + 450)`, so five matches of data gets you halfway to your own rate.
First-choice penalty takers get +0.10 xG90; set-piece takers get an xA multiplier.

**Optimiser.** MILP via PuLP/CBC. Encodes £100.0m, 15 players (2/5/5/3), max 3 per club,
valid XI formations, captain doubling. Solves in under a second. The horizon is
time-decayed (0.86^n) so it doesn't over-commit to GW+5 projections it can't trust.
Bench players count at 12% of their xP and every squad member needs `p_start ≥ 0.30` —
that's deliberate, because the classic 4.0m non-playing bench is fragile the moment you
get an injury.

`transfer_plan` prices each transfer beyond your free ones at −4 and lets the solver
decide. That's the honest way to answer "is this hit worth it": it compares against the
true alternative, which is keeping the squad and rolling.

## Backtest results

Walk-forward over 2025-26, GW8-38, refitting every feature at each gameweek from
prior data only. 9,217 player-gameweeks.

```
predictor     spearman     MAE   top20   GWs  status
------------------------------------------------------------------
fpl_ep          0.6054   2.848   4.068     5  EXCLUDED (see below)
model_xp        0.3261   2.004   4.739    31
ppg              0.291   2.094   4.516    31
form6           0.2725   2.217   4.166    31
price           0.1282   3.428   4.437    31

field mean actual points: 3.043
```

**The model beats every legitimate benchmark, modestly.** Spearman 0.33 sits in the
normal band for FPL projection models — nobody gets 0.6 on this task. The practical
number is `top20`: the mean actual points of each gameweek's top-20 picks by that
predictor. The model's top 20 returned **4.74 points against a field average of 3.04**,
ahead of points-per-game (4.52), form (4.17), and price (4.44).

Probability outputs are calibrated, which matters more than the headline correlation
because the optimiser consumes them directly:

| output | predicted | actual | Brier | vs base rate |
|---|---|---|---|---|
| clean sheet | 0.264 | 0.267 | 0.189 | better (0.196) |
| DefCon | 0.198 | 0.206 | 0.141 | better (0.164) |

By position the model ranks forwards best (0.41) and midfielders well (0.39), defenders
weakly (0.26), goalkeepers barely at all (0.16). Treat GK picks as a fixtures-and-price
decision, not a model decision.

### Why fpl_ep is excluded

FPL's own published expected points looks like it wins at 0.605. It doesn't:

1. It is populated in **only 5 of 31 gameweeks** — the upstream scrape is broken, so it
   was being scored on a different, easier sample.
2. On the gameweeks where it does exist, it assigns **exactly 0.0 to 72.5% of players
   who didn't play, versus 1.1% of players who did.** No pre-deadline projection can
   know who will be benched. That column is recorded after kickoff.

`backtest.py` now detects both failure modes automatically and marks such predictors
EXCLUDED rather than silently flattering them. If you add your own benchmark, it gets
the same scrutiny.

## Verified working

21 tests, covering FPL constraints, season gating, provenance labelling, integrity
detection, and squad-state durability.

```
PASS squad size 15          PASS XI == 11
PASS positions 2/5/5/3      PASS XI 1 GK
PASS budget <= 100          PASS XI >= 3 DEF
PASS max 3 per club         PASS XI >= 1 FWD
PASS one captain
```

Ingest: 27,605 rows (2024-25) + 29,747 rows (2025-26), 38/38 gameweeks each. Those are
mirror-sourced and path-independent.

**Crosswalk numbers are path-dependent — check yours rather than matching mine.**
Live `bootstrap-static` carries more current-season players than the mirror's periodic
`players_raw.csv` snapshot (the mirror set is a strict subset), so the denominator
differs and the match rate moves with it:

| path | current players | matched to 2025-26 | rate |
|---|---|---|---|
| mirror only (`FPLBRAIN_ALLOW_LIVE=0`) | 567 | 457 | 0.806 |
| live current season (default) | 590 | 460 | 0.780 |

Both are correct. The live figure is the fresher one and the one to prefer — the extra
players are registrations made since the mirror snapshot, which in August is exactly
where new signings live. A *lower* rate here is the healthier number.

The invariant that does hold on every path: unmatched players are new signings and
promoted-club players with genuinely no history, **not** a join failure. If the rate
ever hits 1.0, that is not success — see the season-gating section below.

## Known data-quality issues

Three real traps in the upstream data, all handled here, all worth knowing if you build
your own pipeline:

1. **Duplicate rows.** `merged_gw.csv` for 2025-26 contains 10 exact duplicates across
   two players (elements 100 and 391, GW1-9). A naive `to_sql` fails on the primary key.
   Ingest dedupes and reports the count.

2. **No defensive stats before 2025-26.** `tackles`, `recoveries`, and
   `clearances_blocks_interceptions` are **100% NULL for all 27,605 rows of 2024-25** —
   the DefCon rule didn't exist yet. Pooling that season with `fillna(0)` adds tens of
   thousands of guaranteed misses and **halves every DefCon estimate** (measured: DEF
   predicted 0.128 against a true 0.270). The model now filters to seasons where the
   data actually exists.

4. **`defensive_contribution` is an action count, not points.** It is the
   position-correct metric already — verified to equal CBIT for DEF and CBIRT for
   MID/FWD on 100% of 2025-26 rows. Threshold it at 10 (DEF) / 12 (MID, FWD); don't
   treat a non-zero value as "they scored the DefCon points".

## Honest limitations

- **Minutes modelling is the weak link.** Everything downstream inherits its error.
  Team news beats the model every time.
- **No Understat integration by default** — the domain is blocked in sandboxes. The FPL
  API's own `expected_goals`/`expected_assists` are Opta-sourced and adequate. Add
  `soccerdata` locally if you want shot-level data.
- **FBref lost its Opta advanced-stats feed in January 2026**, so progressive passing and
  SCA are no longer freely available at scale. Don't design around them.
- **Bonus modelling is crude.** 2026/27 retuned BPS (CBI now 1 per 3, not 1 per 2). Rather
  than fake precision, expected bonus is tied to goal involvement and clean-sheet
  probability, which is what actually drives top-3 BPS.
- **The model's edge is real but modest.** Spearman 0.33 and a top-20 that returns 4.74
  vs a 3.04 field average. That is worth having, and it is not a licence to switch off
  your judgement. Most of the season is variance.
- **Goalkeeper ranking is near-worthless** (0.16). Pick on fixtures, price, and save
  volume instead.
- The model has never watched a football match. It cannot tell you that a manager hinted
  at rotation. That's what the agent's web search is for.

## Layout

```
fplbrain/
├── fplbrain/
│   ├── sources.py       fetching + fallback chain + connectivity probe
│   ├── schema.sql       tables, indexes, 4 analytical views
│   ├── ingest.py        ETL, validation, cross-season crosswalk
│   ├── models.py        Dixon-Coles, minutes, DefCon, xP assembly
│   ├── optimize.py      MILP squad / transfers / chips
│   ├── backtest.py      walk-forward eval + leakage detection
│   └── mcp_server.py    12 agent tools
├── scripts/refresh.py   one-command update, exits non-zero on failure
├── prompts/AGENT.md     agent operating instructions
├── .github/workflows/   daily refresh (runners CAN reach the FPL API)
└── Makefile
```

## License

Personal use. The FPL API is free for non-commercial use — don't monetise this.
Credit vaastav/Fantasy-Premier-League for the historical data.
