---
name: deadline-brief
description: Prepare the user's Fantasy Premier League decision for the next gameweek deadline in the fplbrain repo — sync the real squad from the FPL API, refresh data, read any new newsletter, check the model's prospective record, gather injury and press-conference news, plan transfers under injury scenarios, check chips against the first-half policy, red-team the recommendation, and record judgement calls. Use when the user asks what to do this week, about transfers, captaincy, chips, a gameweek or deadline, or mentions a "[fplbrain] GWn deadline brief" GitHub issue.
---

# Deadline brief

Produce the decision for the next open gameweek, with evidence the user can
check. The user decides and executes — FPL has no public way to make
transfers, and credentials are never automated.

**Objective: overall rank** (agreed 11 Oct 2026). Expected points decide most
moves, but ownership matters for captaincy and differentials: a captain most
of the field owns protects rank; a low-owned one gains it.

Tools are the `fplbrain` MCP server's. If it is not connected, call the same
functions directly: `.venv/bin/python -c "from fplbrain import mcp_server as m; print(m.plan_transfers())"`.

## 1. Orient
- Read the session-start status: deadline and time left, CI state, squad age.
  If CI is red or an `[fplbrain] CI failing` issue is open, say so first —
  the projections may be stale.
- `status()`. If data predates the last finished gameweek, `refresh_data()`.

## 2. The real squad
- `sync_squad()`. Never plan from a pasted squad: a hand-typed one carried the
  wrong backup keeper for five weeks and a bank £0.6m too high.
- Bank and squad are API facts; selling prices and free transfers are derived
  — say so when they decide a move.
- API down: `load_state()`, and state that the numbers are from saved state.

## 3. Newsletter
If `data/newsletters/` has a PDF newer than the last brief, follow
[newsletter.md](newsletter.md) exactly. It defines what to extract and what to
ignore; do not improvise beyond it.

## 4. How the model and the news have been doing
- `model_scorecard()`. Quote the latest `spearman_start` interval. While it
  spans zero, the points model has not beaten "points so far" among starters —
  the model's proven edge is predicting who plays, so weigh its minutes calls
  more than its points calls.
- `news_scorecard()`. How logged claims did where they disagreed with
  persistence, and the verdict (WAIT / FEED / DROP).

## 5. News the model cannot see
LIMITATIONS §5: transformations of what the model already knows have failed
three times; new information is the input class that can still add value.
- Delegate to the **news-scout** subagent. Give it the gameweek, all 15 players
  with their element ids (from `sync_squad`), the doubtful ones first, and this
  week's transfer targets. It searches, then logs each claim with its source
  via `record_news` — logged before the deadline, so it can be scored after.
- Read what it found with `news_for()` (your squad) and
  `news_for(squad_only=False)` (targets too). Present claims as CONFIRMED or
  RUMOURED, with source and date.
- Check last gameweek's minutes for the 15. A starter who went off early can
  be an injury signal before any flag appears — but **only read minutes from a
  finished match** (`player_gw` holds finished gameweeks; from the live API,
  require the fixture's `finished_provisional`). On 11 Oct 2026 live minutes
  read 15 minutes into two 13:00 kickoffs were taken for early substitutions of
  Gibbs-White and Tarkowski, who then played 90; a day of injury scenarios was
  built on it.
- Fixture congestion (European and cup midweeks) is not in the model. Name it
  where it applies; never adjust numbers for it by feel.
- Claims shape the scenarios below and nothing else: they do not change the
  model's numbers until `news_scorecard()` says FEED.

## 6. Plan under scenarios
- `plan_transfers()` — judge it by the **gain over rolling** it reports.
- For each player with an `out` or `doubtful` claim (or an early substitution):
  `plan_transfers(unavailable="Name:2")` (short absence) and `"Name:5"` (long).
  Combine players when several are doubtful.
- Present a decision tree — "if X is out: A; if fit: B" — so the user can act
  on Friday's news without another session.
- Hit policy (prompts/AGENT.md): a −4 only when gain over rolling exceeds +6.
  Always show rolling as an option, with its cost.

## 7. Chips
`chip_advice()`, then the first-half policy in prompts/AGENT.md: no doubles or
blanks before GW19, so each chip goes in its best remaining week. A wildcard's
modelled gain is inflated (it re-picks all 15 slots — the optimiser's curse);
compare it with the best free-transfer plan, never with rolling.

## 8. Red team — before presenting
Try to break the recommendation. Report what survives and what does not.
1. **Noise.** Over 5 gameweeks, gains under about +3 xP sit inside the
   optimiser's instability (about 7 of 15 picks change under small input
   perturbations — LIMITATIONS §4). Say so when a move is that close.
2. **Minutes risk** of every incoming player: p_start, recent minutes, rotation,
   congestion, news.
3. **Market disagreement** on their fixtures, if a Pinnacle table is available.
4. **Concentration.** Three from one club is a correlated bet (same-club
   defenders correlate at 0.51; the model treats them as independent).
5. **Rank.** Ownership of the captain and each target; is a differential
   warranted at the user's current rank?
6. **Rules.** Club limit, bank after selling prices, the hit threshold, chips.

## 9. Refresh the tracker
The user's dashboard ("fplbrain Tracker", a private Dashboard-type Claude
artifact; find it with `Artifact` list) draws every number from files built
here, so it is only as fresh as its last refresh. Before presenting:
1. `.venv/bin/python scripts/build_tracker.py` (about a minute; it syncs the
   squad too). It prints each file's sha256 prefix.
2. List the artifact's assets (`Artifact` list, scope `assets`) and upload
   (`asset: true`) only the files whose sha256 the store does not have.
3. One `ArtifactData` batch of `update`s on `datasets/<id>`, pinned to the
   versions a list of `datasets` shows: a re-uploaded file gets its whole
   `source` with the new url; every dataset gets `updated: {at, by}` with the
   build time, because an unchanged file was re-checked and an old time reads
   as stale.
4. Delete the assets the new uploads replaced.

To change the page itself, read the type's instructions first (`Artifact`
read of the dashboard). They rule out rendering it to check your work unless
the user asks.

## 10. Present
Lead with the decision. Then Data, Model and Judgement, kept separate, then the
numbers. Give the deadline in UTC and as time remaining.

## 11. Record
After the user decides, `note_decision("...")` for every judgement call that
departs from the model, with its reason. The end-of-season review scores
judgement against the model from these notes; the next `sync_squad()` keeps
them and records what was actually done.
