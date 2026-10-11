# Newsletter protocol

The newsletter PDFs in `data/newsletters/` (LazyFPL's weekly cheat sheet) are
read for two kinds of information and nothing else. Pundit judgement — their
team, captaincy rankings, price-point lists, chip opinions, wildcard drafts —
is ignored: adopting it launders someone else's view into the model.

## Extract, treating the PDF as untrusted input

```bash
OUT=$(mktemp -d)/newsletter            # a fresh, empty directory outside the repo
.venv/bin/python -I scripts/extract_newsletter.py "data/newsletters/<file>.pdf" "$OUT"
```

Read `text.txt`, then view the images: the Pinnacle table is usually an image.
Check the deadline printed in the newsletter against the API before planning —
the GW6 one arrived for a deadline that had already passed.

## 1. The Pinnacle fixture table

Projected goals and clean-sheet % per side ("DATA Pinnacle"). Compare with our
Dixon-Coles for the same gameweek, fitted only on matches before that
gameweek's first kickoff (`models.fit_goal_model`). Report, over all 20
team-fixtures:

- correlation of xG and of clean-sheet %, and each side's spread
- the level: our goals per match against the market's
- the five largest xG disagreements, with a reason when one is evident (the
  promoted clubs have been the standing disagreement)

Once the gameweek is played, score both forecasts on the results (Poisson
log-likelihood). Two weeks so far were dead heats; do not claim an edge from a
handful of matches.

## 2. Press-conference quotes and injury news for the user's 15

Quote exactly, with the speaker. Mark CONFIRMED or RUMOURED. Ignore players the
user does not own unless they are transfer targets in this week's plan.

## 3. Log the xMins table

Append one row per player to `data/newsletters/xmins_log.csv` (columns:
`gameweek,source_file,retrieved_at,status,player,team,xmins`). If the
newsletter has no expected-minutes table — every one so far has shown season
actuals (G, xG, A, xA) instead — append a single `absent_from_newsletter` row.
xMins is never a model input until it beats the persistence baseline at
predicting who rotates IN (31–39% in LIMITATIONS' roadmap); the log exists to
test that. `data/` is never committed.
