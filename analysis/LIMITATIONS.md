# fplbrain — Limitations, Measured

Every claim below is an experiment run against the 2025-26 season with strict
no-look-ahead construction, not an assumption. Each section names the script that
produces its numbers; run it from the repo root against a built `data/fpl.db`.

| section | script |
|---|---|
| §1, §2 | `python analysis/exp1_ceiling.py` |
| §3 | `python analysis/exp4_calibration.py` |
| §4 | `python analysis/exp2_optimiser.py` |
| §5 | `python analysis/exp3b_power.py` (single window: `exp3_robust.py`) |
| Tier 1 baseline | `python -m analysis.baselines` |

---

## 1. The headline number was being read wrong

Spearman 0.33 sounds poor against a scale that tops out at 1.0. But FPL points are
a lumpy, low-count outcome — a player with 0.5 xG either scores or doesn't — so
the right denominator is not 1.0 but **the best any model could achieve**.

We measured it by building oracles that use information no pre-deadline model
could have, and seeing where they stop.

| predictor | Spearman | top-20 actual | vs field |
|---|---|---|---|
| our model | 0.3167 | 4.429 | +1.428 |
| + perfect minutes | 0.5285 | 4.934 | +1.933 |
| + perfect xG/xA | 0.5739 | 6.869 | +3.868 |
| + perfect clean sheet & DefCon | 0.7692 | 8.529 | +5.528 |
| perfect foresight | 1.0000 | 11.100 | +8.099 |

n = 9,348 player-gameweeks, field mean 3.001 pts.

**Even with perfect knowledge of minutes, underlying performance, clean sheets and
DefCon, the ceiling is 0.769.** The remaining 0.231 is finishing, bonus and cards —
irreducible. Against that ceiling we capture **41.2% of achievable signal**, not
33% of a fantasy.

The sobering version is the points column. Perfect foresight beats the field by
8.10 points per pick; we beat it by 1.43. On realised points we capture **17.6% of
the achievable edge**. Both framings are true and worth holding at once.

Two caveats on the ceiling. `oracle_full` still uses MODELLED bonus, so the 0.231
residual includes bonus error as well as finishing. And the sample is filtered to
players who featured (`minutes >= 1`), which removes the easy negatives — so
0.3167 here is not comparable to `python -m fplbrain.backtest`, which scores
everyone. The ratio is what this experiment exists to produce, and both numerator
and denominator use the same sample.

## 2. Minutes is not "a" weakness, it is 47% of the entire gap

Marginal Spearman gain from perfecting each component in turn:

```
minutes              +0.2118   <-- 46.8% of all addressable headroom
clean sheet + DefCon +0.1953
xG/xA rates          +0.0454
```

I had been asserting minutes was the weak link all along. It is worse than that:
**xG/xA modelling is essentially saturated.** All the shrinkage, the penalty-taker
uplift, the set-piece multiplier — perfecting every bit of it buys 0.045 Spearman.
The attacking-rate work is done. Further effort there is close to wasted.

The minutes model is currently an average of recent start rates times an
availability multiplier. It is the crudest component in the system and it carries
the most weight.

## 3. The probability components — better than they looked, but the method has a limit

My first read was that clean-sheet prediction was weak: Brier 0.1891 against a
0.1957 base-rate baseline, only 3.4% better. That read was wrong, and wrong in the
same way as reading 0.33 against 1.0.

For a binary outcome, predicting the base rate gives Brier `p̄(1-p̄)`. Knowing each
case's true probability gives `p̄(1-p̄) - Var(p)`. **`Var(p)` is the entire reducible
portion**; everything else is coin-flip noise.

| component | baseline | floor | ours | naive read | captured |
|---|---|---|---|---|---|
| clean sheet | 0.1957 | 0.1858 | 0.1891 | 3.4% better | **66.5%** |
| DefCon | 0.1637 | 0.1478 | 0.1410 | 13.8% better | **not computable** |

**The DefCon floor is falsified by its own data.** Our Brier (0.1410) comes in
*below* the estimated floor (0.1478), which is impossible if the floor were right.
`Var(p)` is unobservable and was estimated from the model's own predicted
variance; a score beneath it means the model is **under-dispersed** — true
probabilities vary more than our predictions do — so `Var(p)` was underestimated
and the floor set too high. No `captured` figure can be quoted for DefCon. An
earlier draft of this document quoted 56.2%; that number should not be used.

The same problem afflicts clean sheets more mildly, and in the same direction. The
dispersion check:

```
clean-sheet spread across teams
  modelled  0.155 to 0.454   sd 0.072
  realised  0.099 to 0.518   sd 0.099
```

The model's spread is narrower than reality's. So the floor is again somewhat too
high, reducible too small, and **66.5% is an over-estimate of an unknown true
figure.** The safe conclusion is directional: clean-sheet modelling captures a
substantial majority of what is available and is one of the better parts of the
system, not the worst. The precise percentage is not defensible.

## 4. The optimiser has three real pathologies

### Instability

Perturbing projections by 25% of their own stated uncertainty and re-solving, 25
trials over GW20-25:

```
mean overlap with the base squad     8.32 of 15
minimum overlap                      6 of 15
distinct players ever selected       67
selected in EVERY trial              0
selected exactly once                21
```

**Not one player survives every perturbation.** Roughly seven of the fifteen are
noise. "Optimal" is doing far less work than the word implies.

### Selection bias — the estimation-error maximiser

Michaud (1989) showed mean-variance optimisers systematically select assets whose
returns are most *overestimated*, because they cannot distinguish a high estimate
from a high estimation error. Ours has the same shape. Measured over GW20-25:

```
selected 15:              projected 22.619 -> actual 19.467   gap -3.153
comparable pool (n=59):                                       gap +3.175
bias attributable to selection:                               -6.327 per player
```

That is a **~95-point illusion** across a squad over six gameweeks.

An earlier draft reported −5.04 and ~76 points. That was wrong: the comparator
pool was defined as everyone above the selected squad's minimum projection, which
makes it a *superset* of the selected players. Their own negative gap was being
averaged into the benchmark they were measured against, diluting it by about a
quarter. The pool now excludes them; `exp2_optimiser.py` still reports the
contaminated figure under a leading underscore so the error cannot be silently
reintroduced.

### Correlation blindness

The MILP treats players independently. They are not:

```
same-team, all positions      r =  0.1943
same-team defenders           r =  0.5107
different-team control        r = -0.031    (correctly ~0)
```

Three defenders from one club: assumed sd 5.47, actual sd 7.77. **Variance is
understated by 42.2%.** Stacking a defence reads as diversified when it is a
concentrated bet on one team's clean sheet.

## 5. But fixing the optimiser does not help — and this is the important part

The obvious response to §4 is Michaud's own remedy: resampled efficiency. Solve
many times under perturbed inputs, select by frequency rather than a single draw.
We implemented that, plus shrinkage toward positional means, and tested both
against realised points across 8 paired 6-gameweek windows (stride 3, GW10-36, 15
resamples).

```
                                   mean    SE    95% CI            wins   p
overlapping windows (n=8)
  shrunk vs naive                  +0.2   1.0    [-1.7, +2.2]      4/8   0.562
  resampled vs naive               +1.9   7.0    [-11.8, +15.5]    5/8   0.844

non-overlapping windows (n=4, stride 6, zero shared gameweeks)
  shrunk vs naive                  -0.2   2.1    [-4.4, +3.9]      2/4
  resampled vs naive               +2.0  12.7    [-22.9, +26.9]    2/4

moving-block bootstrap (n=8, block 2)
  shrunk vs naive                  +0.2         [-1.6, +2.0]
  resampled vs naive               +1.9         [-8.0, +16.2]

between-window SD                  41.5
between-strategy SD within window   9.3
```

**Neither helps, under all three estimators.** And the variance decomposition
explains why: which six gameweeks you are in swings the outcome by roughly ±42
points; which optimiser you use swings it by ±9.

### A methodological worry that did not survive measurement

The 8 windows overlap by half, so the paired differences ought to be
autocorrelated, making a plain SE anti-conservative. Measured:

```
lag-1 autocorrelation, LEVELS       naive +0.178   shrunk +0.201   resampled +0.040
lag-1 autocorrelation, DIFFERENCES  shrunk -0.167  resampled -0.071
```

The overlap does show up in the levels, exactly as construction implies — but
**pairing removes the window effect, and the dependence with it.** The differences
the test actually runs on are not positively autocorrelated, so the naive SE is
not inflated. The reasoning was sound and the conclusion was still wrong, which is
why `exp3b_power.py` measures this rather than asserting it in either direction.
The non-overlapping and bootstrap estimates are reported regardless, and all three
agree — that agreement, not any single interval, is the reason to believe the
finding.

### Why remedies cannot work here

The pathologies in §4 are **downstream of input error**. The selection bias is
*caused* by projection error, and no selection rule removes bias that originates
in the estimates. Rearranging how you choose from noisy inputs does not make them
less noisy.

**This extends to input SOURCES, with one boundary that matters.** A source
carrying information the model genuinely cannot see — Friday team news, a press
conference — is *new information*, not a rearrangement, and the argument above
does not apply to it. A source that largely re-encodes what the model already
knows is a rearrangement wearing a different hat, and will behave like shrinkage
and resampling did here. The two look identical until you test them, which is
exactly what Tier 1 item 1 below now exists to do.

This is the single most useful finding here, because it redirects effort.
Optimiser sophistication is a dead end. Input quality is everything — but only
input quality that is *new*.

## 6. The limitation no experiment here touches: the objective is wrong

fplbrain maximises expected points. FPL is a **rank tournament**, and maximising EV
is not the same as maximising P(finishing top 10k).

- Against a field where the template scores 60, a 62-EV/low-variance squad and a
  62-EV/high-variance squad are *not* equivalent. If you need to gain rank,
  variance is a tool; if you need to protect it, variance is a threat. EV is blind
  to which situation you are in.
- **Effective ownership** determines rank movement. Captaining a 55%-owned player
  who hauls gains you nothing on most of the field. The model has no concept of
  what anyone else owns, despite `selected_by_percent` sitting in the database.
- §4's correlation finding is the same point from the other side: same-team
  stacking is a *variance instrument*. Right now we take that bet accidentally
  rather than choosing it.

This is structural, not a bug. It is also the largest single conceptual gap.

## 7. Sample limitations, stated plainly

- **One season of backtest** (2025-26), 9,348 player-gameweeks, 31 gameweeks.
- **DefCon fitted on one season of a rule that has existed for one season.** The
  empirical-Bayes prior has no way to know whether 2025-26 was typical. §3 shows
  the model is under-dispersed on DefCon, which is consistent with a prior fitted
  on too little data.
- **Eight windows in the optimiser comparison, of which only four are
  independent.** This is a hard ceiling, not a choice: one season of DefCon-era
  data admits at most about five non-overlapping 6-gameweek windows. No amount of
  statistical care fixes it. It needs 2026-27.
- **No cross-league validation.** Every parameter is tuned on the Premier League.
- **`exp3_robust.py` was reconstructed** to the interface `exp3b_power.py`
  expects, after the original was lost. The methodology follows the description
  above but the implementation is not byte-identical to the one that produced an
  earlier draft's figures (+0.6 / −4.4). §5's numbers are the ones the committed
  code produces. The conclusion is unchanged and slightly strengthened: the
  earlier draft had resampling mildly negative, this has it mildly positive, and
  both are indistinguishable from zero.

---

## Roadmap, ordered by measured value

### Tier 1 — minutes (worth up to +0.2118 Spearman)

Nothing else comes close.

**1. Determine whether any lineup source beats persistence. Do this BEFORE
building an integration.**

An earlier draft of this roadmap recommended "add predicted lineups" on the
strength of a self-reported 87% accuracy figure with no baseline attached. That is
the same denominator error §1 and §3 exist to correct, committed a third time — in
this document's own top recommendation. Writing the lesson down did not install
it. That is probably the most useful single sentence here.

The baseline, from `python -m analysis.baselines`. Persistence = "a player starts
iff they started last week", which costs nothing and needs no source:

| population | n | persistence | majority class | lift |
|---|---|---|---|---|
| all player-gameweeks | 28,906 | 0.889 | 0.737 | +0.151 |
| started ≥1 all season | 16,594 | 0.806 | 0.543 | +0.263 |
| squad-relevant (≥10 starts) | 11,459 | 0.784 | 0.605 | +0.180 |
| regulars (≥20 starts) | 6,634 | 0.809 | 0.757 | +0.051 |
| choice set (≥1 of previous 3) | 9,647 | 0.760 | 0.664 | +0.096 |

**An unbenchmarked 87% is at or below the do-nothing baseline** depending on which
denominator it was computed over — and no published source states one.

Overall accuracy is also the wrong metric, because persistence fails
asymmetrically:

```
regulars     P(drop | started) = 0.127    P(come in | benched) = 0.393
choice set   P(drop | started) = 0.214    P(come in | benched) = 0.313
```

Persistence is already good at who **keeps** a place. It is bad at who **rotates
in**. That 0.31–0.39 is the only number a lineup source has to beat.

The test: log one source's Friday predictions for 4 gameweeks, score on the same
denominators as above, report `P(come in | benched)`. A week of passive logging
against a build of unknown value. Integrate only if it clears the bar; if not,
this item closes and the 0.2118 minutes gap needs a different attack.

**2. Rotation-risk features.** Days since last match, European fixture in the
window, cup involvement, minutes in the last three matches. All derivable from
data already in the database, dependent on no external source.

**3. Manager-level rotation rates.** Some managers rotate heavily and predictably.
A per-club random effect on start probability is cheap.

**4. Model substitution patterns**, not just starts. P(60+) is currently
`p_start × 0.92`, a constant. It should vary by player and by game state.

**5. Structured injury feed.** Currently the agent does this by web search at the
deadline, which works but is unlogged and unbacktestable.

Items 2–4 are promoted above the lineup work deliberately: they depend on data
already held, rather than on an external source clearing a bar it may not clear.

### Tier 2 — treat variance as a decision variable

Not to raise EV — §5 says that will not work — but to make risk *chosen* rather
than accidental.

6. **Add a covariance term to the MILP.** Estimate same-club correlation (0.51 for
   defenders is a solid starting estimate) and let the objective be
   `xP − λ·risk`, with λ exposed. Then "triple up on Arsenal's defence" becomes an
   explicit decision with a stated variance cost.
7. **Report a distribution, not a point.** Monte Carlo the squad forward and give
   the agent P(score > 60), not just the mean. Given §4's instability, a point
   estimate overstates what is known.

### Tier 3 — model the field

8. **Effective ownership.** `selected_by_percent` is already in the database and
   entirely unused. EO-adjusted captaincy is the cheapest real edge available.
9. **Rank-objective mode.** Simulate the field, optimise P(top X%) instead of EV.
   A meaningful build, but it addresses §6 directly.

### Tier 4 — validate what we have

10. **Re-fit DefCon in November** on 2026-27 data and compare priors. §3 shows the
    current model is under-dispersed on DefCon, so this is not merely a check.
11. **Re-run the calibration floor** once two seasons exist, estimating `Var(p)`
    from realised outcomes rather than from the model's own predictions, which §3
    shows is circular.
12. **Widen the optimiser comparison** to 20+ *independent* windows across two
    seasons before concluding anything about §5 beyond "no large effect."

### Explicitly deprioritised

- **xG/xA modelling.** Perfecting it entirely is worth +0.045. It is done.
- **Optimiser sophistication.** Measured, no effect under three estimators,
  dominated by window variance.
- **More data sources**, *unless* they carry information the model cannot already
  infer. The constraint is not data volume; it is that the one thing that matters
  most — who starts — is published as text on Friday, not as a stat.

---

## What would change these conclusions

- If the November DefCon re-fit shows the positional priors moving materially,
  §3's under-dispersion finding becomes a live modelling defect and DefCon
  rejoins Tier 1.
- If a lineup source clears `P(come in | benched) > 0.40` on a stated denominator,
  Tier 1 item 1 could close a large fraction of the 0.2118 minutes gap on its own,
  and §1's ceiling analysis should be re-run immediately after to find the new
  bottleneck.
- If the optimiser comparison at 20+ independent windows shows shrinkage helping
  by even +2 points reliably, that is nearly free and should be adopted despite §5.
- If `Var(p)` estimated from realised outcomes puts our clean-sheet Brier well
  above the floor, §3's "substantial majority captured" is too generous and
  clean sheets rejoin the priority list.
