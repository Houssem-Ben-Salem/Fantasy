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
| minutes prior | `python analysis/validate_minutes_prior.py` |
| goal-model ridge | `python analysis/validate_goal_ridge.py [--stage2]` |

---

## 0. Changes since this document was written — read first

**2026-10-11: the first prospective test, one model gap closed, two infrastructure breaks.**

*The fixes held up on data nobody had seen — but only on minutes.* GW5 2026-27 is
the first gameweek where both the old model (CI's export, 18 Sep 14:19 UTC) and the
fixed one (17 Sep) committed to projections before kickoff. Over 659 players,
team-cluster bootstrap:

| GW5 | Spearman | RMSE | top-50 avg | p_start Brier |
|---|---|---|---|---|
| old model (CI) | 0.620 | 2.41 | 4.00 | 0.195 |
| fixed model | **0.738** | **2.19** | **4.92** | **0.086** |
| naive GW1-4 points/week | 0.689 | 2.35 | 3.68 | 0.115 |

Every gap excludes zero. Two things temper it. The old model was **worse than the
naive baseline** — what CI published every Friday since the mirror stalled was worse
than doing nothing clever. And **among the 220 who actually started, no model ranks
points**: Spearman 0.08-0.14, fixed minus naive +0.009 [-0.088, +0.097]. The whole
edge is predicting who plays. One gameweek of starters' points is mostly noise;
whether the points model is any good needs many weeks of prospective scoring.

*Attacking returns were fixture-blind.* The module docstring said "scaled by opponent
strength"; the code was `xg90 * mins_frac`. Fixed with `ATTACK_FIXTURE_ALPHA = 0.75`,
evidence and caveats in that constant's comment: per fixture the gain replicated
out-of-sample (+0.044, +0.053) but neither stage excludes zero once bootstrapped over
fixtures. Backtest 0.3263 -> 0.3295 (22/31 weeks), concentrated in forwards. Earlier
market corrections (GW3 Guéhi, GW5 Wissa) scaled attack against a baseline that never
contained the fixture, so they over-boosted; the GW5 call survives a correct one.

*Versus the market, on outcomes:* GW5 log-lik -33.75 ours vs -33.81 Pinnacle; GW6
(8 of 10 played) -21.81 vs -21.91. Two dead heats over 18 matches — parity at most,
not an edge. The standing disagreements are the promoted clubs (GW6: EVE at HUL,
Pinnacle 1.52 vs ours 0.76). HUL's defence rating moved -0.305 -> -0.201 after one
more match, the direction `check_promoted_prior.py` predicted. Hull have conceded 4
against 7.40 xGA: the rating is fitted on goals, and goals flatter them.

*Mover minutes* (`check_mover_minutes.py`, movers -0.119 vs stayers, CI excludes 0)
pools four runs of which three are the collapsed pre-fix model. Re-measure on post-fix
runs only before acting on it.

*Infrastructure.* CI failed every daily run for 17 days: `pulp>=2.7` was unbounded and
PuLP 4.0 (25 Sep) stopped bundling CBC. Locally, macOS 27 left PuLP's x86_64-only CBC
unable to execute. Both fixed (`optimize._solve`, capped `requirements.txt`), and the
fallback exposed `int(var.value())` truncating 0.9999999 to 0 — a 14-man "Optimal"
squad. `transfer_plan` also paired its report in element-id order, printing a
midfielder as swapped for a defender. See CLAUDE.md, Optimiser conventions.

**2026-09-15: three defects fixed, each validated out-of-sample before shipping.**
Numbers below §0 were produced before these fixes unless marked. Where a fix
moved a headline figure, it is recorded here rather than silently overwritten.

| fix | commit | out-of-sample evidence |
|---|---|---|
| Minutes: in-season `p_start` collapsed to exactly 0/1, and `player_gw` was stuck on GW1 | `87eac4d` | Brier **+6.5%** on 2025-26 (7/7 weeks), **+10.9%** on 2026-27 with k frozen (3/3) |
| Dixon-Coles: ratings pinned at the ±3 bound on thin promoted clubs | `79e4712` | held-out LL +0.048 (6/8 clean weeks) and +0.053; CS correlation with Pinnacle **0.563 → 0.792** |
| `evaluate_chips` valued an unfieldable XI (top 11 by points) | `2d412c6` | exact; `bench_boost_gain` understated by up to 0.56 where illegal |

**Re-run after the fixes — headline claims that held:** the ceiling analysis
(§1–2) is unchanged to three decimals: 41.2% of achievable signal captured,
minutes still the largest gap at +0.2105. The 2025-26 backtest is unharmed
(Spearman 0.3261 → 0.3263). DefCon calibration (§3) is identical, as it must be —
DefCon does not use the goal model.

### The minutes collapse is §2's warning realised

§2 says minutes is 47% of the addressable gap. From 3 Sep to 15 Sep every live
projection's `p_start` was "did he play 60+ minutes in GW1": `minutes_model`
switched from prior-seasons-only to current-season-only on the first
current-season row, with no prior, while the mirror published GW1 and nothing
after. 92% of players sat at exactly 0 or 1; 85 were provably misclassified.
Same defect class as the pinned ratings — an unshrunk small-sample estimate —
in the most important component. It was caught only because a user transfer
produced a verdict (Semenyo 3.79 xP vs Cherki 1.13) that did not survive a look
at Cherki's minutes.

### Retraction: the clean-sheet over-dispersion finding roughly halves

An earlier version of §3 said clean sheets were mildly over-dispersed (β = 0.844),
and a later session cited our wider-than-market clean-sheet spread as
**independent corroboration** of that. Both need correcting, and the second
correction goes further than first drafted.

1. **The market-spread corroboration is withdrawn.** It was measuring the pinned
   COV/HUL ratings, not a modelling choice. Excluding those fixtures our spread
   was already 0.096 against the market's 0.082.

2. **β itself was partly the same defect.** The first draft of this retraction
   argued β = 0.844 "stands on its own evidence", since it was fitted on
   full-season 2025-26 data where no club had two matches. Measured, that is
   wrong. Refitting the identical backtest with only the ridge changed:

   | clean sheets | raw MLE | ridge λ=2 |
   |---|---|---|
   | calibration slope β | 0.844 | **0.926** |
   | gap to perfect calibration | 0.156 | **0.074** |
   | Var(p) / debiased Var(π) | 0.00991 / 0.0087 | 0.00891 / 0.00941 |
   | Brier | 0.1891 | 0.1884 |

   **53% of the over-dispersion came from the unregularised fit** — not only
   via clubs at the bound, but because raw MLE over-disperses every rating a
   little, reading noise as signal, even with a full season of data.

   What survives is a small residual, and at that size the two estimators no
   longer agree on its sign: β = 0.926 < 1 says slightly over-dispersed, while
   debiased Var(π) > Var(p) says slightly under. Treat clean-sheet calibration as
   close to correct, with no reliable direction. §3's table below still shows the
   pre-ridge figures.

### Minutes do not reallocate within a squad

Found 2026-09-15 preparing GW5. `minutes_model` estimates each player's start
probability independently. When a teammate is suspended or injured, *their*
`p_start` correctly drops to 0 — but nobody else's rises. Minutes are conserved in
real football and not in the model.

Concrete case: Manchester City lost Foden (red card in the GW4 derby, suspended
until 17 Oct, so GW5 and GW6) and Doku (calf, back ~20 Sep). Cherki, a
rotation-risk midfielder in exactly that role, stayed at `p_start` 0.49. The
model therefore recommended selling him for Gibbs-White (+6.62 xP over GW5-9),
with 78% of that gain landing after the international break — and the GW5-6
share overstated, because Cherki's minutes there are understated.

This is the in-season face of Tier 1 item 3 (manager-level rotation). A fix would
redistribute an absent player's expected minutes across same-position teammates
in proportion to their recent usage — a modelling change, so it needs the same
out-of-sample test the minutes prior passed before it ships. Until then, treat
any `p_start` for a player whose direct competitor is flagged as a lower bound,
and let team news decide.

### A measured null that is conditional, not settled

The optimiser commits to one XI for the whole horizon rather than re-picking
weekly. Measured on the 2026-27 squad, weekly re-selection was worth only
**+0.48 xP over GW3–7**, because 3-5-2 was optimal in all five weeks. That is a
statement about that squad's fixture diversity, not about the simplification.
**Re-run `analysis/check_chip_xi_legality.py`'s weekly comparison whenever the
squad's players' fixtures diverge** — around a double or blank gameweek, or after
a wildcard that mixes easy and hard runs.

### The ridge shrinks toward the wrong TARGET for promoted clubs

`GOAL_RIDGE_LAMBDA` fixed the *variance* of a thin club's rating — nothing
diverges to the ±3 bound any more. It did not fix the *centre*. Shrinking toward
zero means shrinking toward league average, and a promoted club's honest prior is
worse than average. Four matches can therefore still carry a promoted side to a
top-of-table rating, which is what 2026-27 HUL did: **defence −0.305 on four
matches, second-best in a 25-team table**, on a record of 2 conceded in 4.

Measured on the only promoted cohort the database holds (2025-26 BUR/LEE/SUN),
the rating after 4 matches vs the full season moved **+0.174 for promoted clubs
and −0.003 for the other 17** — all three the same way, SUN going from
better-than-average to below it. n=3, sign test p=0.25: **this establishes a
direction and a mechanism, not an effect.** `analysis/check_promoted_prior.py`.

**It is not the explanation for the fixture that exposed it.** GW5 2026-27 had
NEW at home to HUL at 1.07 xG against Pinnacle's 2.02. The measured +0.174 buys
0.20 of that, the league-wide level shortfall below buys 0.17, and **0.57 — 60%
of the gap — is unexplained.** Recorded because the pull to close a gap with the
nearest available bias is exactly how §3's clean-sheet claim went wrong.

Not fixed: a nonzero shrinkage target moves every rating in the table and needs
`validate_goal_ridge.py`'s two-stage out-of-sample treatment, which does not fit
between finding it and a deadline.

### The model runs ~0.35 goals/match below the market, and 4 GWs cannot judge it

GW5 2026-27, all 20 team-fixtures against Pinnacle: **xG correlation 0.780, CS
correlation 0.817**, our CS spread 0.118 against the market's 0.106. The CS
figures are the ridge fix showing up against an outside benchmark — the same
comparison in GW3 needed COV and HUL excluded to reach 0.888, and this one
excludes nothing.

The level is the open question. We project **2.68 goals/match, Pinnacle 3.03,
realised 2.85** across 40 matches. With a per-match sd of 1.83 the SE on that
realised figure is 0.29, so **both sit inside the 95% interval [2.28, 3.42]** —
0.6 SE either side. A season of results cannot yet say which is closer, and the
ridge's known level cost (−1.5% at λ=2) accounts for only about 0.04 of the 0.35.
Re-run this each gameweek and let the CI narrow rather than acting on one week.

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

## 3. The probability components — better than they looked, and now measurable

My first read was that clean-sheet prediction was weak: Brier 0.1891 against a
0.1957 base-rate baseline, only 3.4% better. That read was wrong, and wrong in the
same way as reading 0.33 against 1.0.

Writing `π` for the true conditional probability of each case,

```
BS = E[(p - π)²] + E[π(1 - π)] = MSE(p, π) + p̄(1 - p̄) - Var(π)
```

so **`Var(π)` is the entire reducible portion** and `captured = (baseline - BS) / Var(π)`.
Everything below that floor is coin-flip noise no model removes.

### Estimating Var(π) is the whole difficulty

An earlier draft substituted `Var(p)` — the variance of our *own* predictions —
which assumes the model is calibrated, i.e. assumes the answer. Doing that gave
DefCon a "captured" share of 143%, which is not a triumph but a **falsification of
the floor**: a Brier below a floor derived from our own predicted variance can only
happen if `Var(π) > Var(p)`.

That crossing is a *bound*, not merely a sign. Rearranging the identity,

```
Var(π) - Var(p) = [floor_est - BS] + MSE(p, π)    and MSE ≥ 0
```

so `Var(π) > Var(p) + (floor_est - BS)` strictly. Because both terms come from the
same sample their errors are correlated, so the crossing is bootstrapped on
**paired** resamples rather than composed from separate standard errors.

The independent measurement is the **calibration slope**: regress `y` on
`logit(p)`. `β > 1` means the true probabilities are more spread out than the
predictions — under-dispersion — and `β` is the correction factor itself. The
crossing establishes existence; the slope gives magnitude.

`Var(π)` itself is then estimated from a **binned resolution term with the
within-bin sampling variance subtracted**. Without that subtraction, bin noise
inflates resolution and reproduces the identical class of error one level up.

### Results

| | clean sheet | DefCon |
|---|---|---|
| n | 3,049 | 5,694 |
| baseline Brier | 0.1957 | 0.1637 |
| our Brier | 0.1891 | 0.1410 |
| naive read | 3.4% better | 13.8% better |
| `floor(Var(p)) − BS` | −0.0033 | **+0.0068**, 95% CI (0.0043, 0.0093) |
| calibration slope β | 0.844 *(0.926 after ridge — see §0)* | **1.239** |
| Var(p), circular | 0.00991 | 0.01582 |
| Var(π), debiased | 0.00870 | 0.02348 |
| **captured** | ≤ **75.8%** | ≤ **96.3%** |

**DefCon is significantly under-dispersed**, by two independent routes that agree:
the paired-bootstrap crossing excludes zero, giving `Var(π) > 0.02262`, and the
binned estimate lands at 0.02348 — just above that bound, as it must. `β = 1.239`
confirms the direction.

Both `captured` figures are **upper bounds**, because binning coarsens away
within-bin variation and so understates `Var(π)`, which sits in the denominator.

### "captured ≤ 96.3%" and "β = 1.239" are not in tension

They look contradictory — a model capturing 96% of achievable Brier ought to be
well calibrated, and a well-calibrated model has β ≈ 1. The Murphy decomposition
shows they measure **different terms of the same identity**:

```
BS = reliability - resolution + uncertainty

clean sheet   reliability 0.00254   resolution 0.00931   uncertainty 0.19570
DefCon        reliability 0.00167   resolution 0.02373   uncertainty 0.16367
```

`captured` is about **resolution** — how much of the real spread in π the model
detects. `β` is about **reliability** — whether the detected spread is correctly
scaled. A model can be highly informative and slightly compressed at the same
time, and DefCon is exactly that.

This bounds the recalibration prize precisely. **Reliability is the only term any
recalibration can remove, and for DefCon it is 0.00167 — 1.2% of the Brier.**
That is the ceiling, before asking whether it is achievable at all.

### The recalibration does not survive out-of-sample validation

Fitting `(a, b)` on early gameweeks and evaluating on later ones:

```
fit gw<20, test gw>=20:   a=+0.398 b=1.183   Brier 0.1366 -> 0.1367   -0.09%
fit gw<24, test gw>=24:   a=+0.435 b=1.224   Brier 0.1329 -> 0.1335   -0.47%
fit gw<28, test gw>=28:   a=+0.367 b=1.217   Brier 0.1325 -> 0.1321   +0.30%
```

Two of three are negative and all three sit inside ±0.5%. The in-sample gain
(0.1410 → 0.1400) is fitting noise, which the 1.2% reliability ceiling already
predicted. **β is stable and real; correcting for it buys nothing measurable.**

Three further facts make this decisive rather than merely discouraging:

1. **Recalibration cannot reorder anyone.** `sigmoid(a + b·logit p)` is strictly
   monotone, so the Spearman correlation between raw and recalibrated `p_defcon`
   is 1.000000 exactly. It cannot improve which defenders the model prefers — only
   how much weight DefCon carries against goals, assists and clean sheets.
2. **Slope-only recalibration is actively harmful.** Applying `β` without the
   intercept moves the mean to 0.1633 against an actual rate of 0.2062 and makes
   the Brier *worse* (0.1410 → 0.1427). The phrase "scale the logits by β",
   which an earlier draft of this document used, describes a change that degrades
   the model.
3. **It still moves the squad.** 13 of 15 survive; Semenyo and Mykolenko out,
   Anderson and Virgil in, two of them in the XI. So the effect is not nil — it
   perturbs decisions without evidence that it improves them, which is the worst
   available combination.

This is §5's pattern one level down: a principled correction, correctly derived
from a real defect, that does not survive measurement.

### A correction to an earlier correction

A previous draft of this section claimed clean sheets were under-dispersed too,
"more mildly and in the same direction", on this evidence:

```
clean-sheet spread across teams
  modelled  sd 0.072      realised  sd 0.099
```

**That was wrong, and wrong by the same mechanism the section exists to fix.** The
realised spread is a set of proportions and carries sampling noise; the modelled
spread does not. Comparing them directly is the un-debiased comparison all over
again. Correcting it requires knowing the independent sample size per team — and
that is *not* the ~152 player-rows, because every GK and defender from one club in
one fixture shares the same clean-sheet outcome (§4 measures r = 0.51 among
same-team defenders). The independent unit is the **match**, about 30 per team:

```
realised, raw                       sd 0.099
  debiased as if rows independent   sd 0.093   (wrong by ~5x)
  debiased per independent match    sd 0.061   <- correct
```

So the model's spread (0.072) is *wider* than reality's (0.061): clean sheets are
mildly **over**-dispersed, which is exactly what `β = 0.844 < 1` says. *(Since
corrected: 53% of that was the unregularised goal model — see §0.)* The two
measurements agree once the noise correction is right. §4's correlation finding
turns out to bite §3's dispersion check.

Direction matters for how the numbers are read. Where `Var(π) > Var(p)`
(DefCon), a figure computed with `Var(p)` is an upper bound; where
`Var(π) < Var(p)` (clean sheets), it is a lower bound. The sign is measured, not
assumed.

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
points; which optimiser you use swings it by ±9. **Window choice dominates
strategy choice by 4.5×**, which is the most useful number in this section.

### What this design could ever detect

Two-sided α = 0.05, 80% power. The two strategies differ by an order of magnitude
and must not be given a common threshold:

| strategy | SD | SE | MDE at n=8 | n for +2 | n for +5 | n for +15 |
|---|---|---|---|---|---|---|
| shrunk | 2.9 | 1.0 | **2.8 pts** | 17 | 3 | 1 |
| resampled | 19.7 | 7.0 | **19.5 pts** | 760 | 122 | 14 |

Shrinkage barely moves the squad, so its differences are tight and a small effect
is reachable. Resampling reshuffles it wholesale, so its differences are wide and
a small effect is not reachable at any realistic n. Read the `n for` columns
against §7's ceiling of about **five independent windows per season** — so in
seasons, not in windows.

### The autocorrelation cannot settle anything at this n

The 8 windows overlap by half, so the paired differences ought to be
autocorrelated, making a plain SE anti-conservative. The temptation is to check
the sample ACF and declare the matter resolved. It cannot be.

For **independent** draws with an estimated mean, the sample lag-1
autocorrelation is biased downward: `E[r₁] ≈ −1/(n−1)`, which at n = 8 is
**−0.143**, with `SE(r₁) ≈ 1/√n = 0.354`. Against that reference:

```
                              r₁       distance from white noise
levels      naive          +0.178            +0.9 SE
levels      shrunk         +0.201            +1.0 SE
differences shrunk         -0.167            -0.1 SE
differences resampled      -0.071            +0.2 SE
```

The differences sit **on** the white-noise expectation — so there is no negative
dependence to explain, just small-sample bias. And the levels, which *are*
dependent by construction, land under 1 SE from white noise, so the ACF cannot
even detect the dependence we know is there.

An earlier draft read the negative difference-ACFs as evidence that "pairing
removes the dependence." That over-read the statistic: −0.167 at n = 8 is what
independence looks like, not evidence of anything.

**The warrant for using the naive SE is the moving-block bootstrap**, which
tolerates dependence and returns essentially the same interval (+0.2 [−1.6, +2.0]
against +0.2 [−1.7, +2.2]). Two estimators agreeing is the evidence. The ACF
carries no weight here in either direction, and `exp3b_power.py` now prints the
white-noise reference alongside it so it cannot be over-read again.

### Deciding an unresolvable question

"No feasible experiment separates resampling from zero" is a result, not a shrug.
760 independent windows is roughly 150 seasons. The question will never be
settled on evidence, so it has to be settled on other grounds — and those all
point the same way:

- **naive is one solve; resampling is sixteen** (fifteen perturbed plus a final
  frequency solve), so it is ~16× the compute for an effect indistinguishable
  from zero.
- **naive has fewer moving parts.** Resampling adds a perturbation scale, a
  resample count and a seed, none of which can be tuned against evidence for the
  same reason the effect cannot be measured. Untunable parameters are places for
  silent misconfiguration to live.
- **the point estimate is +1.9 with a CI spanning ±16**, so nothing is being
  given up.

**Keep the naive solve.** Written down here so the option is not reopened every
time someone rediscovers Michaud — the argument for resampling is sound in
theory, and that is exactly why it needs a standing answer rather than a
re-derivation.

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

### Three instances, and what they share

This section has now failed three remedies, each principled, each correctly
derived from a real defect this document measured:

| remedy | the defect it targets | verdict |
|---|---|---|
| resampled efficiency | selection bias (§4) | +1.9, CI ±16; unresolvable at any feasible n |
| shrinkage | estimation error in the inputs | +0.2, CI [−1.7, +2.2] |
| DefCon recalibration | under-dispersion, β = 1.24 (§3) | prize capped at 1.2%; −0.09/−0.47/+0.30% out-of-sample |

**What they share is not that they are bad ideas. It is that all three transform
existing estimates rather than adding information.** Resampling reshuffles how we
choose from the projections; shrinkage rescales them; recalibration rescales one
component's probabilities — and being strictly monotone, it cannot even reorder
the players it rescales.

That is the sharp form of the rule, and it is predictive rather than merely
descriptive: **a transformation of what we already know cannot manufacture what we
do not.** Applied to the roadmap, it says the only candidate that has not yet
failed a test is the one that is not a transformation — Friday team news, which
carries information the model provably cannot infer, because §2's oracle shows
minutes are 47% of the addressable gap and no rearrangement of current inputs
closes any of it.

It also explains why Tier 1 item 1 must be a *measurement* rather than a build. A
lineup source that re-encodes what persistence already knows is a transformation
wearing the costume of new information, and will fail exactly as these three did.
`P(come in | benched)` is the test that tells the two apart.

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

10. **Do NOT recalibrate DefCon yet.** A previous version of this list promoted
    it as a free win. §3 now shows the reliability term caps the prize at 1.2%
    of the Brier, three out-of-sample splits deliver none of it, the transform
    cannot reorder players at all, and it still churns 2 of 15 squad places.
    Revisit only with 2026-27 data, and only if `(a, b)` refit there still
    improves a held-out Brier. Left as a live item because the *defect* is real
    and stable (β = 1.18–1.22 across splits) — it is the *fix* that fails.
11. **Re-fit DefCon priors in November** on 2026-27 data. One season of a
    one-season-old rule, and §3 shows the fitted spread is too narrow — both
    point the same way. This remains the higher-value of the two, because more
    data addresses the cause rather than rescaling the symptom.
12. **Widen the optimiser comparison only for shrinkage, and only to ~17
    independent windows** (§5's power table). Resampling is closed: no feasible
    experiment separates it from zero. Do not spend windows on it.

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
- **Shrinkage at +2 points is detectable, but costs ~17 independent windows**
  (≈3–4 seasons at five per season, so **2029 or not at all**). An earlier draft
  set this trigger at "20+ windows", which resolves ±8 at best — the measurement
  would never have concluded. Kept because the effect size is plausible and
  shrinkage is nearly free to adopt, but say the date out loud: a threshold that
  implies a check next spring is worse than no threshold.
- **Resampling is closed on power grounds, not on evidence.** Detecting +2 needs
  ~760 windows, roughly 150 seasons. Even +15 needs 14. No feasible experiment
  will separate it from zero, so the point estimate of +1.9 is where it stays.
- If the November DefCon recalibration (scaling logits by β ≈ 1.24) moves the
  Brier materially, §3's ≤96.3% was too generous and DefCon has more headroom
  than it appears.
