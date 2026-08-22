# LEARNINGS — what building this pipeline actually taught

This is a learning document, not a project-status document — `PROJECT_CONTEXT.md`
already covers what FanTeasy Stats *is* and where it stands. This is about the
ideas behind it: what they mean, why they mattered here specifically, and what
went wrong before it went right. Terms are defined the first time they come up,
on the assumption you're learning them, not just refreshing them.

---

## 1. How the pieces fit together

The pipeline is six layers, each one built on top of the last, and — this is
the thing worth actually internalizing — **each layer was checked against
something external before the next layer was allowed to depend on it.** Not
"the code ran without an error," but "the output matches a fact about the real
world that didn't come from this code." That discipline, repeated six times, is
the real subject of this document; the bugs and reversals below are all
examples of what happens in the layer that skips it.

**Ingestion** (`src/ingest.py`) pulls raw data from two places: `nflreadpy`
(play-by-play, weekly stats, snap counts, Next Gen Stats, the NFL schedule) and
Sleeper's public API (league settings, scoring rules, projections, real
matchups). It produces cached files under `data/raw/`. The thing this layer
lives or dies on is quieter than it sounds: joining Sleeper's player IDs to
nflverse's player IDs. Get that join wrong and every later layer returns
technically-valid, completely-wrong numbers — a join that returns zero rows at
least fails loudly; a join that returns the *wrong* rows doesn't fail at all,
it just lies. This was checked against a known real mapping (a specific
player's Sleeper ID matching their known nflverse ID) before anything else was
built on it.

**Scoring** (`src/features.py`, `compute_custom_score()`) turns raw box-score
stats into this league's actual fantasy points — not generic PPR, this
league's specific rules (0.5 PPR, a stacking fumble penalty, no yardage
bonuses). It produces the `custom_points` column everything downstream trains
on. This is the layer with the highest stakes for silent corruption, so it got
the strictest validation: not "does this look like the settings dict says it
should," but a diff against Sleeper's own **actual awarded points** from real
historical matchups — 739 rostered player-weeks, every position including
kicker, 100% exact match, zero mismatches. Seven non-obvious scoring rules
(documented in `CLAUDE.md`) were only discovered *because* this diff existed;
reading the settings dict alone would have missed all seven.

**Features** (`src/usage.py`) builds ~180 columns describing usage, role,
efficiency, situational tendencies, game context, and rolling trends — plus
**expected fantasy points (xFP)**, defined below. It produces
`weekly_features.parquet`. Validated two ways: leakage tests (does any column
secretly see the future — see Section 2), and direct sanity checks (do target
shares actually sum to ~1.0 per team per week, are snap shares between 0 and 1,
does aDOT — average depth of target — rank sensibly by position). xFP itself
got its own reproduction check: summing xFP's own per-play scores over a
player's real plays and comparing that sum to their real `custom_points`.

**Model** (`src/model.py`) trains one untuned LightGBM regressor per position
(QB/RB/WR/TE) on the safe subset of those ~180 columns, validated with
**walk-forward validation** (Section 2) against four baselines: season-to-date
average, trailing 3-week average, Sleeper's own projection, and trailing xFP. A
model that doesn't beat simple baselines isn't ready, and for a long time this
one didn't — see Section 4.

**Calibration** takes the model's **quantile regression** output (predicted
10th/25th/50th/75th/90th percentiles per player-week) and corrects it using
**conformalized quantile regression (CQR)** — both defined in Section 2 —
because the raw quantile intervals were measurably too narrow. Validated by
checking **coverage** (Section 2) on a calibration split that was strictly
*before*, in time, the weeks being evaluated — never on the fold being
corrected or anything later.

**Simulation** (`src/simulate.py`) samples from those calibrated quantiles,
using a **Gaussian copula** (Section 2) to correlate players who share a real
NFL game, to simulate fantasy matchups and full-season standings. Validated
against 204 real historical matchups and 8 real season-snapshot combinations,
using the same **calibration** idea as the model layer — binning predicted
probabilities and checking the actual rate tracks them.

Read top to bottom, the shape is: ingest → score → describe (features) →
predict (model) → correct the prediction's honesty (calibration) → use it
(simulation). Read for the lesson, the shape is: nothing got trusted just
because it compiled — every layer had a specific, external fact it had to
match first.

---

## 2. Vocabulary

**Point-in-time correctness.** A feature used to predict week N's outcome may
only use information that existed *before week N's kickoff*. This sounds
obvious until you hit the concrete way it breaks: pandas' `.rolling()` window
function includes the *current* row by default. A careless "trailing 3-week
average" for week N therefore quietly includes week N's own result — the model
isn't predicting the future, it's reading the answer off the back of the test.
The fix used throughout this project is mechanical and boring on purpose:
`.shift(1)` every rolling or expanding feature before the window is applied, so
week N's feature can only ever see weeks before N.

**Data leakage — two distinct kinds.** The first kind is leakage in a
**feature's own value**: a column that, for week N, actually encodes
information from week N or later (the failure mode point-in-time correctness
above is about). This project catches it with a specific two-part test
pattern in `tests/test_no_leakage.py`: build the feature table twice — once
from the full dataset, once with everything after some week N removed — and
assert week N's own feature rows are identical either way. If a feature
"changes" depending on whether the future exists, it was reading the future.
The second, subtler kind is leakage in **feature selection** — not any single
column's *value* being wrong, but the *process that chose which columns to
use* having seen information it shouldn't have. This project hit this exact
bug (Section 4) and it's worth sitting with why the first kind of test doesn't
catch the second: every individual feature can be perfectly point-in-time
correct, and the *ranking used to pick the best N of them* can still be
computed using data a real forecasting fold would never have had yet. A
leakage suite built only around checking values will pass cleanly while this
kind of leak sits undetected.

**Walk-forward validation.** The alternative to a random train/test split for
time-ordered data. Sort everything chronologically; for each period you want
to evaluate, train only on data strictly before it, predict that period, then
advance and repeat with a bigger training set. Never shuffle, never use
k-fold cross-validation (which would let the model train on rows that are, in
real time, *after* some of its test rows). This is sometimes called
"expanding-window" validation because the training set keeps growing as you
move forward instead of resetting each time.

**EWMA vs. expanding mean vs. volatility.** Three different ways to summarize
"how has this player been doing," each answering a different question.
*Expanding mean* is the average of everything so far this season — a
season-to-date rate, weighted equally regardless of how long ago it happened.
*EWMA* (exponentially weighted moving average) is a trailing average that
weights recent weeks more than older ones — a "~3-week half-life" means
roughly half the weight in the number comes from the last three weeks, so it
reacts to a role change faster than a flat expanding mean would. *Volatility*
here means expanding standard deviation — not a location (where's the
average) but a spread (how much has this player's own output bounced around)
— a consistency signal, useful for telling the model how much to trust a given
player's other numbers.

**Quantile regression.** Instead of training a model to predict one number
(the expected/average outcome), train it to predict a specific *percentile* of
the outcome's distribution. LightGBM does this with `objective='quantile'`
and an `alpha` parameter naming which percentile (0.1 for the 10th, 0.9 for
the 90th, etc.). Training five separate models at alpha = 0.10, 0.25, 0.50,
0.75, 0.90 gives five points that sketch the whole shape of a player's
possible range — "usually 10-18, occasionally 30" — instead of one flat
number that hides all of that.

**Coverage.** The honesty check for a quantile prediction. If a model's
"10th-percentile" prediction is well-calibrated, then across many real
outcomes, roughly 10% of them should actually fall at or below it — no more,
no less. If 19% of real outcomes fall below the stated 10th percentile, the
model's floor isn't pessimistic enough; the number is lying about how bad
things can get.

**Conformal prediction / CQR.** A statistical correction applied *after* a
model is already trained — no retraining involved. Take a held-out
**calibration split** (data the correction is derived from, always
chronologically earlier than what it corrects), measure exactly how far real
outcomes fell outside the model's stated interval (a **conformity score**),
and widen (or narrow) the interval by precisely the amount needed to hit the
target coverage rate — with a mathematical guarantee behind that correction,
not just an empirical hope that it worked. **CQR** (conformalized quantile
regression) is this technique applied specifically to quantile-regression
intervals, which is what this project used.

**Calibration**, the general idea. Whether a model's stated confidence
matches reality, at whatever layer you're asking it. A model that says "70%"
a hundred times should be right about seventy of those times. Checked by
binning predictions into probability ranges and comparing the *predicted*
rate in each bin against the *actual* rate — the exact same check applies
whether the "70%" is a quantile floor, a fantasy matchup's win probability, or
a team's odds of making the playoffs.

**SHAP** (SHapley Additive exPlanations). A way of explaining one prediction
by fairly splitting credit for it among the input features that produced it.
It borrows a concept from game theory — the *Shapley value* — for "how much
did adding this player change the outcome, averaged over every possible order
players could join a team." Applied to a model, "players" become features:
how much did knowing this feature's value change the prediction, averaged over
every way the features could have been considered. Averaging `|SHAP value|`
across many predictions gives a ranking of which features the model actually
leans on — useful both for explaining a model and, as this project
discovered, for catching leaks (a feature that "shouldn't matter" showing up
high on this list is a leak or a bug, not a discovery).

**Expected fantasy points (xFP).** Score the *opportunity*, not the outcome.
For every kind of play a player could get (a target at a given depth, a carry
inside the 5-yard line, etc.), compute what that opportunity is worth *on
average across the whole league*, then apply those league-average rates to a
player's *actual* opportunities that week. The result — xFP — strips out
touchdown luck and randomness, leaving a number that reflects how much
opportunity a player got, independent of whether it converted. The gap
between what actually happened and xFP (`fp_over_expected`) is a rough proxy
for how lucky or unlucky a player has been.

**Gaussian copula and correlation.** A *copula* is a way of gluing two random
things together so they move *with* each other, without forcing either one
to have a particular shape on its own. Concretely, in this project: draw one
shared random number representing "how did the game environment go" (a
shootout vs. a defensive slog), blend it with each player's own individual
random draw, and the two blended results end up correlated by an amount you
control (`rho`) — while each one, on its own, still has whatever realistic
shape it's supposed to have (each player's own calibrated quantiles).
"Gaussian" just names which bell-curve-shaped randomness is doing the gluing
underneath (via z-scores and the normal distribution's CDF), even though the
actual fantasy-point distributions being correlated aren't bell-curve shaped
themselves.

**MAE vs. RMSE vs. Spearman — and when each matters.** *MAE* (mean absolute
error) averages `|prediction − actual|` — every miss counts the same
regardless of size, and the number is directly interpretable ("off by 4
points on average"). *RMSE* (root mean squared error) squares each error
before averaging, then takes the square root — this punishes a few
disastrous misses much harder than being slightly off everywhere, which
matters when blowout weeks are what actually decide outcomes. *Spearman
correlation* ignores the size of errors entirely and only asks whether the
*ranking* is right — did the model correctly say who'd outscore whom. For a
start/sit decision, getting the order right is often what actually matters,
even if the exact point totals are off.

---

## 3. Pitfalls, and how each was caught

The catching method is the point here more than the bug itself — each of
these is a worked example of a specific *kind* of check that generalizes well
beyond this project.

- **`pass_attempt` firing on sacks.** The raw play-by-play data flags a sack
  as a pass attempt, which doesn't match what a real published box score
  calls a "pass attempt." *Caught by*: computing a stat from the raw data and
  checking it against a real, independently-published box score for a
  specific game — a fact from outside the pipeline, not a property of the
  pipeline's own internal consistency.

- **Kneels padding designed rushes.** A QB kneel-down (running out the clock)
  is flagged `rush_attempt=1` in the raw data, which would otherwise inflate
  a "designed rush attempts" feature with plays that carry zero fantasy
  meaning. *Caught by*: domain knowledge applied *while building* the
  feature, not a test that fired later — a reminder that some pitfalls are
  best avoided by knowing the sport, not by writing more assertions.

- **The xFP double-count from copying the pbp slice.** An early version of
  the xFP feature let the raw play-by-play's own play-level `passing_yards`
  column leak into the synthetic per-play frame used to score a *receiver's*
  expected points — silently adding a slice of the *quarterback's* passing
  yards onto every target's expected score. *Caught by*: the xFP-vs-real-play
  reproduction check (Section 1) turning up a mismatch, which led back to the
  bug — an example of a validation check earning its keep by catching
  something nobody was specifically looking for.

- **The two-point-conversion mismatch, found by a full team-week audit.**
  Team-level denominators (used for target share and situational shares) were
  silently including two-point conversion attempts alongside normal pass
  attempts, distorting the denominator by a small, consistent amount.
  *Caught by*: checking that shares summed to 1.0 for *every* team-week that
  existed, not a sample of plausible-looking ones — the discrepancy was small
  enough that a handful of spot-checks plausibly would have missed it
  entirely; an exhaustive, cheap-to-run check found it because it looked at
  everything.

- **The biased ablation ("peaks at 25 features").** Covered in full in
  Section 4 — a feature-count experiment that looked like textbook
  overfitting and wasn't.

- **`<` vs. `<=` on a zero-inflated column.** Two coverage-reporting
  functions asked the logically identical question — "is the actual outcome
  at or below this predicted quantile" — using different comparison
  operators. For almost any column this distinction is invisible, since a
  floating-point tie is astronomically unlikely. But 19.4% of TE (tight end)
  player-weeks score *exactly* 0.0 (inactive, no recorded stats), so ties at
  the boundary are common, not rare, and the two functions' reported figures
  for the same statistic differed by about 13 percentage points. *Caught by*:
  computing the same quantity two different ways and noticing the two
  answers didn't match — a general-purpose check worth remembering: if two
  code paths are supposed to agree, actually run them both and compare,
  rather than trusting that they will.

- **Cross-pair quantile crossing after CQR.** CQR widens the 10th-90th and
  25th-75th prediction intervals by *different* constant amounts, since each
  pair is calibrated to correct exactly its own miscoverage. That can push
  the corrected 25th percentile *below* the corrected 10th percentile, even
  though each pair was individually in order before the correction was
  applied. *Caught by*: reasoning through what could go wrong *before*
  writing the code that would consume these five points (the simulator's
  inverse-CDF sampler) — a proactive catch, not a symptom-driven one. Worth
  naming as its own category: some bugs are worth thinking your way to,
  rather than waiting to trip over them in production.

- **The stale-cache and warmup-anchoring confounds in the data-volume test.**
  Two separate problems, both found while *designing* the 2-vs-4-vs-8-season
  comparison, before it was ever run for real. First: the walk-forward
  warm-up period anchors to the first season present in whatever dataframe
  you hand it, so a "2-season" run and an "8-season" run would silently
  evaluate *different* sets of weeks unless the evaluated fold set is
  explicitly locked identical across every condition. Second: nflverse's own
  snapshot of a season can be revised after the fact — a fresh fetch can
  return extra rows a stale cached fetch doesn't have — so comparing a
  "2-season, old cache" condition against an "8-season, freshly fetched" one
  would confound "more data" with "different (revised) data for the same
  weeks." *Caught by*: thinking through what has to be held fixed for the
  comparison to mean anything, before running it — neither problem would
  have been visible in the resulting numbers on its own; only reasoning about
  the experiment's design would surface them.

- **The Sleeper-projection false alarm.** A real, reasonable worry going in:
  does Sleeper's *projections* endpoint use the same stat-name vocabulary as
  the league's own scoring rules, the way the raw play-by-play data needs an
  entire translation layer (`compute_custom_score()`) to get there? *Checked
  directly* — and it turned out Sleeper's projections already use Sleeper's
  own key names, the same ones `scoring_settings` is keyed by, so no
  translation layer was needed at all. Worth keeping as its own category
  distinct from the others: verifying is just as valuable when it rules a
  problem *out* as when it confirms one — the alternative would have been
  building unnecessary translation machinery to solve a problem that didn't
  exist.

---

## 4. The two reversals, in detail

### "These features are too weak to forecast anything" — overturned by data volume

**The original conclusion**, reached training on 2 seasons of history: the
model lost to nearly every simple baseline at every position. The diagnosis
at the time was a "signal-to-noise ceiling" — not a bug, not a tuning gap,
just an honest conclusion that these features didn't carry enough forecasting
signal to beat a three-line moving average.

**What made this convincing at the time** is worth dwelling on, because it
wasn't a lazy first guess — it was reasoned from *converging* evidence. SHAP
rankings showed the model leaning on exactly the kind of features you'd want
it to (role share, opportunity, efficiency) — so it wasn't garbage-in,
garbage-out. A feature-count ablation was run to rule out "too many noisy
features" as the culprit (see the second reversal, below — this ablation
itself later turned out to be biased, but at the time it looked like one more
piece of confirming evidence). And a direct residual test — training the
model to predict `custom_points − season_to_date_average` instead of
`custom_points` directly, isolating whether the feature set added anything
*on top of* a simple trailing average — found a real, non-degenerate signal
(predicted and actual residuals correlated at r≈0.11-0.15, clearly not zero)
that *still* made the final prediction worse than the raw baseline once
reconstructed. Every one of these checks pointed the same direction. That's
exactly why the conclusion held for as long as it did: it wasn't sloppy, it
was thorough — and still wrong.

**What actually flipped it**: extending the training window from 2 seasons to
8 (first verifying that snap counts and the other data sources actually
covered the wider range, and explicitly deciding how to handle 2020's
disrupted schedule rather than silently including or excluding it), then
re-running the *identical* model, features, and baselines — holding the
*evaluated* weeks fixed across every condition (see the warmup-anchoring
pitfall above) so the only thing changing was how much training history the
model got to learn from. At 8 seasons, the same model, same code, same
evaluation weeks, cleanly beat the two weaker baselines at every position and
closed real ground on the strongest one.

**The lesson**: "there isn't enough signal in these features" and "there
isn't enough training data to learn the signal that's actually there" produce
*identical* symptoms — the model loses to baselines either way — but call for
opposite responses (give up on the feature set, versus go get more data).
The only way to tell them apart is to deliberately vary the one variable that
distinguishes them while holding everything else fixed. No amount of
diagnostic reasoning about the *existing* result, however careful, can
substitute for that.

### "Performance peaks at 25 features" — an artifact of biased selection, not a real finding

**The original finding**: rank all ~180 candidate features by SHAP importance
using one model trained on the *entire* dataset, then test keeping just the
top 10, 25, 50, or all of them in separate walk-forward evaluations. The
result was a clean, consistent "peaks at 25, degrades with more features"
curve at every position — the exact shape a textbook would predict from
overfitting on too many noisy features.

**What made this convincing**: it wasn't a fluke in one position — the
pattern held across all four, and it matched a well-known, intuitively
plausible mechanism (more features, more overfitting risk) that any ML
practitioner would find unremarkable on its face. That combination — a
consistent result, matching a familiar and expected story — is precisely the
situation where it's easiest to stop checking. Nothing about it looked wrong.

**What actually happened**: the feature *ranking* itself was computed using
SHAP importances from a model trained on the whole dataset — including the
same weeks later used to *score* the ablation. That means the "top 25" list
was chosen partly using information from the exact weeks being used to judge
whether 25 was the right number to pick — a leak in the *selection process*,
the second kind described in Section 2, not a leak in any single feature's
value (every individual feature was still perfectly point-in-time correct).
Redone properly — deriving the SHAP ranking freshly *inside* each
walk-forward fold, from only that fold's own training data — the "peaks at
25" pattern disappeared entirely. Twenty-five features turned out to be the
single *worst* option at one position, and the full feature set was
preferred everywhere.

**The lesson**: a result that is both clean *and* matches what you already
expected to find deserves *more* scrutiny, not less — both properties are
exactly the conditions under which a subtle methodology bug is least likely
to get a second look. And separately: a leakage test suite built entirely
around checking feature *values* provides zero protection against leakage in
the *process that selects which features to use* — these are different bugs,
caught by different tests, and having one kind of test passing cleanly says
nothing about the other.

---

## 5. What didn't work, and why

- **Formulation B** (predicting `custom_points − sleeper_projection`, i.e.
  the *residual* against Sleeper's own number, instead of `custom_points`
  directly). The idea was sound in principle — start from a number that
  already encodes real information (Sleeper's beat-writer and depth-chart
  knowledge) and ask only "where is Sleeper wrong," rather than re-learning
  everything Sleeper already knows from scratch. It didn't work: the residual
  itself carried almost no learnable week-to-week signal (predicted-vs-actual
  residual correlation of only ~0.03-0.05), and reconstructing a final
  prediction from it performed *worse* than predicting `custom_points`
  directly against nearly every baseline. Why: Sleeper's projection already
  captures most of the explainable structure — role, matchup, recent form —
  that this feature set *also* captures. Once Sleeper's number is subtracted
  out, what's left over is dominated by the part *neither* Sleeper nor this
  feature set can see (touchdown variance, game-script randomness), and
  that part isn't learnable from either source.

- **Markov chains**, considered for the simulator and explicitly rejected
  rather than tried and failed. A Markov chain models a system where the
  *next* state depends only on the *current* state and nothing further back.
  Weekly fantasy production doesn't fit that shape — next week's output
  depends on next week's opponent, health, and game script, not on "what
  state" a player's production was in the week before. The one place football
  genuinely is Markovian is drive-level modeling (down, distance, field
  position → next state), but nflverse's own EPA and win-probability columns
  are already a far better version of exactly that model, built from more
  data than this project could realistically rebuild — so a Markov chain here
  would just be a worse copy of something already sitting in the data.

- **Beating Sleeper's own projection.** At no data volume and in neither
  formulation did the model catch Sleeper's projection — the strongest of the
  four baselines, at every position, at every volume tested. Sleeper's number
  encodes real-world information (injury reports, depth-chart moves,
  beat-writer signals) that simply isn't present anywhere in play-by-play-
  derived statistics, however well the features are engineered. The honest
  conclusion: this feature set is good at *explaining* what already happened
  — genuinely useful for the dashboard's analytical panels — but isn't
  positioned to *forecast* better than a source with information it
  structurally can't see.

- **The simulator's original acceptance criterion**: "simulated win
  probabilities beat a naive baseline of whichever team has the higher
  projected total." This turned out to be the wrong test, not a failed one.
  What a Monte Carlo simulation actually adds over a plain point-estimate
  comparison is a better account of *spread* — variance and correlation — not
  a different *average*. Since the simulated favorite and the naive favorite
  are determined by essentially the same averages, they agree by construction
  in the overwhelming majority of matchups; they can only possibly disagree
  in genuine near-toss-up games. Asking the simulator to "beat" a method it
  structurally agrees with almost every time was measuring the wrong thing.
  The criterion was rewritten to ask whether the probabilities themselves are
  *calibrated* — whether a stated "60%" comes true about 60% of the time —
  which is the property a simulation can actually add that a point estimate
  cannot.

---

## 6. What I'd do differently, with hindsight

- **Start at 8 seasons of training history from the beginning, not 2.** The
  entire "features are too weak" detour — the diagnostic apparatus, the
  residual test, the feature-count ablation and its own later reversal —
  spent real effort investigating a conclusion that more data alone would
  have avoided reaching in the first place. This isn't a free lunch (a bigger
  cache is slower to build and refresh, which is exactly why Phase 8's
  automation now needs incremental fetching instead of a full refetch every
  run) — but the data-volume question deserved to be asked explicitly and
  first, rather than defaulted small and revisited only after a wrong
  conclusion had already been reached and half-investigated.

- **Guard feature *selection* against look-ahead bias in the leakage suite
  from day one, not just feature *values*.** `tests/test_no_leakage.py` is
  thorough about individual columns, but nothing in its original design was
  built to catch "was the feature list itself chosen using information a
  fold wouldn't have had yet" — and that exact gap is what let the biased
  ablation happen and go unnoticed until it was redone by hand. A parallel
  check — does the selection *process*, not just each feature's value, only
  ever see what a fold could have seen — deserved to be a standing category
  in the suite from the start, not something added reactively after finding
  one instance of it.

- **When a check is cheap enough to run on everything, run it on everything,
  not a sample.** The two-point-conversion bug surfaced through an
  exhaustive team-week audit; a handful of representative spot-checks
  plausibly would have missed a discrepancy that small. Exhaustive is
  usually cheap for exactly the kind of "does this sum to what it should"
  check that catches this class of bug — there's rarely a good reason to
  sample when the full check is nearly free.

- **Treat "this result is clean and matches what I expected" as a prompt to
  look harder, not a reason to stop.** Both reversals in this document were
  convincing *in part because* they matched a familiar, plausible-sounding
  story — a signal-to-noise ceiling, a classic overfitting curve. A
  surprising result invites scrutiny on its own; an unsurprising one needs
  someone to apply that same scrutiny on purpose, since nothing about the
  result itself will prompt it.

- **Design a comparison so the one thing you're testing is the only thing
  that changes, and do that thinking before running anything.** The
  data-volume experiment needed the evaluation weeks locked and the caches
  freshly and identically fetched before it could produce a trustworthy
  answer — and both of the confounds that would have wrecked it (warmup
  anchoring to the wrong season, a stale-vs-fresh cache mismatch) would have
  been invisible in the resulting numbers themselves. They were only visible
  by reasoning about the experiment's design in advance, not by inspecting
  its output afterward.
