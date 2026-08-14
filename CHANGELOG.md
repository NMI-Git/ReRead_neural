# Update log

Changes from the original version of this project (`master`, commit `69af313`)
to `version2`. Written for reviewers: it records what changed, why, and what
moved as a result.

The work was prompted by reviewer feedback on the original submission, and then
by a line-by-line comparison against the source paper:

> Dandurand, F., Grainger, J., & Hannagan, T. (2013). Computational models of
> location-invariant orthographic processing. *Connection Science, 25*(1), 1–26.

---

## 1. The repository can now be run from a clone

*Commit `076a03b`.* This addresses the reviewer's first and most concrete
complaint — that the code could not be run, that filenames were hardcoded and
pointed at files that were not in the repository.

- **Added `config.py`.** Every script previously carried its own copy of a
  `FilePathEnums` enum plus a `corpus_instantiation()` function returning a plain
  list whose *positions* encoded meaning — `chosen_corpus[4]` meant "the lower
  deck mapping", but only in some scripts, because the orderings had drifted
  apart between files. All of that is replaced by one registry of named fields.
- **Fixed a corpus mismatch this had caused.** The lookup table paired `FIN`
  with the French label file and `FR` with the Finnish one. Training either deck
  would have fitted a model against another language's word indices.
- **Committed the files that were missing.** The corpora, the derived training
  data, the trained models and the character mappings are all in version control
  now. A clone runs without generating anything first.
- **Added early validation.** `config.require()` checks a script's inputs up
  front and names what is missing and which script produces it, instead of
  failing later with a bare `FileNotFoundError`.
- **Rewrote `README.md`** with a step-by-step guide covering both routes: running
  the shipped models, and rebuilding everything from a raw vocabulary.

Verified by cloning the branch to a clean directory and running it with no setup
beyond creating the conda environment.

---

## 2. The model now matches the source paper

*Commit `f72a79d`.* A parameter-by-parameter comparison against the paper found
several departures. The largest of them was a cost function.

### Cost function

The paper trains with "cross-entropy as a cost function (Hinton, 1989)" in the
LENS simulator (§2.4). For a layer of independent sigmoid units that is binary
cross-entropy **summed** over the units of a pattern.

The upper deck used Keras `categorical_crossentropy`, which normalises the
prediction by its own sum, giving `−log(p_target / Σp)`. That maximises the
target *relative to the total*, so each of the 1999 non-target units received
only 1/1999 of the downward pressure and none was ever driven towards zero. The
symptom was visible in the trained models: non-target units sat at 0.80–0.93
instead of near zero, leaving a target-to-competitor gap of about 0.06. No
threshold could separate 2000 patterns inside a gap that narrow.

Both decks now use summed cross-entropy (`losses.py`). Non-target units settle
near 0.006 and the gap widens to about 0.97.

### Regularisation

The L2 term on the upper deck was removed. The paper reports none. It had been
compensating for the cost function — holding non-target units below the 0.9
criterion by shrinking every weight, which pulled the target units down with
them. With the loss corrected, reinstating L2 collapses the model entirely.

This also resolves an inconsistency in the shipped models: the Finnish upper
deck had been trained at L2 = 0.0008 and the French at 0.0005, so their
published results were not comparable to each other.

### Learning rate

The reviewer questioned the lower deck's learning rate of 100 against the
paper's 0.9. This turns out to be a normalisation artifact rather than a
departure. Scaling a loss by a constant is exactly equivalent to scaling the
learning rate by that constant under gradient descent, momentum included. Keras
averages over output units; LENS sums. Expressed in the paper's convention, 100
was **0.60** for Finnish (168 units) and **0.38** for French (266). Both decks
now use the paper's 0.9 with momentum 0.2 directly.

### Other parameters

| | Was | Now | Source |
|---|---|---|---|
| Hidden units | 118 (floor) | **119** (ceiling) | §2.1 footnote 3 |
| Momentum | 0.5 lower deck | **0.2** both | §2.4 |
| Weight init | Keras default on upper deck | **U(−0.5, 0.5)** both | §2.4 |
| Priming threshold | 0.87 | **0.5** | §3.2 |
| Priming reads | winning unit | **target word's unit** | §3.2 |

The 0.87 priming threshold was not an independent choice — it was the only band
in which any priming signal survived while non-target units were stuck near
0.85. It was a workaround for the cost function.

### Blank slots

The paper encodes an empty slot as an all-zero vector (§2.3.1.1). The
implementation reached the same input numerically — `#` was one-hot encoded and
then blanked before training — but kept the filler as index 0 of the mapping.
That cost a permanently dead unit in every input slot and every output block,
and made the lower deck's vocabulary one wider than the upper deck's.

The filler is now absent from the mapping. Finnish went from 24 units per slot
to 23, French from 38 to 37 — matching the paper's stated "vector of 37 values
(26 base letters and 11 accentuated French letters)" exactly, with 7 × 37 = 259
output units against the 260-bit vector §2.3.1.2 describes including its
explicit bias.

### Deck-1 → deck-2 handoff

§2.5 is explicit: "network recall in deck 2 is performed with the actual,
continuous outputs of deck 1 without any threshold or other transformation".
The implementation took the argmax of each letter block, assembled a string, and
re-encoded it as clean one-hot.

That discarded the lower deck's uncertainty. For a four-letter priming stimulus
it also invented a complete word the lower deck never proposed — argmax still
returns a letter for a slot holding 0.01 of activation. Removing the filler made
both decks share one representation space, so the activations can now be passed
straight through.

Measured against the quantised path on identical models: nonword rejection
improves on every condition, most on letter transposition (+11.7 points Finnish,
+9.2 French), while relative-position priming weakens. Both were measured across
two prime placements in both languages before adopting the change.

---

## 3. Training stops at the criterion, not at a fixed epoch count

*Commit `d5f8fba`.* Added `stopping.py`.

The paper trains "until they could correctly classify all training patterns,
that is, reach perfect accuracy" (§2.5), where correct means the target unit
above 0.9 **and every other unit below it** — a definition footnote 7 notes is
deliberately stricter than requiring the target merely to win.

The implementation trained for a fixed 2000 epochs and never tested the
criterion. It is now the stopping condition, with `--epochs` as a ceiling.

| Corpus | Lower deck | Upper deck |
|---|---|---|
| `FIN` | epoch 421 | epoch 270 |
| `FR` | epoch 404 | epoch 279 |
| `FIRND` | not reached (32.1% at the 2000 cap) | epoch 40 |

The paper also reports the SSE values *its* networks showed on reaching the
criterion (100 for both decks of a two-deck network). That figure is not used
here as a stopping condition: it is specific to their simulator, initialisation
and corpus. Ours reach the criterion at a very different SSE, and `FIRND` never
approaches 100 at all.

Stopping at the criterion also produced better results than training to 2000
epochs — less saturation lowers nonword activations without dropping real words
below the line. Finnish letter-transposition false positives fell from 867 to
659, French from 463 to 248.

---

## 4. Results

*Commit `10c8249`, re-run under criterion stopping in `d5f8fba`.* All six models
retrained; every analysis regenerated.

**Recognition errors are eliminated on all three corpora.**

| Corpus | Errors before | Errors now | Reaching 0.9 before | Now |
|---|---|---|---|---|
| `FIN` | 518/14000 | **0** | 3.95% | 99.95% |
| `FR` | 55/13895 | **0** | 100% | 99.96% |
| `FIRND` | 337/14000 | **0** | 100% | 39.24% |

Note the Finnish figure in the third column. Before the change, **only 4% of
Finnish training words reached the 0.9 threshold the model uses to decide
something is a word** — the model was failing its own recognition criterion on
its own vocabulary.

**The paper's two-deck signature reproduces.** §3.1 reports that only the
two-deck topology gets single-repeated-letter rejection above double-letter
substitution. With only the upper deck corrected it did not; with both decks
corrected it does, in both languages.

**A spurious result disappeared.** The old `FIRND` model showed
relative-position priming of −2.10 points — the wrong direction, on a corpus of
random strings that has no relative-position structure to prime with. It now
shows exactly 0.00, which is the correct answer for a control.

Comparing nonword false-positive counts directly against the old figures is
misleading and the README says so: the old Finnish model held everything below
0.9, so its count of zero recorded a dead scale rather than discrimination.
Measured as separation between real words and nonwords at the same threshold,
Finnish letter transposition went from 2.9 to 67.0 percentage points.

**The French representation analysis exists for the first time.** `config.py`
had always referenced four French analysis files; none had ever been generated.

---

## 5. Repository layout

*Commit `f209f9d`.* 84 of the 88 tracked files had been in a single directory,
with 21 Python modules interleaved alphabetically among 52 generated data and
model files.

```
data/corpora/           the vocabularies you supply
data/generated/<slug>/  derived training data
models/<slug>/          trained decks and character mappings
results/<slug>/         analysis output
archive/                superseded material, with a README explaining each item
```

Each corpus owns one directory in each tree, so nothing is distinguished by
filename prefix any more. `finnish_lower_deck.h5` became
`models/finnish/lower_deck.h5`.

This let `CorpusConfig` shrink from 15 path fields to four values plus derived
properties, and made the original bug class structurally impossible — a corpus
can no longer point at another's files, because there are no per-file paths to
get wrong. It also retired an inconsistency the old config had to document in a
comment: `FIRND` used a `fin_random_` prefix for most files but
`finnish_random_` for two label files.

All moves used `git mv`, so `git log --follow` still traces each file's history.

---

## 6. Corrections to documentation

Several statements in the original README were wrong and have been fixed:

- **Fixation weighting was described as "the project's extension beyond
  Dandurand et al."** It is not. §2.2 takes within-word visibility from Stevens
  and Grainger (2003) "for strings of 7 letters, and different fixation
  positions", which is the same scheme.
- **`analytics.py` documented the letter-transposition condition as
  `'1234567' -> '1235467'`,** which is the *transposed-letter priming* pattern.
  The code was correct (`1234657`, positions 5 and 6, matching the paper's
  `SILECNE` example); only the docstring was wrong.
- **`zero_deck.py` and `one_deck.py` were not dead code.** They are the paper's
  other two topologies, and the paper's central claim is the comparison between
  all three. They are archived rather than deleted, with the reason recorded.
- **`lower_deck_evaluation.py` had already been broken** before any of this
  work — it calls `output_evaluation.output_eval`, which does not exist.

---

## 7. Known gaps

Three differences from the paper remain. None is a defect; all three are worth
stating before a reviewer finds them.

1. **One replication where the paper runs three.** §2.4 states "Three networks
   are trained for each network topology" and Table 4 lists "Replications per
   topology: 3"; Figure 4 carries standard error bars. Every number here is a
   single sample with no error estimate.
2. **One topology where the paper compares three.** `zero_deck.py` and
   `one_deck.py` are archived and not runnable, so the paper's comparative
   claim — that two-deck discriminates best while using the fewest weights — is
   not reproduced here.
3. **The French corpus is 1985 words against the paper's 2000.** It is
   demonstrably the same Lexique selection minus a short list rather than an
   independent one: of the four anagram triplets the paper names, two are
   present verbatim and two are demoted to pairs by exactly one absent member
   each (`renégat`, `scepter`).

For reference, anagram content across the three corpora, since the paper calls
anagram segregation "one of the most difficult aspects of the present task":

| Corpus | Anagram sets | Words involved | Largest set |
|---|---|---|---|
| `FIN` | 79 (3.95%) | 168 (8.40%) | 4 (`asunnot / osannut / sanonut / sanotun`) |
| `FR` | 60 (3.02%) | 125 (6.30%) | 3 |
| `FIRND` | 4 (0.20%) | 8 (0.40%) | 2 |
| *paper, `FR`* | *67 (3.35%)* | *138 (6.9%)* | *3* |

Finnish is the harder corpus on this dimension, and the control corpus confirms
the anagram density is a property of the languages rather than of the alphabet
or word length.
