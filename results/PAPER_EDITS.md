# Paper edits required by the version 2 model

Every value in the paper that the model changes invalidate, with the replacement.
Ordered by section. Values come from the committed models; regenerate with
`python two_deck.py`, `python corpus_measures.py` and
`python representation_stats.py`.

Legend: **REPLACE** a value · **REWRITE** a claim that is no longer true ·
**ADD** something missing · **NO CHANGE** where the existing text is already
correct (listed so you do not edit it unnecessarily).

---

## 2.1 Network

**REPLACE** — "The subsequent hidden layer's node size equals the square root of
number of input patterns."

> …equals the square root of the number of input patterns, **rounded up to the
> nearest integer** (119 for a 2000-word corpus).

Previously 118 (rounded down); Dandurand et al. footnote 3 rounds up.

**REWRITE** — "The second network maps the word-centered letters directly onto
the output layer of lexical representations."

The second network is now *trained* on idealised binary word-centred letters but
*receives the first network's continuous activations at recall*, unmodified,
following Dandurand et al. §2.5. State both.

**NO CHANGE** — "node size corresponds to a number of locations (13) x the
number of different letters in the training corpus" and "7 locations x number of
letters in a training corpus". Both were previously off by one letter because
the filler token held its own unit. They are now exactly correct.

**NO CHANGE** — the Stevens and Grainger (2003) weighting description.

---

## 2.3 Network training

**REPLACE** — "using cross-entropy as a cost function for the upper deck and
mean squared error for the lower deck"

> using cross-entropy as a cost function, summed over output units, **for both
> decks**

**REPLACE** — "we used a learning rate of 100 and a momentum term of 0.5 for the
lower deck and a learning rate of 0.9 and a momentum term of 0.2 for the upper
deck"

> we used a learning rate of 0.9 and a momentum term of 0.2 **for both decks**

Optional footnote: the previous lower-deck rate of 100 was not a departure from
0.9. Scaling a loss by a constant is equivalent to scaling the learning rate by
it, and Keras averages over output units where LENS sums; expressed in the
summed convention that rate was 0.60.

**NO CHANGE** — "connection weights were initialized with random values within a
range of -.5 and .5". Now true of both decks; the upper deck previously used the
framework default.

**ADD** — a stopping criterion. The section does not currently state one.

> Each deck was trained until it correctly classified all training patterns —
> the target unit above 0.9 with all other units below it (Dandurand et al.,
> 2013, §2.5). The Finnish decks reached this criterion at epochs 421 and 270,
> the French at 404 and 279. The random-corpus upper deck reached it at epoch
> 40; its lower deck did not reach it within a 2000-epoch ceiling, correctly
> classifying 32.1% of patterns.

**NOTE** — "We trained three two-deck networks, one for each corpus" is one
network per corpus. Dandurand et al. train three replications per topology and
report means with standard error bars. This remains a difference.

---

## 2.4 Computation of output activations

**ADD** — the section is empty. It should cover: acuity-weighted input; deck 1
producing continuous word-centred activations; those activations passing
unmodified into deck 2; the 0.9 recognition criterion; and the separate 0.5
priming threshold read on the target word's own unit.

---

## Table 1

**NO CHANGE.** Table 1 describes the corpora, which are untouched by the model
work. Keep the published LexiCAL values. An independent recomputation
(`corpus_measures.py`) reproduced French and Finnish to two decimal places on
all five measures; only the two FIN-random Levenshtein cells differed, which is
most likely a difference between LexiCAL and that implementation.

---

## Table 2 — all values change

| Test Condition | French | Finnish | Random Finnish |
|---|---|---|---|
| Real Words | 100% / 13895 / **0** | 100% / 14000 / **0** | 100% / 14000 / **0** |
| Random String | 100% / 1000 / 0 | **100%** / 1000 / **0** | **100%** / 1000 / **0** |
| Single Repeated Letter | 100% / 1000 / 0 | **100%** / 1000 / **0** | 100% / 1000 / 0 |
| Double Letter Substitution | **99.7%** / 1985 / **5** | **99.7%** / 2000 / **6** | **100%** / 2000 / **0** |
| Letter Transposition | **87.5%** / 1985 / **248** | **67.0%** / 2000 / **659** | **98.5%** / 2000 / **29** |

**The priming rows now measure a different quantity.** They previously counted
priming activations as errors at a 0.87 threshold on the winning unit. They now
report the proportion of primes activating **their own target's** unit above
0.5, which is how Dandurand et al. §3.2 define priming — an effect to be
measured, not a failure. Relabel the columns.

| Condition | French | Finnish | Random Finnish |
|---|---|---|---|
| SILE–SILENCE | 1.21% / 24 | 0.75% / 15 | 0.00% / 0 |
| SLNE–SILENCE | 0.10% / 2 | 0.20% / 4 | 0.00% / 0 |
| SILNECE–SILENCE | 30.43% / 604 | 25.95% / 519 | 18.95% / 379 |
| SILOPCE–SILENCE | 4.94% / 98 | 6.75% / 135 | 0.05% / 1 |

Both predicted orderings hold in every corpus that primes at all:
`SILE > SLNE` and `SILNECE > SILOPCE`.

---

## 3.1 Nonword processing

**REPLACE** — the letter-transposition example. "letter transposition (e.g.
KATITLA)" shows positions 4 and 5 swapped, which is the *transposed-letter
priming* pattern. Dandurand's letter-transposition nonword swaps positions 5 and
6 (`SILENCE` → `SILECNE`), which for `KATTILA` gives **`KATTLIA`**. The code
follows Dandurand.

**REWRITE** — "the random model could not discriminate the transposed letter
nonwords from their base words."

No longer true. Random-model letter-transposition accuracy went from 42.8% to
**98.5%**.

**REWRITE** — "In relative position priming, the random model behaved reversely
to natural models, with 1357 primes activating the target words more likely than
1234 primes."

No longer true. The random model now shows **0/2000 in both conditions** — no
relative-position priming at all, which is the correct result for a corpus with
no relative-position structure. The previous reversal was an artifact.

**REPLACE** — "the transposed letter primes activated the target words with a
high probability (0.73) and the 123DD67 items with a probability of 0.15."

> …with a probability of **0.19** and the 123DD67 items with a probability of
> **0.0005**.

**REWRITE** — the confound paragraph. "Due to frequent double letters in
positions 4 and 5, 5.3 percent of the transposed letter items derived from the
French corpus were identical to their base words, whereas the same value in
Finnish corpus was 9.4 percent."

Those figures are correct for positions 4 and 5 (measured: 5.3% French, 9.3%
Finnish) — but that is the *priming* transposition. The letter-transposition
**nonwords** swap positions 5 and 6, where the rate is **4.6% French and 28.1%
Finnish**. The confound is far larger than reported, and removing it reverses
the comparison:

| | LT items | identical to base | false positives | of which on identical items | genuine rate |
|---|---|---|---|---|---|
| French | 1985 | 92 | 248 | 92 | **156/1893 = 8.2%** |
| Finnish | 2000 | 563 | 659 | 563 | **96/1437 = 6.7%** |
| Random | 2000 | 74 | 29 | 20 | 9/1926 = 0.5% |

Every identical item necessarily reaches threshold, because it *is* the base
word. Excluding them, **Finnish (6.7%) outperforms French (8.2%)** — the
opposite of the raw figures (33.0% versus 12.5%). This strengthens rather than
weakens the existing argument that the French/Finnish difference is not of
theoretical interest.

**NO CHANGE** — "above a threshold value of 0.5". The text already specified
0.5; the implementation previously used 0.87 and now matches.

**NO CHANGE** — "the same threshold value of 0.9" for nonwords.

---

## 3.2 Proximity effects

| | Published | Replace with |
|---|---|---|
| French | M = 2.28, SD = 0.71 | **M = 2.45, SD = 0.29** |
| Finnish | M = 2.67, SD = 0.85 | **M = 2.97, SD = 0.36** |
| Random | M = 2.80, SD = 0.99 | **M = 3.91, SD = 0.55** |
| French vs Finnish | F(1, 310) = 19.13 | **F(1, 310) = 203.57**, p < .001 |

**REPLACE** — "The artificial model produced **slightly** higher distance values
than the Finnish model." The gap is no longer slight (3.91 versus 2.97).

**REWRITE** — "These results indicate that a training model with a random input
scheme may improve rather than hamper letter position coding."

This is now confounded by training length and does not survive a control.
Criterion stopping halts Finnish at 421 epochs and French at 404, but the random
corpus never meets the criterion and runs to the 2000-epoch ceiling — roughly
five times the training. Retraining the random lower deck for 421 epochs to
match Finnish:

| Random Finnish lower deck | proximity M (SD) | letter cluster M (SD) |
|---|---|---|
| 2000 epochs (as committed) | 3.91 (0.55) | 4.16 (0.15) |
| **421 epochs (matched)** | **2.92 (0.48)** | **3.24 (0.16)** |
| *Finnish at 421, for reference* | *2.97 (0.36)* | *3.31 (0.44)* |

At equal training the random model is indistinguishable from Finnish and
marginally **lower** on both measures. The apparent advantage is training
length, not corpus structure.

---

## 3.3 Letter cluster effects

| | Published | Replace with |
|---|---|---|
| French | M = 2.84, SD = 0.70 | **M = 2.89, SD = 0.70** |
| Finnish | M = 3.19, SD = 0.46 | **M = 3.31, SD = 0.44** |
| Random | M = 3.57, SD = 0.19 | **M = 4.16, SD = 0.15** |
| French vs Finnish | F(1, 1896) = 47.54 | **F(1, 1896) = 164.66**, p < .001 |

**REWRITE** — "These results suggest that the random training scheme may
improve, rather than hamper, also letter identity coding." Same confound as
§3.2; the matched-epoch control removes the effect.

**NOTE** — the French letter-cluster value barely moved (2.84 → 2.89, SD
unchanged at 0.70), which suggests the measure is stable and the shifts
elsewhere are real.

---

## 3.4 Error analysis — the section no longer has a subject

**REPLACE** — "The lower deck of the Finnish model reproduced perfect letter
sequences only for 89 percent of words" → **100 percent**.

**REPLACE** — "the word accuracy of the upper deck (95.6 percent)" → **100
percent**.

The discriminant analysis predicted word accuracy from psycholinguistic
properties. With no errors left there is no variance in the outcome to predict,
so the analysis as written cannot be run. Two options:

1. **Reframe as a result.** Report that the corrected model reaches perfect
   classification and that the previously reported error analysis therefore no
   longer applies, retaining it as an account of what the under-trained model
   struggled with.
2. **Move it to the random corpus,** which still fails — 32.1% by the strict
   criterion, 96.6% by argmax reconstruction. The predictors would necessarily
   differ, since a random corpus has no consonant-vowel structure or bigram
   regularities, which arguably makes it the more informative comparison: it
   isolates what the linguistic structure was contributing.

---

## Figures

Figures 2 and 3 are generated from the distance matrices and must be
regenerated. The underlying data is in `results/<corpus>/proximity_effect.csv`
and `results/<corpus>/clustering_effect.csv`.

Note that `results/french/` did not exist before this work — `config.py`
referenced four French analysis files that had never been generated, so the
provenance of the previously published French figures is unclear.
