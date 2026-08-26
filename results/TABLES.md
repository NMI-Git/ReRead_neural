# Results tables

Generated from the committed corpora and models.

| Table | Regenerate with |
|---|---|
| 1 — corpus descriptives | `python corpus_measures.py` |
| 2 — model performance | `python two_deck.py --corpus <KEY> --mode <N>` |
| 3, 4 — hidden-layer distances | `python matched_representations.py` |

LaTeX for Tables 1 and 2 is in [`tables.tex`](tables.tex).

## Table 1. Descriptive information about the corpora

| Orthographic Measure | French M | SD | FIN M | SD | FIN random M | SD | t | p |
|---|---|---|---|---|---|---|---|---|
| Neighbour words # | 0.40 | 0.78 | 0.67 | 0.88 | 0.00 | 0.00 | -10.25 | <.001 |
| Levenshtein distance 20, mean | 3.12 | 0.48 | 3.05 | 0.45 | 4.66 | 0.14 | 5.11 | <.001 |
| Levenshtein distance 20, sd | 0.54 | 0.16 | 0.59 | 0.17 | 0.49 | 0.09 | -10.99 | <.001 |
| Spread (#letters yielding a neighbour) | 0.34 | 0.59 | 0.58 | 0.74 | 0.00 | 0.00 | -11.62 | <.001 |
| Uniqueness point | 4.34 | 1.04 | 5.07 | 1.31 | 3.15 | 0.45 | -19.76 | <.001 |

N = 1985 French, 2000 in both Finnish corpora. Welch's *t*, French against the
non-random Finnish corpus. These describe the word lists, not the model, so the
published Table 1 needs no change — keep the LexiCAL values. This recomputation
reproduced French and Finnish to two decimals; only the two FIN-random
Levenshtein cells differ, most likely LexiCAL versus this implementation.

## Table 2. Model performance

| Test Condition | FR Accur. | # | Err. | FIN Accur. | # | Err. | FIRND Accur. | # | Err. |
|---|---|---|---|---|---|---|---|---|---|
| Real Words | 100.0% | 13895 | 0 | 100.0% | 14000 | 0 | 100.0% | 14000 | 0 |
| Random String | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 |
| Single Repeated Letter | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 |
| Double Letter Substitution | 99.7% | 1985 | 5 | 99.7% | 2000 | 6 | 100.0% | 2000 | 0 |
| Letter Transposition | 91.8% | 1893 | 156 | 93.3% | 1437 | 96 | 99.5% | 1926 | 9 |

**Letter transposition excludes degenerate items.** Where the two swapped
positions hold the same letter the transposition is a no-op and the stimulus is
the base word, not a nonword — it necessarily reaches threshold. Excluded:
**92 French, 563 Finnish, 74 FIN random**. Finnish has by far the most because it
has far more double letters. Before excluding them Finnish scored 67.0% here,
below French; after, it scores 93.3%, above French.

### Priming

Proportion of primes activating **their own target word's** unit above 0.5.
Effects, not errors: the prediction is `SILE > SLNE` and `SILNECE > SILOPCE`.

| Condition | FR % | # | n | FIN % | # | n | FIRND % | # | n |
|---|---|---|---|---|---|---|---|---|---|
| SILE-SILENCE | 1.21% | 24 | 1985 | 0.75% | 15 | 2000 | 0.00% | 0 | 2000 |
| SLNE-SILENCE | 0.10% | 2 | 1985 | 0.20% | 4 | 2000 | 0.00% | 0 | 2000 |
| SILNECE-SILENCE | 26.54% | 499 | 1880 | 18.36% | 333 | 1814 | 14.69% | 279 | 1899 |
| SILOPCE-SILENCE | 5.05% | 95 | 1880 | 6.50% | 118 | 1814 | 0.05% | 1 | 1899 |

Both transposed-letter rows use the same reduced *n* (1880 French, 1814 Finnish,
1899 FIN random). `SILNECE` items are dropped because a degenerate transposition
makes them the target rather than a prime; `SILOPCE` items are always valid but
are dropped alongside their partners so the contrast stays within-item.

| Contrast | French | Finnish | FIN random |
|---|---|---|---|
| RPP (`SILE` − `SLNE`) | +1.11 pp | +0.55 pp | 0.00 pp |
| TLP (`SILNECE` − `SILOPCE`) | +21.49 pp | +11.85 pp | +14.64 pp |

## Table 3. Proximity effect (position x position)

Off-diagonal cells of the 13 x 13 matrix, 156 per corpus.

| Model | M | SD | n |
|---|---|---|---|
| French | 2.45 | 0.29 | 156 |
| Finnish | 2.97 | 0.36 | 156 |
| Random Finnish | 2.92 | 0.48 | 156 |

French vs Finnish: *F*(1, 310) = 198.48, *p* < .001.

## Table 4. Letter cluster effect (letter x letter)

Full letter-by-letter matrix, diagonal included.

| Model | M | SD | n |
|---|---|---|---|
| French | 2.90 | 0.70 | 1369 |
| Finnish | 3.31 | 0.44 | 529 |
| Random Finnish | 3.24 | 0.16 | 529 |

French vs Finnish: *F*(1, 1896) = 159.42, *p* < .001.

### These use lower decks trained to a common epoch count

Criterion stopping halts each model at a different epoch — Finnish 421, French
404, and the random corpus never, so it runs to the 2000-epoch ceiling. Longer
training increases representational separation on its own, so comparing hidden
layers across models trained for different lengths is confounded.

Tables 3 and 4 therefore come from lower decks all trained for **421 epochs**,
the point at which Finnish meets its criterion. Neither real corpus is cut short
of its criterion; French simply trains 17 epochs longer than it needs.

The effect of not controlling this:

| Random Finnish lower deck | proximity M (SD) | letter cluster M (SD) |
|---|---|---|
| 2000 epochs (its own stopping point) | 3.91 (0.55) | 4.16 (0.15) |
| **421 epochs (matched)** | **2.92 (0.48)** | **3.24 (0.16)** |

Matched, the random model falls **below** Finnish on both measures rather than
above it. Any claim that a random training scheme improves letter position or
identity coding is an artifact of training length.

The models behind Tables 3 and 4 are trained on demand and not saved; the
distance matrices are written to `results/matched/<slug>/`.
