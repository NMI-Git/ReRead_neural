# Statistical inputs

Every value behind a statistical test reported in the manuscript, saved so each
test can be re-run without retraining. Regenerate with
`python export_statistics.py`.

## Table 1 — corpus descriptives

`table1_<slug>.csv`, one row per word:

| column | meaning |
|---|---|
| `word` | the corpus item |
| `neighbours` | Coltheart's N — words differing by one letter substitution |
| `old20_mean`, `old20_sd` | mean and sd of the Levenshtein distance to the 20 closest words |
| `spread` | number of letter positions at which a substitution yields a neighbour |
| `uniqueness_point` | position at which the word's prefix stops being shared |

The *t*-tests compare the `french` and `finnish` files column by column. Note
the manuscript's Table 1 values were computed with LexiCAL; these reproduce the
French and Finnish columns to two decimals but differ on two FIN-random
Levenshtein cells.

## Table 2 — model performance

`table2_<slug>_<condition>.csv`, one row per stimulus. Conditions: `realwords`,
`rs`, `srl`, `dls`, `lt`, `rpp1234`, `rpp1357`, `tlp1235467`, `tlp123dd67`.

| column | meaning |
|---|---|
| `input` | the stimulus as fed to the lower deck, padding included |
| `lower_deck_output` | what the lower deck reconstructed, decoded to letters |
| `upper_deck_winner`, `winner_activation` | the most active lexical unit |
| `target`, `target_activation` | the word the stimulus came from, and its own unit |
| `excluded` | 1 if the item does not instantiate its condition |
| `over_threshold` | 1 if it passed — 0.9 for nonwords, 0.5 for priming |

Nonword conditions are scored on the winning unit, priming on the target's own
unit, following Dandurand et al. (2013, §3.2).

`target` is empty for `rs` and `srl`: those stimuli are not derived from any
particular word, so there is no target unit to read.

`excluded` marks degenerate transpositions — where the two swapped positions
held the same letter, so the stimulus is identical to the word it came from. In
`lt` that means it is not a nonword; in `tlp1235467` it is the target rather
than a prime. `tlp123dd67` items are always valid but carry the same mask, so
the transposed-letter contrast is computed within-item.

These use the shipped models, which stop at their own criterion.

## Proximity and letter cluster effects

| file | contents |
|---|---|
| `proximity_values_<slug>.csv` | the 156 off-diagonal cells of the 13 x 13 position matrix |
| `cluster_values_<slug>.csv` | letter-pair distances, long format |
| `cluster_matrix_<slug>.csv` | the same as a labelled square matrix — what Figure 3 plots |

The *F*-tests compare the `french` and `finnish` files.

**These use the matched-epoch lower decks in `results/matched/`, not the shipped
models.** Criterion stopping halts each model at a different epoch — Finnish
421, French 404, and the random corpus never, so it runs to the 2000-epoch
ceiling. Longer training increases representational separation on its own, so
comparing hidden layers across models requires equal training. All three were
trained for 421 epochs for these files; see `matched_representations.py`.

## Figures

`results/figures/figure2_proximity.png` and `figure3_letter_clusters.png`,
plotted from the matrices in `results/matched/<slug>/`. Regenerate with
`python plot_effects.py`, which needs matplotlib — not required by the model
itself, so it is not in `requirements.txt`.
