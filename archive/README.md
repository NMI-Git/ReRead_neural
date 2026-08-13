# Archive

Superseded material, kept because it documents how the project got here. None of
it is part of the current model and none of it runs from a clone.

## Earlier topologies

`zero_deck.py` and `one_deck.py` are the other two network topologies Dandurand,
Grainger & Hannagan (2013) compare against the two-deck model. The paper's
central claim is that comparison — two-deck discriminates words from nonwords
best while using the fewest connection weights — so these are worth keeping even
though neither currently runs: both predate `config.py` and still carry
hardcoded paths, and `one_deck.py` builds its own encoding in which the filler
token owns index 0, which the current `weight_multiplier` no longer expects.

Reproducing the paper's topology comparison would mean porting both to
`config.py`, `config.encode_words` and `losses.summed_cross_entropy`.

## Superseded scripts

- `lower_deck_evaluation.py` — scratch harness from before the upper deck
  existed. Calls `output_evaluation.output_eval`, which no longer exists.
- `char_seq.py` — exploratory character-sequence work from the tutorial phase.

## Superseded data

- `*.rtf` — the original corpus and label files, before they were converted to
  the UTF-16 plain text the model reads.
- `lower_deck.h5`, `upper_deck.h5` and their mappings — an earlier French
  training run (vocabulary 38/37, 1985 units), superseded by
  `models/french/`.
- `activation_vectors.csv`, `euclidean_calculations.csv` — unprefixed analysis
  output from a run whose model is no longer in the repository, so they cannot
  be regenerated or checked.
- `study1_data/` — curated result tables prepared for analysis in R.
