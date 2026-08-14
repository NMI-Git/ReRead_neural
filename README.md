# reread_neural

## Neural network for the ReRead project

A connectionist model of visual word recognition, implemented in Keras. The
model recognises a written word regardless of where it falls in the visual
field, and reports how confidently it identifies it.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate reread

python two_deck.py --corpus FIN --mode 1
```

That runs the pre-trained Finnish model over its whole vocabulary and prints a
recognition error count. Nothing needs training first: the trained models, the
vocabularies and all derived data files are in this repository. Expect it to
finish in a few seconds.

---

## Synopsis

First iterations are based on the work of
[Dandurand et al. (2013)](https://www.tandfonline.com/doi/pdf/10.1080/09540091.2013.801934).
This project aims first to develop a neural network implementation of the
dual-stage view on visual word recognition (Hautala et al., 2021) and then to
extend this model into continuous reading. This later goal will be enabled by
implementing mechanisms of preview processing of upcoming word and forward shift
in text input simulating forward saccade length in reading. The architecture of
the neural network for visual word recognition is expected to take a form of a
hybrid autoencoder consisting of parallel processing encoding layers simulating
orthographic processing and followed by recurrent decoding processing layers
simulating phonological decoding. The design of encoder network is based on
earlier work by Dandurand et al. (2013) and the design of decoding network is
based on earlier work by Sibley et al. (2012). The architecture of the planned
continuous reading model will build on theoretical understanding provided by
existing integrative models (Snell et al., 2018; Li & Pollatsek, 2020) combining
connectionist visual word recognition modules with eye movement control modules.

The work has been funded by grant 317030 from the Academy of Finland to Jarkko
Hautala. The site of the work is Niilo Mäki Institute, Jyväskylä, Finland. The
programming work is conducted by Kiril Khalil under supervision of Jarkko
Hautala, and the team from the Faculty of Information Technology at University of
Jyväskylä consisting of Paavo Nieminen, Mirka Saarela and Tommi Kärkkäinen.

---

## How the model works

The model has two stages, referred to throughout the code as the **lower deck**
and the **upper deck**.

```
   ###aaltola###          aaltola              unit #42 = 0.97
  ---------------      ------------          ----------------
   padded input   -->   lower deck    -->      upper deck
   13 slots             word-centred           one unit per
                        orthography            lexical entry
```

**Lower deck — location-invariant orthographic encoding.** A 7-letter word is
placed at one of 7 positions in a 13-slot input window, the unused slots filled
with `#`. The network must output the same word-centred representation for all 7
placements, which is what forces the hidden layer to encode letter identity
independently of retinal position. This replicates Dandurand et al. (2013).

**Upper deck — lexical identification.** Takes the word-centred letters and
activates one localist unit per lexicon entry. The *activation value* of the
winning unit, not merely which unit wins, is the measure the test batteries
report: a familiar word should drive its unit close to 1, while a nonword or a
distorted word should not.

**Fixation weighting.** Before entering the lower deck, each letter is scaled by
a visual acuity gradient that depends on which letter falls on the fixation
point (the centre of the window). Dandurand et al. use the same scheme — §2.2
takes within-word visibility from Stevens and Grainger (2003) "for strings of 7
letters, and different fixation positions" — so this is a replication rather
than an extension.

---

## Requirements

Verified 2026-08-13 on macOS 15 / Apple M2 (arm64), Python 3.9.

- Python 3.9
- TensorFlow 2.11.0 — bundles Keras 2.11. **Keras 3 cannot load the `.h5`
  models in this repository**, so do not upgrade past TensorFlow 2.15.
- NumPy 1.24.4
- scikit-learn 1.3.2 (only for `letter_category_effect.py`)

Install with either:

```bash
conda env create -f environment.yml && conda activate reread   # recommended
python -m pip install -r requirements.txt                      # existing env
```

On Apple Silicon the TensorFlow 2.11 package is published as `tensorflow-macos`
rather than `tensorflow`; `requirements.txt` selects the right one
automatically via environment markers.

`pydot`, `visualkeras` and Graphviz are **not** required. They are needed only
by the optional `plot_models.py`, which regenerates the architecture diagrams.

---

## Step-by-step guide

Two routes through the project. **Route A** runs the models that ship with the
repository and is what you want to check the published results. **Route B**
rebuilds everything from a raw vocabulary, and is what you want to train the
model on a new language.

Timings are for a 2000-word vocabulary on an Apple M2.

### Route A — run the shipped models (about 2 minutes)

**1. Create and activate the environment.**

```bash
conda env create -f environment.yml
conda activate reread
```

If you would rather use an existing interpreter, `python -m pip install -r
requirements.txt` into a Python 3.9 environment does the same job. Verify with:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"   # expect 2.11.0
```

**2. Run the model over a whole corpus.**

```bash
python two_deck.py --corpus FIN --mode 1
```

Takes a few seconds and prints a recognition error count. Compare it against the
table in "Reproducing the published analyses" below — `FIN` should report
0 errors out of 14000.

**3. Run any of the test batteries.**

```bash
python two_deck.py --corpus FIN --mode 5              # letter transposition
python two_deck.py --corpus FIN --mode 6 --sub-mode 2 # relative position priming
```

Nothing needs training first: the trained models, the vocabularies and all
derived data files are in the repository.

### Route B — rebuild everything from a vocabulary (about 10 minutes)

Run these **in order**. Each step consumes what the previous one produced, and
every script takes the same `--corpus` flag. Substitute your own corpus key if
you have registered one (see "Using your own vocabulary").

```
   data/corpora/<name>_corpus.txt     the raw vocabulary you supply
              |
   1. mod_lower_deck_inputs.py        -> data/generated/<slug>/positional_corpus.txt
   2. mod_upper_deck_inputs.py        -> data/generated/<slug>/target_words.txt
              |
   3. lower_deck.py                   -> models/<slug>/lower_deck.h5  + mapping
   4. upper_deck.py                   -> models/<slug>/upper_deck.h5  + mapping
              |
   5. two_deck.py                     -> predictions, error counts,
                                         results/analysis.txt
```

```bash
python mod_lower_deck_inputs.py --corpus FIN     # instant
python mod_upper_deck_inputs.py --corpus FIN     # instant
python lower_deck.py --corpus FIN                # ~2 minutes (stops at epoch 421)
python upper_deck.py --corpus FIN                # ~3 minutes (stops at epoch 270)
python two_deck.py --corpus FIN --mode 1         # a few seconds
```

Steps 3 and 4 **overwrite the trained models for that corpus**. Add `--epochs 20`
to either one for a quick end-to-end check before committing to a full run.

### Route B (continued) — the representation analysis

Only needed to reproduce the hidden-layer results. Requires a trained lower deck
from step 3, and again runs in order:

```bash
python testbed_input_modding.py --corpus FIN            # build letter probes
python testbed_target_words.py  --corpus FIN            # probe labels for R
python testbed_lower_deck.py    --corpus FIN            # ~3 s, writes 4 CSVs
python letter_category_effect.py --corpus FIN --n-init 10
```

`letter_category_effect.py` without `--n-init` reproduces the published k-means
setting, which takes about an hour; `--n-init 10` gives the same clustering in
seconds.

### If something goes wrong

| Symptom | Cause |
|---|---|
| `Missing required input file(s): ...` | A pipeline step was skipped. The message names the missing file and which script produces it. |
| `argument -c/--corpus: invalid choice` | The corpus key is not registered in `config.py`. The message lists the valid keys. |
| `UnicodeDecodeError`, or letters come out garbled | A corpus file is not UTF-16. See "Using your own vocabulary". |
| Errors loading the `.h5` models | Keras 3 is installed. These are Keras 2 models; use TensorFlow 2.11 as pinned. |

---

## Corpora

Three vocabularies ship with the project. Every script takes `--corpus`:

| Key     | Vocabulary                                          | Words | Alphabet |
|---------|-----------------------------------------------------|-------|----------|
| `FIN`   | Real Finnish 7-letter words                         | 2000  | 23 + `#` |
| `FR`    | Real French 7-letter words (Dandurand et al., 2013) | 1985  | 37 + `#` |
| `FIRND` | Random strings over the Finnish alphabet            | 2000  | 23 + `#` |

`FIRND` is a control: same alphabet and word length as `FIN`, but with the
orthographic structure of Finnish removed. Comparing it against `FIN` isolates
how much of the model's behaviour depends on real orthographic regularities.

---

## Reproducing the published analyses

### Recognition performance and the Dandurand test batteries

```bash
python two_deck.py --corpus FIN   --mode 1
python two_deck.py --corpus FIRND --mode 5
python two_deck.py --corpus FIRND --mode 6 --sub-mode 2
python two_deck.py --corpus FIRND --mode 8 --sub-mode 1 --letter a
```

Run with no arguments to be prompted for the mode interactively, as before.

| Mode | Battery                          | Sub-modes                              |
|------|----------------------------------|----------------------------------------|
| 1    | Corpus run, no alteration        | –                                      |
| 2    | RS — random strings              | –                                      |
| 3    | SRL — single repeated letter     | –                                      |
| 4    | DLS — double letter substitution | –                                      |
| 5    | LT — letter transposition        | –                                      |
| 6    | RPP — relative position priming  | 1 = `1234`, 2 = `1357`                 |
| 7    | TLP — transposed letter priming  | 1 = `1235467`, 2 = `123DD67`           |
| 8    | Letter proximity analysis        | 1 = random filler, 2 = `#` filler      |

Two thresholds are used, following Dandurand et al. (2013). Modes 2–5 count an
activation ≥ 0.9 as a false positive, the same criterion used for recognition.
Modes 6–7 use ≥ 0.5, because the point of a priming measure is to detect weaker
activation *before* a string would be classified as a word (§3.2).

The two families also read different units. For a nonword there is no correct
answer, so modes 2–5 read the winning unit: any unit above threshold is an
error. For a prime there is a correct answer — the word it was built from — so
modes 6–7 read that word's own unit, as the paper defines priming.

**Expected output.** These are the values this repository produces as shipped.

Recognition (mode 1). "Reaching 0.9" is the recognition criterion itself: the
target word's unit above 0.9 with no other unit above it.

| Corpus  | Errors     | Reaching 0.9  |
|---------|------------|---------------|
| `FIN`   | 0/14000    | 13993/14000   |
| `FR`    | 0/13895    | 13889/13895   |
| `FIRND` | 0/14000    | 5494/14000    |

Nonword rejection (modes 2–5): false positives at ≥ 0.9, so **lower is better**.

| Corpus  | 2 (RS)  | 3 (SRL) | 4 (DLS) | 5 (LT)   |
|---------|---------|---------|---------|----------|
| `FIN`   | 0/1000  | 0/1000  | 6/2000  | 659/2000 |
| `FR`    | 0/1000  | 0/1000  | 5/1985  | 248/1985 |
| `FIRND` | 0/1000  | 0/1000  | 0/2000  | 29/2000  |

Priming (modes 6–7): stimuli reaching ≥ 0.5 on the target's unit. What matters
is the *contrast* within each pair — the paper predicts `1234` > `1357` and
`1235467` > `123DD67`, and both hold for every corpus that shows priming at all.

| Corpus  | 6.1 (`1234`) | 6.2 (`1357`) | 7.1 (`1235467`) | 7.2 (`123DD67`) |
|---------|--------------|--------------|-----------------|-----------------|
| `FIN`   | 15/2000      | 4/2000       | 519/2000        | 135/2000        |
| `FR`    | 24/1985      | 2/1985       | 604/1985        | 98/1985         |
| `FIRND` | 0/2000       | 0/2000       | 379/2000        | 1/2000          |

`FIRND` shows no relative-position priming at all, which is the expected result
for a control corpus: its strings are random, so a prime made of letters 1, 3, 5
and 7 carries no regularity the network could use to reconstruct the rest. It
still shows transposed-letter priming, because a transposition preserves every
letter of the string.

Comparing these counts against the ones this repository produced before the cost
function was corrected needs care. The previous Finnish model kept almost every
activation below 0.9 — only 4% of its own training words reached the criterion —
so its false-positive count of zero recorded a dead scale, not discrimination.
The meaningful quantity is the separation between real words and nonwords at the
same threshold. On the hardest condition, letter transposition, Finnish went
from 2.9 to 67.0 percentage points and French from 44.2 to 87.5.

`FIRND` is the exception, and the reason is instructive. Its upper deck reaches
the criterion at epoch 40 — random strings have no orthographic neighbours, so
they are trivially separable at the lexical layer — but its lower deck never
reaches it at all. Only 5494 of its patterns clear 0.9 end to end, and that
shortfall is entirely attributable to the first deck. Training the upper deck
longer would mask the problem rather than fix it.

### Stopping criterion

Training does not run for a fixed number of epochs. Following Dandurand et al.
(2013) §2.5, each deck is trained "until they could correctly classify all
training patterns" — every pattern with its target unit above 0.9 and every
other unit below it. `--epochs` is a ceiling, not a target.

| Corpus  | Lower deck        | Upper deck |
|---------|-------------------|------------|
| `FIN`   | epoch 421         | epoch 270  |
| `FR`    | epoch 404         | epoch 279  |
| `FIRND` | not reached (32.1% at 2000) | epoch 40 |

The paper also reports the SSE values *its* networks showed on reaching the
criterion (100 for both decks of a two-deck network). That figure is specific to
their simulator, initialisation and corpus, and is not used here as a stopping
condition: our decks reach the criterion at a very different SSE, and `FIRND`
never approaches 100 at all.

### Internal representation analysis

```bash
python testbed_input_modding.py --corpus FIN     # build the letter probes
python testbed_lower_deck.py   --corpus FIN      # extract hidden activations
python letter_category_effect.py --corpus FIN --n-init 10
```

`testbed_lower_deck.py` feeds one probe per (letter, position) pair — each letter
isolated in an otherwise empty window — through the lower deck and reads the
119-unit hidden layer. That is 299 probes for the 23-letter Finnish alphabet and
481 for the 37-letter French one. It writes four files per corpus: the raw
activations, a per-letter distance matrix, the averaged proximity effect, and
the full clustering matrix.

`letter_category_effect.py` runs k-means over those activations to test whether
representations group by letter identity rather than by position. Its default
`--n-init` reproduces the value used for the published run, which takes about an
hour; pass `--n-init 10` for a result in seconds.

### A note on exact reproducibility

Predictions are batched by default, which is roughly 250× faster than the
original one-input-at-a-time loop. Batching changes the decoded letter in a
handful of near-tied cases — 1 row in 14000 for `FIRND` mode 1 — without
altering any lexical output or error count. Pass `--batch-size 1` to reproduce
the original numbers exactly.

Results also vary by a row or two across hardware and TensorFlow builds, because
floating-point summation order differs. The `results/analysis.txt` committed here was
generated on the original project machine and differs from a current Apple
Silicon run in 3 of 14000 rows.

---

## Using your own vocabulary

**Your word list must be UTF-16 encoded, one word per whitespace-separated
token, with every word the same length and no characters outside the alphabet
you intend to model.** UTF-16 is not incidental: the character mappings inside
the trained models were derived from UTF-16 reads, so a UTF-8 file will produce
a different mapping and meaningless predictions. Convert an existing file with:

```bash
iconv -f UTF-8 -t UTF-16 my_words.txt > data/corpora/my_words.txt
```

Then:

**1. Register the corpus.** Add one entry to `CORPORA` in `config.py`, copying
an existing block and changing the filenames. This is the only file you need to
edit — every script picks the new key up automatically.

```python
'MYLANG': CorpusConfig(
    key='MYLANG',
    description='My 2000-word vocabulary.',
    slug='mylang',                    # names its directories
    corpus_file='my_words.txt',       # inside data/corpora/
),
```

That is the whole entry. Every path — generated data, models, results — is
derived from `slug`, and the directories are created on first write.

**2. Generate the training data.**

```bash
python mod_lower_deck_inputs.py --corpus MYLANG
python mod_upper_deck_inputs.py --corpus MYLANG
```

**3. Train both decks.** Training stops when every pattern is correctly
classified — a few minutes each on a 2000-word vocabulary (Apple M2). Pass
`--epochs 20` first for a
quick end-to-end check.

```bash
python lower_deck.py --corpus MYLANG
python upper_deck.py --corpus MYLANG
```

**4. Run it.**

```bash
python two_deck.py --corpus MYLANG --mode 1
```

If your words are not 7 letters long, also change `WORD_LENGTH` and
`WINDOW_LENGTH` in `config.py`. Note that `weight_multiplier.py` defines one
acuity gradient per fixation position and would need a matching number of
gradients.

---

## Repository layout

**Configuration**

| File | Purpose |
|------|---------|
| `config.py` | Corpus registry, file paths, word geometry, UTF-16 I/O. The only file to edit when adding a vocabulary. |

**Pipeline** — run in this order for a new corpus

| File | Purpose |
|------|---------|
| `mod_lower_deck_inputs.py` | Vocabulary → positionally-shifted training inputs |
| `mod_upper_deck_inputs.py` | Vocabulary → word-centred targets and class labels |
| `lower_deck.py` | Trains the location-invariance stage |
| `upper_deck.py` | Trains the lexical-identification stage |
| `two_deck.py` | **Main entry point.** Runs both decks and the test batteries |

**Model internals**

| File | Purpose |
|------|---------|
| `weight_multiplier.py` | Fixation-dependent visual acuity gradients |
| `output_evaluation.py` | Decodes lower-deck activations into letters |
| `analytics.py` | Input generators for the Dandurand test batteries |

**Representation analysis**

| File | Purpose |
|------|---------|
| `testbed_input_modding.py` | Builds the single-letter probe set |
| `testbed_target_words.py` | Probe labels for reading results into R |
| `testbed_lower_deck.py` | Extracts hidden-layer activations, computes distances |
| `euclidean_distance.py` | Proximity and clustering distance measures |
| `letter_category_effect.py` | k-means over hidden representations |

**Optional**

| File | Purpose |
|------|---------|
| `plot_models.py` | Regenerates architecture diagrams (needs Graphviz) |

**Data, models and results**

Each corpus owns one directory in each of three trees, named by its slug —
`finnish`, `french`, `fin_random`. Nothing is distinguished by filename prefix,
so a file cannot belong to the wrong corpus.

```
data/corpora/          the vocabularies you supply, one word per token
data/generated/<slug>/ positional_corpus.txt  labels.txt
                       target_words.txt       upper_deck_labels.txt
                       probes.txt
models/<slug>/         lower_deck.h5  lower_deck_mapping.pkl
                       upper_deck.h5  upper_deck_mapping.pkl
results/<slug>/        activation_vectors.csv   euclidean_calculations.csv
                       proximity_effect.csv     clustering_effect.csv
results/               analysis.txt, figures/
archive/               superseded material, kept for provenance
```

Everything under `data/generated/`, `models/` and `results/` is reproducible
from `data/corpora/` by the pipeline above. It is committed anyway so that a
clone can be run and checked without retraining first.

**Superseded** — `archive/`. Kept for reference only; none of it is part of the
current model and none of it runs from a clone.

- `zero_deck.py` (2023-04) — first proof of concept. A single Dense layer that
  classifies which padded string it was shown, with no hidden layer and no
  location invariance. Superseded once inputs and targets were separated and
  the explicit character mapping replaced `TextVectorization`.
- `one_deck.py` (2023-05) — the direct ancestor of the current model:
  `Flatten → Dense(60) → Dense(501)` over a 500-word corpus, where 60 is the
  same `sqrt(words × letters)` rule that gives 119 today. It maps input
  straight to word identity in one network. Splitting it into `lower_deck.py`
  and `upper_deck.py` is what makes the word-centred orthographic code an
  explicit, inspectable representation — which every test battery in
  `two_deck.py` reads.
- `lower_deck_evaluation.py` (2023-05) — scratch harness used before the upper
  deck existed. Superseded by `two_deck.py --mode 1`. It cannot run: it
  reshapes to a vocabulary of 27 that matches no corpus, and calls
  `output_eval()` with one of its three required arguments.
- `lower_deck.h5`, `upper_deck.h5`, `lower_deck_mapping.pkl`,
  `upper_deck_mapping.pkl` — an earlier training run of the **French** model
  under the pre-prefix naming scheme (identical alphabet and architecture; the
  mappings are byte-identical to `french_*_mapping.pkl`, the weights differ).
  Superseded by `models/french/lower_deck.h5` and `models/french/upper_deck.h5`.
- `*.rtf` files — corpora and labels from before the UTF-16 `.txt` convention.

These read RTF inputs, two of which (`positional_corpus.rtf`, `word_list.rtf`)
live outside the repository, so they cannot run for anyone who clones it.

---

## Project evolution

- 01.03.2023: Project started on/off. Basic Keras/TF tutorials and theory regarding NeuralNets
- 03.04.2023: Working implementation of 'zero-deck-topology'. Prep work for one-deck-topology.
- 24.04.2023: Preliminary implementation of 'one-deck-topology'.
- 12.05.2023: Lower deck of 'two-deck-topology' produces wanted outputs.
- 30.05.2023: Upper deck of 'two-deck-topology' produces wanted outputs and started work on the flow through both decks.
- 20.06.2023: Two deck replication based on Dandurand et. al. (2013) complete.
- 30.08.2023: Test cases replicated for French corpus.
- 05.09.2023: Working Finnish implementation sans test cases.
- 13.02.2025: Random Finnish control corpus and Study 1 result tables in CSV format.
- 10.08.2026: Second iteration of the project with user friendly CLI implementation for running and building. Reproducibility pass. Corpora and trained models committed, per-script
  hard-coded paths replaced by `config.py` and `--corpus`, dependencies pinned,
  documentation added. Verified to reproduce the previous implementation's
  output on all eight test modes.

---

## For more information

Hautala, J., Hawelka, S., & Aro, M. (2021). Dual-stage and dual-deficit? Word recognition processes during text reading
across the reading fluency continuum. Reading and Writing, 1-24. https://doi.org/10.1007/s11145-021-10201-1

Dandurand, F., Hannagan, T., & Grainger, J. (2013). Computational models of location-invariant orthographic processing.
Connection Science, 25(1), 1-26.

Sibley, D. E., & Kello, C. T. (2012). Learned Orthographic Representations Facilitates Large-Scale Modeling of Word
Recognition. In Visual Word Recognition Volume 1 (pp. 28-51). Psychology Press.

Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements
in text reading. Psychological review, 125(6), 969.

Li, X., & Pollatsek, A. (2020). An integrated model of word processing and eye-movement control during Chinese reading.
Psychological Review, 127(6), 1139.
