"""Export every data file behind the statistical tests in the manuscript.

Writes into results/stats/, one file per corpus per analysis, so each reported
test can be re-run from the saved values without retraining anything.

    table1_<slug>.csv            per-word orthographic measures. The t-tests in
                                 Table 1 compare the French and Finnish columns
                                 of these files.
    table2_<slug>_<cond>.csv     per-item model output for every test condition
                                 in Table 2: the stimulus, what the lower deck
                                 reconstructed, which lexical unit won and how
                                 strongly, the target word's own unit, whether
                                 the item was excluded, and whether it passed
                                 threshold.
    proximity_values_<slug>.csv  the 156 off-diagonal cells of the 13 x 13
                                 position matrix -- the input to the F-test
                                 reported for proximity effects.
    cluster_values_<slug>.csv    the letter-by-letter cells -- the input to the
                                 F-test reported for letter cluster effects.
    cluster_matrix_<slug>.csv    those cells as a square matrix, which is what
                                 Figure 3 plots.

Table 2 uses the shipped models. The proximity and cluster files use the
matched-epoch lower decks in results/matched/, because comparing hidden layers
across models requires equal training -- see matched_representations.py.

Usage::

    python export_statistics.py
    python export_statistics.py --skip-table1     # the slow one
"""

import argparse
import os
from pickle import load

import numpy as np
import tensorflow as tf
from keras.models import load_model

import config
import corpus_measures
import two_deck

OUT = 'results/stats'
ORDER = ('FR', 'FIN', 'FIRND')

#: label, mode, sub-mode. Matches the rows of Table 2.
CONDITIONS = [
    ('realwords', 1, None),
    ('rs', 2, None),
    ('srl', 3, None),
    ('dls', 4, None),
    ('lt', 5, None),
    ('rpp1234', 6, '1'),
    ('rpp1357', 6, '2'),
    ('tlp1235467', 7, '1'),
    ('tlp123dd67', 7, '2'),
]


def write(rows, header, name):
    path = config.ensure_parent(os.path.join(OUT, name))
    with open(path, 'w') as handle:
        handle.write(','.join(header) + '\n')
        for row in rows:
            handle.write(','.join(str(value) for value in row) + '\n')
    print('  wrote {:44} {} rows'.format(
        str(path.relative_to(config.PROJECT_ROOT)), len(rows)))


# ---------------------------------------------------------------- Table 1
def export_table1(key):
    corpus = config.get(key)
    words = config.load_doc(corpus.source_corpus).split()
    stats = corpus_measures.describe(key)
    rows = [
        (word,
         int(stats['Neighbour words #'][i]),
         '{:.6f}'.format(stats['Levenshtein distance 20, mean'][i]),
         '{:.6f}'.format(stats['Levenshtein distance 20, sd'][i]),
         int(stats['Spread (#letters yielding a neighbour)'][i]),
         int(stats['Uniqueness point'][i]))
        for i, word in enumerate(words)
    ]
    write(rows,
          ['word', 'neighbours', 'old20_mean', 'old20_sd', 'spread',
           'uniqueness_point'],
          'table1_{}.csv'.format(corpus.slug))


# ---------------------------------------------------------------- Table 2
def export_table2(key):
    corpus = config.get(key)
    lower = load_model(config.PROJECT_ROOT / corpus.lower_deck_model,
                       compile=False)
    upper = load_model(config.PROJECT_ROOT / corpus.upper_deck_model,
                       compile=False)
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        mapping = load(handle)
    lexicon = config.load_doc(corpus.source_corpus).split()

    for label, mode, sub in CONDITIONS:
        # Same seed the batteries use, so the random stimuli in modes 2, 3, 4
        # and 7.2 are the exact ones the reported results came from.
        tf.keras.utils.set_random_seed(24)
        raw_inputs = two_deck.build_inputs(mode, sub, None, corpus, mapping)
        decoded, lower_out = two_deck.run_lower_deck(
            lower, raw_inputs, mapping, None)
        blocks = lower_out.reshape(
            len(raw_inputs), config.WORD_LENGTH, len(mapping))
        winners, winning_act, raw_upper = two_deck.run_upper_deck(
            upper, blocks, None)

        # Random strings and single-repeated-letter items are not derived
        # from any particular word, so they have no target unit to read.
        if mode == 1:
            targets = np.array(
                config.load_doc(corpus.upper_deck_labels).split(), dtype=int)
        elif mode in (4, 5, 6, 7):
            targets = two_deck.priming_targets(corpus)
        else:
            targets = None

        if targets is None:
            target_names = [''] * len(raw_inputs)
            target_act = np.full(len(raw_inputs), np.nan)
        else:
            target_names = [lexicon[t] for t in targets]
            target_act = raw_upper[np.arange(len(raw_upper)), targets]

        excluded, _ = two_deck.excluded_items(corpus, mode, sub)
        if excluded is None:
            excluded = np.zeros(len(raw_inputs), dtype=bool)

        if mode in (6, 7):
            threshold, scored = two_deck.PRIMING_THRESHOLD, target_act
        else:
            threshold, scored = two_deck.NONWORD_THRESHOLD, winning_act
        passed = scored >= threshold

        rows = [
            (raw_inputs[i], decoded[i], lexicon[winners[i]],
             '{:.6f}'.format(winning_act[i]), target_names[i],
             '' if np.isnan(target_act[i]) else '{:.6f}'.format(target_act[i]),
             int(excluded[i]), int(passed[i]))
            for i in range(len(raw_inputs))
        ]
        write(rows,
              ['input', 'lower_deck_output', 'upper_deck_winner',
               'winner_activation', 'target', 'target_activation',
               'excluded', 'over_threshold'],
              'table2_{}_{}.csv'.format(corpus.slug, label))


# ------------------------------------------------- proximity / letter cluster
def export_effects(key):
    corpus = config.get(key)
    base = os.path.join('results/matched', corpus.slug)

    proximity = np.loadtxt(
        str(config.PROJECT_ROOT / base / 'proximity_effect.csv'),
        delimiter=',')
    off_diagonal = ~np.eye(proximity.shape[0], dtype=bool)
    rows = [(r + 1, c + 1, '{:.6f}'.format(proximity[r, c]))
            for r in range(proximity.shape[0])
            for c in range(proximity.shape[1]) if off_diagonal[r, c]]
    write(rows, ['position_a', 'position_b', 'distance'],
          'proximity_values_{}.csv'.format(corpus.slug))

    clustering = np.loadtxt(
        str(config.PROJECT_ROOT / base / 'clustering_effect.csv'),
        delimiter=',')
    positions = config.WINDOW_LENGTH
    letters = clustering.shape[0] // positions
    collapsed = clustering.reshape(
        letters, positions, letters, positions).mean(axis=(1, 3))

    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        alphabet = sorted(load(handle))

    rows = [(alphabet[r], alphabet[c], '{:.6f}'.format(collapsed[r, c]))
            for r in range(letters) for c in range(letters)]
    write(rows, ['letter_a', 'letter_b', 'distance'],
          'cluster_values_{}.csv'.format(corpus.slug))

    path = config.ensure_parent(
        os.path.join(OUT, 'cluster_matrix_{}.csv'.format(corpus.slug)))
    with open(path, 'w') as handle:
        handle.write(',' + ','.join(alphabet) + '\n')
        for r in range(letters):
            handle.write(alphabet[r] + ',' + ','.join(
                '{:.6f}'.format(v) for v in collapsed[r]) + '\n')
    print('  wrote {:44} {}x{}'.format(
        str(path.relative_to(config.PROJECT_ROOT)), letters, letters))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--skip-table1', action='store_true',
                        help='Skip the per-word measures (slowest step).')
    args = parser.parse_args()

    for key in ORDER:
        print('\n{}'.format(config.get(key).description))
        if not args.skip_table1:
            export_table1(key)
        export_table2(key)
        export_effects(key)
    print('\nAll statistical inputs written to {}/'.format(OUT))


if __name__ == '__main__':
    main()
