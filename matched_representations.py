"""Tables 3 and 4 from lower decks trained to a common epoch count.

Why this exists
---------------
The shipped models stop at their own criterion, which halts each at a different
epoch: Finnish 421, French 404, and the random control never at all -- it runs
to the 2000-epoch ceiling. That is the right rule for the recognition and
nonword results, where each model should be trained until it can do its job.

It is the wrong basis for comparing hidden-layer geometry across models. Longer
training increases representational separation on its own, so a five-fold
difference in training length confounds any cross-model distance comparison.
Measured directly: the random corpus scores 3.91 on proximity at 2000 epochs but
2.92 at 421, which is the whole of its apparent advantage over Finnish (2.97).

This script therefore trains all three lower decks for the same number of epochs
and reports the distance statistics from those. 421 is used because it is where
Finnish meets the criterion -- the latest of the two natural stopping points, so
neither real corpus is cut short of its criterion, and French simply trains 17
epochs longer than it needs.

Nothing here overwrites a shipped model. The distance matrices are written to
results/matched/<slug>/ so the figures and statistics can be regenerated without
retraining.

Usage::

    python matched_representations.py
    python matched_representations.py --epochs 421 --out results/matched
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from scipy import stats

import config
import euclidean_distance
import losses
import weight_multiplier

#: Where Finnish meets the criterion; see the module docstring.
MATCHED_EPOCHS = 421
HIDDEN_UNITS = 119
SEED = 24
ORDER = ('FR', 'FIN', 'FIRND')


def train_lower_deck(corpus, epochs):
    """Train one lower deck for a fixed number of epochs, without saving it."""
    positional_text = config.load_doc(corpus.positional_corpus)
    positional_words = positional_text.split()
    mapping = config.build_character_mapping(positional_text)
    vocab_size = len(mapping)

    inputs = weight_multiplier.apply_input_weights(
        config.encode_words(positional_words, mapping, vocab_size))
    target_words = config.load_doc(corpus.target_words).split()
    targets = config.encode_words(
        target_words, mapping, vocab_size).reshape(len(target_words), -1)

    tf.keras.utils.set_random_seed(SEED)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(
            input_shape=(max(len(w) for w in positional_words), vocab_size)),
        tf.keras.layers.Dense(HIDDEN_UNITS, activation='sigmoid',
                              kernel_initializer=initializer,
                              name='hidden_layer'),
        tf.keras.layers.Dense(config.WORD_LENGTH * vocab_size,
                              activation='sigmoid'),
    ])
    model.compile(loss=losses.summed_cross_entropy,
                  optimizer=tf.keras.optimizers.legacy.SGD(
                      learning_rate=0.9, momentum=0.2))
    model.fit(inputs, targets, epochs=epochs, verbose=0)
    return model, mapping


def hidden_activations(model, corpus, mapping):
    """Hidden-layer response to the single-letter probes."""
    probes = config.load_doc(corpus.testbed_inputs).split()
    encoded = config.encode_words(probes, mapping, len(mapping))
    hidden = tf.keras.Model(inputs=model.input,
                            outputs=model.get_layer('hidden_layer').output)
    return np.array(hidden(encoded))


def distance_matrices(activations):
    proximity_raw = euclidean_distance.calculate_euclidean_distance(
        activations, config.WINDOW_LENGTH)
    proximity = euclidean_distance.count_average_euclidean_distance(
        proximity_raw)
    clustering = euclidean_distance.calculate_euclidean_distance(
        activations, activations.shape[0])
    return proximity, clustering


def proximity_values(matrix):
    """Off-diagonal cells; the diagonal is a position against itself."""
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def letter_cluster_values(matrix):
    """Probe-by-probe distances collapsed to letter by letter."""
    positions = config.WINDOW_LENGTH
    letters = matrix.shape[0] // positions
    return matrix.reshape(
        letters, positions, letters, positions).mean(axis=(1, 3)).ravel()


def report(title, values, section):
    print('\n{}  ({})'.format(title, section))
    print('-' * 66)
    for key in ORDER:
        v = values[key]
        print('  {:16} M = {:.2f}   SD = {:.2f}   (n = {})'.format(
            key, v.mean(), v.std(ddof=1), len(v)))
    f, p = stats.f_oneway(values['FR'], values['FIN'])
    df2 = len(values['FR']) + len(values['FIN']) - 2
    print('  French vs Finnish: F(1, {}) = {:.2f}, p {}'.format(
        df2, f, '< .001' if p < 0.001 else '= {:.3f}'.format(p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--epochs', type=int, default=MATCHED_EPOCHS,
                        help='Common epoch count for all three lower decks.')
    parser.add_argument('--out', default='results/matched',
                        help='Directory for the distance matrices.')
    args = parser.parse_args()

    proximity, clustering = {}, {}
    for key in ORDER:
        corpus = config.get(key)
        print('training {} lower deck for {} epochs...'.format(
            key, args.epochs), flush=True)
        model, mapping = train_lower_deck(corpus, args.epochs)
        activations = hidden_activations(model, corpus, mapping)
        prox, clus = distance_matrices(activations)
        proximity[key] = proximity_values(prox)
        clustering[key] = letter_cluster_values(clus)

        target = os.path.join(args.out, corpus.slug)
        for data, name in ((prox, 'proximity_effect.csv'),
                           (clus, 'clustering_effect.csv')):
            path = config.ensure_parent(os.path.join(target, name))
            np.savetxt(str(path), data, delimiter=',', fmt='%1.8f')
        print('  wrote {}/'.format(target))

    print('\nAll three lower decks trained for {} epochs.'.format(args.epochs))
    report('Proximity effect', proximity,
           'position x position, diagonal excluded')
    report('Letter cluster effect', clustering,
           'letter x letter, diagonal included')
    print()


if __name__ == '__main__':
    main()
