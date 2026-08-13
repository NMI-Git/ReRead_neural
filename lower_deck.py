"""Train the lower deck: location-invariant orthographic encoding.

The lower deck is the first stage of the two-deck model. It receives a word
placed at an arbitrary position in a WINDOW_LENGTH-slot retinotopic input
window and must reproduce that word in a fixed, word-centred output frame.
Learning to do this for every placement is what makes the representation
location-invariant, following Dandurand et al. (2013).

Input
    One-hot encoding of a padded word, shape (WINDOW_LENGTH, vocab_size),
    scaled by a fixation-dependent acuity gradient (see weight_multiplier.py).
Output
    A flat vector of WORD_LENGTH * vocab_size units: one block of vocab_size
    units per letter of the word-centred word.

Running this overwrites the trained model and character mapping for the chosen
corpus. Training takes a while (2000 epochs over 14000 examples); the repository
ships pre-trained models so you do not have to run this to reproduce results.

Usage::

    python lower_deck.py --corpus FIN
    python lower_deck.py --corpus FIN --epochs 100     # quick smoke test
"""

import argparse
from pickle import dump

import numpy as np
import tensorflow as tf

import config
import losses
import stopping
import weight_multiplier

# Fixed seed so that a retrained model reproduces the published one.
tf.keras.utils.set_random_seed(24)
np.set_printoptions(threshold=np.inf)

#: Hidden layer width. Dandurand et al. (2013) set it to "the square root of the
#: number of training patterns rounded up to the closest integer", and footnote 3
#: spells the calculation out as sqrt(2000 words * 7 positions) = 119. This
#: project previously rounded down to 118.
HIDDEN_UNITS = 119

DEFAULT_EPOCHS = 2000

#: Dandurand et al. (2013) S2.4. The project previously used 100 with Keras's
#: averaged mean squared error, which is the same effective rate expressed in a
#: different normalisation -- see losses.py.
LEARNING_RATE = 0.9
MOMENTUM = 0.2


def build_model(window_length, vocab_size):
    """Flatten -> sigmoid hidden layer -> sigmoid word-centred output layer."""
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
    output_units = config.WORD_LENGTH * vocab_size

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(window_length, vocab_size)))
    model.add(tf.keras.layers.Dense(
        HIDDEN_UNITS, activation='sigmoid',
        kernel_initializer=initializer, name='hidden_layer'))
    model.add(tf.keras.layers.Dense(
        output_units, activation='sigmoid', name='output_layer'))
    model.compile(
        loss=losses.summed_cross_entropy,
        optimizer=tf.keras.optimizers.legacy.SGD(
            learning_rate=LEARNING_RATE, momentum=MOMENTUM),
        # Squared error is not the training objective any more, but the paper
        # reports SSE as its stopping signal ("we empirically found that the
        # following SSE values yielded such accuracy: 100 ... for decks 1 and 2
        # of two-deck networks"), so it stays visible during training.
        metrics=['mean_squared_error'],
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_corpus_argument(parser)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Maximum training epochs. Training normally stops '
                             'earlier, when every training pattern is '
                             'correctly classified.')
    args = parser.parse_args()
    corpus = config.get(args.corpus)

    config.require(corpus.positional_corpus, corpus.target_words)

    # --- Inputs: words at every position in the window -------------------
    positional_text = config.load_doc(corpus.positional_corpus)
    positional_words = positional_text.split()
    mapping = config.build_character_mapping(positional_text)
    vocab_size = len(mapping)
    window_length = max(len(word) for word in positional_words)

    inputs = config.encode_words(positional_words, mapping, vocab_size)
    weighted_inputs = weight_multiplier.apply_input_weights(inputs)

    # --- Targets: the same words, word-centred ---------------------------
    # Encoded with the *input* mapping so that output block i of the network
    # indexes the same alphabet the input does.
    target_words = config.load_doc(corpus.target_words).split()
    targets = config.encode_words(target_words, mapping, vocab_size)
    flattened_targets = targets.reshape(len(target_words), -1)

    if len(positional_words) != len(target_words):
        raise SystemExit(
            'Input/target mismatch: {} rows in {} but {} rows in {}. '
            'Regenerate both with mod_lower_deck_inputs.py and '
            'mod_upper_deck_inputs.py.'.format(
                len(positional_words), corpus.positional_corpus,
                len(target_words), corpus.target_words))

    print('corpus          : {} ({})'.format(corpus.key, corpus.description))
    print('vocabulary      : {} characters {}'.format(
        vocab_size, sorted(mapping)))
    print('input shape     : {}'.format(weighted_inputs.shape))
    print('target shape    : {}'.format(flattened_targets.shape))

    model = build_model(window_length, vocab_size)
    model.summary()

    # Dandurand et al. (2013) train "until they could correctly classify all
    # training patterns", not for a fixed number of epochs. See stopping.py.
    target_letters = np.array(
        [[mapping[char] for char in word] for word in target_words])
    criterion = stopping.CriterionStopping(
        weighted_inputs, target_letters, 'letters', vocab_size=vocab_size)

    model.fit(weighted_inputs, flattened_targets, epochs=args.epochs,
              callbacks=[criterion])
    print(criterion.summary(args.epochs))

    model.save(config.PROJECT_ROOT / corpus.lower_deck_model)
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'wb') as handle:
        dump(mapping, handle)
    print('saved {} and {}'.format(
        corpus.lower_deck_model, corpus.lower_deck_mapping))


if __name__ == '__main__':
    main()
