"""Train the upper deck: word-centred orthography to lexical identity.

The upper deck is the second stage of the two-deck model. It takes the
word-centred letter string the lower deck produces and activates a single
localist unit standing for one entry in the lexicon. The activation value of
that unit -- not just which unit wins -- is the quantity the Dandurand test
batteries in two_deck.py measure, since it expresses how confidently the model
recognises the string as a real word.

Input
    One-hot encoding of a WORD_LENGTH-letter word, shape
    (WORD_LENGTH, vocab_size). Note this vocabulary excludes the filler token,
    which is why it is one smaller than the lower deck's.
Output
    One sigmoid unit per word in the lexicon.

Running this overwrites the trained model and character mapping for the chosen
corpus. The repository ships pre-trained models.

Usage::

    python upper_deck.py --corpus FIN
    python upper_deck.py --corpus FIN --epochs 100     # quick smoke test
"""

import argparse
from pickle import dump

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

import config

tf.keras.utils.set_random_seed(24)
np.set_printoptions(threshold=np.inf)

DEFAULT_EPOCHS = 2000
LEARNING_RATE = 0.9
MOMENTUM = 0.2
L2_REGULARIZATION = 0.0005


def build_character_mapping(text):
    """Map every character in ``text`` to an integer index."""
    chars = sorted(set(text))
    chars.remove(' ')
    return {char: index for index, char in enumerate(chars)}


def build_model(word_length, vocab_size, lexicon_size):
    """Flatten -> one sigmoid unit per lexical entry."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(word_length, vocab_size)))
    model.add(tf.keras.layers.Dense(
        lexicon_size, activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)))
    model.compile(
        loss=tf.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.legacy.SGD(
            learning_rate=LEARNING_RATE, momentum=MOMENTUM),
        metrics=['accuracy'],
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_corpus_argument(parser)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Training epochs.')
    args = parser.parse_args()
    corpus = config.get(args.corpus)

    # These two files must belong to the same corpus. They previously did not:
    # the corpus lookup table paired FIN with the French label file and FR with
    # the Finnish one, which would train a model against another language's
    # word indices. Selecting them by name from one config entry makes that
    # class of mistake impossible.
    config.require(corpus.target_words, corpus.upper_deck_labels)

    word_text = config.load_doc(corpus.target_words)
    words = word_text.split()
    mapping = build_character_mapping(word_text)
    vocab_size = len(mapping)
    word_length = max(len(word) for word in words)

    sequences = np.array([[mapping[char] for char in word] for word in words])
    inputs = to_categorical(sequences, vocab_size)

    labels = config.load_doc(corpus.upper_deck_labels).split()
    lexicon_size = len(set(labels))
    targets = to_categorical(np.array(labels), lexicon_size)

    if len(words) != len(labels):
        raise SystemExit(
            'Input/label mismatch: {} words in {} but {} labels in {}. '
            'Both are produced by mod_upper_deck_inputs.py -- regenerate '
            'them together.'.format(
                len(words), corpus.target_words,
                len(labels), corpus.upper_deck_labels))

    print('corpus          : {} ({})'.format(corpus.key, corpus.description))
    print('vocabulary      : {} characters {}'.format(
        vocab_size, sorted(mapping)))
    print('input shape     : {}'.format(inputs.shape))
    print('lexicon size    : {}'.format(lexicon_size))

    model = build_model(word_length, vocab_size, lexicon_size)
    model.summary()
    model.fit(inputs, targets, epochs=args.epochs)

    loss, accuracy = model.evaluate(inputs, targets, batch_size=128)
    print('training-set loss: {:.6f}  accuracy: {:.6f}'.format(loss, accuracy))

    model.save(config.PROJECT_ROOT / corpus.upper_deck_model)
    with open(config.PROJECT_ROOT / corpus.upper_deck_mapping, 'wb') as handle:
        dump(mapping, handle)
    print('saved {} and {}'.format(
        corpus.upper_deck_model, corpus.upper_deck_mapping))


if __name__ == '__main__':
    main()
