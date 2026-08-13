"""Extract and analyse the lower deck's internal letter representations.

Feeds the single-letter probes (one letter isolated at one position, see
testbed_input_modding.py) through a trained lower deck and reads the 118-unit
hidden layer rather than the output layer. That hidden layer is where location
invariance has to live, so its geometry is the evidence for whether the model
learned a position-tolerant orthographic code.

Note that the acuity gradient is deliberately *not* applied here. The probes are
meant to reveal the network's own positional coding, so scaling them by a
fixation profile first would confound the two.

Writes four files, all named after the corpus:
    activation_vectors      299 x 118 raw hidden-layer activations
    euclidean_calculations  299 x 13  distance to the same letter elsewhere
    proximity_effect        13 x 13   the above averaged over letters
    clustering_effect       299 x 299 distance between all probe pairs

Usage::

    python testbed_lower_deck.py --corpus FIN
"""

from pickle import load

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical

import config
import euclidean_distance
import weight_multiplier

tf.keras.utils.set_random_seed(24)
np.set_printoptions(threshold=np.inf)

HIDDEN_LAYER_NAME = 'hidden_layer'


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(
        corpus.testbed_inputs, corpus.lower_deck_model,
        corpus.lower_deck_mapping)

    # Use the model's own mapping rather than re-deriving one from the probe
    # text. Re-deriving it happened to work only because the probe file was
    # generated from this same mapping; it would silently mis-encode if the two
    # ever drifted apart, or if probes from one corpus met another's model.
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        mapping = load(handle)
    vocab_size = len(mapping)

    probes = config.load_doc(corpus.testbed_inputs).split()
    sequences = np.array([[mapping[char] for char in probe] for probe in probes])
    inputs = to_categorical(sequences, vocab_size)
    inputs = weight_multiplier.turn_hashes_into_zero_vectors(inputs)

    model = load_model(config.PROJECT_ROOT / corpus.lower_deck_model)
    hidden_layer_model = tf.keras.Model(
        inputs=model.input, outputs=model.get_layer(HIDDEN_LAYER_NAME).output)
    activations = np.array(hidden_layer_model(inputs))

    print('corpus     : {}'.format(corpus.key))
    print('probes     : {} ({} letters x {} positions)'.format(
        len(probes), len(probes) // config.WINDOW_LENGTH,
        config.WINDOW_LENGTH))
    print('activations: {}'.format(activations.shape))

    proximity_raw = euclidean_distance.calculate_euclidean_distance(
        activations, config.WINDOW_LENGTH)
    proximity = euclidean_distance.count_average_euclidean_distance(
        proximity_raw, vocab_size)
    clustering = euclidean_distance.calculate_euclidean_distance(
        activations, activations.shape[0])

    for data, name in (
        (activations, corpus.activation_vectors),
        (proximity_raw, corpus.euclidean_calculations),
        (proximity, corpus.proximity_effect),
        (clustering, corpus.clustering_effect),
    ):
        np.savetxt(str(config.PROJECT_ROOT / name), data,
                   delimiter=',', fmt='%1.8f')
        print('  wrote {:45s} {}'.format(name, data.shape))


if __name__ == '__main__':
    main()
