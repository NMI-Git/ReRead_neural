"""One-deck topology from Dandurand et al. (2013): a single hidden layer
mapping location-specific letters straight onto lexical units.

Kept because the paper's central claim compares this against the zero-deck and
two-deck topologies -- see zero_deck.py and two_deck.py.

NOT CURRENTLY RUNNABLE. This script predates the config.py refactor and still
carries hardcoded paths, and it builds its own one-hot encoding in which the
filler token owns index 0. weight_multiplier.apply_input_weights no longer
blanks that index, because index 0 is a real letter for every other caller now.
Before running this, port it to config.build_character_mapping and
config.encode_words as lower_deck.py does.
"""

import numpy as np
import tensorflow as tf
import weight_multiplier
from tensorflow.keras import layers
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


raw_input_text = load_doc('../positional_corpus.rtf')
raw_labels = load_doc('labels.rtf')
label_lines = raw_labels.split()
label_lines = np.array(label_lines)
input_lines = raw_input_text.split()
chars = sorted(list(set(raw_input_text)))  # All the separate chars found in input text
chars.remove(' ')
chars.remove('\n')
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
sequences = list()
for line in input_lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
vocab_size = len(mapping)  # Size of vocabulary
sequences = np.array(sequences)

raw_output_text = load_doc('../word_list.rtf')
output_lines = raw_output_text.split()
input_hot = to_categorical(sequences, vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)
output_hot = to_categorical(label_lines)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(13, 27)))
model.add(tf.keras.layers.Dense(60,
                                activation='sigmoid'))  # Number of neurons with 500 input words rounded up to INT:
# sqrt(500 * 7) = 60
model.add(tf.keras.layers.Dense(501, activation='sigmoid'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
              metrics=['accuracy'],
              )
epochs = 50
history = model.fit(weighted_inputs, output_hot, epochs=epochs)
print("Evaluate model on test data")
results = model.evaluate(weighted_inputs, output_hot, batch_size=128)
print("test loss, test acc:", results)
test_input = weighted_inputs[0].reshape(1, 13, 27)
output = model(test_input)
output = np.array(output)
print(output)
print(np.argmax(output))
