import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import pad_sequences
from pickle import load
import weight_multiplier
import output_evaluation


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


model = load_model('lower_deck.h5')
mapping = load(open('lower_deck_mapping.pkl', 'rb'))
raw_input_text = load_doc('../positional_corpus.rtf')
input_lines = raw_input_text.split()
test_input = input_lines[0:1]  # Words change every 7 indexes.
vocab_size = len(mapping)  # Size of vocabulary

sequences = list()
for word in test_input:
    encoded_seq = [mapping[char] for char in word]
    sequences.append(encoded_seq)
sequences = np.array(sequences)
input_hot = to_categorical(sequences, vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)
for x in range(len(weighted_inputs)):
    test_input = weighted_inputs[x].reshape(1, 13, 27)
    output = model.predict(test_input)
    output = np.array(output)
    output_evaluation.output_eval(output)

