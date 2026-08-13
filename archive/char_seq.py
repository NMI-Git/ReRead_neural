import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from pickle import dump

np.set_printoptions(threshold=np.inf)


# Turns out this network is likely optimized to predict the next most likely word in a sequence.
# Need to re-think the network more in the lines of a Reconstruction LSTM autoencoder instead of a prediction one


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8-sig')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load text
raw_text = load_doc('../poem.txt')

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i - length:i + 1]
    # store
    sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)


in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')


chars = sorted(list(set(raw_text)))
# chars contains line change and \ufeff values so we remove those as we don't want to map those.
chars.remove('\n')
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)

vocab_size = len(mapping)
# Last element is empty due to using txt file and weird encoding stuff, so we pop it for now to make life easier.
sequences.pop()
print(sequences)
sequences = np.array(sequences)
print(sequences)
#print(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
print(sequences)
X = np.array(sequences)
print(X)
y = to_categorical(y, num_classes=vocab_size)

#model = Sequential()
#model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
#model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
#model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
#              metrics=['accuracy'])
#model.fit(X, y, epochs=110, verbose=2)
#
#
## save the model to file
#model.save('lstm_auto_enc.h5')
## save the mapping
#dump(mapping, open('mapping.pkl', 'wb'))