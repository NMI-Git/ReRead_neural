"""The cost function Dandurand et al. (2013) train with.

The paper uses "cross-entropy as a cost function (Hinton, 1989)" as implemented
in the LENS simulator (S2.4). For a layer of independent sigmoid units that is
binary cross-entropy **summed** over the units of a pattern.

Keras averages instead, and for this architecture the difference is not
cosmetic. The upper deck has one output unit per lexical entry, so each training
pattern carries one target of 1 and 1999 targets of 0. Averaging scales the
single positive term down by 1/2000 relative to the summed form, leaving the
negatives to dominate: training collapses to the trivial "every unit off"
solution and no lexical unit ever activates.

Summing restores the target unit's weight. Measured on the Finnish corpus, the
same architecture and learning rate reaches 100% of the paper's recognition
criterion with this loss, against 3.95% for the previously used
``categorical_crossentropy``.

Note also that scaling a loss by a constant is exactly equivalent to scaling the
learning rate by that constant under gradient descent, momentum included. This
is why the lower deck previously needed a learning rate of 100 under Keras's
averaged mean squared error: expressed in the paper's summed convention that is
0.60, close to the 0.9 the paper reports. The learning rate was never the real
departure -- the cost function was.
"""

import tensorflow as tf


def summed_cross_entropy(y_true, y_pred):
    """Binary cross-entropy over sigmoid units, summed within each pattern."""
    epsilon = tf.keras.backend.epsilon()
    predicted = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return tf.reduce_sum(
        -(y_true * tf.math.log(predicted)
          + (1.0 - y_true) * tf.math.log(1.0 - predicted)),
        axis=-1,
    )
