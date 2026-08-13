"""Stop training when the network correctly classifies all training patterns.

Dandurand et al. (2013) S2.5 state the rule and then report an observation:

    "Networks were trained until they could correctly classify all training
     patterns, that is, reach perfect accuracy. We empirically found that the
     following SSE values yielded such accuracy: 100 for zero-deck networks and
     for decks 1 and 2 of two-deck networks, and 50 for one-deck networks."

The rule is perfect classification. The SSE figures are what *their* networks
happened to show when they got there, and are specific to their initialisation,
their simulator and their corpus -- ours reach the criterion at a very different
SSE, and the FIRND control never approaches 100 at all. So this module stops on
the criterion itself rather than on their number.

"Correctly classified" is defined in the same section: "the lexical output unit
corresponding to the target word is activated above a threshold value of 0.9,
while all other outputs are activated below that same threshold". Footnote 7
notes this is deliberately stricter than simply requiring the target to win.

That definition is written for lexical units. The lower deck has no lexical
output, so the same *form* of test is applied to each of its letter blocks: the
target letter's unit above threshold, every other unit in that block below it.
Requiring the threshold rather than just the argmax matters here, because the
upper deck now receives the lower deck's activations directly -- a letter that
is merely winning at 0.4 would be passed on as weak evidence.

Training always stops at ``--epochs`` regardless, so a network that never meets
the criterion (the random-string control does not) still terminates.
"""

import numpy as np
import tensorflow as tf

import config

#: The paper's recognition threshold. Also two_deck.NONWORD_THRESHOLD.
THRESHOLD = 0.9


class CriterionStopping(tf.keras.callbacks.Callback):
    """Halt training once every training pattern is correctly classified.

    Parameters
    ----------
    inputs
        Training inputs, exactly as passed to ``fit``.
    targets
        Lexical: an (n_patterns,) array of target unit indices.
        Letters: an (n_patterns, WORD_LENGTH) array of target letter indices.
    kind
        ``'lexical'`` for the upper deck, ``'letters'`` for the lower deck.
    vocab_size
        Width of one letter block. Required when ``kind='letters'``.
    check_every
        Epochs between checks. Each check is one forward pass over the training
        set, so this trades precision in the reported epoch against runtime.
    """

    def __init__(self, inputs, targets, kind, vocab_size=None,
                 threshold=THRESHOLD, check_every=1, verbose=True):
        super().__init__()
        if kind not in ('lexical', 'letters'):
            raise ValueError(
                "kind must be 'lexical' or 'letters', got {!r}".format(kind))
        if kind == 'letters' and vocab_size is None:
            raise ValueError("vocab_size is required when kind='letters'")

        self.inputs = inputs
        self.targets = targets
        self.kind = kind
        self.vocab_size = vocab_size
        self.threshold = threshold
        self.check_every = check_every
        self.verbose = verbose

        #: Epoch at which the criterion was first met, or None.
        self.epoch_met = None
        #: Proportion correct at the last check.
        self.last_accuracy = 0.0

    # -- scoring ----------------------------------------------------------
    def _correct(self, predictions):
        """Boolean array: is each pattern correctly classified?"""
        if self.kind == 'lexical':
            return self._threshold_test(predictions, self.targets)

        blocks = predictions.reshape(
            len(predictions), config.WORD_LENGTH, self.vocab_size)
        # Every letter block must pass the test independently.
        per_block = np.stack([
            self._threshold_test(blocks[:, i, :], self.targets[:, i])
            for i in range(config.WORD_LENGTH)
        ], axis=1)
        return per_block.all(axis=1)

    def _threshold_test(self, activations, targets):
        """Target unit above threshold, every other unit below it."""
        rows = np.arange(len(activations))
        target_activation = activations[rows, targets]
        others = activations.copy()
        others[rows, targets] = -np.inf
        return ((target_activation > self.threshold)
                & (others.max(axis=1) < self.threshold))

    def accuracy(self):
        predictions = self.model.predict(
            self.inputs, batch_size=512, verbose=0)
        return float(self._correct(predictions).mean())

    # -- keras hook -------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        if epoch_number % self.check_every:
            return

        self.last_accuracy = self.accuracy()
        if logs is not None:
            logs['criterion'] = self.last_accuracy

        if self.last_accuracy >= 1.0:
            self.epoch_met = epoch_number
            self.model.stop_training = True
            if self.verbose:
                print('\ncriterion reached at epoch {}: all {} training '
                      'patterns correctly classified (target unit > {}, all '
                      'others below)'.format(
                          epoch_number, len(self.inputs), self.threshold))

    # -- reporting --------------------------------------------------------
    def summary(self, max_epochs):
        """One line for the training log and for a methods section."""
        if self.epoch_met is not None:
            return ('Stopped at the criterion: all training patterns correctly '
                    'classified at epoch {}.'.format(self.epoch_met))
        return ('Criterion not reached within {} epochs; {:.2f}% of training '
                'patterns correctly classified at the last check.'.format(
                    max_epochs, 100 * self.last_accuracy))
