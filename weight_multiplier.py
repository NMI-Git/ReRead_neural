"""Apply a fixation-dependent visual acuity gradient to the model's input.

This is the extension that takes the model beyond a plain replication of
Dandurand et al. (2013). A reader does not see every letter of a word equally
well: acuity falls off away from the fixation point. Rather than presenting each
letter as a clean one-hot vector, the active unit of each letter is scaled by a
weight that depends on where that letter falls relative to fixation.

The fixation point is the centre of the input window, slot 6 of 13. Because a
word can sit at any of seven offsets in that window, a different letter of the
word lands on the fixation point in each case, and a different gradient applies:

    window slot   0 1 2 3 4 5 6 7 8 9 10 11 12
                              ^ fixation
    ######aaltola             a               -> 1st letter fixated (FIRST)
    ###aaltola###             t               -> 4th letter fixated (FOURTH)
    aaltola######             a               -> 7th letter fixated (SEVENTH)

Each gradient lists the weight for the word's letters in order, from its first
letter to its last. The profiles combine two effects visible in the numbers: a
peak of acuity at or near the fixated letter, and a persistent advantage for the
word-initial letter, which stays high (0.6-0.95) in every profile even when
fixation is at the other end of the word.

Both effects are in Dandurand et al. (2013) S2.2, which cites Stevens and
Grainger (2003) for "within-word visibility ... for strings of 7 letters, and
different fixation positions". Note that visibility genuinely cannot be reduced
to a fixed function of window slot: the seven profiles disagree about 11 of the
13 slots, and the fixation slot alone takes values from 0.75 to 0.95 depending
on which letter of the word lands there. That is the outer-letter crowding
advantage, and it is why the gradient is selected by where the word starts.

Filler slots carry no visual information. They arrive as all-zero vectors --
config.build_character_mapping gives the filler no unit of its own -- so they
are simply left alone by the gradient below.
"""

from enum import Enum


class FixationMultipliers(Enum):
    """Per-letter acuity weights, named for which letter of the word is fixated."""

    FIRST = [0.95, 0.65, 0.5, 0.45, 0.4, 0.35, 0.5]
    SECOND = [0.95, 0.8, 0.6, 0.5, 0.45, 0.4, 0.55]
    THIRD = [0.85, 0.75, 0.8, 0.6, 0.55, 0.5, 0.6]
    FOURTH = [0.8, 0.6, 0.7, 0.75, 0.65, 0.6, 0.65]
    FIFTH = [0.75, 0.5, 0.6, 0.7, 0.8, 0.65, 0.7]
    SIXTH = [0.65, 0.4, 0.45, 0.6, 0.7, 0.8, 0.75]
    SEVENTH = [0.6, 0.3, 0.3, 0.45, 0.55, 0.7, 0.85]


#: Maps the window slot the word starts at to the gradient that applies.
#: A word starting at slot 6 has its first letter on the fixation point; a word
#: starting at slot 0 has its seventh.
GRADIENT_BY_WORD_START = {
    6: FixationMultipliers.FIRST,
    5: FixationMultipliers.SECOND,
    4: FixationMultipliers.THIRD,
    3: FixationMultipliers.FOURTH,
    2: FixationMultipliers.FIFTH,
    1: FixationMultipliers.SIXTH,
    0: FixationMultipliers.SEVENTH,
}

def multiplication(word_start, weights, word_array):
    """Scale the active unit of each letter by its weight, in place."""
    for slot in range(word_start, word_start + len(weights)):
        active = word_array[slot].nonzero()
        word_array[slot][active] = weights[slot - word_start]
    return word_array


def weight_applier(word_array, word_start):
    """Apply the gradient for a word beginning at ``word_start``.

    Only slots 0-6 can start a WORD_LENGTH-letter word inside the window, so
    only those have a gradient. The letter-proximity analysis (two_deck.py mode
    8) deliberately places single letters at every slot including 7-12; those
    inputs are left unweighted rather than treated as an error, which is the
    behaviour all published results were produced with.
    """
    gradient = GRADIENT_BY_WORD_START.get(word_start)
    if gradient is None:
        return word_array
    return multiplication(word_start, gradient.value, word_array)


def seek_fixation(word_array):
    """Find where the word starts and apply the matching gradient.

    Filler slots are all-zero, so the first row containing a non-zero value is
    the word's first letter.
    """
    non_zero_rows = word_array.nonzero()[0]
    return weight_applier(word_array, non_zero_rows[0])


def apply_input_weights(inputs):
    """Apply the acuity gradient to every word in ``inputs``.

    This used to blank index 0 first, back when the filler token owned that
    index and arrived one-hot encoded. It must not do so any more: index 0 is
    now the first letter of the alphabet, so blanking it would silently delete
    every word-initial 'a'. Blanks are produced by the encoder instead, in
    config.encode_words.
    """
    for word_array in inputs:
        seek_fixation(word_array)
    return inputs
