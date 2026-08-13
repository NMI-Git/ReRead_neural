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

Filler slots carry no visual information and are zeroed out entirely rather than
weighted, so the network receives an all-zero vector for empty positions.
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

FILLER_INDEX = 0


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

    Filler slots have already been zeroed, so the first row containing a
    non-zero value is the word's first letter.
    """
    non_zero_rows = word_array.nonzero()[0]
    return weight_applier(word_array, non_zero_rows[0])


def turn_hashes_into_zero_vectors(inputs):
    """Blank the one-hot unit that stands for the filler token, in place.

    The filler is index 0 of the character mapping, so a padded slot arrives as
    a vector with a 1 in position 0. Clearing it leaves an all-zero vector,
    which is what the network should see for an empty position.
    """
    for word_array in inputs:
        for slot in word_array:
            if slot[FILLER_INDEX] == 1:
                slot[FILLER_INDEX] = 0
    return inputs


def apply_input_weights(inputs):
    """Zero the filler slots, then apply the acuity gradient to every word."""
    turn_hashes_into_zero_vectors(inputs)
    for word_array in inputs:
        seek_fixation(word_array)
    return inputs
