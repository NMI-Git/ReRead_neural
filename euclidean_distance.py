"""Euclidean distance measures over the lower deck's hidden representations.

Two different questions are asked of the same 299 x 118 matrix of hidden-layer
activations produced by the single-letter probes:

Proximity effect
    How far apart are the representations of the *same* letter at *different*
    positions? Compares each probe against the block of probes for its own
    letter, giving a 299 x 13 matrix that is then averaged down to 13 x 13.
    Small distances mean the code is position-tolerant.

Clustering effect
    How far apart is every probe from every other probe? A full 299 x 299
    matrix, used to test whether representations group by letter identity
    rather than by position.
"""

import math

import numpy as np


def calculate_euclidean_distance(intermediate_output, block_length):
    """Distances from each row to the rows of its own block.

    ``block_length`` selects which comparison is made: pass WINDOW_LENGTH (13)
    to compare each probe only against the other positions of the same letter
    (the proximity effect), or the total number of rows to compare every probe
    against every other one (the clustering effect).
    """
    rows = len(intermediate_output)
    distances = np.zeros((rows, block_length))
    block_start = 0
    for row in range(rows):
        if row != 0 and row % block_length == 0:
            block_start = row
        for offset in range(block_length):
            distances[row][offset] = math.dist(
                intermediate_output[row],
                intermediate_output[block_start + offset])
    return distances


def count_average_euclidean_distance(euclidean_output):
    """Average a (letters x positions) x 13 proximity matrix down to 13 x 13.

    Sums the distance for each (position, position) pair across all letters,
    then divides by the number of letters.

    The letter count is derived from the data rather than passed in. It used to
    be taken from the character mapping minus one, because the mapping carried a
    filler token that is never probed. The filler no longer has a unit (see
    config.build_character_mapping), so that subtraction would now divide by one
    letter too few and inflate every cell.
    """
    block_length = euclidean_output.shape[1]
    letter_count = len(euclidean_output) // block_length
    averages = np.zeros((block_length, block_length))
    for row in range(block_length):
        for column in range(block_length):
            index = row
            while index < len(euclidean_output):
                averages[row][column] += euclidean_output[index][column]
                index += block_length

    return averages / letter_count
