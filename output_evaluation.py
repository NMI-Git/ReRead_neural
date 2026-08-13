"""Decode the lower deck's raw activations back into a letter string.

The lower deck's output layer is one flat vector of WORD_LENGTH * vocab_size
sigmoid units, laid out as WORD_LENGTH consecutive blocks of vocab_size units.
Block *i* represents the letter in position *i* of the word-centred word, and
the most active unit within a block names that letter::

    [ block 0 ][ block 1 ][ block 2 ] ...   vocab_size units each
      ^winner    ^winner    ^winner
        'a'        'a'        'l'      ->  "aal..."

The decoded string is what gets fed to the upper deck, so this module sits on
the boundary between the two stages of the model.
"""

import numpy as np

#: Index 0 is always the filler token '#' (see config.build_character_mapping).
FILLER_INDEX = 0


def invert_mapping(mapping):
    """Turn a ``{char: index}`` mapping into ``{index: char}``.

    Build this once and reuse it. The previous implementation re-read the
    pickled mapping from disk and linearly scanned it for every single letter
    of every single prediction, which dominated the runtime of a full run.
    """
    return {index: char for char, index in mapping.items()}


def split_into_letter_blocks(flat_output, vocab_size, word_length):
    """Split a flat output vector into one activation block per letter slot."""
    expected = word_length * vocab_size
    if len(flat_output) < expected:
        raise ValueError(
            'Output vector has {} units, expected at least {} '
            '({} letters x {} vocabulary).'.format(
                len(flat_output), expected, word_length, vocab_size))
    return [flat_output[i * vocab_size:(i + 1) * vocab_size]
            for i in range(word_length)]


def winning_index(block):
    """Index of the most active unit in one letter block.

    The filler token is never a legal output: the upper deck's vocabulary does
    not contain it, so emitting one would raise a KeyError when the decoded
    string is re-encoded. When the filler unit wins, the next index is used
    instead. This reproduces the original behaviour exactly.
    """
    index = int(np.argmax(block))
    return 1 if index == FILLER_INDEX else index


def decode_word(flat_output, vocab_size, index_to_char, word_length):
    """Decode one flat output vector into a word_length-character string."""
    blocks = split_into_letter_blocks(flat_output, vocab_size, word_length)
    return ''.join(index_to_char[winning_index(block)] for block in blocks)


def letter_block_activations(flat_output, vocab_size, word_length):
    """Return the raw per-letter activation blocks, for analysis rather than decoding.

    Used by the analysis mode of two_deck.py, which reports the full activation
    profile of each letter slot instead of just the winner.
    """
    return split_into_letter_blocks(flat_output, vocab_size, word_length)
