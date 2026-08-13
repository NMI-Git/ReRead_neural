"""Central configuration for the ReRead two-deck word-recognition model.

Every script in this project used to carry its own copy of a ``FilePathEnums``
enum plus a ``corpus_instantiation()`` function that returned a plain list whose
*positions* encoded meaning (``chosen_corpus[4]`` was "the lower deck mapping",
but only in some scripts -- the ordering differed between files). That design
made it impossible to add a corpus without editing every script, and it silently
mismatched files when two orderings drifted apart.

This module replaces all of that with one registry of named fields. To add a new
language you add a single ``CorpusConfig`` entry here and nothing else changes.

Paths are resolved relative to this file, not the current working directory, so
the scripts can be run from anywhere.
"""

import argparse
import codecs
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
# Every corpus and label file in this project is UTF-16 encoded. This is not
# incidental: the trained models shipped in this repository were built from
# character mappings derived from these exact files, so reading them with a
# different encoding produces a different mapping and therefore meaningless
# predictions. If you supply your own vocabulary it must be UTF-16 as well --
# see "Using your own vocabulary" in README.md for a conversion command.
CORPUS_ENCODING = 'utf-16'


def load_doc(filename):
    """Read a whitespace-separated word list from a UTF-16 text file.

    Accepts a path relative to the project root or an absolute path.
    """
    path = _resolve(filename)
    with codecs.open(str(path), 'r', encoding=CORPUS_ENCODING) as handle:
        return handle.read()


def save_doc(data, filename):
    """Write ``data`` to a UTF-16 text file, creating parent directories."""
    path = _resolve(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(str(path), 'w', encoding=CORPUS_ENCODING) as handle:
        handle.write(data)


def _resolve(filename):
    path = Path(filename)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Word geometry
# ---------------------------------------------------------------------------
# These three constants explain the bare numbers that used to appear inline in
# lower_deck.py, mod_lower_deck_inputs.py and analytics.py.
#
# Every word in every corpus is exactly WORD_LENGTH letters. During training the
# word is slid across a WINDOW_LENGTH-slot retinotopic input window and the
# unused slots are filled with FILLER_TOKEN, which yields FIXATION_POSITIONS
# distinct placements per word:
#
#     ######aaltola   (word at the far right of the window)
#     #####aaltola#
#     ...
#     aaltola######   (word at the far left)
#
# Learning to map all FIXATION_POSITIONS placements onto the same word-centred
# output is what makes the lower deck location-invariant, following
# Dandurand et al. (2013).
WORD_LENGTH = 7
WINDOW_LENGTH = 13
FILLER_TOKEN = '#'

#: Number of distinct positions a word can occupy in the input window (7).
FIXATION_POSITIONS = WINDOW_LENGTH - WORD_LENGTH + 1

#: Number of filler slots in any single padded input (6). The lower deck's
#: output layer is (WINDOW_LENGTH - PADDING_SLOTS) * vocab_size units wide,
#: i.e. one block of vocab_size units per letter of the word-centred output.
PADDING_SLOTS = WINDOW_LENGTH - WORD_LENGTH


# ---------------------------------------------------------------------------
# Character mapping and encoding
# ---------------------------------------------------------------------------
# Both decks must agree exactly on which letter owns which index, so the two
# functions below live here rather than being copied into lower_deck.py and
# upper_deck.py. A previous version of this project kept per-script copies of
# its configuration and they drifted apart, which is how the corpus lookup
# table came to pair FIN with the French label file.


def build_character_mapping(text):
    """Map every real letter in ``text`` to an index.

    FILLER_TOKEN is deliberately excluded. Dandurand et al. (2013) encode an
    empty slot as an all-zero vector -- "Slots in which no letter is present
    [0 0 0... 0] represent blanks" (S2.3.1.1) -- not as a unit of its own.

    Giving the filler an index cost a permanently dead unit in every input slot
    and every output block, and made the lower deck's vocabulary one wider than
    the upper deck's. That mismatch is what prevented the lower deck's output
    from being fed directly to the upper deck; with the filler gone both decks
    share one WORD_LENGTH x vocab_size space.
    """
    letters = set(text) - {' ', '\n', '\t', FILLER_TOKEN}
    return {char: index for index, char in enumerate(sorted(letters))}


def encode_words(words, mapping, vocab_size=None):
    """One-hot encode equal-length strings, filler slots becoming all zeros.

    ``to_categorical`` cannot express this: it needs an index for every slot,
    which is precisely what a blank does not have.
    """
    if vocab_size is None:
        vocab_size = len(mapping)
    encoded = np.zeros((len(words), len(words[0]), vocab_size), dtype='float32')
    for row, word in enumerate(words):
        for slot, char in enumerate(word):
            index = mapping.get(char)
            if index is not None:
                encoded[row, slot, index] = 1.0
    return encoded


@dataclass(frozen=True)
class CorpusConfig:
    """All file paths and metadata belonging to one training corpus."""

    key: str
    description: str

    # -- Inputs you provide -------------------------------------------------
    #: The vocabulary itself: one WORD_LENGTH-letter word per whitespace token.
    source_corpus: str

    # -- Generated by mod_lower_deck_inputs.py ------------------------------
    #: Each source word repeated at all FIXATION_POSITIONS placements.
    positional_corpus: str
    #: Word-identity class index for each row of positional_corpus.
    positional_labels: str

    # -- Generated by mod_upper_deck_inputs.py ------------------------------
    #: Word-centred training targets: each source word repeated
    #: FIXATION_POSITIONS times so it aligns row-for-row with positional_corpus.
    target_words: str
    #: Word-identity class index for each row of target_words.
    upper_deck_labels: str

    # -- Produced by training ----------------------------------------------
    lower_deck_model: str
    lower_deck_mapping: str
    upper_deck_model: str
    upper_deck_mapping: str

    # -- Produced by the representation analysis (testbed_lower_deck.py) ----
    # These are prefixed per corpus. testbed_lower_deck.py previously wrote
    # fixed filenames regardless of which corpus was selected, so analysing a
    # second corpus silently overwrote the first one's results.
    activation_vectors: str
    euclidean_calculations: str
    proximity_effect: str
    clustering_effect: str

    # -- Single-letter probe inputs (testbed_input_modding.py) --------------
    # One string per (letter, position) pair with the letter isolated in an
    # otherwise empty window. These are corpus-specific because they depend on
    # the corpus alphabet -- the Finnish alphabet has 23 letters and the French
    # one 37, so a probe set built for one cannot be fed to the other's model.
    testbed_inputs: str
    testbed_target_words: str


CORPORA = {
    'FIN': CorpusConfig(
        key='FIN',
        description='2000 real Finnish 7-letter words.',
        source_corpus='corpora/finnish_corpus.txt',
        positional_corpus='finnish_positional_supervised_corpus.txt',
        positional_labels='finnish_labels.txt',
        target_words='finnish_two_deck_target_words.txt',
        upper_deck_labels='finnish_upper_deck_labels.txt',
        lower_deck_model='finnish_lower_deck.h5',
        lower_deck_mapping='finnish_lower_deck_mapping.pkl',
        upper_deck_model='finnish_upper_deck.h5',
        upper_deck_mapping='finnish_upper_deck_mapping.pkl',
        activation_vectors='finnish_activation_vectors.csv',
        euclidean_calculations='finnish_euclidean_calculations.csv',
        proximity_effect='finnish_proximity_effect.csv',
        clustering_effect='finnish_clustering_effect.csv',
        testbed_inputs='finnish_testbed_lower_deck_inputs.txt',
        testbed_target_words='finnish_testbed_target_words.txt',
    ),
    'FR': CorpusConfig(
        key='FR',
        description='1985 real French 7-letter words (Dandurand et al., 2013).',
        source_corpus='corpora/french_corpus.txt',
        positional_corpus='french_positional_supervised_corpus.txt',
        positional_labels='french_labels.txt',
        target_words='french_two_deck_target_words.txt',
        upper_deck_labels='french_upper_deck_labels.txt',
        lower_deck_model='french_lower_deck.h5',
        lower_deck_mapping='french_lower_deck_mapping.pkl',
        upper_deck_model='french_upper_deck.h5',
        upper_deck_mapping='french_upper_deck_mapping.pkl',
        activation_vectors='french_activation_vectors.csv',
        euclidean_calculations='french_euclidean_calculations.csv',
        proximity_effect='french_proximity_effect.csv',
        clustering_effect='french_clustering_effect.csv',
        testbed_inputs='french_testbed_lower_deck_inputs.txt',
        testbed_target_words='french_testbed_target_words.txt',
    ),
    # Control corpus: same alphabet and word length as FIN but with the
    # orthographic structure of Finnish removed, used to test how much of the
    # model's behaviour depends on real orthographic regularities.
    # NOTE: the two label files below use a 'finnish_random_' prefix while the
    # rest of this corpus uses 'fin_random_'. The inconsistency is historical;
    # the names are recorded here so it cannot cause a mismatch.
    'FIRND': CorpusConfig(
        key='FIRND',
        description='2000 random 7-letter strings over the Finnish alphabet.',
        source_corpus='corpora/fin_random_corpus.txt',
        positional_corpus='fin_random_positional_supervised_corpus.txt',
        positional_labels='finnish_random_labels.txt',
        target_words='fin_random_two_deck_target_words.txt',
        upper_deck_labels='finnish_random_upper_deck_labels.txt',
        lower_deck_model='fin_random_lower_deck.h5',
        lower_deck_mapping='fin_random_lower_deck_mapping.pkl',
        upper_deck_model='fin_random_upper_deck.h5',
        upper_deck_mapping='fin_random_upper_deck_mapping.pkl',
        activation_vectors='fin_random_activation_vectors.csv',
        euclidean_calculations='fin_random_euclidean_calculations.csv',
        proximity_effect='fin_random_proximity_effect.csv',
        clustering_effect='fin_random_clustering_effect.csv',
        # Unprefixed names retained: these are the probe files already in the
        # repository, generated from the FIRND mapping.
        testbed_inputs='testbed_lower_deck_inputs.txt',
        testbed_target_words='testbed_target_words.txt',
    ),
}


def get(key):
    """Look up a corpus by key, with a helpful error for unknown names."""
    try:
        return CORPORA[key.upper()]
    except KeyError:
        raise SystemExit(
            "Unknown corpus '{}'. Available: {}".format(
                key, ', '.join(sorted(CORPORA))
            )
        )


def add_corpus_argument(parser):
    """Attach the shared ``--corpus`` option to an argparse parser."""
    choices = sorted(CORPORA)
    helptext = 'Corpus to use. ' + ' '.join(
        '{}: {}'.format(k, CORPORA[k].description) for k in choices
    )
    parser.add_argument(
        '-c', '--corpus',
        choices=choices,
        default='FIN',
        help=helptext,
    )
    return parser


def parse_corpus_arg(description):
    """Build a parser with only ``--corpus`` and return the chosen config."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_corpus_argument(parser)
    return get(parser.parse_args().corpus)


def require(*paths):
    """Fail early with an actionable message if an input file is missing.

    The published version of this project could not be run by others because
    scripts referenced files that were never committed, and the resulting
    ``FileNotFoundError`` gave no hint about how to produce them. Every script
    now checks its inputs up front.
    """
    missing = [str(p) for p in paths if not _resolve(p).exists()]
    if missing:
        raise SystemExit(
            'Missing required input file(s):\n'
            + '\n'.join('  - ' + m for m in missing)
            + '\n\nSee README.md: these are either shipped with the repository '
              'or generated by mod_lower_deck_inputs.py / '
              'mod_upper_deck_inputs.py before training.'
        )
