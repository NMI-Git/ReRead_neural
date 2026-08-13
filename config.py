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
    """All file paths and metadata belonging to one training corpus.

    Only four things are stored. Every path is derived from ``slug``, so a
    corpus cannot end up with one of its files pointing at another corpus's
    data -- which is not hypothetical: this project shipped a lookup table that
    paired FIN with the French label file, and its FIRND entry mixed two
    different filename prefixes because the prefixes were doing the job that
    directories should do.

    The layout each corpus occupies::

        data/corpora/<corpus_file>          the vocabulary you supply
        data/generated/<slug>/              derived training data
        models/<slug>/                      trained decks and char mappings
        results/<slug>/                     analysis output
    """

    key: str
    description: str
    #: Directory name used under data/generated, models and results.
    slug: str
    #: Filename of the vocabulary within data/corpora.
    corpus_file: str

    def _generated(self, name):
        return 'data/generated/{}/{}'.format(self.slug, name)

    # -- Input you provide --------------------------------------------------
    @property
    def source_corpus(self):
        """The vocabulary: one WORD_LENGTH-letter word per whitespace token."""
        return 'data/corpora/{}'.format(self.corpus_file)

    # -- Generated by mod_lower_deck_inputs.py ------------------------------
    @property
    def positional_corpus(self):
        """Each source word repeated at all FIXATION_POSITIONS placements."""
        return self._generated('positional_corpus.txt')

    @property
    def positional_labels(self):
        """Word-identity class index for each row of positional_corpus."""
        return self._generated('labels.txt')

    # -- Generated by mod_upper_deck_inputs.py ------------------------------
    @property
    def target_words(self):
        """Word-centred training targets, aligned row-for-row with the above."""
        return self._generated('target_words.txt')

    @property
    def upper_deck_labels(self):
        """Word-identity class index for each row of target_words."""
        return self._generated('upper_deck_labels.txt')

    # -- Single-letter probes (testbed_input_modding.py) --------------------
    # One string per (letter, position) pair, the letter isolated in an
    # otherwise empty window. Corpus-specific because they depend on the
    # alphabet: 23 letters for Finnish, 37 for French, so a probe set built for
    # one cannot be fed to the other's model.
    @property
    def testbed_inputs(self):
        return self._generated('probes.txt')

    @property
    def testbed_target_words(self):
        return self._generated('probe_target_words.txt')

    # -- Produced by training ----------------------------------------------
    @property
    def lower_deck_model(self):
        return 'models/{}/lower_deck.h5'.format(self.slug)

    @property
    def lower_deck_mapping(self):
        return 'models/{}/lower_deck_mapping.pkl'.format(self.slug)

    @property
    def upper_deck_model(self):
        return 'models/{}/upper_deck.h5'.format(self.slug)

    @property
    def upper_deck_mapping(self):
        return 'models/{}/upper_deck_mapping.pkl'.format(self.slug)

    # -- Produced by the representation analysis (testbed_lower_deck.py) ----
    @property
    def activation_vectors(self):
        return 'results/{}/activation_vectors.csv'.format(self.slug)

    @property
    def euclidean_calculations(self):
        return 'results/{}/euclidean_calculations.csv'.format(self.slug)

    @property
    def proximity_effect(self):
        return 'results/{}/proximity_effect.csv'.format(self.slug)

    @property
    def clustering_effect(self):
        return 'results/{}/clustering_effect.csv'.format(self.slug)


CORPORA = {
    'FIN': CorpusConfig(
        key='FIN',
        description='2000 real Finnish 7-letter words.',
        slug='finnish',
        corpus_file='finnish_corpus.txt',
    ),
    'FR': CorpusConfig(
        key='FR',
        description='1985 real French 7-letter words (Dandurand et al., 2013).',
        slug='french',
        corpus_file='french_corpus.txt',
    ),
    # Control corpus: same alphabet and word length as FIN but with the
    # orthographic structure of Finnish removed, used to test how much of the
    # model's behaviour depends on real orthographic regularities.
    'FIRND': CorpusConfig(
        key='FIRND',
        description='2000 random 7-letter strings over the Finnish alphabet.',
        slug='fin_random',
        corpus_file='fin_random_corpus.txt',
    ),
}


def ensure_parent(filename):
    """Create the directory a generated file will be written into.

    Training and analysis scripts write into models/<slug>/ and results/<slug>/,
    which will not exist yet for a corpus added after checkout.
    """
    path = _resolve(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
