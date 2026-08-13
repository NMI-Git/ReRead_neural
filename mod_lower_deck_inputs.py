"""Generate the lower deck's training inputs from a source vocabulary.

The lower deck learns location invariance: it must map a word to the same
word-centred representation no matter where that word falls in the visual
input window. This script builds the training set for that by taking each word
of the source corpus and writing it out once per possible placement in the
window, padding the unused slots with the filler token.

For a 7-letter word in a 13-slot window that yields 7 rows::

    aaltola  ->  ######aaltola
                 #####aaltola#
                 ####aaltola##
                 ###aaltola###
                 ##aaltola####
                 #aaltola#####
                 aaltola######

Each row is labelled with the index of the word it came from, so the label file
lines up row-for-row with the corpus file.

Usage::

    python mod_lower_deck_inputs.py --corpus FIN
"""

import config


def build_positional_corpus(words):
    """Return (padded_rows, word_index_labels) for every placement of every word."""
    rows = []
    labels = []
    for word_index, word in enumerate(words):
        for offset in range(config.FIXATION_POSITIONS):
            left = config.FILLER_TOKEN * (config.PADDING_SLOTS - offset)
            right = config.FILLER_TOKEN * offset
            rows.append(left + word + right)
            labels.append(str(word_index))
    return rows, labels


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(corpus.source_corpus)

    words = config.load_doc(corpus.source_corpus).split()
    unexpected = [w for w in words if len(w) != config.WORD_LENGTH]
    if unexpected:
        raise SystemExit(
            'Corpus {} contains {} word(s) that are not {} letters long '
            '(first offender: {!r}). All words must be the same length.'.format(
                corpus.source_corpus, len(unexpected), config.WORD_LENGTH,
                unexpected[0]
            )
        )

    rows, labels = build_positional_corpus(words)
    config.save_doc(' '.join(rows), corpus.positional_corpus)
    config.save_doc(' '.join(labels), corpus.positional_labels)

    print('{} words -> {} positional rows'.format(len(words), len(rows)))
    print('  wrote {}'.format(corpus.positional_corpus))
    print('  wrote {}'.format(corpus.positional_labels))


if __name__ == '__main__':
    main()
