"""Generate the word-centred target words and the upper deck's class labels.

The lower deck's *targets* are the word-centred (unpadded) form of whatever the
positional corpus holds. Because mod_lower_deck_inputs.py emits one row per
placement, each source word has to be repeated the same number of times here so
that the two files align row-for-row::

    positional corpus        target words
    ------------------       ------------
    ######aaltola            aaltola
    #####aaltola#            aaltola
    ...                      ...
    aaltola######            aaltola

The same repeated words are also the upper deck's training inputs, and the
label file gives each one its word-identity class index.

Usage::

    python mod_upper_deck_inputs.py --corpus FIN
"""

import config


def build_target_words(words):
    """Repeat each word once per fixation position, with matching class labels."""
    rows = []
    labels = []
    for word_index, word in enumerate(words):
        for _ in range(config.FIXATION_POSITIONS):
            rows.append(word)
            labels.append(str(word_index))
    return rows, labels


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(corpus.source_corpus)

    words = config.load_doc(corpus.source_corpus).split()
    rows, labels = build_target_words(words)

    config.save_doc(' '.join(rows), corpus.target_words)
    config.save_doc(' '.join(labels), corpus.upper_deck_labels)

    print('{} words -> {} target rows ({} classes)'.format(
        len(words), len(rows), len(set(labels))))
    print('  wrote {}'.format(corpus.target_words))
    print('  wrote {}'.format(corpus.upper_deck_labels))


if __name__ == '__main__':
    main()
