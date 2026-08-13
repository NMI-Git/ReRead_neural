"""Generate the single-letter probe inputs for the representation analysis.

Produces one input string per (letter, position) pair, with a single letter
isolated in an otherwise empty window::

    a############
    #a###########
    ##a##########
    ...

Feeding these through the lower deck and reading the hidden layer shows how the
network encodes letter identity and letter position independently of any word.
For a 23-letter alphabet in a 13-slot window this is 299 probes.

The alphabet comes from the trained lower deck's own character mapping, so the
probes always match the model they will be fed to.

Usage::

    python testbed_input_modding.py --corpus FIN
"""

from pickle import load

import analytics
import config


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(corpus.lower_deck_mapping)

    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        mapping = load(handle)

    probes = analytics.single_letter_positional_input(
        mapping, config.WINDOW_LENGTH)
    config.save_doc(' '.join(probes), corpus.testbed_inputs)

    print('{} letters x {} positions = {} probes'.format(
        len(probes) // config.WINDOW_LENGTH, config.WINDOW_LENGTH,
        len(probes)))
    print('  wrote {}'.format(corpus.testbed_inputs))


if __name__ == '__main__':
    main()
