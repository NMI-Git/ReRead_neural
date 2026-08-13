"""Strip the filler padding from the probe inputs to get their bare letters.

Companion to testbed_input_modding.py: turns 'a############' into 'a'. The
result is a plain list of the letters the probes test, in probe order, used to
label rows when the analysis output is read into R.

Usage::

    python testbed_target_words.py --corpus FIN
"""

import config


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(corpus.testbed_inputs)

    probes = config.load_doc(corpus.testbed_inputs).split()
    letters = [probe.replace(config.FILLER_TOKEN, '') for probe in probes]

    config.save_doc(' '.join(letters), corpus.testbed_target_words)
    print('{} probe labels -> {}'.format(
        len(letters), corpus.testbed_target_words))


if __name__ == '__main__':
    main()
