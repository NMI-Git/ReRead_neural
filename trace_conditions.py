"""Per-item trace: stimulus -> lower deck output -> upper deck output.

Prints what each deck actually does with a stimulus, for the letter
transposition battery and the four priming conditions. Useful for seeing why an
item counts as a false positive, or why a prime does or does not activate its
target.

Columns
    input        the padded stimulus as fed to the lower deck
    LD output    what the lower deck reconstructs, decoded to letters
    UD winner    the most active lexical unit, and its activation
    target       the word the stimulus was derived from, and that unit's
                 activation (priming is scored on this, not on the winner)
    flag         ! excluded, the transposition was a no-op
                 * counted as a false positive / as primed

Usage::

    python trace_conditions.py --corpus FIN
    python trace_conditions.py --corpus FIN --rows 40 --out results/traces
"""

import argparse
import os
from pickle import load

import numpy as np
from keras.models import load_model
import tensorflow as tf

import analytics
import config
import two_deck

CONDITIONS = [
    ('LT  letter transposition', 5, None, two_deck.NONWORD_THRESHOLD, 'nonword'),
    ('RPP 1234', 6, '1', two_deck.PRIMING_THRESHOLD, 'prime'),
    ('RPP 1357', 6, '2', two_deck.PRIMING_THRESHOLD, 'prime'),
    ('TLP 1235467', 7, '1', two_deck.PRIMING_THRESHOLD, 'prime'),
    ('TLP 123DD67', 7, '2', two_deck.PRIMING_THRESHOLD, 'prime'),
]


def trace(corpus, mode, sub_mode, lower, upper, mapping):
    tf.keras.utils.set_random_seed(24)
    raw_inputs = two_deck.build_inputs(mode, sub_mode, None, corpus, mapping)
    decoded, lower_out = two_deck.run_lower_deck(lower, raw_inputs, mapping, None)
    blocks = lower_out.reshape(len(raw_inputs), config.WORD_LENGTH, len(mapping))
    winners, winning_act, raw_upper = two_deck.run_upper_deck(upper, blocks, None)

    lexicon = config.load_doc(corpus.source_corpus).split()
    targets = two_deck.priming_targets(corpus)
    target_act = raw_upper[np.arange(len(raw_upper)), targets]
    excluded, _ = two_deck.excluded_items(corpus, mode, sub_mode)
    if excluded is None:
        excluded = np.zeros(len(raw_inputs), dtype=bool)
    return (raw_inputs, decoded, [lexicon[i] for i in winners], winning_act,
            [lexicon[i] for i in targets], target_act, excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_corpus_argument(parser)
    parser.add_argument('--rows', type=int, default=25,
                        help='Rows to print per condition.')
    parser.add_argument('--out', default=None,
                        help='Directory to write the full trace into.')
    args = parser.parse_args()
    corpus = config.get(args.corpus)

    lower = load_model(config.PROJECT_ROOT / corpus.lower_deck_model,
                       compile=False)
    upper = load_model(config.PROJECT_ROOT / corpus.upper_deck_model,
                       compile=False)
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        mapping = load(handle)

    header = ('{:<15}{:<11}{:<11}{:>7}   {:<11}{:>7}  {}'.format(
        'input', 'LD output', 'UD winner', 'act', 'target', 'act', 'flag'))

    for label, mode, sub, threshold, kind in CONDITIONS:
        data = trace(corpus, mode, sub, lower, upper, mapping)
        inputs, decoded, winners, wact, tgts, tact, excluded = data
        scored = tact if kind == 'prime' else wact
        hit = scored >= threshold

        print('\n{}  --  {}   (threshold {}, scored on the {} unit)'.format(
            corpus.key, label, threshold,
            'target' if kind == 'prime' else 'winning'))
        print('-' * len(header))
        print(header)
        print('-' * len(header))
        for i in range(min(args.rows, len(inputs))):
            flag = '!' if excluded[i] else ('*' if hit[i] else '')
            print('{:<15}{:<11}{:<11}{:>7.4f}   {:<11}{:>7.4f}  {}'.format(
                inputs[i], decoded[i], winners[i], wact[i],
                tgts[i], tact[i], flag))
        kept = ~excluded
        print('-' * len(header))
        print('{}: {}/{} over threshold'.format(
            'primed' if kind == 'prime' else 'false positives',
            int(hit[kept].sum()), int(kept.sum())), end='')
        print('   ({} excluded)'.format(int(excluded.sum()))
              if excluded.any() else '')

        if args.out:
            os.makedirs(config.PROJECT_ROOT / args.out, exist_ok=True)
            name = '{}_{}.csv'.format(
                corpus.slug, label.split()[0].lower() + (sub or ''))
            path = config.PROJECT_ROOT / args.out / name
            with open(path, 'w') as fh:
                fh.write('input,lower_deck_output,upper_deck_winner,'
                         'winner_activation,target,target_activation,'
                         'excluded,over_threshold\n')
                for i in range(len(inputs)):
                    fh.write('{},{},{},{:.6f},{},{:.6f},{},{}\n'.format(
                        inputs[i], decoded[i], winners[i], wact[i],
                        tgts[i], tact[i], int(excluded[i]), int(hit[i])))
            print('wrote {}'.format(path.relative_to(config.PROJECT_ROOT)))


if __name__ == '__main__':
    main()
