"""Run the full two-deck model and the Dandurand et al. (2013) test batteries.

This is the main entry point for reproducing the published analyses. It loads
the two pre-trained decks for a corpus and pushes inputs through them:

    padded word  ->  [lower deck]  ->  word-centred letters
                 ->  [upper deck]  ->  lexical unit + activation value

The activation value of the winning lexical unit is the measure of interest.
A real word the model knows drives its unit close to 1; a nonword or a
distorted word should not, and each test mode below probes a different kind of
distortion to check the model behaves like human readers do.

Modes
    1  Corpus run, no alteration. Reports recognition errors.
    2  RS   -- random strings (nonword discrimination).
    3  SRL  -- single repeated letter.
    4  DLS  -- double letter substitution.
    5  LT   -- letter transposition.
    6  RPP  -- relative position priming (sub-modes 1-2).
    7  TLP  -- transposed letter priming (sub-modes 1-2).
    8  Letter-proximity analysis suite (sub-modes 1-2).

Usage::

    python two_deck.py --corpus FIN --mode 1
    python two_deck.py --corpus FIN --mode 6 --sub-mode 2
    python two_deck.py --corpus FIN --mode 8 --sub-mode 1 --letter a
    python two_deck.py                       # interactive prompts, as before
"""

import argparse

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from pickle import load

import analytics
import config
import output_evaluation
import weight_multiplier

# Seeds Python's random module as well as NumPy's and TensorFlow's. Modes 2, 3,
# 4, 7 (sub-mode 2) and 8 (sub-mode 1) build their inputs with random.choice
# and random.sample, so without this they would produce different stimuli --
# and different results -- on every run.
tf.keras.utils.set_random_seed(24)

np.set_printoptions(threshold=np.inf)

#: For modes 2-5 an activation at or above this counts as a false positive:
#: the model has accepted a nonword as a word.
NONWORD_THRESHOLD = 0.9

#: For the priming modes 6-7. Note this is 0.87, not the 0.5 mentioned in some
#: older comments; 0.87 is the value all published results were produced with.
PRIMING_THRESHOLD = 0.87

#: Index of the centred placement within each word's block of
#: FIXATION_POSITIONS rows: '###aaltola###'. The distortion batteries operate
#: on the centred form only.
CENTRED_OFFSET = 3


def slice_centred_words(positional_words):
    """Take the centred placement of each word from the positional corpus."""
    return positional_words[CENTRED_OFFSET::config.FIXATION_POSITIONS]


def encode(words, mapping, vocab_size):
    """One-hot encode words using a character mapping."""
    sequences = np.array([[mapping[char] for char in word] for word in words])
    return to_categorical(sequences, vocab_size)


def run_lower_deck(model, words, mapping, batch_size=None):
    """Encode, apply the fixation acuity gradient, predict, and decode.

    Returns (decoded_strings, raw_output_matrix).
    """
    vocab_size = len(mapping)
    inputs = encode(words, mapping, vocab_size)
    weighted = weight_multiplier.apply_input_weights(inputs)

    raw = model.predict(weighted, batch_size=batch_size, verbose=0)

    index_to_char = output_evaluation.invert_mapping(mapping)
    decoded = [
        output_evaluation.decode_word(
            row, vocab_size, index_to_char, config.WORD_LENGTH)
        for row in raw
    ]
    return decoded, raw


def run_upper_deck(model, decoded_words, mapping, batch_size=None):
    """Predict lexical units for decoded words.

    Returns (winning_indices, winning_activations, raw_output_matrix).
    """
    vocab_size = len(mapping)
    inputs = encode(decoded_words, mapping, vocab_size)
    raw = model.predict(inputs, batch_size=batch_size, verbose=0)

    winners = np.argmax(raw, axis=1)
    activations = raw[np.arange(len(raw)), winners]
    return winners, activations, raw


def transcribe(winning_indices, source_corpus_path):
    """Map winning lexical unit indices back to the words they stand for."""
    lexicon = config.load_doc(source_corpus_path).split()
    return [lexicon[index] for index in winning_indices]


def build_inputs(mode, sub_mode, letter, corpus, lower_deck_mapping):
    """Produce the raw input strings for the requested test mode."""
    if mode in (1, 4, 5, 6, 7):
        config.require(corpus.positional_corpus)
        positional_words = config.load_doc(corpus.positional_corpus).split()

    if mode == 1:
        return positional_words
    if mode == 2:
        return analytics.non_word_discrimination(
            1000, config.WORD_LENGTH, lower_deck_mapping)
    if mode == 3:
        return analytics.single_letter_repeat(
            1000, config.WORD_LENGTH, lower_deck_mapping)
    if mode == 4:
        return analytics.double_letter_substitution(
            slice_centred_words(positional_words), lower_deck_mapping)
    if mode == 5:
        return analytics.letter_transposition(
            slice_centred_words(positional_words))
    if mode == 6:
        return analytics.relative_position_priming(
            slice_centred_words(positional_words), str(sub_mode))
    if mode == 7:
        return analytics.transposed_letter_priming(
            slice_centred_words(positional_words), str(sub_mode),
            lower_deck_mapping)
    if mode == 8:
        return analytics.letter_proximity_effect(
            letter, lower_deck_mapping, config.WORD_LENGTH, str(sub_mode))
    raise SystemExit('Unknown mode {}'.format(mode))


def report_corpus_run(raw_inputs, decoded, transcribed, activations):
    """Mode 1: list every prediction, count errors, write analysis.txt."""
    predictions = {}
    misses = {}
    for i, raw_input in enumerate(raw_inputs):
        record = (
            'LD output: ' + decoded[i],
            'UDT output: ' + transcribed[i],
        )
        predictions['Raw input: ' + raw_input] = record
        # The padded input contains the target word as a substring when the
        # model is correct, e.g. 'aaltola' in '###aaltola###'.
        if transcribed[i] not in raw_input:
            misses['Raw input: ' + raw_input] = record

    print('Error count: {}  Total prediction count: {}'.format(
        len(misses), len(predictions)))
    confident = int(np.sum(np.asarray(activations) >= NONWORD_THRESHOLD))
    print('Predictions with activation >= {}: {}'.format(
        NONWORD_THRESHOLD, confident))

    with open(config.PROJECT_ROOT / 'analysis.txt', 'w') as handle:
        handle.write('{\n')
        for key, value in predictions.items():
            handle.write("'{}':'{}'\n".format(key, value))
        handle.write('}')
    print('wrote analysis.txt')


def report_false_positives(activations, threshold):
    """Modes 2-7: count inputs the model wrongly accepted as known words."""
    activations = np.asarray(activations)
    false_positives = int(np.sum(activations >= threshold))
    print('False positives (activation >= {}): {} / {}'.format(
        threshold, false_positives, len(activations)))
    print('Mean winning activation: {:.6f}'.format(float(activations.mean())))
    return false_positives


def report_letter_proximity(raw_inputs, decoded, raw_lower_output,
                            vocab_size):
    """Mode 8: dump the full per-letter activation profile of each input.

    The layout is preserved exactly as previously published: the complete list
    of raw inputs and of decoded outputs is repeated ahead of *every* input's
    activation blocks, and each block is written one activation per line. It is
    redundant, but downstream analysis reads this shape, so it is reproduced
    rather than tidied.
    """
    out_path = config.PROJECT_ROOT / 'outfile.txt'
    with open(out_path, 'wb') as handle:
        for row in raw_lower_output:
            np.savetxt(handle, raw_inputs, delimiter=' ', fmt='%s')
            np.savetxt(handle, decoded, delimiter=' ', fmt='%s')
            for block in output_evaluation.letter_block_activations(
                    row, vocab_size, config.WORD_LENGTH):
                np.savetxt(handle, block, fmt='%1.10f')
    print('wrote {}'.format(out_path.name))


def two_deck(corpus, mode, sub_mode=None, letter=None, batch_size=None):
    """Run one full pass of the model in the requested mode."""
    config.require(
        corpus.lower_deck_model, corpus.lower_deck_mapping,
        corpus.upper_deck_model, corpus.upper_deck_mapping,
        corpus.source_corpus,
    )

    lower_deck_model = load_model(config.PROJECT_ROOT / corpus.lower_deck_model)
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        lower_deck_mapping = load(handle)

    raw_inputs = build_inputs(
        mode, sub_mode, letter, corpus, lower_deck_mapping)
    print('corpus {} | mode {} | {} input(s)'.format(
        corpus.key, mode, len(raw_inputs)))

    decoded, raw_lower_output = run_lower_deck(
        lower_deck_model, raw_inputs, lower_deck_mapping, batch_size)

    upper_deck_model = load_model(config.PROJECT_ROOT / corpus.upper_deck_model)
    with open(config.PROJECT_ROOT / corpus.upper_deck_mapping, 'rb') as handle:
        upper_deck_mapping = load(handle)

    winners, activations, _ = run_upper_deck(
        upper_deck_model, decoded, upper_deck_mapping, batch_size)
    transcribed = transcribe(winners, corpus.source_corpus)

    if mode == 1:
        report_corpus_run(raw_inputs, decoded, transcribed, activations)
    elif 2 <= mode <= 5:
        analytics.progress_printout(
            raw_inputs, decoded, transcribed, activations, len(activations))
        report_false_positives(activations, NONWORD_THRESHOLD)
    elif 6 <= mode <= 7:
        report_false_positives(activations, PRIMING_THRESHOLD)
    elif mode == 8:
        report_letter_proximity(
            raw_inputs, decoded, raw_lower_output, len(lower_deck_mapping))


MODE_PROMPT = (
    "Choose one of the following modes to proceed:\n"
    "1 - Run using the defined corpus without alterations.\n"
    "2 - Run using Dandurand et. al. (2013) RS random string.\n"
    "3 - Run using Dandurand et. al. (2013) SRL (single repeated letter) evaluation.\n"
    "4 - Run using Dandurand et. al. (2013) DLS (double letter substitution) evaluation.\n"
    "5 - Run using Dandurand et. al. (2013) LT (letter transposition) evaluation.\n"
    "6 - Run using Dandurand et. al. (2013) RPP (relative position priming) evaluation.\n"
    "7 - Run using Dandurand et. al. (2013) TLP (transposed letter priming) evaluation.\n"
    "8 - Run using analysis suite. \n"
)

SUB_MODE_PROMPTS = {
    6: ("Choose one of the following sub modes to proceed:\n"
        "1 - Original word '1234567' changed to '1234'.\n"
        "2 - Original word '1234567' changed to '1357'.\n"),
    7: ("Choose one of the following sub modes to proceed:\n"
        "1 - Original word '1234567' changed to '1235467'.\n"
        "2 - Original word '1234567' changed to '123DD67' where 'D' is a char "
        "that was not present in the original word.\n"),
    8: ("Choose one of the following sub modes to proceed:\n"
        "1 - Analysis of input letter effect based on letter proximity effect "
        "using randomized strings. \n"
        "2 - Analysis of input letter effect based on letter proximity effect "
        "using #-symbol as filler for all other position than the chosen "
        "letter. \n"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MODE_PROMPT)
    config.add_corpus_argument(parser)
    parser.add_argument('-m', '--mode', type=int, choices=range(1, 9),
                        help='Test mode 1-8. Prompted for if omitted.')
    parser.add_argument('-s', '--sub-mode', type=int, choices=(1, 2),
                        help='Sub-mode for modes 6, 7 and 8.')
    parser.add_argument('-l', '--letter',
                        help='Focal letter for mode 8.')
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help='Prediction batch size. The published results '
                             'were produced one input at a time; pass 1 to '
                             'reproduce them exactly. Batching is ~250x '
                             'faster and changes the decoded letter in a '
                             'handful of near-tied cases (2 of 14000 for '
                             'FIRND mode 1), without altering any lexical '
                             'output.')
    args = parser.parse_args()
    corpus = config.get(args.corpus)

    mode = args.mode
    if mode is None:
        mode = int(input(MODE_PROMPT).strip())
        if not 1 <= mode <= 8:
            raise SystemExit('Please choose a mode between 1 and 8.')

    sub_mode = args.sub_mode
    if mode in SUB_MODE_PROMPTS and sub_mode is None:
        sub_mode = int(input(SUB_MODE_PROMPTS[mode]).strip())
        if sub_mode not in (1, 2):
            raise SystemExit('Please choose sub-mode 1 or 2.')

    letter = args.letter
    if mode == 8 and letter is None:
        letter = input(
            'Input letter to be used as the focal point for letter '
            'proximity analysis. \n').strip()

    two_deck(corpus, mode, sub_mode, letter, args.batch_size)


if __name__ == '__main__':
    main()
