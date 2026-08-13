"""Input generators for the Dandurand et al. (2013) test batteries.

Each function here builds a set of distorted or artificial input strings that
probe one property of the model's orthographic code. They are driven from
two_deck.py, which feeds them through both decks and reports how strongly the
lexical layer responded.

All generated strings are padded to WINDOW_LENGTH so they can be fed to the
lower deck directly. Distortions are applied to the *centred* placement of a
word, so index 3 of a padded string is the first letter of the word itself::

    ###aaltola###
       ^ index 3

Several functions take ``alphabet``, which is the lower deck's ``{char: index}``
mapping. Only its keys are used, and the filler token is always excluded so
distortions never introduce padding into the middle of a word.
"""

import random

import config

#: First index of the actual word inside a centred, padded input string.
WORD_START = 3


def letters_of(alphabet):
    """Return the usable letters of a character mapping, excluding the filler."""
    return ''.join(
        char for char in alphabet if char != config.FILLER_TOKEN)


def apply_filler_token(word_list, filler_count):
    """Pad each word with filler tokens, split as evenly as possible.

    An odd ``filler_count`` puts the extra token on the right, which keeps a
    7-letter word in a 13-slot window centred at index 3.
    """
    left = filler_count // 2
    right = filler_count - left
    return [
        config.FILLER_TOKEN * left + word + config.FILLER_TOKEN * right
        for word in word_list
    ]


def non_word_discrimination(word_count, letter_count, alphabet):
    """RS: random letter strings.

    These are orthographically legal but meaningless. The lexical layer should
    not respond strongly to any of them.
    """
    letters = letters_of(alphabet)
    nonwords = [
        ''.join(random.choice(letters) for _ in range(letter_count))
        for _ in range(word_count)
    ]
    return apply_filler_token(nonwords, config.PADDING_SLOTS)


def single_letter_repeat(word_count, letter_count, alphabet):
    """SRL: one letter repeated to fill the word, e.g. 'aaaaaaa'.

    A degenerate input that carries letter identity but no positional
    structure.
    """
    letters = letters_of(alphabet)
    strings = [
        random.choice(letters) * letter_count
        for _ in range(word_count)
    ]
    return apply_filler_token(strings, config.PADDING_SLOTS)


def double_letter_substitution(words, alphabet):
    """DLS: replace two letters of each word with different letters.

    Substitutes at two distinct random positions within the word, drawing
    replacements that differ from both original letters.
    """
    letters = letters_of(alphabet)
    result = []
    for word in words:
        indexes = random.sample(
            range(WORD_START, WORD_START + config.WORD_LENGTH - 1), 2)
        banned = word[indexes[0]] + word[indexes[1]]
        new_word = list(word)
        for index in indexes:
            new_word[index] = random.choice(
                [c for c in letters if c not in banned])
        result.append(''.join(new_word))
    return result


def letter_transposition(words):
    """LT: swap the two middle letters of each word ('1234567' -> '1235467')."""
    result = []
    for word in words:
        new_word = list(word)
        new_word[7], new_word[8] = new_word[8], new_word[7]
        result.append(''.join(new_word))
    return result


def relative_position_priming(words, sub_mode):
    """RPP: keep a subset of letters in their relative order.

    sub_mode '1' -> '1234' (first four letters)
    sub_mode '2' -> '1357' (every other letter)

    Human readers show priming from both, because relative order is preserved
    even though absolute positions change.
    """
    result = []
    for word in words:
        if sub_mode == '1':
            result.append(word[WORD_START:WORD_START + 4])
        else:
            result.append(word[WORD_START:WORD_START + 7:2])
    return apply_filler_token(result, config.WINDOW_LENGTH - 4)


def transposed_letter_priming(words, sub_mode, alphabet):
    """TLP: transposition versus substitution at the same two positions.

    sub_mode '1' -> '1235467', the two letters swapped.
    sub_mode '2' -> '123DD67', the same two positions replaced by letters that
                    do not occur anywhere in the original word.

    The contrast matters because human readers treat a transposition as far
    more word-like than a substitution.
    """
    letters = letters_of(alphabet)
    indexes = [6, 7]
    result = []
    for word in words:
        new_word = list(word)
        if sub_mode == '1':
            new_word[6], new_word[7] = new_word[7], new_word[6]
        else:
            banned = list(set(word))
            for index in indexes:
                new_word[index] = random.choice(
                    [c for c in letters if c not in banned])
        result.append(''.join(new_word))
    return result


def letter_proximity_effect(chosen_letter, alphabet, position_count, mode):
    """Place one letter at each position to probe positional coding.

    mode '1' -- the rest of the string is random letters (the chosen letter
                appears once, at a different index in each string).
    mode '2' -- the rest of the string is filler, isolating the chosen letter
                completely.
    """
    letters = letters_of(alphabet)
    result = []

    if mode == '1':
        for index in range(position_count):
            filler = ''.join(
                random.choice([c for c in letters if c != chosen_letter])
                for _ in range(position_count))
            result.append(filler[:index] + chosen_letter + filler[index + 1:])
        return apply_filler_token(result, config.PADDING_SLOTS)

    for index in range(config.WINDOW_LENGTH):
        blank = config.FILLER_TOKEN * config.WINDOW_LENGTH
        result.append(blank[:index] + chosen_letter + blank[index + 1:])
    return result


def single_letter_positional_input(alphabet, window_length):
    """Build the probe set used by the representation analysis.

    Returns one string per (letter, position) pair, with the letter isolated in
    an otherwise empty window. For a 23-letter alphabet and a 13-slot window
    that is 299 strings.
    """
    return [
        config.FILLER_TOKEN * index + char
        + config.FILLER_TOKEN * (window_length - 1 - index)
        for char in alphabet if char != config.FILLER_TOKEN
        for index in range(window_length)
    ]


def progress_printout(ld_input, ld_output, ud_output, ud_activation_values,
                      iteration_count):
    """Print each input alongside what both decks made of it."""
    for i in range(iteration_count):
        print("Raw input: {} | LD output: {} | UDT output: {} | "
              "UD activation: {}".format(
                  ld_input[i], ld_output[i], ud_output[i],
                  ud_activation_values[i]))


def alphabet_counter(corpus_key='FR'):
    """Report how often each character occurs in a corpus.

    Used when checking that a new vocabulary covers its alphabet evenly. The
    original version hard-coded the French corpus path and listed all forty
    characters by hand; this derives the alphabet from the text instead, so it
    works for any corpus.
    """
    corpus = config.get(corpus_key)
    text = config.load_doc(corpus.positional_corpus)
    for char in sorted(set(text)):
        if char in (' ', config.FILLER_TOKEN):
            continue
        print('Count of letter {}: {}'.format(char, text.count(char)))
