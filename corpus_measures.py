"""Orthographic descriptives for each corpus (Table 1).

These describe the word lists, not the model, so they should be unchanged by the
model updates. Computing them is a check that the corpora in the repository are
the ones the published Table 1 was built from.

Measures, as standard in the visual word recognition literature:

  Neighbour words     Coltheart's N -- words differing by exactly one letter
                      substitution at the same length.
  OLD20 mean / sd     Mean and sd of the Levenshtein distance to the 20
                      orthographically closest words (Yarkoni et al., 2008).
  Spread              Number of letter positions at which some substitution
                      yields a neighbour.
  Uniqueness point    Position at which the word's prefix stops being shared
                      with any other word in the corpus.

Usage::

    python corpus_measures.py
"""
import numpy as np

import config

LENGTH = config.WORD_LENGTH
NEAREST = 20


def encode(words):
    alphabet = sorted({c for w in words for c in w})
    index = {c: i for i, c in enumerate(alphabet)}
    return np.array([[index[c] for c in w] for w in words], dtype=np.int16)


def neighbours_and_spread(codes):
    """Coltheart's N and spread, from position-wise mismatch counts."""
    n = len(codes)
    neighbour_count = np.zeros(n, dtype=np.int32)
    spread = np.zeros(n, dtype=np.int32)
    chunk = 250
    for start in range(0, n, chunk):
        block = codes[start:start + chunk]                      # (b, L)
        diff = block[:, None, :] != codes[None, :, :]           # (b, n, L)
        mismatches = diff.sum(axis=2)                           # (b, n)
        is_neighbour = mismatches == 1                          # (b, n)
        np.fill_diagonal(is_neighbour[:, start:start + len(block)], False)
        neighbour_count[start:start + len(block)] = is_neighbour.sum(axis=1)
        # which position differs, for neighbours only
        positions = np.where(is_neighbour[:, :, None], diff, False).any(axis=1)
        spread[start:start + len(block)] = positions.sum(axis=1)
    return neighbour_count, spread


def levenshtein_matrix(a_codes, b_codes):
    """Levenshtein distance between every row of a_codes and every row of b."""
    na, nb = len(a_codes), len(b_codes)
    la, lb = a_codes.shape[1], b_codes.shape[1]
    prev = np.tile(np.arange(lb + 1, dtype=np.int16), (na, nb, 1))
    prev = prev[:, :, :]                                        # (na, nb, lb+1)
    for i in range(1, la + 1):
        cur = np.empty_like(prev)
        cur[:, :, 0] = i
        ai = a_codes[:, None, i - 1][:, :, None]                # (na, 1, 1)
        for j in range(1, lb + 1):
            cost = (ai[:, :, 0] != b_codes[None, :, j - 1]).astype(np.int16)
            cur[:, :, j] = np.minimum(
                np.minimum(prev[:, :, j] + 1, cur[:, :, j - 1] + 1),
                prev[:, :, j - 1] + cost)
        prev = cur
    return prev[:, :, lb]


def old20(codes):
    n = len(codes)
    means = np.zeros(n)
    sds = np.zeros(n)
    chunk = 100
    for start in range(0, n, chunk):
        block = codes[start:start + chunk]
        d = levenshtein_matrix(block, codes).astype(np.float64)
        for r in range(len(block)):
            d[r, start + r] = np.inf                            # exclude self
        nearest = np.sort(d, axis=1)[:, :NEAREST]
        means[start:start + len(block)] = nearest.mean(axis=1)
        sds[start:start + len(block)] = nearest.std(axis=1, ddof=1)
    return means, sds


def uniqueness_point(words):
    """1 + length of the longest prefix shared with any other word."""
    ordered = sorted(range(len(words)), key=lambda i: words[i])
    up = np.full(len(words), LENGTH + 1, dtype=np.int32)

    def shared(a, b):
        k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]:
            k += 1
        return k

    for rank, idx in enumerate(ordered):
        best = 0
        for other in (rank - 1, rank + 1):
            if 0 <= other < len(ordered):
                best = max(best, shared(words[idx], words[ordered[other]]))
        up[idx] = min(best + 1, LENGTH + 1)
    return up


def describe(key):
    corpus = config.get(key)
    words = config.load_doc(corpus.source_corpus).split()
    codes = encode(words)
    n_count, spread = neighbours_and_spread(codes)
    old_mean, old_sd = old20(codes)
    up = uniqueness_point(words)
    return {
        'n': len(words),
        'Neighbour words #': n_count.astype(float),
        'Levenshtein distance 20, mean': old_mean,
        'Levenshtein distance 20, sd': old_sd,
        'Spread (#letters yielding a neighbour)': spread.astype(float),
        'Uniqueness point': up.astype(float),
    }


def main():
    stats = {k: describe(k) for k in ('FR', 'FIN', 'FIRND')}
    rows = [k for k in stats['FR'] if k != 'n']

    try:
        from scipy import stats as sps
        have_scipy = True
    except ImportError:
        have_scipy = False

    print('Table 1. Descriptive information about the corpora.\n')
    print('{:<40}{:>16}{:>16}{:>16}{:>12}'.format(
        'Orthographic Measure', 'French', 'FIN', 'FIN random', 't'))
    print('{:<40}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>12}'.format(
        '', 'M', 'SD', 'M', 'SD', 'M', 'SD', ''))
    print('-' * 100)
    for row in rows:
        cells = []
        for k in ('FR', 'FIN', 'FIRND'):
            v = stats[k][row]
            cells += [v.mean(), v.std(ddof=1)]
        if have_scipy:
            t, p = sps.ttest_ind(stats['FR'][row], stats['FIN'][row],
                                 equal_var=False)
            tcell = '{:.2f}'.format(t)
        else:
            tcell = 'n/a'
        print('{:<40}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>12}'
              .format(row, *cells, tcell))
    print('-' * 100)
    print('N = {} French, {} FIN, {} FIN random'.format(
        stats['FR']['n'], stats['FIN']['n'], stats['FIRND']['n']))
    print('t compares French against the non-random Finnish corpus (Welch).')


if __name__ == '__main__':
    main()
