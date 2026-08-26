"""Summary statistics for the proximity and letter-cluster distance matrices.

These reproduce the two analyses reported in the results section, and the
degrees of freedom identify exactly which cells each one uses.

Proximity effect (13 x 13, position by position)
    The published F(1, 310) implies N = 312 = 2 x 156, and 156 = 13^2 - 13, so
    the comparison is over the off-diagonal cells of the two matrices. The
    diagonal is a position compared with itself and is necessarily zero.

Letter cluster effect (letters x letters)
    The published F(1, 1896) implies N = 1898 = 23^2 + 37^2, so the comparison
    is over the full letter-by-letter matrix, diagonal included, for the
    Finnish (23 letters) and French (37) alphabets. The stored clustering matrix
    is probe by probe, ordered letter-major, so each letter pair is the mean of
    a 13 x 13 block.

Usage::

    python representation_stats.py
"""

import numpy as np
from scipy import stats

import config


def load(path):
    return np.loadtxt(str(config.PROJECT_ROOT / path), delimiter=',')


def proximity_values(corpus):
    """Off-diagonal cells of the 13 x 13 position matrix."""
    matrix = load(corpus.proximity_effect)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def letter_cluster_values(corpus):
    """Probe-by-probe distances collapsed to a letter-by-letter matrix."""
    matrix = load(corpus.clustering_effect)
    positions = config.WINDOW_LENGTH
    letters = matrix.shape[0] // positions
    collapsed = matrix.reshape(letters, positions, letters, positions)
    return collapsed.mean(axis=(1, 3)).ravel()


def report(title, values, section):
    print('\n{}  ({})'.format(title, section))
    print('-' * 62)
    for key in ('FR', 'FIN', 'FIRND'):
        v = values[key]
        print('  {:24} M = {:.2f}   SD = {:.2f}   (n = {})'.format(
            config.get(key).description.split('(')[0].strip()[:24],
            v.mean(), v.std(ddof=1), len(v)))
    f, p = stats.f_oneway(values['FR'], values['FIN'])
    df2 = len(values['FR']) + len(values['FIN']) - 2
    print('  French vs Finnish: F(1, {}) = {:.2f}, p {}'.format(
        df2, f, '< .001' if p < 0.001 else '= {:.3f}'.format(p)))


def main():
    corpora = {k: config.get(k) for k in ('FR', 'FIN', 'FIRND')}
    report('Proximity effect', {k: proximity_values(c) for k, c in corpora.items()},
           'position x position, diagonal excluded')
    report('Letter cluster effect',
           {k: letter_cluster_values(c) for k, c in corpora.items()},
           'letter x letter, diagonal included')
    print()


if __name__ == '__main__':
    main()
