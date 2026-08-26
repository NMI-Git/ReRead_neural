"""Figures 2 and 3: proximity and letter cluster distance matrices.

Figure 2 plots the 13 x 13 position-by-position matrix for each model. A dark
diagonal means the hidden layer separates a letter's own location sharply; the
graded band around it is the proximity effect.

Figure 3 plots the letter-by-letter matrix, collapsed from the probe-by-probe
distances by averaging over positions.

Both read the matched-epoch matrices in results/matched/, because comparing
hidden layers across models is only meaningful when all three were trained for
the same number of epochs -- see matched_representations.py.

Requires matplotlib, which the model itself does not, so it is not in
requirements.txt. Install it separately to regenerate the figures::

    python -m pip install matplotlib
    python plot_effects.py
"""

import argparse
import os
from pickle import load

import numpy as np

import config

ORDER = ('FR', 'FIN', 'FIRND')
TITLES = {'FR': 'French', 'FIN': 'Finnish', 'FIRND': 'Random Finnish'}


def matched(corpus, name):
    return np.loadtxt(
        str(config.PROJECT_ROOT / 'results/matched' / corpus.slug / name),
        delimiter=',')


def alphabet_of(corpus):
    with open(config.PROJECT_ROOT / corpus.lower_deck_mapping, 'rb') as handle:
        return sorted(load(handle))


def letter_matrix(corpus):
    clustering = matched(corpus, 'clustering_effect.csv')
    positions = config.WINDOW_LENGTH
    letters = clustering.shape[0] // positions
    return clustering.reshape(
        letters, positions, letters, positions).mean(axis=(1, 3))


def draw(figure_data, labels, title, filename, cbar_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    low = min(m.min() for m, _ in figure_data)
    high = max(m.max() for m, _ in figure_data)
    widths = [m.shape[0] for m, _ in figure_data]

    fig, axes = plt.subplots(
        1, len(figure_data), figsize=(4.2 * len(figure_data), 4.6),
        gridspec_kw={'width_ratios': widths})
    if len(figure_data) == 1:
        axes = [axes]

    for ax, (matrix, name), ticks in zip(axes, figure_data, labels):
        image = ax.imshow(matrix, cmap='gray_r', vmin=low, vmax=high,
                          interpolation='nearest')
        ax.set_title(name, fontsize=11)
        if ticks is not None and len(ticks) <= 40:
            ax.set_xticks(range(len(ticks)))
            ax.set_yticks(range(len(ticks)))
            ax.set_xticklabels(ticks, fontsize=5)
            ax.set_yticklabels(ticks, fontsize=5)
        ax.tick_params(length=0)

    fig.suptitle(title, fontsize=12)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02,
                 label=cbar_label)
    path = config.ensure_parent(filename)
    fig.savefig(str(path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote {}'.format(path.relative_to(config.PROJECT_ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results/figures')
    args = parser.parse_args()

    corpora = {k: config.get(k) for k in ORDER}

    proximity = [(matched(corpora[k], 'proximity_effect.csv'), TITLES[k])
                 for k in ORDER]
    positions = [list(range(1, config.WINDOW_LENGTH + 1)) for _ in ORDER]
    draw(proximity, positions,
         'Figure 2. Proximity effects (position x position)',
         os.path.join(args.out, 'figure2_proximity.png'),
         'Mean Euclidean distance')

    clusters = [(letter_matrix(corpora[k]), TITLES[k]) for k in ORDER]
    letters = [alphabet_of(corpora[k]) for k in ORDER]
    draw(clusters, letters,
         'Figure 3. Letter cluster effects (letter x letter)',
         os.path.join(args.out, 'figure3_letter_clusters.png'),
         'Mean distance across locations')


if __name__ == '__main__':
    main()
