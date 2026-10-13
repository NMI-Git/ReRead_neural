"""Figures 2 and 3: proximity and letter cluster distance matrices.

Figure 2 plots the 13 x 13 position-by-position matrix for each model. A dark
diagonal means the hidden layer separates a letter's own location sharply; the
graded band around it is the proximity effect.

Figure 3 plots the letter-by-letter matrix, collapsed from the probe-by-probe
distances by averaging over positions.

Both read the matched-epoch matrices in results/matched/, because comparing
hidden layers across models is only meaningful when all three were trained for
the same number of epochs -- see matched_representations.py.

The figures are prepared to the journal's artwork requirements. Each is drawn
at its printed size, the 34pc text width of Taylor & Francis's Interact layout,
so the font sizes below are the sizes on the page. Each is written three ways::

    figure2_proximity.tif   600 dpi grayscale TIFF -- submit this
    figure2_proximity.eps   vector text with embedded fonts -- also accepted
    figure2_proximity.png   the same raster as the TIFF, for viewing

600 dpi is the journal's minimum for grayscale art. Captions are not drawn into
the figures; they go in the manuscript, where the typesetter sets them.

Requires matplotlib, which the model itself does not, so it is not in
requirements.txt. Install it separately to regenerate the figures::

    python -m pip install matplotlib
    python plot_effects.py
"""

import argparse
import io
import os
from pickle import load

import numpy as np

import config

ORDER = ('FR', 'FIN', 'FIRND')
TITLES = {'FR': 'French', 'FIN': 'Finnish', 'FIRND': 'Random Finnish'}

# Printed width in inches (34pc). Change it if the journal sets figures at a
# different width; the font sizes are chosen to stay legible at this one.
TEXT_WIDTH = 34 / 6
DPI = 600


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


def draw(figure_data, labels, rows, height, axis_label, filename, cbar_label):
    """Plot the matrices on one shared grayscale and save the figure.

    rows lists the panels on each row of the figure by index; a row with fewer
    panels than the widest spreads them across its columns. Columns and rows
    are sized by their matrices, so one matrix cell is the same size in every
    panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'ps.fonttype': 42,  # embed as TrueType; publishers reject Type 3
    })

    low = min(m.min() for m, _ in figure_data)
    high = max(m.max() for m, _ in figure_data)

    fig = plt.figure(figsize=(TEXT_WIDTH, height), layout='constrained')
    columns = max(len(row) for row in rows)
    widest = next(row for row in rows if len(row) == columns)
    grid = fig.add_gridspec(
        len(rows), columns,
        width_ratios=[figure_data[i][0].shape[0] for i in widest],
        height_ratios=[max(figure_data[i][0].shape[0] for i in row)
                       for row in rows])

    axes = []
    for r, row in enumerate(rows):
        span = columns // len(row)
        for column, i in enumerate(row):
            matrix, name = figure_data[i]
            ticks = labels[i]
            ax = fig.add_subplot(
                grid[r, column * span:(column + 1) * span])
            # 'none' embeds the matrix unresampled in the EPS, so every cell
            # stays a sharp square at any zoom; rasters get nearest-neighbour.
            image = ax.imshow(matrix, cmap='gray_r', vmin=low, vmax=high,
                              interpolation='none')
            ax.set_title(name, fontsize=8)
            if ticks is not None and len(ticks) <= 40:
                ax.set_xticks(range(len(ticks)))
                ax.set_yticks(range(len(ticks)))
                ax.set_xticklabels(ticks, fontsize=6)
                ax.set_yticklabels(ticks, fontsize=6)
            ax.tick_params(length=0, pad=2)
            ax.set_xlabel(axis_label)
            if column == 0:
                ax.set_ylabel(axis_label)
            axes.append(ax)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.9 / len(rows),
                            aspect=20 / len(rows))
    colorbar.set_label(cbar_label)
    colorbar.ax.tick_params(labelsize=6)
    save(fig, filename)
    plt.close(fig)


def save(fig, filename):
    """Write filename as a 600 dpi grayscale TIFF and PNG, and as an EPS."""
    from PIL import Image

    stem = config.ensure_parent(filename)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=DPI)
    raster = Image.open(buffer).convert('L')

    paths = [stem.with_suffix(s) for s in ('.tif', '.eps', '.png')]
    raster.save(str(paths[0]), dpi=(DPI, DPI), compression='tiff_lzw')
    fig.savefig(str(paths[1]), format='eps')
    raster.save(str(paths[2]), dpi=(DPI, DPI))
    for path in paths:
        print('  wrote {}'.format(os.path.relpath(path, config.PROJECT_ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results/figures')
    args = parser.parse_args()

    corpora = {k: config.get(k) for k in ORDER}

    proximity = [(matched(corpora[k], 'proximity_effect.csv'), TITLES[k])
                 for k in ORDER]
    positions = [list(range(1, config.WINDOW_LENGTH + 1)) for _ in ORDER]
    # Heights (inches) are fitted to TEXT_WIDTH: the shortest that keeps every
    # panel full size, which for Figure 3 is also where its cells match up.
    draw(proximity, positions, [[0, 1, 2]], 1.9, 'Position',
         os.path.join(args.out, 'figure2_proximity'),
         'Mean Euclidean distance')

    # French has 36 letters to the Finnish 26. Three across at this width
    # would shrink its labels below legibility, so it gets a row of its own.
    clusters = [(letter_matrix(corpora[k]), TITLES[k]) for k in ORDER]
    letters = [alphabet_of(corpora[k]) for k in ORDER]
    draw(clusters, letters, [[0], [1, 2]], 6.1, 'Letter',
         os.path.join(args.out, 'figure3_letter_clusters'),
         'Mean distance across locations')


if __name__ == '__main__':
    main()
