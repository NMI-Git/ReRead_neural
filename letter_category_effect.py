"""Test whether hidden-layer representations cluster by letter identity.

Runs k-means over the 299 x 118 hidden-layer activations produced by
testbed_lower_deck.py, with k set to the number of letters in the alphabet. If
the lower deck has learned a location-invariant code then the 13 probes for a
given letter should fall into the same cluster regardless of position, and the
cluster labels should therefore recover letter identity.

Prints the cluster assignment for each probe together with the letter it came
from, plus a purity score: the proportion of probes whose cluster is the one
most of its letter's probes ended up in.

Usage::

    python letter_category_effect.py --corpus FIN
    python letter_category_effect.py --corpus FIN --n-init 10   # fast check

The default ``--n-init`` reproduces the value the published run used. It is
extremely large and takes on the order of an hour; k-means++ with a fixed
random state converges to the same solution far sooner, so pass ``--n-init 10``
when you only want to check the result.
"""

import argparse
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

import config

RANDOM_STATE = 23

#: Values used for the published analysis. Retained as defaults so the result
#: reproduces exactly; see --n-init above for a practical alternative.
PUBLISHED_N_INIT = 1000000
PUBLISHED_MAX_ITER = 10000000000


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_corpus_argument(parser)
    parser.add_argument('--n-init', type=int, default=PUBLISHED_N_INIT,
                        help='k-means restarts.')
    parser.add_argument('--max-iter', type=int, default=PUBLISHED_MAX_ITER,
                        help='Maximum iterations per restart.')
    args = parser.parse_args()
    corpus = config.get(args.corpus)

    config.require(corpus.activation_vectors, corpus.testbed_inputs)

    activations = np.loadtxt(
        str(config.PROJECT_ROOT / corpus.activation_vectors), delimiter=',')
    probes = config.load_doc(corpus.testbed_inputs).split()
    letters = [probe.replace(config.FILLER_TOKEN, '') for probe in probes]

    n_clusters = len(set(letters))
    kmeans = KMeans(
        init='k-means++', n_clusters=n_clusters, random_state=RANDOM_STATE,
        n_init=args.n_init, max_iter=args.max_iter)
    kmeans.fit(activations)

    print('corpus    : {}'.format(corpus.key))
    print('probes    : {}  clusters: {}'.format(len(letters), n_clusters))
    print('inertia   : {:.6f}  iterations: {}'.format(
        kmeans.inertia_, kmeans.n_iter_))

    # How cleanly does each letter map onto a single cluster?
    correct = 0
    by_letter = {}
    for letter, label in zip(letters, kmeans.labels_):
        by_letter.setdefault(letter, []).append(int(label))
    for letter in sorted(by_letter):
        labels = by_letter[letter]
        dominant, count = Counter(labels).most_common(1)[0]
        correct += count
        print('  {}  cluster {:3d}  {}/{} probes'.format(
            letter, dominant, count, len(labels)))

    print('purity    : {:.4f}'.format(correct / len(letters)))


if __name__ == '__main__':
    main()
