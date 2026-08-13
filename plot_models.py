"""Regenerate the architecture diagrams for a corpus's two decks.

Optional utility, kept separate from the analysis pipeline because it is the
only thing in the project that needs pydot, visualkeras and a system Graphviz
install. Nothing else imports it, so you can ignore it entirely unless you want
to rebuild the figures.

    pip install pydot visualkeras     # plus Graphviz from your package manager

Usage::

    python plot_models.py --corpus FIN
"""

import config


def main():
    corpus = config.parse_corpus_arg(__doc__.splitlines()[0])
    config.require(corpus.lower_deck_model, corpus.upper_deck_model)

    try:
        import visualkeras
        from keras.models import load_model
        from keras.utils.vis_utils import plot_model
    except ImportError as exc:
        raise SystemExit(
            'Diagram support is optional and is not installed: {}\n'
            'Install it with:  pip install pydot visualkeras\n'
            '(you also need Graphviz, e.g. "brew install graphviz").'.format(exc))

    for model_path, stem in (
        (corpus.lower_deck_model, 'lower_deck'),
        (corpus.upper_deck_model, 'upper_deck'),
    ):
        model = load_model(config.PROJECT_ROOT / model_path)

        layers_png = '{}_{}.png'.format(corpus.key.lower(), stem)
        plot_model(model, to_file=str(config.PROJECT_ROOT / layers_png),
                   show_shapes=True, show_layer_names=True)
        print('wrote {}'.format(layers_png))

        stacked_png = '{}_{}_visualisation.png'.format(corpus.key.lower(), stem)
        visualkeras.layered_view(
            model, to_file=str(config.PROJECT_ROOT / stacked_png),
            legend=True, scale_xy=1, scale_z=1, max_z=1000)
        print('wrote {}'.format(stacked_png))


if __name__ == '__main__':
    main()
