"""Visualize MedMNIST dataset embeddings via t-SNE, colored by class.

Usage:
    python -m miccai.run.visualize_datasets --datasets organsmnist pneumoniamnist --embeddings imagenet
    python -m miccai.run.visualize_datasets --datasets organsmnist --embeddings imagenet uni --size 224
    python -m miccai.run.visualize_datasets --datasets tissuemnist --embeddings imagenet --max-samples 10000
"""

import argparse
import numpy as np
from pathlib import Path

from .data import load_dataset, get_labels, DATASETS
from .embeddings import load_or_compute_embeddings

try:
    from cuml.manifold import TSNE
    USING_GPU_TSNE = True
except ImportError:
    from sklearn.manifold import TSNE
    USING_GPU_TSNE = False


def run_tsne(embeddings, n_samples, perplexity=None, seed=42):
    """Run t-SNE with perplexity scaled to dataset size."""
    if perplexity is None:
        # Scale perplexity with dataset size: ~30 for 5k, ~50 for 50k+
        perplexity = min(50, max(15, int(n_samples ** 0.5 / 3)))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    if USING_GPU_TSNE:
        coords = tsne.fit_transform(embeddings.astype(np.float32))
        coords = np.array(coords)
    else:
        coords = tsne.fit_transform(embeddings)
    return coords


def subsample_proportional(labels, max_samples, seed=42):
    """Proportional subsampling that preserves class distribution."""
    rng = np.random.RandomState(seed)
    ratio = max_samples / len(labels)
    indices = []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        n_take = max(1, int(len(class_idx) * ratio))
        indices.extend(rng.choice(class_idx, n_take, replace=False))
    return np.array(indices)


def plot_single(ax, coords, labels, dataset_name, embedding_name, num_classes):
    """Plot a single t-SNE scatter with size/alpha adapted to sample count."""
    import matplotlib.pyplot as plt

    n = len(labels)
    # Scale point size and alpha with sample count
    s = max(1, min(15, 5000 / n * 5))
    alpha = max(0.1, min(0.6, 3000 / n))

    unique_classes = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', max(len(unique_classes), 2))

    for c in unique_classes:
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(c)], alpha=alpha, s=s, rasterized=True)

    ax.set_title(f'{dataset_name} | {embedding_name} (n={n}, {num_classes}c)',
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description='Visualize MedMNIST dataset embeddings via t-SNE')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help=f'Datasets to visualize. Available: {", ".join(DATASETS)}')
    parser.add_argument('--embeddings', nargs='+', default=['imagenet'],
                        help='Embedding sources (default: imagenet)')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perplexity', type=int, default=None,
                        help='t-SNE perplexity (default: auto-scaled by dataset size)')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Max samples for subsampled figure (default: 10000)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--output-dir', type=str, default='miccai/figures')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"t-SNE backend: {'GPU (cuML)' if USING_GPU_TSNE else 'CPU (sklearn)'}")

    n_embeddings = len(args.embeddings)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset, info = load_dataset(dataset_name, args.split, size=args.size, verbose=True)
        labels = get_labels(dataset)
        num_classes = len(info['label'])
        in_channels = info['n_channels']

        # Compute embeddings once per source (reused for both figures)
        all_embeddings = {}
        for emb_source in args.embeddings:
            print(f"\n[{dataset_name} | {emb_source}] Computing embeddings...")
            emb_data = load_or_compute_embeddings(
                dataset_name=dataset_name, split=args.split, source=emb_source,
                dataset=dataset, num_classes=num_classes, in_channels=in_channels,
                size=args.size, seed=args.seed,
                cache_dir=Path('miccai/cache/embeddings'), verbose=True)
            all_embeddings[emb_source] = emb_data['embeddings']

        emb_str = '_'.join(args.embeddings)
        size_str = f'_s{args.size}'
        needs_subsample = len(labels) > args.max_samples

        # --- Figure 1: Full dataset ---
        fig, axes = plt.subplots(1, n_embeddings, figsize=(5 * n_embeddings, 4), squeeze=False)
        for col, emb_source in enumerate(args.embeddings):
            print(f"[{dataset_name} | {emb_source}] t-SNE on full ({len(labels)} samples)...")
            coords = run_tsne(all_embeddings[emb_source], len(labels),
                              perplexity=args.perplexity, seed=args.seed)
            plot_single(axes[0, col], coords, labels, dataset_name, emb_source, num_classes)
        plt.tight_layout()
        out_path = out_dir / f'tsne_{dataset_name}_{emb_str}{size_str}_{args.split}_full.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

        # --- Figure 2: Subsampled (only if dataset is larger than max_samples) ---
        if needs_subsample:
            sub_idx = subsample_proportional(labels, args.max_samples, seed=args.seed)
            sub_labels = labels[sub_idx]

            fig, axes = plt.subplots(1, n_embeddings, figsize=(5 * n_embeddings, 4), squeeze=False)
            for col, emb_source in enumerate(args.embeddings):
                sub_emb = all_embeddings[emb_source][sub_idx]
                print(f"[{dataset_name} | {emb_source}] t-SNE on subsample ({len(sub_labels)} samples)...")
                coords = run_tsne(sub_emb, len(sub_labels),
                                  perplexity=args.perplexity, seed=args.seed)
                plot_single(axes[0, col], coords, sub_labels, dataset_name, emb_source, num_classes)
            plt.tight_layout()
            out_path = out_dir / f'tsne_{dataset_name}_{emb_str}{size_str}_{args.split}_{args.max_samples}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")

        print()


if __name__ == '__main__':
    main()
