"""Compare selected subsets across graph_a variants without training."""

import numpy as np
from pathlib import Path
from miccai.run.data import load_dataset, get_labels
from miccai.run.embeddings import load_or_compute_embeddings
from miccai.run.selection import select


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def overlap_pct(a, b):
    return len(set(a) & set(b)) / len(a) * 100


def visualize(embeddings, labels, selections, args):
    """PCA projection with selected samples highlighted per method."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(embeddings)

    methods = list(selections.keys())
    n_methods = len(methods)
    unique_classes = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))

    # Row 1: each method's selection
    # Row 2: pairwise diffs (a1 only, shared, a_last only) for first vs last
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 9))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    # Row 1: selections per method
    for col, method in enumerate(methods):
        ax = axes[0, col]
        sel_set = set(selections[method])
        is_selected = np.array([i in sel_set for i in range(len(labels))])

        # Background: all samples, light
        for c in unique_classes:
            mask = (labels == c) & ~is_selected
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(c)], alpha=0.08, s=3, rasterized=True)
        # Selected: bold
        for c in unique_classes:
            mask = (labels == c) & is_selected
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(c)], alpha=0.9, s=25, edgecolors='black',
                       linewidths=0.3)

        ax.set_title(f'{method} ({len(selections[method])} sel.)', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: pairwise diff between consecutive methods
    for col in range(n_methods):
        ax = axes[1, col]
        if col == 0:
            # First method: show samples unique to a1 (not in a2)
            if n_methods > 1:
                m1, m2 = methods[0], methods[1]
                s1, s2 = set(selections[m1]), set(selections[m2])
                only_m1 = s1 - s2
                only_m2 = s2 - s1
                shared = s1 & s2

                # Background
                all_sel = s1 | s2
                unsel = np.array([i not in all_sel for i in range(len(labels))])
                ax.scatter(coords[unsel, 0], coords[unsel, 1],
                           c='lightgray', alpha=0.08, s=3, rasterized=True)
                # Shared
                shared_idx = np.array(list(shared))
                if len(shared_idx) > 0:
                    ax.scatter(coords[shared_idx, 0], coords[shared_idx, 1],
                               c='gray', alpha=0.5, s=15, label=f'shared ({len(shared)})')
                # Only m1
                m1_idx = np.array(list(only_m1))
                if len(m1_idx) > 0:
                    ax.scatter(coords[m1_idx, 0], coords[m1_idx, 1],
                               c='blue', alpha=0.8, s=30, edgecolors='black',
                               linewidths=0.3, label=f'{m1} only ({len(only_m1)})')
                # Only m2
                m2_idx = np.array(list(only_m2))
                if len(m2_idx) > 0:
                    ax.scatter(coords[m2_idx, 0], coords[m2_idx, 1],
                               c='red', alpha=0.8, s=30, edgecolors='black',
                               linewidths=0.3, label=f'{m2} only ({len(only_m2)})')
                ax.legend(fontsize=7, loc='lower right')
                ax.set_title(f'{m1} vs {m2}', fontsize=11)
            else:
                ax.set_visible(False)
        elif col < n_methods:
            # Consecutive pairs: method[col-1] vs method[col]
            m1, m2 = methods[col - 1], methods[col]
            s1, s2 = set(selections[m1]), set(selections[m2])
            only_m1 = s1 - s2
            only_m2 = s2 - s1
            shared = s1 & s2

            all_sel = s1 | s2
            unsel = np.array([i not in all_sel for i in range(len(labels))])
            ax.scatter(coords[unsel, 0], coords[unsel, 1],
                       c='lightgray', alpha=0.08, s=3, rasterized=True)
            shared_idx = np.array(list(shared))
            if len(shared_idx) > 0:
                ax.scatter(coords[shared_idx, 0], coords[shared_idx, 1],
                           c='gray', alpha=0.5, s=15, label=f'shared ({len(shared)})')
            m1_idx = np.array(list(only_m1))
            if len(m1_idx) > 0:
                ax.scatter(coords[m1_idx, 0], coords[m1_idx, 1],
                           c='blue', alpha=0.8, s=30, edgecolors='black',
                           linewidths=0.3, label=f'{m1} only ({len(only_m1)})')
            m2_idx = np.array(list(only_m2))
            if len(m2_idx) > 0:
                ax.scatter(coords[m2_idx, 0], coords[m2_idx, 1],
                           c='red', alpha=0.8, s=30, edgecolors='black',
                           linewidths=0.3, label=f'{m2} only ({len(only_m2)})')
            ax.legend(fontsize=7, loc='lower right')
            ax.set_title(f'{m1} vs {m2}', fontsize=11)

        ax.set_xticks([])
        ax.set_yticks([])

    global_str = '_global' if args.global_selection else ''
    fig.suptitle(f'{args.dataset} | {args.embedding} | ratio={args.ratio} | '
                 f'k={args.k_neighbors}{global_str}', fontsize=13)
    plt.tight_layout()

    out_path = f'miccai/figures/compare_{args.dataset}_{args.embedding}_k{args.k_neighbors}_r{args.ratio}{global_str}.png'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='organsmnist')
    parser.add_argument('--embedding', default='uni')
    parser.add_argument('--methods', nargs='+', default=['graph_a1', 'graph_a2', 'graph_a3', 'graph_a4', 'graph_a5'])
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('-k', '--k-neighbors', type=int, default=10)
    parser.add_argument('--global', dest='global_selection', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    # Load data
    dataset, info = load_dataset(args.dataset, 'train', size=args.size, verbose=True)
    labels = get_labels(dataset)
    num_classes = len(info['label'])
    in_channels = info['n_channels']

    emb_data = load_or_compute_embeddings(
        dataset_name=args.dataset, split='train', source=args.embedding,
        dataset=dataset, num_classes=num_classes, in_channels=in_channels,
        size=args.size, seed=args.seed,
        cache_dir=Path('miccai/cache/embeddings'), verbose=True)
    embeddings = emb_data['embeddings']

    budget_per_class = max(1, int(len(labels) * args.ratio / num_classes))
    print(f"\nDataset: {args.dataset} | Embedding: {args.embedding} | "
          f"Ratio: {args.ratio} | Budget/class: {budget_per_class} | "
          f"k={args.k_neighbors} | global={args.global_selection}\n")

    # Run selection for each method
    selections = {}
    for method in args.methods:
        indices = select(
            method=method, labels=labels, budget_per_class=budget_per_class,
            embeddings=embeddings, seed=args.seed, verbose=False,
            k_neighbors=args.k_neighbors,
            global_selection=args.global_selection)
        selections[method] = indices
        print(f"{method}: {len(indices)} samples selected")

    # Pairwise comparison
    methods = list(selections.keys())
    print(f"\n{'':20s}", end='')
    for m in methods:
        print(f"{m:>12s}", end='')
    print()

    for i, m1 in enumerate(methods):
        print(f"{m1:20s}", end='')
        for j, m2 in enumerate(methods):
            if i == j:
                print(f"{'---':>12s}", end='')
            else:
                pct = overlap_pct(selections[m1], selections[m2])
                print(f"{pct:>11.1f}%", end='')
        print()

    # Show unique samples per method (not in any other method)
    print(f"\nUnique samples (not selected by any other variant):")
    all_sets = {m: set(s) for m, s in selections.items()}
    for m in methods:
        others = set()
        for m2 in methods:
            if m2 != m:
                others |= all_sets[m2]
        unique = all_sets[m] - others
        print(f"  {m}: {len(unique)} unique ({len(unique)/len(selections[m])*100:.1f}%)")

    # Visualization
    if not args.no_viz:
        visualize(embeddings, labels, selections, args)


if __name__ == '__main__':
    main()
