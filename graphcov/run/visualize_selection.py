"""
Visualize selected subsets using t-SNE on embeddings.

Runs selection (no training) and plots the selections overlaid on the full dataset.
Uses the same arguments as graphcov.run to ensure identical selections.

Usage:
    python -m graphcov.run.visualize_selection \
        --dataset dermamnist \
        --methods graph_a2 graph_a2 \
        --labels "per-class" "global" \
        --global-flags 0 1 \
        --embedding uni \
        --ratio 0.02 \
        -k 10 \
        --size 224

    # Single method visualization
    python -m graphcov.run.visualize_selection \
        --dataset dermamnist \
        --methods graph_a2 \
        --embedding uni \
        --ratio 0.02 \
        -k 10 \
        --size 224
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from .data import load_dataset, get_labels
from .embeddings import load_or_compute_embeddings
from .selection import select

# CVD-safe palette consistent with project figures
CLASS_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
    '#2F9E8F', '#B5542B',
]


def run_visualization(args):
    dataset_name = args.dataset
    ds, info = load_dataset(dataset_name, 'train', args.size, verbose=True)
    num_classes = len(info['label'])
    in_channels = info['n_channels']
    labels = get_labels(ds)

    # Load embeddings
    emb_data = load_or_compute_embeddings(
        dataset_name=dataset_name,
        split='train',
        source=args.embedding,
        dataset=ds,
        num_classes=num_classes,
        in_channels=in_channels,
        size=args.size,
        seed=args.seed,
        cache_dir=Path(args.cache),
        verbose=True,
    )
    embeddings = emb_data['embeddings']
    n_total = len(labels)
    n_select = int(n_total * args.ratio)
    budget_per_class = n_select // num_classes

    print(f"\nDataset: {dataset_name} | Embedding: {args.embedding}")
    print(f"Ratio: {args.ratio} | Budget/class: {budget_per_class} | Total: {budget_per_class * num_classes}")

    # Run selections
    methods = args.methods
    global_flags = args.global_flags or [0] * len(methods)
    method_labels = args.labels or methods

    if len(global_flags) != len(methods):
        print(f"Error: --global-flags length ({len(global_flags)}) must match --methods length ({len(methods)})")
        sys.exit(1)
    if len(method_labels) != len(methods):
        print(f"Error: --labels length ({len(method_labels)}) must match --methods length ({len(methods)})")
        sys.exit(1)

    selections = {}
    for method, is_global, label in zip(methods, global_flags, method_labels):
        # Only pass k_hops if explicitly set by user (otherwise use method registry default)
        sel_kwargs = dict(
            method=method,
            labels=labels,
            budget_per_class=budget_per_class,
            embeddings=embeddings,
            seed=args.seed,
            verbose=True,
            k_neighbors=args.k_neighbors,
            global_selection=bool(is_global),
            sparse_cpu=args.sparse_cpu,
        )
        if args.k_hops is not None:
            sel_kwargs['k_hops'] = args.k_hops
        sel = select(**sel_kwargs)
        selections[label] = sel
        print(f"  {label}: {len(sel)} selected (global={bool(is_global)})")

    # t-SNE
    print(f"\nRunning t-SNE (n={n_total}, perplexity={args.perplexity})...")
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(embeddings)
    print("  t-SNE done.")

    # --- Plotting ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee', 'no-latex'])
    except (ImportError, OSError):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 6,
        })

    n_methods = len(selections)
    n_panels = n_methods + 1
    textwidth = 7.0
    fig_height = min(max(textwidth / n_panels + 0.4, 1.8), 3.0)

    fig, axes = plt.subplots(1, n_panels, figsize=(textwidth, fig_height))
    if n_panels == 1:
        axes = [axes]

    class_colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(num_classes)]
    class_names = (
        list(info['label'].values()) if isinstance(info['label'], dict)
        else [str(i) for i in range(num_classes)]
    )

    # --- Panel (a): Full dataset ---
    ax = axes[0]
    for c in range(num_classes):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=class_colors[c], s=3, alpha=0.35, rasterized=True,
        )
    ax.set_title(f'Full dataset ($n$={n_total:,})', fontsize=8)
    ax.text(
        0.03, 0.97, '(a)', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Selection panels ---
    for idx, (label, sel) in enumerate(selections.items()):
        ax = axes[idx + 1]
        panel_letter = chr(ord('b') + idx)

        # Faint grey background — spatial reference only
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c='#e0e0e0', s=1, alpha=0.15, rasterized=True,
        )

        # Selected points (bold, colored by class)
        sel_arr = np.array(sel)
        sel_labels = labels[sel_arr]
        for c in range(num_classes):
            c_mask = sel_labels == c
            if c_mask.any():
                ax.scatter(
                    coords[sel_arr[c_mask], 0], coords[sel_arr[c_mask], 1],
                    c=class_colors[c], s=18, alpha=0.95,
                    edgecolors='black', linewidths=0.3, rasterized=True,
                )

        ax.set_title(f'{label} ($n$={len(sel):,})', fontsize=8)
        ax.text(
            0.03, 0.97, f'({panel_letter})', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    if not args.legend_off:
        legend_elements = [
            Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=class_colors[c], markersize=4,
                label=class_names[c], markeredgewidth=0,
            )
            for c in range(num_classes)
        ]
        fig.legend(
            handles=legend_elements, loc='lower center',
            ncol=min(num_classes, 10), fontsize=6, frameon=False,
            handletextpad=0.3, columnspacing=0.8,
            bbox_to_anchor=(0.5, -0.01),
        )

    plt.tight_layout()
    if not args.legend_off:
        plt.subplots_adjust(bottom=0.14)

    # Save (PNG + PDF)
    output_stem = args.output or f'selection_viz_{dataset_name}'
    output_stem = str(Path(output_stem).with_suffix(''))
    fig.savefig(f'{output_stem}.png', dpi=args.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_stem}.pdf', bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_stem}.png and {output_stem}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize selected subsets using t-SNE')

    # Selection args (match graphcov.run)
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--methods', nargs='+', required=True, help='Selection methods')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Display labels for each method (default: method names)')
    parser.add_argument('--global-flags', type=int, nargs='+', default=None,
                        help='0/1 per method for global selection (default: all 0)')
    parser.add_argument('--embedding', default='uni', help='Embedding source')
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('-k', '--k-neighbors', type=int, default=10)
    parser.add_argument('--k-hops', type=int, default=None,
                        help='Override k_hops (default: use method registry default)')
    parser.add_argument('--sparse-cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--cache', default='graphcov/cache/embeddings')

    # Plot args
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--legend-off', action='store_true', help='Hide class legend')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: selection_viz_{dataset}.png)')

    args = parser.parse_args()
    run_visualization(args)


if __name__ == '__main__':
    main()
