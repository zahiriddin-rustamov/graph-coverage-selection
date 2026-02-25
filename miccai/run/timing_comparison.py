"""
Timing comparison: selection methods vs dataset size.

Shows how facility location (dense N×N matrix) scales compared to
graph_a2 (sparse k-NN), herding, and FPS.

Usage:
    python -m miccai.run.timing_comparison
    python -m miccai.run.timing_comparison --dataset tissuemnist --sizes 1000 5000 10000 20000 50000
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from miccai.run.data import load_dataset, get_labels
from miccai.run.embeddings import load_or_compute_embeddings
from miccai.run.selection import select


METHODS = ['facility', 'graph_a2', 'herding', 'fps']
DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 50000]


def time_selection(method, embeddings, labels, budget_per_class, seed=42, k_neighbors=5):
    """Time a single selection call. Returns wall-clock seconds."""
    kwargs = dict(
        embeddings=embeddings, seed=seed, verbose=False,
        global_selection=True, k_neighbors=k_neighbors,
    )
    t0 = time.time()
    selected = select(method, labels, budget_per_class, **kwargs)
    elapsed = time.time() - t0
    return elapsed, len(selected)


def subsample_stratified(embeddings, labels, n_target, seed=42):
    """Subsample to n_target preserving class ratios."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n_target >= n:
        return embeddings, labels

    classes, counts = np.unique(labels, return_counts=True)
    ratios = counts / counts.sum()
    per_class = np.maximum(1, (ratios * n_target).astype(int))
    # Adjust to hit target exactly
    diff = n_target - per_class.sum()
    if diff > 0:
        for i in range(diff):
            per_class[i % len(classes)] += 1
    elif diff < 0:
        for i in range(-diff):
            idx = len(classes) - 1 - (i % len(classes))
            per_class[idx] = max(1, per_class[idx] - 1)

    indices = []
    for c, n_c in zip(classes, per_class):
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, min(n_c, len(class_idx)), replace=False)
        indices.extend(chosen)

    indices = np.array(sorted(indices))
    return embeddings[indices], labels[indices]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tissuemnist')
    parser.add_argument('--sizes', nargs='+', type=int, default=DEFAULT_SIZES)
    parser.add_argument('--budget-ratio', type=float, default=0.05)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='miccai/figures')
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    print(f"Loading {args.dataset}...")
    dataset, info = load_dataset(args.dataset, split='train', size=224, verbose=True)
    labels_full = get_labels(dataset)
    num_classes = len(np.unique(labels_full))
    in_channels = info['n_channels']

    emb_data = load_or_compute_embeddings(
        args.dataset, 'train', 'uni', dataset,
        num_classes, in_channels, size=224, seed=args.seed, verbose=True
    )
    embeddings_full = emb_data['embeddings']
    n_full = len(embeddings_full)
    print(f"Full dataset: n={n_full}, classes={num_classes}, dim={embeddings_full.shape[1]}")

    # Filter sizes that are <= dataset size
    sizes = [s for s in args.sizes if s <= n_full]
    print(f"Testing sizes: {sizes}")
    print(f"Methods: {METHODS}")
    print(f"Budget ratio: {args.budget_ratio:.0%}, k={args.k}")
    print("=" * 70)

    # Time each method at each size
    results = {m: [] for m in METHODS}
    for n in sizes:
        emb, lab = subsample_stratified(embeddings_full, labels_full, n, seed=args.seed)
        actual_n = len(lab)
        budget_per_class = max(1, int(actual_n * args.budget_ratio / num_classes))
        print(f"\nn={actual_n}, budget={budget_per_class}/class")

        for method in METHODS:
            try:
                elapsed, n_sel = time_selection(
                    method, emb, lab, budget_per_class,
                    seed=args.seed, k_neighbors=args.k
                )
                results[method].append((actual_n, elapsed))
                print(f"  {method:<15} {elapsed:>8.2f}s  (selected {n_sel})")
            except Exception as e:
                print(f"  {method:<15} FAILED: {e}")
                results[method].append((actual_n, float('nan')))

    # Summary table
    print("\n" + "=" * 70)
    header = f"{'n':>8}" + "".join(f"  {m:>12}" for m in METHODS)
    print(header)
    print("-" * 70)
    for i, n in enumerate(sizes):
        row = f"{n:>8}"
        for method in METHODS:
            if i < len(results[method]):
                _, t = results[method][i]
                row += f"  {t:>11.2f}s" if not np.isnan(t) else f"  {'OOM':>12}"
            else:
                row += f"  {'—':>12}"
        print(row)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    markers = {'facility': 's', 'graph_a2': 'o', 'herding': '^', 'fps': 'D'}
    colors = {'facility': '#d63031', 'graph_a2': '#0984e3', 'herding': '#00b894', 'fps': '#fdcb6e'}
    labels_map = {'facility': 'Facility Location', 'graph_a2': 'Graph A+A² (ours)',
                  'herding': 'Herding', 'fps': 'FPS'}

    for method in METHODS:
        ns = [r[0] for r in results[method] if not np.isnan(r[1])]
        ts = [r[1] for r in results[method] if not np.isnan(r[1])]
        if ns:
            ax.plot(ns, ts, marker=markers[method], color=colors[method],
                    label=labels_map[method], linewidth=2, markersize=7)

    ax.set_xlabel('Number of samples (N)', fontsize=12)
    ax.set_ylabel('Selection time (seconds)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    path_png = output_path / 'timing_comparison.png'
    path_pdf = output_path / 'timing_comparison.pdf'
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    print(f"\nSaved: {path_png}, {path_pdf}")
    plt.close(fig)


if __name__ == '__main__':
    main()
