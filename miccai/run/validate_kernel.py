"""
Validate that K = A_sym + A_sym^2 matches the full heat kernel exp(-tL)
on medical imaging embeddings.

Compares at the SELECTION level — do both kernels select the same samples?
Also compares per-sample coverage scores (row sums).

Usage:
    python -m miccai.run.validate_kernel
    python -m miccai.run.validate_kernel --datasets organsmnist bloodmnist --k 5 10
"""

import numpy as np
from scipy.sparse import diags
from scipy.stats import spearmanr
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from miccai.run.data import load_dataset, get_labels
from miccai.run.embeddings import load_or_compute_embeddings
from miccai.run.graph import build_knn_graph, build_adjacency_matrix


DATASETS = ['organsmnist', 'organamnist', 'pathmnist', 'tissuemnist', 'bloodmnist']


def compute_heat_kernel(A_sym_dense, t=1.0):
    """Full heat kernel via eigendecomposition: K = exp(-tL), L = I - A_sym."""
    n = A_sym_dense.shape[0]
    L = np.eye(n) - A_sym_dense
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    exp_eigenvalues = np.exp(-t * eigenvalues)
    K = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
    return K


def compute_a2_kernel(A_sym):
    """K = A_sym + A_sym^2 (sparse)."""
    return A_sym + A_sym @ A_sym


def greedy_facility_location(K, labels, budget_per_class):
    """Run greedy facility location with per-class budgets. Returns selected indices."""
    n = K.shape[0]
    # Dense kernel for selection
    if hasattr(K, 'toarray'):
        K_dense = K.toarray().astype(np.float64)
    else:
        K_dense = np.array(K, dtype=np.float64)

    max_coverage = np.zeros(n)
    selected = []
    class_counts = {}

    total_budget = budget_per_class * len(np.unique(labels))
    for _ in range(total_budget):
        marginal = np.maximum(K_dense - max_coverage[:, None], 0)
        gains = marginal.sum(axis=0)

        # Mask out already-selected and classes at budget
        for idx in selected:
            gains[idx] = -np.inf
        for idx in range(n):
            c = labels[idx]
            if class_counts.get(c, 0) >= budget_per_class:
                gains[idx] = -np.inf

        best = np.argmax(gains)
        if gains[best] == -np.inf:
            break

        selected.append(best)
        c = labels[best]
        class_counts[c] = class_counts.get(c, 0) + 1
        max_coverage = np.maximum(max_coverage, K_dense[:, best])

    return selected


def compare_kernels(dataset_name, k_neighbors=10, t=1.0, budget_ratio=0.05,
                    embedding_source='uni', size=224, seed=42, max_n=5000):
    """Compare heat kernel and A+A^2 on one dataset."""
    # Load
    dataset, info = load_dataset(dataset_name, split='train', size=size, verbose=False)
    labels = get_labels(dataset)
    num_classes = len(np.unique(labels))
    in_channels = info['n_channels']

    emb_data = load_or_compute_embeddings(
        dataset_name, 'train', embedding_source, dataset,
        num_classes, in_channels, size=size, seed=seed, verbose=False
    )
    embeddings = emb_data['embeddings']
    n = len(embeddings)

    # Subsample if too large for eigendecomposition
    if n > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_n, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        n = max_n
        print(f"  Subsampled to {n} (eigendecomposition limit)")

    # Build graph
    knn_indices, knn_distances = build_knn_graph(embeddings, k_neighbors, verbose=False)
    A = build_adjacency_matrix(knn_indices, knn_distances, n)

    # Symmetric normalization
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt

    # Heat kernel (dense, O(n^3))
    t0 = time.time()
    A_sym_dense = A_sym.toarray().astype(np.float64)
    K_heat = compute_heat_kernel(A_sym_dense, t=t)
    time_heat = time.time() - t0

    # A + A^2 (sparse)
    t0 = time.time()
    K_a2 = compute_a2_kernel(A_sym)
    time_a2 = time.time() - t0

    # --- Comparison 1: Per-sample coverage scores (row sums) ---
    # Row sum = total influence of each sample = how much it covers / is covered
    heat_row_sums = K_heat.sum(axis=1)
    if hasattr(K_a2, 'toarray'):
        a2_row_sums = np.array(K_a2.sum(axis=1)).flatten()
    else:
        a2_row_sums = K_a2.sum(axis=1)
    rho_rowsums, _ = spearmanr(heat_row_sums, a2_row_sums)

    # --- Comparison 2: Selection overlap ---
    budget_per_class = max(1, int(n * budget_ratio / num_classes))
    total_selected = budget_per_class * num_classes

    t0 = time.time()
    sel_heat = greedy_facility_location(K_heat, labels, budget_per_class)
    time_sel_heat = time.time() - t0

    t0 = time.time()
    sel_a2 = greedy_facility_location(K_a2, labels, budget_per_class)
    time_sel_a2 = time.time() - t0

    set_heat = set(sel_heat)
    set_a2 = set(sel_a2)
    overlap = len(set_heat & set_a2)
    jaccard = overlap / len(set_heat | set_a2) if len(set_heat | set_a2) > 0 else 0
    overlap_pct = overlap / total_selected * 100

    # --- Comparison 3: Selection order rank correlation ---
    # Map each selected index to its selection rank
    rank_heat = {idx: rank for rank, idx in enumerate(sel_heat)}
    rank_a2 = {idx: rank for rank, idx in enumerate(sel_a2)}
    common = set_heat & set_a2
    if len(common) > 2:
        common_list = sorted(common)
        ranks_h = [rank_heat[i] for i in common_list]
        ranks_a = [rank_a2[i] for i in common_list]
        rho_order, _ = spearmanr(ranks_h, ranks_a)
    else:
        rho_order = float('nan')

    return {
        'dataset': dataset_name,
        'n': n,
        'k': k_neighbors,
        't': t,
        'budget_per_class': budget_per_class,
        'total_selected': total_selected,
        'rho_rowsums': rho_rowsums,
        'overlap': overlap,
        'overlap_pct': overlap_pct,
        'jaccard': jaccard,
        'rho_order': rho_order,
        'time_heat_s': round(time_heat, 3),
        'time_a2_s': round(time_a2, 3),
        'time_sel_heat_s': round(time_sel_heat, 3),
        'time_sel_a2_s': round(time_sel_a2, 3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate A+A^2 ≈ heat kernel on medical data')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--k', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--budget-ratio', type=float, default=0.05)
    parser.add_argument('--max-n', type=int, default=5000,
                        help='Max samples for eigendecomposition (subsamples if larger)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Validating K = A_sym + A_sym^2 vs exp(-tL)")
    print(f"Datasets: {args.datasets}")
    print(f"k values: {args.k}, t = {args.t}, budget = {args.budget_ratio:.0%}")
    print(f"max_n = {args.max_n}")
    print("=" * 90)

    results = []
    for dataset_name in args.datasets:
        for k in args.k:
            print(f"\n{dataset_name} (k={k})...")
            result = compare_kernels(
                dataset_name, k_neighbors=k, t=args.t,
                budget_ratio=args.budget_ratio,
                seed=args.seed, max_n=args.max_n
            )
            results.append(result)
            print(f"  Row-sum rank correlation:  ρ = {result['rho_rowsums']:.4f}")
            print(f"  Selection overlap:         {result['overlap']}/{result['total_selected']} "
                  f"({result['overlap_pct']:.1f}%), Jaccard = {result['jaccard']:.3f}")
            print(f"  Selection order correlation: ρ = {result['rho_order']:.4f}")
            print(f"  Time — kernel: heat={result['time_heat_s']}s, a2={result['time_a2_s']}s | "
                  f"selection: heat={result['time_sel_heat_s']}s, a2={result['time_sel_a2_s']}s")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Dataset':<15} {'k':>3} {'n':>6} {'ρ(rowsum)':>10} {'Overlap%':>10} {'Jaccard':>9} {'ρ(order)':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['dataset']:<15} {r['k']:>3} {r['n']:>6} "
              f"{r['rho_rowsums']:>10.4f} {r['overlap_pct']:>9.1f}% "
              f"{r['jaccard']:>9.3f} {r['rho_order']:>10.4f}")

    # Averages
    print("-" * 90)
    avg_rho = np.mean([r['rho_rowsums'] for r in results])
    avg_overlap = np.mean([r['overlap_pct'] for r in results])
    avg_jaccard = np.mean([r['jaccard'] for r in results])
    rho_orders = [r['rho_order'] for r in results if not np.isnan(r['rho_order'])]
    avg_order = np.mean(rho_orders) if rho_orders else float('nan')
    print(f"{'Mean':<15} {'':>3} {'':>6} {avg_rho:>10.4f} {avg_overlap:>9.1f}% "
          f"{avg_jaccard:>9.3f} {avg_order:>10.4f}")


if __name__ == '__main__':
    main()
