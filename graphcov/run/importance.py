"""
Importance computation module.

Computes per-sample importance scores using various metrics.
Importance scores are used to weight coverage in selection methods.

Available metrics:
- uniform: All samples equally important
- el2n_easy: Low EL2N = easy = important (foundation samples)
- el2n_hard: High EL2N = hard = important (boundary samples)
- variance_low: Low confidence variance = stable = important
- variance_high: High confidence variance = ambiguous = important
- test_attention: Similarity to validation distribution = important (uses val set, not test)
- density: High local density in k-NN = prototypical = important
- centrality: High graph centrality = hub = important
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Available importance methods
IMPORTANCE_METHODS = [
    'uniform',
    'el2n_easy',
    'el2n_hard',
    'el2n_mid',
    'variance_low',
    'variance_high',
    'test_attention',
    'density',
    'centrality',
]


@dataclass
class ImportanceResult:
    """Result of importance computation."""
    scores: np.ndarray  # (n,) importance scores
    method: str
    metadata: Dict  # Method-specific metadata


def compute_importance(
    method: str,
    labels: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    el2n_scores: Optional[np.ndarray] = None,
    variance_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    test_embeddings: Optional[np.ndarray] = None,
    graph: Optional = None,  # scipy sparse matrix
    temperature: float = 0.1,
    k_neighbors: int = 10,
    normalize: bool = True,
    verbose: bool = False,
) -> ImportanceResult:
    """
    Compute per-sample importance scores.

    Args:
        method: Importance method (see IMPORTANCE_METHODS)
        labels: (n,) array of labels
        embeddings: (n, d) array of embeddings (needed for test_attention, density, centrality)
        el2n_scores: (n,) array of EL2N scores (needed for el2n_* methods)
        variance_stats: (conf_variance, conf_mean) tuple (needed for variance_* methods)
        test_embeddings: (n_test, d) array (needed for test_attention)
        graph: scipy sparse adjacency matrix (needed for centrality, can be used for density)
        temperature: Temperature for test_attention softmax
        k_neighbors: k for density computation
        normalize: If True, normalize scores to [0, 1] range
        verbose: Print score statistics

    Returns:
        ImportanceResult with scores and metadata
    """
    if method not in IMPORTANCE_METHODS:
        raise ValueError(f"Unknown importance method: {method}. Available: {IMPORTANCE_METHODS}")

    n_samples = len(labels)
    metadata = {'method': method}

    if method == 'uniform':
        scores = np.ones(n_samples)
        metadata['description'] = 'All samples equally important'

    elif method == 'el2n_easy':
        if el2n_scores is None:
            raise ValueError("el2n_easy requires el2n_scores (use 'trained' embedding)")
        # Low EL2N = easy = high importance
        scores = -el2n_scores  # Negate so low EL2N -> high score
        metadata['description'] = 'Low EL2N (easy samples) = high importance'

    elif method == 'el2n_hard':
        if el2n_scores is None:
            raise ValueError("el2n_hard requires el2n_scores (use 'trained' embedding)")
        # High EL2N = hard = high importance
        scores = el2n_scores.copy()
        metadata['description'] = 'High EL2N (hard samples) = high importance'

    elif method == 'el2n_mid':
        if el2n_scores is None:
            raise ValueError("el2n_mid requires el2n_scores (use 'trained' embedding)")
        # Middle EL2N = ambiguous = high importance
        # Score peaks at median, decreases toward extremes
        median = np.median(el2n_scores)
        scores = -np.abs(el2n_scores - median)  # Negate absolute deviation
        metadata['description'] = 'Mid EL2N (ambiguous samples) = high importance'
        metadata['median_el2n'] = float(median)

    elif method == 'variance_low':
        if variance_stats is None:
            raise ValueError("variance_low requires variance_stats (use 'trained' embedding)")
        conf_var, conf_mean = variance_stats
        # Low variance + high confidence = stable = important
        scores = -conf_var + conf_mean
        metadata['description'] = 'Low variance, high confidence = high importance'

    elif method == 'variance_high':
        if variance_stats is None:
            raise ValueError("variance_high requires variance_stats (use 'trained' embedding)")
        conf_var, _ = variance_stats
        # High variance = ambiguous = important
        scores = conf_var.copy()
        metadata['description'] = 'High variance (ambiguous) = high importance'

    elif method == 'test_attention':
        if embeddings is None or test_embeddings is None:
            raise ValueError("test_attention requires embeddings and test_embeddings (val set)")
        scores = _compute_test_attention(embeddings, test_embeddings, temperature, verbose=verbose)
        metadata['description'] = 'High attention from validation distribution = high importance'
        metadata['temperature'] = temperature
        metadata['n_val_samples'] = len(test_embeddings)

    elif method == 'density':
        if embeddings is None:
            raise ValueError("density requires embeddings")
        scores = _compute_density(embeddings, k_neighbors, verbose=verbose)
        metadata['description'] = 'High local density = prototypical = high importance'
        metadata['k_neighbors'] = k_neighbors

    elif method == 'centrality':
        if graph is None:
            raise ValueError("centrality requires graph")
        scores = _compute_centrality(graph, verbose=verbose)
        metadata['description'] = 'High PageRank centrality = hub = high importance'

    else:
        raise ValueError(f"Method {method} not implemented")

    # Normalize to [0, 1] range
    if normalize:
        scores = _normalize_scores(scores)

    if verbose:
        percentiles = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
        print(f"  [Importance] {method}: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"  [Importance] Percentiles [0,10,25,50,75,90,100]: {[f'{p:.4f}' for p in percentiles]}")
        # Distribution shape info
        skewness = ((scores - scores.mean()) ** 3).mean() / (scores.std() ** 3 + 1e-8)
        kurtosis = ((scores - scores.mean()) ** 4).mean() / (scores.std() ** 4 + 1e-8) - 3
        print(f"  [Importance] Distribution: skewness={skewness:.4f}, kurtosis={kurtosis:.4f}")
        # Top/bottom sample info
        n_high = (scores > np.percentile(scores, 90)).sum()
        n_low = (scores < np.percentile(scores, 10)).sum()
        print(f"  [Importance] Outliers: {n_high} samples above 90th percentile, {n_low} below 10th")

    return ImportanceResult(scores=scores, method=method, metadata=metadata)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val > 1e-8:
        return (scores - min_val) / (max_val - min_val)
    else:
        return np.ones_like(scores)  # All same value -> uniform


def _compute_test_attention(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    temperature: float = 0.1,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute test attention: how much test samples attend to each train sample.

    Uses cross-attention where test samples are queries, train samples are keys.
    Each train sample's importance = sum of attention it receives from all test samples.

    Args:
        train_embeddings: (n_train, d) array
        test_embeddings: (n_test, d) array
        temperature: Softmax temperature (lower = sharper attention)
        verbose: Print attention statistics

    Returns:
        (n_train,) importance scores
    """
    # Move to GPU for efficiency
    train_t = torch.from_numpy(train_embeddings).float().to(device)
    test_t = torch.from_numpy(test_embeddings).float().to(device)

    # Normalize for cosine similarity
    train_norm = F.normalize(train_t, dim=1)
    test_norm = F.normalize(test_t, dim=1)

    # Compute attention: test attends to train
    # similarity[i, j] = how similar test sample i is to train sample j
    similarity = test_norm @ train_norm.T  # (n_test, n_train)

    if verbose:
        sim_vals = similarity.flatten().cpu().numpy()
        print(f"    [TestAttention] Similarity: min={sim_vals.min():.4f}, max={sim_vals.max():.4f}, "
              f"mean={sim_vals.mean():.4f}")

    # Softmax over train samples (for each test sample)
    attention = F.softmax(similarity / temperature, dim=1)  # (n_test, n_train)

    if verbose:
        # Attention entropy (higher = more uniform)
        entropy = -((attention * torch.log(attention + 1e-10)).sum(dim=1)).mean().item()
        max_entropy = np.log(train_embeddings.shape[0])
        # Attention concentration (how many samples get most attention)
        top1_mass = attention.max(dim=1).values.mean().item()
        top10_mass = attention.topk(min(10, attention.shape[1]), dim=1).values.sum(dim=1).mean().item()
        print(f"    [TestAttention] Temperature={temperature}, entropy={entropy:.4f} (max={max_entropy:.4f})")
        print(f"    [TestAttention] Concentration: top1={top1_mass:.4f}, top10={top10_mass:.4f}")

    # Aggregate: sum attention from all test samples
    importance = attention.sum(dim=0).cpu().numpy()  # (n_train,)

    return importance


def _compute_density(
    embeddings: np.ndarray,
    k: int = 10,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute local density: average similarity to k nearest neighbors.

    High density = sample is in a dense region = prototypical.

    Args:
        embeddings: (n, d) array
        k: Number of neighbors
        verbose: Print density statistics

    Returns:
        (n,) density scores
    """
    from .graph import build_knn_graph

    # Build k-NN graph
    knn_indices, knn_distances = build_knn_graph(embeddings, k)

    # Density = inverse of mean distance to neighbors
    # Lower distance = higher density
    mean_dist = knn_distances.mean(axis=1)
    density = 1.0 / (mean_dist + 1e-8)

    if verbose:
        print(f"    [Density] k={k}, n={len(embeddings)}")
        print(f"    [Density] Mean distance to neighbors: min={mean_dist.min():.4f}, "
              f"max={mean_dist.max():.4f}, mean={mean_dist.mean():.4f}")
        print(f"    [Density] Density values: min={density.min():.4f}, max={density.max():.4f}")

    return density


def _compute_centrality(graph, verbose: bool = False) -> np.ndarray:
    """
    Compute PageRank centrality on the graph.

    High centrality = sample is a hub = connects many regions.

    Args:
        graph: scipy sparse adjacency matrix
        verbose: Print convergence info

    Returns:
        (n,) centrality scores
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigs

    n = graph.shape[0]

    # Convert to transition matrix (row-stochastic)
    row_sums = np.array(graph.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    D_inv = diags(1.0 / row_sums)
    P = D_inv @ graph

    # Power iteration for PageRank
    damping = 0.85
    teleport = np.ones(n) / n

    scores = np.ones(n) / n
    converged_iter = 50
    for i in range(50):  # Usually converges in ~20 iterations
        scores_new = damping * (P.T @ scores) + (1 - damping) * teleport
        diff = np.abs(scores_new - scores).max()
        if diff < 1e-6:
            converged_iter = i + 1
            break
        scores = scores_new

    if verbose:
        print(f"    [Centrality] PageRank converged in {converged_iter} iterations, damping={damping}")
        print(f"    [Centrality] Scores: min={scores.min():.6f}, max={scores.max():.6f}, "
              f"sum={scores.sum():.4f}")

    return scores


def get_importance_requirements(method: str) -> set:
    """Get data requirements for an importance method."""
    requirements = {
        'uniform': set(),
        'el2n_easy': {'el2n_scores'},
        'el2n_hard': {'el2n_scores'},
        'el2n_mid': {'el2n_scores'},
        'variance_low': {'variance_stats'},
        'variance_high': {'variance_stats'},
        'test_attention': {'embeddings', 'test_embeddings'},
        'density': {'embeddings'},
        'centrality': {'graph'},
    }
    if method not in requirements:
        raise ValueError(f"Unknown method: {method}")
    return requirements[method]


def get_available_importance_methods() -> List[str]:
    """Get list of available importance methods."""
    return IMPORTANCE_METHODS.copy()
