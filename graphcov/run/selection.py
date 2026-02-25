"""
Sample selection methods with unified registry.

All selection methods are registered in METHODS dict with their requirements
and importance handling mode.

Importance modes:
- 'ignore': Method has its own selection logic, ignores --importance
- 'optional': Can use importance weights if provided, defaults to uniform
- 'required': Must have importance weights (errors or defaults to uniform)
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Set, Dict, List, Optional, Tuple, Any

from .graph import build_knn_graph, compute_k_hop_adjacency

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Method Registry
# =============================================================================

@dataclass
class MethodSpec:
    """Specification for a selection method."""
    fn: Callable
    needs: Set[str]  # {'embeddings', 'el2n_scores', 'variance_stats', 'eva_scores', 'aum_scores', 'forgetting_scores'}
    importance: str  # 'ignore', 'optional', 'required'
    kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# Registry populated at module load
METHODS: Dict[str, MethodSpec] = {}


def register_method(
    name: str,
    needs: Set[str],
    importance: str = 'ignore',
    kwargs: Optional[Dict] = None,
    description: str = ""
):
    """Decorator to register a selection method."""
    if importance not in ('ignore', 'optional', 'required'):
        raise ValueError(f"importance must be 'ignore', 'optional', or 'required', got '{importance}'")

    def decorator(fn: Callable):
        METHODS[name] = MethodSpec(
            fn=fn,
            needs=needs,
            importance=importance,
            kwargs=kwargs or {},
            description=description
        )
        return fn
    return decorator


def get_available_methods() -> List[str]:
    """Get list of all registered method names."""
    return sorted(METHODS.keys())


def get_method_requirements(method: str) -> Set[str]:
    """Get the data requirements for a method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")
    return METHODS[method].needs


def get_method_info(method: str) -> MethodSpec:
    """Get full method specification."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")
    return METHODS[method]


def get_method_importance_mode(method: str) -> str:
    """Get how a method handles importance: 'ignore', 'optional', or 'required'."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")
    return METHODS[method].importance


# =============================================================================
# Unified Selection Interface
# =============================================================================

def select(
    method: str,
    labels: np.ndarray,
    budget_per_class: int,
    embeddings: Optional[np.ndarray] = None,
    el2n_scores: Optional[np.ndarray] = None,
    variance_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eva_scores: Optional[np.ndarray] = None,
    aum_scores: Optional[np.ndarray] = None,
    forgetting_scores: Optional[np.ndarray] = None,
    importance: Optional[np.ndarray] = None,
    importance_name: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,  # 1=summary, 2=detailed, 3=per-iteration
    **overrides
) -> List[int]:
    """
    Unified selection interface.

    Args:
        method: Method name
        labels: (n,) array of labels
        budget_per_class: Number of samples to select per class
        embeddings: (n, d) embedding array (if needed by method)
        el2n_scores: (n,) EL2N scores (if needed by method)
        variance_stats: (conf_var, conf_mean) tuple (if needed by method)
        eva_scores: (n,) EVA scores (if needed by method)
        aum_scores: (n,) AUM scores (if needed by method)
        forgetting_scores: (n,) Forgetting scores (if needed by method)
        importance: (n,) importance weights (for methods that use importance)
        importance_name: Name of importance method (for logging only)
        seed: Random seed
        verbose: Print selection details
        **overrides: Override method-specific kwargs (e.g., k_neighbors, k_hops)

    Returns:
        List of selected indices
    """
    spec = get_method_info(method)

    # Check data requirements
    if 'embeddings' in spec.needs and embeddings is None:
        raise ValueError(f"Method '{method}' requires embeddings")
    if 'el2n_scores' in spec.needs and el2n_scores is None:
        raise ValueError(f"Method '{method}' requires el2n_scores (use 'trained' embedding)")
    if 'variance_stats' in spec.needs and variance_stats is None:
        raise ValueError(f"Method '{method}' requires variance_stats (use 'trained' embedding)")
    if 'eva_scores' in spec.needs and eva_scores is None:
        raise ValueError(f"Method '{method}' requires eva_scores")
    if 'aum_scores' in spec.needs and aum_scores is None:
        raise ValueError(f"Method '{method}' requires aum_scores")
    if 'forgetting_scores' in spec.needs and forgetting_scores is None:
        raise ValueError(f"Method '{method}' requires forgetting_scores")

    # Handle importance based on method's importance mode
    n_samples = len(labels)
    if spec.importance == 'ignore':
        # Method ignores importance, don't pass it
        effective_importance = None
    elif spec.importance == 'optional':
        # Use provided importance or default to uniform
        effective_importance = importance if importance is not None else np.ones(n_samples)
    elif spec.importance == 'required':
        # Must have importance
        if importance is None:
            if verbose:
                print(f"  [Warning] Method '{method}' requires importance, using uniform")
            effective_importance = np.ones(n_samples)
        else:
            effective_importance = importance

    # Build kwargs: start with method defaults, then apply overrides
    kwargs = {
        'labels': labels,
        'budget_per_class': budget_per_class,
        'seed': seed,
        **spec.kwargs
    }

    # Apply overrides (only for keys that exist in spec.kwargs)
    for key, value in overrides.items():
        if key in spec.kwargs:
            kwargs[key] = value

    # Pass flags that methods accept via **kwargs
    if 'global_selection' in overrides:
        kwargs['global_selection'] = overrides['global_selection']
    if 'sparse_cpu' in overrides:
        kwargs['sparse_cpu'] = overrides['sparse_cpu']

    # Add data requirements
    if 'embeddings' in spec.needs:
        kwargs['embeddings'] = embeddings
    if 'el2n_scores' in spec.needs:
        kwargs['el2n_scores'] = el2n_scores
    if 'variance_stats' in spec.needs:
        kwargs['variance_stats'] = variance_stats
    if 'eva_scores' in spec.needs:
        kwargs['eva_scores'] = eva_scores
    if 'aum_scores' in spec.needs:
        kwargs['aum_scores'] = aum_scores
    if 'forgetting_scores' in spec.needs:
        kwargs['forgetting_scores'] = forgetting_scores

    # Add importance for methods that use it
    if spec.importance in ('optional', 'required'):
        kwargs['importance'] = effective_importance

    # Pass verbose to method functions
    kwargs['verbose'] = verbose
    kwargs['_verbose_level'] = _verbose_level

    selected = spec.fn(**kwargs)

    if verbose:
        n_classes = len(np.unique(labels))
        imp_str = ""
        if spec.importance != 'ignore':
            if importance is not None:
                imp_label = importance_name if importance_name else 'provided'
            else:
                imp_label = 'uniform'
            imp_str = f", importance={imp_label}"
        print(f"  [Selection] {method}: budget={budget_per_class}/class, "
              f"n_classes={n_classes}, total_selected={len(selected)}{imp_str}")

    return selected


# =============================================================================
# Helper Functions
# =============================================================================

def compute_class_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """Compute centroid for each class."""
    centroids = {}
    for c in np.unique(labels):
        class_mask = labels == c
        centroids[c] = embeddings[class_mask].mean(axis=0)
    return centroids


def farthest_point_sampling(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42
) -> List[int]:
    """Select k points using FPS - GPU accelerated."""
    np.random.seed(seed)
    n = len(embeddings)
    if k >= n:
        return list(range(n))

    emb_t = torch.from_numpy(embeddings).float().to(device)
    emb_t = F.normalize(emb_t, dim=1)

    first_idx = np.random.randint(n)
    selected = [first_idx]
    min_distances = torch.full((n,), float('inf'), device=device)
    min_distances[first_idx] = float('-inf')

    for _ in range(k - 1):
        last_selected = selected[-1]
        distances = 1.0 - emb_t @ emb_t[last_selected]
        min_distances = torch.minimum(min_distances, distances)
        next_idx = torch.argmax(min_distances).item()
        selected.append(next_idx)
        min_distances[next_idx] = float('-inf')

    return selected


def _greedy_facility_global_sparse(
    kernel,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    verbose: bool = False,
    _verbose_level: int = 1,
    method_name: str = 'Global',
) -> List[int]:
    """
    Sparse CPU implementation of greedy facility location using lazy greedy.

    Mathematically identical to the dense GPU path — same selections, same order.
    Two key optimizations:

    1. Sparse: operates only on nonzero entries. For K[i,j]=0,
       max(0 - max_coverage[i], 0) = 0 since max_coverage >= 0 always.

    2. Lazy greedy (Minoux 1978): submodularity guarantees marginal gains
       only decrease. A max-heap of upper bounds avoids recomputing all N
       candidates each iteration — typically only 1-5 recomputations needed.
    """
    import scipy.sparse as sp
    import heapq

    n = len(labels)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    total_budget = budget_per_class * n_classes

    K_csc = sp.csc_matrix(kernel, dtype=np.float32)
    imp = importance.astype(np.float32)

    # Precompute CSC structure arrays
    kdata = K_csc.data
    row_idx = K_csc.indices
    indptr = K_csc.indptr
    col_idx = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))

    eligible = np.ones(n, dtype=bool)
    selected = []
    class_counts = {int(c): 0 for c in unique_classes}
    max_coverage = np.zeros(n, dtype=np.float32)

    # Initial full pass to compute all gains — O(nnz), done once
    marginal = np.maximum(kdata, 0.0) * imp[row_idx]  # max_coverage is 0
    gains_init = np.bincount(col_idx, weights=marginal, minlength=n)

    # Build max-heap (negate for Python's min-heap)
    # last_eval[j] tracks when j's gain was last computed
    last_eval = np.zeros(n, dtype=np.int64)
    heap = [(-gains_init[j], j) for j in range(n)]
    heapq.heapify(heap)

    for iter_idx in range(total_budget):
        # Lazy greedy: pop candidates, recompute only if stale
        best_j = -1
        while heap:
            neg_gain, j = heapq.heappop(heap)
            if not eligible[j]:
                continue
            if last_eval[j] == iter_idx:
                # Gain is exact for this iteration — select it
                best_j = j
                best_gain = -neg_gain
                break
            # Recompute gain for candidate j from its CSC column
            s, e = indptr[j], indptr[j + 1]
            col_rows = row_idx[s:e]
            m = np.maximum(kdata[s:e] - max_coverage[col_rows], 0.0)
            gain_j = np.dot(imp[col_rows], m)
            last_eval[j] = iter_idx
            heapq.heappush(heap, (-gain_j, j))

        if best_j == -1:
            if verbose:
                print(f"  [{method_name}] Early stop at iter={iter_idx}: no eligible candidates")
            break

        selected.append(best_j)
        c = int(labels[best_j])
        class_counts[c] += 1
        eligible[best_j] = False

        if class_counts[c] >= budget_per_class:
            eligible[labels == c] = False

        # Update max_coverage from column best_j
        s, e = indptr[best_j], indptr[best_j + 1]
        col_rows = row_idx[s:e]
        max_coverage[col_rows] = np.maximum(max_coverage[col_rows], kdata[s:e])

        if verbose and _verbose_level >= 3 and (iter_idx < 10 or iter_idx % 50 == 0):
            coverage_pct = (max_coverage * imp).sum() / imp.sum()
            print(f"    [{method_name}] iter={iter_idx+1}: idx={best_j}, class={c}, "
                  f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

    if verbose:
        final_coverage = (max_coverage * imp).sum() / imp.sum()
        print(f"  [{method_name}] Global: selected={len(selected)}, coverage={final_coverage:.4f}")
        print(f"  [{method_name}] Class distribution: {dict(class_counts)}")

    return selected


def _greedy_facility_global(
    kernel,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    verbose: bool = False,
    _verbose_level: int = 1,
    method_name: str = 'Global',
    sparse_cpu: bool = False,
) -> List[int]:
    """
    Greedy facility location on a global kernel with per-class budget constraints.

    Selects samples one at a time, choosing the one with highest marginal coverage
    gain, while ensuring exactly budget_per_class samples from each class.

    Uses sparse CPU path when --sparse-cpu is set or when the dense matrix
    would be too large (n > 50K with sparse kernel).

    Args:
        kernel: (n, n) kernel/similarity matrix - scipy sparse, numpy, or torch tensor
        labels: (n,) class labels
        budget_per_class: Samples to select per class
        importance: (n,) importance weights
        verbose: Print progress
        _verbose_level: Detail level
        method_name: Name for verbose output
        sparse_cpu: Force sparse CPU path regardless of size

    Returns:
        List of selected global indices
    """
    import scipy.sparse as sp

    n = len(labels)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    total_budget = budget_per_class * n_classes

    # Use sparse CPU path if forced or if kernel is sparse and too large
    if sp.issparse(kernel) and (sparse_cpu or n > 50_000):
        if verbose:
            print(f"  [{method_name}] Sparse CPU path (n={n:,}, nnz={kernel.nnz:,})")
        return _greedy_facility_global_sparse(
            kernel, labels, budget_per_class, importance,
            verbose, _verbose_level, method_name)

    if sp.issparse(kernel):
        K = torch.from_numpy(kernel.toarray().astype(np.float32)).to(device)
    elif isinstance(kernel, np.ndarray):
        K = torch.from_numpy(kernel.astype(np.float32)).to(device)
    elif isinstance(kernel, torch.Tensor):
        K = kernel.float().to(device)
    else:
        raise ValueError(f"Unsupported kernel type: {type(kernel)}")

    imp_t = torch.from_numpy(importance.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)
    eligible = torch.ones(n, dtype=torch.bool, device=device)

    selected = []
    class_counts = {int(c): 0 for c in unique_classes}
    max_coverage = torch.zeros(n, device=device)

    for iter_idx in range(total_budget):
        # Marginal gain: how much does adding candidate j improve total coverage?
        marginal = (K - max_coverage.unsqueeze(1)).clamp(min=0)
        gains = (marginal * imp_t.unsqueeze(1)).sum(dim=0)
        gains[~eligible] = float('-inf')

        best_j = torch.argmax(gains).item()
        if gains[best_j] == float('-inf'):
            if verbose:
                print(f"  [{method_name}] Early stop at iter={iter_idx}: no eligible candidates")
            break

        selected.append(best_j)
        c = int(labels[best_j])
        class_counts[c] += 1
        eligible[best_j] = False

        # Disable class when budget is met
        if class_counts[c] >= budget_per_class:
            eligible[labels_t == c] = False

        max_coverage = torch.maximum(max_coverage, K[:, best_j])

        if verbose and _verbose_level >= 3 and (iter_idx < 10 or iter_idx % 50 == 0):
            coverage_pct = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
            print(f"    [{method_name}] iter={iter_idx+1}: idx={best_j}, class={c}, "
                  f"gain={gains[best_j].item():.4f}, coverage={coverage_pct:.4f}")

    if verbose:
        final_coverage = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
        print(f"  [{method_name}] Global: selected={len(selected)}, coverage={final_coverage:.4f}")
        print(f"  [{method_name}] Class distribution: {dict(class_counts)}")

    return selected


# =============================================================================
# Methods that IGNORE importance (self-contained logic)
# =============================================================================

@register_method('random', needs=set(), importance='ignore', description="Random balanced selection (equal per class)")
def _select_random(
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Random balanced selection - equal samples per class."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        k = min(budget_per_class, len(class_indices))
        selected.extend(np.random.choice(class_indices, k, replace=False))
    return selected


@register_method('random_proportional', needs=set(), importance='ignore',
                description="Random proportional selection (preserves class distribution)")
def _select_random_proportional(
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Random proportional selection - preserves original class distribution."""
    np.random.seed(seed)
    n_classes = len(np.unique(labels))
    total_budget = budget_per_class * n_classes
    # Pure random sampling without class stratification
    indices = np.random.choice(len(labels), total_budget, replace=False)
    return indices.tolist()


@register_method('eva', needs={'eva_scores'}, importance='ignore', description="EVA: high dual-window variance samples")
def _select_eva(
    eva_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """
    Select samples with highest EVA scores (per-class balanced).

    Note: The EVA paper describes "global top-M" selection, but this creates
    severe class imbalance. Per-class selection is required to match their
    reported results (~97% accuracy). The paper's description appears incomplete.
    """
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = eva_scores[class_indices]
        k = min(budget_per_class, len(class_indices))
        top_k_local = np.argsort(class_scores)[-k:]
        selected.extend(class_indices[top_k_local])
    return selected


@register_method('aum', needs={'aum_scores'}, importance='ignore', description="AUM: low margin samples (hard/noisy)")
def _select_aum(
    aum_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """
    Select samples with lowest AUM scores (per-class balanced).

    AUM (Area Under Margin) = average of (P(true_class) - max(P(other_class))) across epochs.
    - Low AUM = sample is often misclassified or has low confidence = hard/noisy
    - High AUM = sample is consistently correct with high margin = easy

    Following EVA paper convention: select LOW AUM samples (hard examples).

    Reference: Pleiss et al., "Identifying Mislabeled Data using the Area Under the Margin Ranking"
    """
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = aum_scores[class_indices]
        k = min(budget_per_class, len(class_indices))
        # Select LOWEST AUM (hardest samples)
        bottom_k_local = np.argsort(class_scores)[:k]
        selected.extend(class_indices[bottom_k_local])
    return selected


@register_method('forgetting', needs={'forgetting_scores'}, importance='ignore', description="Forgetting: high forgetting event samples")
def _select_forgetting(
    forgetting_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """
    Select samples with highest forgetting scores (per-class balanced).

    Forgetting score = count of correct→incorrect transitions during training.
    - High forgetting = sample is frequently "forgotten" = important for learning
    - Zero forgetting = sample is either always correct (easy) or always wrong (noise)

    Following EVA paper convention: select HIGH forgetting samples.

    Reference: Toneva et al., "An Empirical Study of Example Forgetting during Deep Neural Network Learning"
    """
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = forgetting_scores[class_indices]
        k = min(budget_per_class, len(class_indices))
        # Select HIGHEST forgetting (most forgotten samples)
        top_k_local = np.argsort(class_scores)[-k:]
        selected.extend(class_indices[top_k_local])
    return selected


@register_method('el2n_top', needs={'el2n_scores'}, importance='ignore', description="Hardest samples (high EL2N)")
def _select_el2n_top(
    el2n_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select hardest samples (highest EL2N)."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = el2n_scores[class_indices]
        sorted_order = np.argsort(class_scores)
        k = min(budget_per_class, len(class_indices))
        selected_local = sorted_order[-k:]
        selected.extend(class_indices[selected_local])
    return selected


@register_method('el2n_bottom', needs={'el2n_scores'}, importance='ignore', description="Easiest samples (low EL2N)")
def _select_el2n_bottom(
    el2n_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select easiest samples (lowest EL2N)."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = el2n_scores[class_indices]
        sorted_order = np.argsort(class_scores)
        k = min(budget_per_class, len(class_indices))
        selected_local = sorted_order[:k]
        selected.extend(class_indices[selected_local])
    return selected


@register_method('el2n_mid', needs={'el2n_scores'}, importance='ignore', description="Ambiguous samples (mid EL2N)")
def _select_el2n_mid(
    el2n_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select ambiguous samples (middle EL2N)."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = el2n_scores[class_indices]
        sorted_order = np.argsort(class_scores)
        n_class = len(class_indices)
        k = min(budget_per_class, n_class)
        mid_start = (n_class - k) // 2
        selected_local = sorted_order[mid_start:mid_start + k]
        selected.extend(class_indices[selected_local])
    return selected


@register_method('easy_samples', needs={'variance_stats'}, importance='ignore', description="Low variance, high confidence")
def _select_easy_samples(
    variance_stats: Tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select easy samples: low variance, high confidence."""
    np.random.seed(seed)
    conf_var, conf_mean = variance_stats
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_var = conf_var[class_indices]
        class_conf = conf_mean[class_indices]
        score = -class_var + class_conf
        sorted_order = np.argsort(score)
        k = min(budget_per_class, len(class_indices))
        selected_local = sorted_order[-k:]
        selected.extend(class_indices[selected_local])
    return selected


@register_method('high_variance', needs={'variance_stats'}, importance='ignore', description="High variance samples")
def _select_high_variance(
    variance_stats: Tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select high variance samples (ambiguous)."""
    np.random.seed(seed)
    conf_var, _ = variance_stats
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_var = conf_var[class_indices]
        sorted_order = np.argsort(class_var)
        k = min(budget_per_class, len(class_indices))
        selected_local = sorted_order[-k:]
        selected.extend(class_indices[selected_local])
    return selected


@register_method('prototype', needs={'embeddings'}, importance='ignore', description="Samples closest to class centroid")
def _select_prototype(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select samples closest to class centroid."""
    np.random.seed(seed)
    centroids = compute_class_centroids(embeddings, labels)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        k = min(budget_per_class, len(class_indices))
        centroid = centroids[c]
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:k]
        selected.extend(class_indices[closest_indices])
    return selected


@register_method('anti_prototype', needs={'embeddings'}, importance='ignore', description="Samples farthest from centroid")
def _select_anti_prototype(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select samples farthest from class centroid."""
    np.random.seed(seed)
    centroids = compute_class_centroids(embeddings, labels)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        k = min(budget_per_class, len(class_indices))
        centroid = centroids[c]
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        farthest_indices = np.argsort(distances)[-k:]
        selected.extend(class_indices[farthest_indices])
    return selected


@register_method('graph_density', needs={'embeddings'}, importance='ignore',
                kwargs={'k_neighbors': 10}, description="Select by local density in k-NN graph")
def _select_graph_density(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    k_neighbors: int = 10,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select samples by local density in k-NN graph."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        n_class = len(class_indices)
        k_budget = min(budget_per_class, n_class)
        if k_budget >= n_class:
            selected.extend(class_indices)
            continue
        _, knn_distances = build_knn_graph(class_embeddings, k_neighbors)
        mean_dist = knn_distances.mean(axis=1)
        density = 1.0 / (mean_dist + 1e-8)
        top_k_indices = np.argsort(density)[-k_budget:]
        selected.extend(class_indices[top_k_indices])
    return selected


# =============================================================================
# Methods that OPTIONALLY use importance (default to uniform)
# =============================================================================

@register_method('fps', needs={'embeddings'}, importance='optional', description="Farthest Point Sampling")
def _select_fps(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    **kwargs
) -> List[int]:
    """
    FPS within each class.

    When importance is non-uniform, uses importance-weighted FPS:
    Priority = (1 - α) × min_distance + α × importance
    """
    np.random.seed(seed)
    selected = []

    # Check if importance is uniform
    is_uniform = np.allclose(importance, importance[0])

    if verbose:
        print(f"  [FPS] is_uniform={is_uniform}, importance range=[{importance.min():.4f}, {importance.max():.4f}]")
        if not is_uniform:
            imp_percentiles = np.percentile(importance, [25, 50, 75, 90, 99])
            print(f"  [FPS] importance percentiles [25,50,75,90,99]: {[f'{p:.4f}' for p in imp_percentiles]}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        k = min(budget_per_class, len(class_indices))

        if k >= len(class_indices):
            selected.extend(class_indices)
            continue

        if is_uniform:
            # Standard FPS
            fps_indices = farthest_point_sampling(class_embeddings, k, seed=seed + c)
            selected.extend(class_indices[fps_indices])
            if verbose and _verbose_level >= 2:
                print(f"    [FPS] class={c}: selected {k} via standard FPS")
        else:
            # Importance-weighted FPS
            class_importance = importance[class_indices]
            imp_min, imp_max = class_importance.min(), class_importance.max()
            if imp_max - imp_min > 1e-8:
                imp_norm = (class_importance - imp_min) / (imp_max - imp_min)
            else:
                imp_norm = np.ones(len(class_indices))

            imp_t = torch.from_numpy(imp_norm).float().to(device)
            emb_t = torch.from_numpy(class_embeddings).float().to(device)
            emb_t = F.normalize(emb_t, dim=1)

            importance_weight = 0.5  # Balance between diversity and importance

            first_idx = torch.argmax(imp_t).item()
            chosen = [first_idx]
            min_distances = 1.0 - emb_t @ emb_t[first_idx]
            min_distances[first_idx] = float('-inf')

            dist_max = min_distances[min_distances > float('-inf')].max()
            if dist_max > 1e-8:
                min_distances_norm = min_distances / dist_max
            else:
                min_distances_norm = min_distances

            if verbose and _verbose_level >= 2:
                print(f"    [FPS] class={c}: n={len(class_indices)}, k={k}, importance_weight={importance_weight}")
                print(f"    [FPS] class={c}: class_importance range=[{imp_min:.4f}, {imp_max:.4f}]")
                print(f"    [FPS] class={c}: first_idx={first_idx} (highest importance)")

            for iter_idx in range(k - 1):
                priority = (1 - importance_weight) * min_distances_norm + importance_weight * imp_t
                priority[chosen] = float('-inf')
                next_idx = torch.argmax(priority).item()
                chosen.append(next_idx)
                new_distances = 1.0 - emb_t @ emb_t[next_idx]
                min_distances = torch.minimum(min_distances, new_distances)
                dist_max = min_distances[min_distances > float('-inf')].max()
                if dist_max > 1e-8:
                    min_distances_norm = min_distances / dist_max

                if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                    valid_priority = priority[priority > float('-inf')]
                    print(f"      [FPS] iter={iter_idx+1}: selected idx={next_idx}, "
                          f"priority={priority[next_idx].item():.4f}, "
                          f"imp={imp_t[next_idx].item():.4f}, "
                          f"dist={min_distances_norm[next_idx].item():.4f}")

            selected.extend(class_indices[chosen])

            if verbose and _verbose_level >= 2:
                chosen_importance = imp_t[chosen].cpu().numpy()
                print(f"    [FPS] class={c}: selected importance mean={chosen_importance.mean():.4f}, "
                      f"std={chosen_importance.std():.4f}")

    return selected


@register_method('facility', needs={'embeddings'}, importance='optional', description="Greedy Facility Location")
def _select_facility(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs
) -> List[int]:
    """
    Greedy facility location within each class (or globally with --global).

    Coverage objective: f(S) = Σᵢ importance[i] × max_{j∈S} sim(i,j)
    With uniform importance, equivalent to standard facility location.
    """
    np.random.seed(seed)
    is_uniform = np.allclose(importance, importance[0])

    if global_selection:
        if verbose:
            print(f"  [Facility] Global facility location: n={len(embeddings)}, is_uniform={is_uniform}")
        emb_t = torch.from_numpy(embeddings).float().to(device)
        emb_t = F.normalize(emb_t, dim=1)
        similarity = emb_t @ emb_t.T
        return _greedy_facility_global(
            similarity, labels, budget_per_class, importance,
            verbose, _verbose_level, 'Facility',
            sparse_cpu=kwargs.get('sparse_cpu', False))

    selected = []

    if verbose:
        print(f"  [Facility] is_uniform={is_uniform}, importance range=[{importance.min():.4f}, {importance.max():.4f}]")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        k = min(budget_per_class, len(class_indices))
        n = len(class_indices)

        if k >= n:
            selected.extend(class_indices)
            continue

        class_emb_t = torch.from_numpy(embeddings[class_indices]).float().to(device)
        class_emb_t = F.normalize(class_emb_t, dim=1)
        similarity = class_emb_t @ class_emb_t.T

        class_importance = importance[class_indices]
        imp_t = torch.from_numpy(class_importance).float().to(device)

        max_sim_to_selected = torch.zeros(n, device=device)
        chosen = []

        if verbose and _verbose_level >= 2:
            sim_stats = similarity[similarity < 1.0]  # Exclude self-similarity
            print(f"    [Facility] class={c}: n={n}, k={k}, "
                  f"sim_range=[{sim_stats.min().item():.4f}, {sim_stats.max().item():.4f}], "
                  f"sim_mean={sim_stats.mean().item():.4f}")

        total_coverage = 0.0
        for iter_idx in range(k):
            marginal_sim = (similarity - max_sim_to_selected.unsqueeze(1)).clamp(min=0)
            gains = (marginal_sim * imp_t.unsqueeze(1)).sum(dim=0)
            if chosen:
                gains[chosen] = float('-inf')
            best = torch.argmax(gains).item()
            best_gain = gains[best].item()
            chosen.append(best)
            max_sim_to_selected = torch.maximum(max_sim_to_selected, similarity[:, best])
            total_coverage += best_gain

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                coverage_pct = (max_sim_to_selected * imp_t).sum().item() / imp_t.sum().item()
                print(f"      [Facility] iter={iter_idx+1}: selected idx={best}, "
                      f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            final_coverage = (max_sim_to_selected * imp_t).sum().item() / imp_t.sum().item()
            chosen_importance = imp_t[chosen].cpu().numpy()
            print(f"    [Facility] class={c}: final_coverage={final_coverage:.4f}, "
                  f"selected_imp_mean={chosen_importance.mean():.4f}")

    return selected


@register_method('herding', needs={'embeddings'}, importance='optional', description="Herding (mean-matching)")
def _select_herding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    **kwargs
) -> List[int]:
    """
    Herding selects samples so the subset mean matches the full class mean.

    At each step, pick the sample that minimizes the distance between
    the running subset mean and the full class mean.

    Ref: Welling (2009), "Herding Dynamic Weights to Learn";
         Chen et al. (2012) for coreset application.
    """
    np.random.seed(seed)
    selected = []

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_emb = embeddings[class_indices]
        k = min(budget_per_class, len(class_indices))

        if k >= len(class_indices):
            selected.extend(class_indices)
            continue

        # L2-normalize for consistency with other methods
        norms = np.linalg.norm(class_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        class_emb = class_emb / norms

        class_mean = class_emb.mean(axis=0)
        chosen = []
        running_sum = np.zeros_like(class_mean)

        for i in range(k):
            # Target: (i+1) * class_mean
            # We want argmin || running_sum + x_j - (i+1)*class_mean ||
            # = argmax <x_j, (i+1)*class_mean - running_sum>
            residual = (i + 1) * class_mean - running_sum
            scores = class_emb @ residual
            # Exclude already chosen
            for j in chosen:
                scores[j] = float('-inf')
            best = np.argmax(scores)
            chosen.append(best)
            running_sum += class_emb[best]

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            subset_mean = running_sum / k
            cosine = np.dot(subset_mean, class_mean) / (
                np.linalg.norm(subset_mean) * np.linalg.norm(class_mean) + 1e-8)
            print(f"    [Herding] class={c}: n={len(class_indices)}, k={k}, "
                  f"mean_cosine={cosine:.4f}")

    if verbose:
        print(f"  [Herding] Total selected: {len(selected)}")

    return selected


@register_method('graph_fps', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 2}, description="Graph-constrained FPS")
def _select_graph_fps(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    **kwargs
) -> List[int]:
    """Graph-constrained FPS. Non-neighbors are treated as infinitely far."""
    np.random.seed(seed)
    selected = []
    is_uniform = np.allclose(importance, importance[0])

    if verbose:
        print(f"  [GraphFPS] k_neighbors={k_neighbors}, k_hops={k_hops}, is_uniform={is_uniform}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        n_class = len(class_indices)
        k_budget = min(budget_per_class, n_class)

        if k_budget >= n_class:
            selected.extend(class_indices)
            continue

        knn_indices, _ = build_knn_graph(class_embeddings, k_neighbors)
        k_hop_adj = compute_k_hop_adjacency(knn_indices, k_hops, n_class)

        emb_t = torch.from_numpy(class_embeddings).float().to(device)
        emb_t = F.normalize(emb_t, dim=1)
        cos_sim = emb_t @ emb_t.T
        cos_dist = 1.0 - cos_sim

        large_dist = 2.0
        graph_dist = torch.where(k_hop_adj, cos_dist, torch.full_like(cos_dist, large_dist))

        if verbose and _verbose_level >= 2:
            adj_density = k_hop_adj.float().mean().item()
            reachable_dists = graph_dist[k_hop_adj]
            print(f"    [GraphFPS] class={c}: n={n_class}, k={k_budget}, "
                  f"adj_density={adj_density:.4f}, "
                  f"reachable_dist_mean={reachable_dists.mean().item():.4f}")

        if is_uniform:
            first_idx = np.random.randint(n_class)
        else:
            class_importance = importance[class_indices]
            first_idx = np.argmax(class_importance)

        chosen = [first_idx]
        min_dist_to_selected = graph_dist[:, first_idx].clone()

        for iter_idx in range(k_budget - 1):
            min_dist_to_selected[chosen] = float('-inf')
            next_idx = torch.argmax(min_dist_to_selected).item()
            chosen.append(next_idx)
            min_dist_to_selected = torch.minimum(min_dist_to_selected, graph_dist[:, next_idx])

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                valid_dists = min_dist_to_selected[min_dist_to_selected > float('-inf')]
                print(f"      [GraphFPS] iter={iter_idx+1}: idx={next_idx}, "
                      f"max_min_dist={valid_dists.max().item():.4f}")

        selected.extend(class_indices[chosen])

    return selected


@register_method('graph_facility', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 2}, description="Graph-constrained facility location")
def _select_graph_facility(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    **kwargs
) -> List[int]:
    """Graph-constrained facility location. Similarity only counts between k-hop neighbors."""
    np.random.seed(seed)
    selected = []

    is_uniform = np.allclose(importance, importance[0])
    if verbose:
        print(f"  [GraphFacility] k_neighbors={k_neighbors}, k_hops={k_hops}, is_uniform={is_uniform}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        n_class = len(class_indices)
        k_budget = min(budget_per_class, n_class)

        if k_budget >= n_class:
            selected.extend(class_indices)
            continue

        knn_indices, _ = build_knn_graph(class_embeddings, k_neighbors)
        k_hop_adj = compute_k_hop_adjacency(knn_indices, k_hops, n_class)

        emb_t = torch.from_numpy(class_embeddings).float().to(device)
        emb_t = F.normalize(emb_t, dim=1)
        similarity = emb_t @ emb_t.T
        masked_similarity = similarity * k_hop_adj.float()

        class_importance = importance[class_indices]
        imp_t = torch.from_numpy(class_importance).float().to(device)

        if verbose and _verbose_level >= 2:
            adj_density = k_hop_adj.float().mean().item()
            masked_sim_vals = masked_similarity[masked_similarity > 0]
            print(f"    [GraphFacility] class={c}: n={n_class}, k={k_budget}, "
                  f"adj_density={adj_density:.4f}, "
                  f"masked_sim_mean={masked_sim_vals.mean().item():.4f}")

        max_coverage = torch.zeros(n_class, device=device)
        chosen = []

        for iter_idx in range(k_budget):
            marginal = (masked_similarity - max_coverage.unsqueeze(1)).clamp(min=0)
            gains = (marginal * imp_t.unsqueeze(1)).sum(dim=0)
            if chosen:
                gains[chosen] = float('-inf')
            best = torch.argmax(gains).item()
            best_gain = gains[best].item()
            chosen.append(best)
            max_coverage = torch.maximum(max_coverage, masked_similarity[:, best])

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                coverage_pct = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
                print(f"      [GraphFacility] iter={iter_idx+1}: idx={best}, "
                      f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            final_coverage = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
            print(f"    [GraphFacility] class={c}: final_coverage={final_coverage:.4f}")

    return selected


# =============================================================================
# Methods that REQUIRE importance
# =============================================================================

@register_method('greedy_importance', needs=set(), importance='required', description="Select highest importance samples")
def _select_greedy_importance(
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Greedy selection by importance: select highest importance samples per class."""
    np.random.seed(seed)
    selected = []
    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_importance = importance[class_indices]
        k = min(budget_per_class, len(class_indices))
        top_k_local = np.argsort(class_importance)[-k:]
        selected.extend(class_indices[top_k_local])
    return selected


@register_method('graph_coverage', needs={'embeddings'}, importance='required',
                kwargs={'k_neighbors': 10, 'walk_length': 5, 'gamma': 0.85},
                description="Graph coverage with random walk reachability")
def _select_graph_coverage(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    walk_length: int = 5,
    gamma: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs
) -> List[int]:
    """Graph-based coverage maximization with random walk reachability."""
    from .graph import build_adjacency_matrix, compute_reachability_matrix

    np.random.seed(seed)

    if global_selection:
        n = len(embeddings)
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        total_budget = budget_per_class * n_classes

        if verbose:
            print(f"  [GraphCoverage] Global graph coverage: k={k_neighbors}, "
                  f"walk_length={walk_length}, gamma={gamma}, n={n}")
        knn_indices, knn_distances = build_knn_graph(
            embeddings, k_neighbors, verbose=(verbose and _verbose_level >= 2))
        graph = build_adjacency_matrix(
            knn_indices, knn_distances, n,
            verbose=(verbose and _verbose_level >= 2))
        reachability = compute_reachability_matrix(
            graph, walk_length, gamma,
            verbose=(verbose and _verbose_level >= 2))

        reach_dense = torch.from_numpy(reachability.toarray().astype(np.float32)).to(device)
        imp_t = torch.from_numpy(importance.astype(np.float32)).to(device)
        labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)
        eligible = torch.ones(n, dtype=torch.bool, device=device)
        covered = torch.zeros(n, device=device)
        selected = []
        class_counts = {int(c): 0 for c in unique_classes}

        for iter_idx in range(total_budget):
            uncovered = (1 - covered) * imp_t
            gains = reach_dense @ uncovered
            gains[~eligible] = float('-inf')

            best_j = torch.argmax(gains).item()
            if gains[best_j] == float('-inf'):
                if verbose:
                    print(f"  [GraphCoverage] Early stop at iter={iter_idx}: no eligible candidates")
                break

            selected.append(best_j)
            c = int(labels[best_j])
            class_counts[c] += 1
            eligible[best_j] = False
            if class_counts[c] >= budget_per_class:
                eligible[labels_t == c] = False
            covered = torch.maximum(covered, reach_dense[best_j])

            if verbose and _verbose_level >= 3 and (iter_idx < 10 or iter_idx % 50 == 0):
                coverage_pct = (covered * imp_t).sum().item() / imp_t.sum().item()
                print(f"    [GraphCoverage] iter={iter_idx+1}: idx={best_j}, class={c}, "
                      f"gain={gains[best_j].item():.4f}, coverage={coverage_pct:.4f}")

        if verbose:
            final_coverage = (covered * imp_t).sum().item() / imp_t.sum().item()
            print(f"  [GraphCoverage] Global: selected={len(selected)}, coverage={final_coverage:.4f}")
            print(f"  [GraphCoverage] Class distribution: {dict(class_counts)}")

        return selected

    selected = []

    if verbose:
        print(f"  [GraphCoverage] Per-class graph coverage: k={k_neighbors}, "
              f"walk_length={walk_length}, gamma={gamma}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        class_importance = importance[class_indices]
        n_class = len(class_indices)
        k = min(budget_per_class, n_class)

        if k >= n_class:
            selected.extend(class_indices)
            continue

        knn_indices, knn_distances = build_knn_graph(class_embeddings, k_neighbors)
        graph = build_adjacency_matrix(knn_indices, knn_distances, n_class)
        reachability = compute_reachability_matrix(graph, walk_length, gamma, verbose=(verbose and _verbose_level >= 2))

        if verbose and _verbose_level >= 2:
            nnz = reachability.nnz
            sparsity = 1.0 - nnz / (n_class * n_class)
            print(f"    [GraphCoverage] class={c}: n={n_class}, k={k}, "
                  f"reachability_nnz={nnz}, sparsity={sparsity:.4f}")

        # Convert to dense once for vectorized operations
        reach_dense = torch.from_numpy(reachability.toarray().astype(np.float32)).to(device)
        imp_t = torch.from_numpy(class_importance.astype(np.float32)).to(device)
        covered = torch.zeros(n_class, device=device)
        eligible = torch.ones(n_class, dtype=torch.bool, device=device)
        chosen = []

        for iter_idx in range(k):
            uncovered = (1 - covered) * imp_t
            # Vectorized: compute gains for all candidates at once
            gains = reach_dense @ uncovered
            gains[~eligible] = float('-inf')

            best_idx = torch.argmax(gains).item()
            best_gain = gains[best_idx].item()

            if best_gain == float('-inf'):
                if verbose:
                    print(f"    [GraphCoverage] class={c}: early stop at iter={iter_idx}")
                break

            chosen.append(best_idx)
            eligible[best_idx] = False
            covered = torch.maximum(covered, reach_dense[best_idx])

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                coverage_pct = (covered * imp_t).sum().item() / imp_t.sum().item()
                print(f"      [GraphCoverage] iter={iter_idx+1}: idx={best_idx}, "
                      f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            final_coverage = (covered * imp_t).sum().item() / imp_t.sum().item()
            print(f"    [GraphCoverage] class={c}: final_coverage={final_coverage:.4f}")

    return selected


@register_method('graph_a1', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 1},
                description="A symmetric coverage (1-hop only)")
@register_method('graph_a2', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 2},
                description="A+A² symmetric coverage (heat kernel decomposition)")
@register_method('graph_a3', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 3},
                description="A+A²+A³ symmetric coverage (3-hop)")
@register_method('graph_a4', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 4},
                description="Σ A^i up to 4-hop")
@register_method('graph_a5', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 'k_hops': 5},
                description="Σ A^i up to 5-hop")
def _select_graph_a2(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs
) -> List[int]:
    """
    Graph coverage using K = Σ_{i=1}^{k_hops} A_sym^i (symmetric normalized adjacency).

    From heat kernel decomposition: polynomial terms with symmetric normalization.
    k_hops=2 matches full heat kernel performance. No decay, no eigendecomposition needed.
    """
    from .graph import build_adjacency_matrix
    from scipy.sparse import diags

    np.random.seed(seed)
    is_uniform = np.allclose(importance, importance[0])

    if global_selection:
        n = len(embeddings)
        if verbose:
            print(f"  [GraphA{k_hops}] Global graph, global selection: k={k_neighbors}, hops={k_hops}, n={n}, is_uniform={is_uniform}")

        # Build global k-NN graph (cross-class edges included)
        knn_indices, knn_distances = build_knn_graph(
            embeddings, k_neighbors, verbose=(verbose and _verbose_level >= 2))
        A = build_adjacency_matrix(knn_indices, knn_distances, n)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
        A_sym = D_inv_sqrt @ A @ D_inv_sqrt

        # K = Σ_{i=1}^{k_hops} A_sym^i
        K = A_sym.copy()
        A_power = A_sym.copy()
        for _ in range(k_hops - 1):
            A_power = A_power @ A_sym
            K = K + A_power

        if verbose and _verbose_level >= 2:
            print(f"    [GraphA{k_hops}] Global K: nnz={K.nnz}, sparsity={1.0 - K.nnz / (n ** 2):.4f}")

        # Global greedy selection with per-class budget constraints
        # Coverage is measured over ALL N samples (cross-class included),
        # so boundary samples get credit for covering nearby other-class samples.
        return _greedy_facility_global(
            kernel=K,
            labels=labels,
            budget_per_class=budget_per_class,
            importance=importance,
            verbose=verbose,
            _verbose_level=_verbose_level,
            method_name=f'GraphA{k_hops}',
            sparse_cpu=kwargs.get('sparse_cpu', False),
        )

    selected = []

    if verbose:
        print(f"  [GraphA{k_hops}] Per-class coverage: k={k_neighbors}, hops={k_hops}, is_uniform={is_uniform}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        class_importance = importance[class_indices]
        n_class = len(class_indices)
        k = min(budget_per_class, n_class)

        if k >= n_class:
            selected.extend(class_indices)
            continue

        knn_indices, knn_distances = build_knn_graph(class_embeddings, k_neighbors)
        A = build_adjacency_matrix(knn_indices, knn_distances, n_class)

        # Symmetric normalization: A_sym = D^(-1/2) A D^(-1/2)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
        A_sym = D_inv_sqrt @ A @ D_inv_sqrt

        # K = Σ_{i=1}^{k_hops} A_sym^i
        K = A_sym.copy()
        A_power = A_sym.copy()
        for _ in range(k_hops - 1):
            A_power = A_power @ A_sym
            K = K + A_power

        if verbose and _verbose_level >= 2:
            nnz = K.nnz
            sparsity = 1.0 - nnz / (n_class * n_class)
            print(f"    [GraphA{k_hops}] class={c}: n={n_class}, k={k}, "
                  f"K_nnz={nnz}, sparsity={sparsity:.4f}")

        # Greedy facility location on K
        K_dense = torch.from_numpy(K.toarray().astype(np.float32)).to(device)
        imp_t = torch.from_numpy(class_importance.astype(np.float32)).to(device)
        max_coverage = torch.zeros(n_class, device=device)
        eligible = torch.ones(n_class, dtype=torch.bool, device=device)
        chosen = []

        for iter_idx in range(k):
            marginal = (K_dense - max_coverage.unsqueeze(1)).clamp(min=0)
            gains = (marginal * imp_t.unsqueeze(1)).sum(dim=0)
            gains[~eligible] = float('-inf')

            best_idx = torch.argmax(gains).item()
            best_gain = gains[best_idx].item()

            if best_gain == float('-inf'):
                if verbose:
                    print(f"    [GraphA{k_hops}] class={c}: early stop at iter={iter_idx}")
                break

            chosen.append(best_idx)
            eligible[best_idx] = False
            max_coverage = torch.maximum(max_coverage, K_dense[:, best_idx])

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                coverage_pct = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
                print(f"      [GraphA{k_hops}] iter={iter_idx+1}: idx={best_idx}, "
                      f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            final_coverage = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
            chosen_importance = imp_t[chosen].cpu().numpy()
            print(f"    [GraphA{k_hops}] class={c}: final_coverage={final_coverage:.4f}, "
                  f"selected_imp_mean={chosen_importance.mean():.4f}")

    return selected


@register_method('heat_kernel', needs={'embeddings'}, importance='optional',
                kwargs={'k_neighbors': 10, 't': 1.0},
                description="Heat kernel exp(-tL) via eigendecomposition")
def _select_heat_kernel(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    t: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs
) -> List[int]:
    """
    Graph coverage using heat kernel K = exp(-tL) via eigendecomposition.

    Full heat kernel on normalized Laplacian. O(n³) per class due to eigendecomposition.
    Used as ablation baseline to validate that A+A² (graph_a2) loses nothing.
    """
    from .graph import build_adjacency_matrix
    from scipy.sparse import diags

    np.random.seed(seed)
    is_uniform = np.allclose(importance, importance[0])

    if global_selection:
        n = len(embeddings)
        if verbose:
            print(f"  [HeatKernel] Global heat kernel coverage: k={k_neighbors}, t={t}, n={n}, is_uniform={is_uniform}")
        knn_indices, knn_distances = build_knn_graph(
            embeddings, k_neighbors, verbose=(verbose and _verbose_level >= 2))
        A = build_adjacency_matrix(knn_indices, knn_distances, n)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
        A_sym = D_inv_sqrt @ A @ D_inv_sqrt
        A_sym_dense = A_sym.toarray().astype(np.float64)
        L = np.eye(n) - A_sym_dense
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        exp_eigenvalues = np.exp(-t * eigenvalues)
        K = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
        if verbose and _verbose_level >= 2:
            print(f"    [HeatKernel] Global: eigenvalue_range=[{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        return _greedy_facility_global(
            K, labels, budget_per_class, importance,
            verbose, _verbose_level, 'HeatKernel',
            sparse_cpu=kwargs.get('sparse_cpu', False))

    selected = []

    if verbose:
        print(f"  [HeatKernel] Per-class heat kernel coverage: k={k_neighbors}, t={t}, is_uniform={is_uniform}")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        class_importance = importance[class_indices]
        n_class = len(class_indices)
        k = min(budget_per_class, n_class)

        if k >= n_class:
            selected.extend(class_indices)
            continue

        knn_indices, knn_distances = build_knn_graph(class_embeddings, k_neighbors)
        A = build_adjacency_matrix(knn_indices, knn_distances, n_class)

        # Symmetric normalization: A_sym = D^(-1/2) A D^(-1/2)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
        A_sym = D_inv_sqrt @ A @ D_inv_sqrt

        # Normalized Laplacian: L = I - A_sym
        A_sym_dense = A_sym.toarray().astype(np.float64)
        L = np.eye(n_class) - A_sym_dense

        # Eigendecomposition: L = V Λ V^T
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Heat kernel: K = V diag(exp(-t*λ)) V^T
        exp_eigenvalues = np.exp(-t * eigenvalues)
        K = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T

        if verbose and _verbose_level >= 2:
            print(f"    [HeatKernel] class={c}: n={n_class}, k={k}, "
                  f"eigenvalue_range=[{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

        # Greedy facility location on K
        K_t = torch.from_numpy(K.astype(np.float32)).to(device)
        imp_t = torch.from_numpy(class_importance.astype(np.float32)).to(device)
        max_coverage = torch.zeros(n_class, device=device)
        eligible = torch.ones(n_class, dtype=torch.bool, device=device)
        chosen = []

        for iter_idx in range(k):
            marginal = (K_t - max_coverage.unsqueeze(1)).clamp(min=0)
            gains = (marginal * imp_t.unsqueeze(1)).sum(dim=0)
            gains[~eligible] = float('-inf')

            best_idx = torch.argmax(gains).item()
            best_gain = gains[best_idx].item()

            if best_gain == float('-inf'):
                if verbose:
                    print(f"    [HeatKernel] class={c}: early stop at iter={iter_idx}")
                break

            chosen.append(best_idx)
            eligible[best_idx] = False
            max_coverage = torch.maximum(max_coverage, K_t[:, best_idx])

            if verbose and _verbose_level >= 3 and (iter_idx < 5 or iter_idx % 10 == 0):
                coverage_pct = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
                print(f"      [HeatKernel] iter={iter_idx+1}: idx={best_idx}, "
                      f"gain={best_gain:.4f}, coverage={coverage_pct:.4f}")

        selected.extend(class_indices[chosen])

        if verbose and _verbose_level >= 2:
            final_coverage = (max_coverage * imp_t).sum().item() / imp_t.sum().item()
            chosen_importance = imp_t[chosen].cpu().numpy()
            print(f"    [HeatKernel] class={c}: final_coverage={final_coverage:.4f}, "
                  f"selected_imp_mean={chosen_importance.mean():.4f}")

    return selected


@register_method('global_influence', needs={'embeddings'}, importance='required',
                kwargs={'k_neighbors': 10, 'walk_length': 5, 'gamma': 0.85, 'coverage_mode': 'prob'},
                description="Global graph influence with class balance constraint")
def _select_global_influence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    walk_length: int = 5,
    gamma: float = 0.85,
    coverage_mode: str = 'prob',
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    **kwargs
) -> List[int]:
    """
    Global graph influence-based selection with class balance constraint.

    Builds GLOBAL k-NN graph across ALL samples and uses random walk reachability.

    Coverage modes:
    - 'prob': Probabilistic OR. Redundancy has diminishing returns.
    - 'max': Max-based. Only best-covering sample counts.
    """
    from .graph import build_knn_graph, build_adjacency_matrix, compute_reachability_matrix

    np.random.seed(seed)
    n = len(embeddings)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    total_budget = budget_per_class * n_classes

    if verbose:
        print(f"  [GlobalInfluence] Building GLOBAL graph: n={n}, k={k_neighbors}, "
              f"walk_length={walk_length}, gamma={gamma}, mode={coverage_mode}")
        print(f"  [GlobalInfluence] Budget: {budget_per_class}/class × {n_classes} classes = {total_budget} total")
        imp_percentiles = np.percentile(importance, [0, 25, 50, 75, 100])
        print(f"  [GlobalInfluence] Importance percentiles [0,25,50,75,100]: {[f'{p:.4f}' for p in imp_percentiles]}")

    knn_indices, knn_distances = build_knn_graph(embeddings, k_neighbors, verbose=verbose)
    graph = build_adjacency_matrix(knn_indices, knn_distances, n, verbose=verbose)
    reachability = compute_reachability_matrix(graph, walk_length, gamma, verbose=(verbose and _verbose_level >= 2))

    if verbose:
        nnz = reachability.nnz
        sparsity = 1.0 - nnz / (n * n)
        reach_data = reachability.data
        print(f"  [GlobalInfluence] Reachability matrix: nnz={nnz}, sparsity={sparsity:.4f}")
        print(f"  [GlobalInfluence] Reachability values: min={reach_data.min():.4f}, "
              f"max={reach_data.max():.4f}, mean={reach_data.mean():.4f}")

    reach_dense = torch.from_numpy(reachability.toarray().astype(np.float32)).to(device)
    row_max = reach_dense.max(dim=1, keepdim=True).values
    row_max = torch.where(row_max == 0, torch.ones_like(row_max), row_max)
    reach_dense = reach_dense / row_max

    importance_t = torch.from_numpy(importance.astype(np.float32)).to(device)
    eligible = torch.ones(n, dtype=torch.bool, device=device)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)

    selected = []
    class_counts = {c: 0 for c in unique_classes}

    if coverage_mode == 'prob':
        uncovered = importance_t.clone()
        initial_total = uncovered.sum().item()
        for iter_idx in range(total_budget):
            gains = reach_dense @ uncovered
            gains[~eligible] = float('-inf')
            best_j = torch.argmax(gains).item()
            if gains[best_j] == float('-inf'):
                if verbose:
                    print(f"  [GlobalInfluence] Early stop at iter={iter_idx}: no eligible candidates")
                break
            selected.append(best_j)
            c = labels[best_j]
            class_counts[c] += 1
            eligible[best_j] = False
            if class_counts[c] >= budget_per_class:
                eligible[labels_t == c] = False

            old_uncovered = uncovered.sum().item()
            uncovered = uncovered * (1 - reach_dense[best_j])
            new_uncovered = uncovered.sum().item()
            coverage_gain = old_uncovered - new_uncovered

            if verbose and _verbose_level >= 3 and (iter_idx < 10 or iter_idx % 50 == 0):
                coverage_pct = 1.0 - new_uncovered / initial_total
                print(f"    [GlobalInfluence] iter={iter_idx+1}: idx={best_j}, class={c}, "
                      f"gain={gains[best_j].item():.4f}, coverage_gain={coverage_gain:.4f}, "
                      f"total_coverage={coverage_pct:.4f}, class_counts={dict(class_counts)}")

        if verbose:
            final_coverage = 1.0 - uncovered.sum().item() / initial_total
            print(f"  [GlobalInfluence] Final: selected={len(selected)}, coverage={final_coverage:.4f}")
            print(f"  [GlobalInfluence] Class distribution: {dict(class_counts)}")

    elif coverage_mode == 'max':
        max_coverage = torch.zeros(n, device=device)
        for iter_idx in range(total_budget):
            marginal = (reach_dense - max_coverage.unsqueeze(0)).clamp(min=0)
            gains = (marginal * importance_t.unsqueeze(0)).sum(dim=1)
            gains[~eligible] = float('-inf')
            best_j = torch.argmax(gains).item()
            if gains[best_j] == float('-inf'):
                if verbose:
                    print(f"  [GlobalInfluence] Early stop at iter={iter_idx}: no eligible candidates")
                break
            selected.append(best_j)
            c = labels[best_j]
            class_counts[c] += 1
            eligible[best_j] = False
            if class_counts[c] >= budget_per_class:
                eligible[labels_t == c] = False
            max_coverage = torch.maximum(max_coverage, reach_dense[best_j])

            if verbose and _verbose_level >= 3 and (iter_idx < 10 or iter_idx % 50 == 0):
                current_coverage = (max_coverage * importance_t).sum().item() / importance_t.sum().item()
                print(f"    [GlobalInfluence] iter={iter_idx+1}: idx={best_j}, class={c}, "
                      f"gain={gains[best_j].item():.4f}, coverage={current_coverage:.4f}")

        if verbose:
            final_coverage = (max_coverage * importance_t).sum().item() / importance_t.sum().item()
            print(f"  [GlobalInfluence] Final: selected={len(selected)}, coverage={final_coverage:.4f}")
            print(f"  [GlobalInfluence] Class distribution: {dict(class_counts)}")
    else:
        raise ValueError(f"Unknown coverage_mode: {coverage_mode}. Use 'prob' or 'max'.")

    return selected


# =============================================================================
# Hybrid Methods (ignore importance - have their own logic)
# =============================================================================

@register_method('easy_then_diverse', needs={'embeddings', 'variance_stats'}, importance='ignore',
                kwargs={'easy_ratio': 0.5}, description="Easy samples + diverse samples")
def _select_easy_then_diverse(
    embeddings: np.ndarray,
    variance_stats: Tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    budget_per_class: int,
    easy_ratio: float = 0.5,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select easy samples first, then add diverse samples."""
    np.random.seed(seed)
    conf_var, conf_mean = variance_stats
    selected = []

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]
        class_var = conf_var[class_indices]
        class_conf = conf_mean[class_indices]

        k = min(budget_per_class, len(class_indices))
        k_easy = int(k * easy_ratio)
        k_diverse = k - k_easy

        easy_score = -class_var + class_conf
        easy_order = np.argsort(easy_score)
        easy_local = list(easy_order[-k_easy:]) if k_easy > 0 else []

        if k_diverse > 0:
            remaining_mask = np.ones(len(class_indices), dtype=bool)
            remaining_mask[easy_local] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) > 0:
                remaining_embeddings = class_embeddings[remaining_indices]
                diverse_local_in_remaining = farthest_point_sampling(
                    remaining_embeddings, k_diverse, seed=seed + c
                )
                diverse_local = list(remaining_indices[diverse_local_in_remaining])
            else:
                diverse_local = []
        else:
            diverse_local = []

        selected_local = easy_local + diverse_local
        selected.extend(class_indices[selected_local])

    return selected


@register_method('proto_then_diverse', needs={'embeddings'}, importance='ignore',
                kwargs={'proto_ratio': 0.5}, description="Prototypes + diverse samples")
def _select_proto_then_diverse(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    proto_ratio: float = 0.5,
    seed: int = 42,
    **kwargs
) -> List[int]:
    """Select prototypes first, then add diverse samples."""
    np.random.seed(seed)
    centroids = compute_class_centroids(embeddings, labels)
    selected = []

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]

        k = min(budget_per_class, len(class_indices))
        k_proto = int(k * proto_ratio)
        k_diverse = k - k_proto

        centroid = centroids[c]
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        proto_order = np.argsort(distances)
        proto_local = list(proto_order[:k_proto]) if k_proto > 0 else []

        if k_diverse > 0:
            remaining_mask = np.ones(len(class_indices), dtype=bool)
            remaining_mask[proto_local] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) > 0:
                remaining_embeddings = class_embeddings[remaining_indices]
                diverse_local_in_remaining = farthest_point_sampling(
                    remaining_embeddings, k_diverse, seed=seed + c
                )
                diverse_local = list(remaining_indices[diverse_local_in_remaining])
            else:
                diverse_local = []
        else:
            diverse_local = []

        selected_local = proto_local + diverse_local
        selected.extend(class_indices[selected_local])

    return selected


