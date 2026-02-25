"""
Graph construction utilities.

Provides k-NN graph construction using FAISS (GPU/CPU) with sklearn fallback,
and k-hop adjacency computation for graph-constrained selection.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check FAISS availability
FAISS_AVAILABLE = False
FAISS_GPU_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    if faiss.get_num_gpus() > 0:
        FAISS_GPU_AVAILABLE = True
except ImportError:
    pass


def build_knn_graph_faiss(
    embeddings: np.ndarray,
    k: int,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build exact k-NN graph using FAISS.

    Args:
        embeddings: (n, d) array of embeddings
        k: Number of neighbors
        use_gpu: Whether to use GPU (if available)

    Returns:
        indices: (n, k) array of neighbor indices
        distances: (n, k) array of squared L2 distances on unit vectors.
                   Range [0, 4]. Related to cosine: cos_sim = 1 - dist/2.
    """
    n, d = embeddings.shape
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    # Normalize for cosine similarity (L2 on normalized = cosine distance)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    # Create exact L2 index
    index = faiss.IndexFlatL2(d)

    if use_gpu and FAISS_GPU_AVAILABLE:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings_norm)

    # Search for k+1 neighbors (first is self)
    # FAISS IndexFlatL2 returns squared L2 distances
    distances, indices = index.search(embeddings_norm, k + 1)

    # Remove self from neighbors
    return indices[:, 1:], distances[:, 1:]


def build_knn_graph_sklearn(
    embeddings: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph using sklearn (CPU fallback).

    Args:
        embeddings: (n, d) array
        k: Number of neighbors

    Returns:
        indices: (n, k) array of neighbor indices
        distances: (n, k) array of squared L2 distances on unit vectors.
                   Range [0, 4]. Related to cosine: cos_sim = 1 - dist/2.
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
    nn.fit(embeddings_norm)
    distances, indices = nn.kneighbors(embeddings_norm)

    # sklearn returns L2 distances; square to match FAISS convention
    distances = distances ** 2

    return indices[:, 1:], distances[:, 1:]


def build_knn_graph(
    embeddings: np.ndarray,
    k: int,
    use_gpu: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph using best available method.

    Args:
        embeddings: (n, d) array
        k: Number of neighbors
        use_gpu: Whether to use GPU (if FAISS GPU available)
        verbose: Print debug info

    Returns:
        indices: (n, k_actual) array of neighbor indices
        distances: (n, k_actual) array of squared L2 distances on unit vectors.
                   Range [0, 4]. Related to cosine: cos_sim = 1 - dist/2.

    Note:
        k is clamped to n-1 (max possible neighbors excluding self).
    """
    n = len(embeddings)

    # Clamp k to max possible neighbors (n-1, excluding self)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        # Edge case: only 1 sample, no neighbors possible
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)

    if k_actual < k and verbose:
        print(f"  [Graph] Warning: k={k} > n-1={n-1}, using k={k_actual}")

    if FAISS_AVAILABLE:
        backend = "faiss_gpu" if (use_gpu and FAISS_GPU_AVAILABLE) else "faiss_cpu"
        if verbose:
            print(f"  [Graph] k-NN: n={n}, k={k_actual}, backend={backend}")
        return build_knn_graph_faiss(embeddings, k_actual, use_gpu=use_gpu)
    else:
        if verbose:
            print(f"  [Graph] k-NN: n={n}, k={k_actual}, backend=sklearn")
        return build_knn_graph_sklearn(embeddings, k_actual)


def compute_k_hop_adjacency(
    knn_indices: np.ndarray,
    k_hops: int,
    n_samples: int,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute k-hop adjacency matrix.

    For small datasets (<15k), uses dense GPU operations.
    For larger datasets, uses sparse scipy operations.

    Args:
        knn_indices: (n, k) array of neighbor indices
        k_hops: Number of hops
        n_samples: Total number of samples
        verbose: Print debug info

    Returns:
        adjacency: (n, n) boolean tensor indicating k-hop connectivity
    """
    n, k = knn_indices.shape

    # For small datasets, use dense GPU operations
    if n_samples < 15000:
        if verbose:
            print(f"  [Graph] k-hop adjacency: k_hops={k_hops}, n={n_samples}, method=dense_gpu")
        adj = torch.zeros(n_samples, n_samples, device=device, dtype=torch.float32)
        rows = torch.arange(n_samples, device=device).unsqueeze(1).expand(-1, k).flatten()
        cols = torch.from_numpy(knn_indices.flatten()).to(device)
        adj[rows, cols] = 1.0

        # Make symmetric (undirected graph)
        adj = torch.maximum(adj, adj.T)

        # Compute k-hop reachability
        reachable = adj.clone()
        current = adj.clone()

        for _ in range(k_hops - 1):
            current = torch.mm(current, adj)
            reachable = torch.maximum(reachable, (current > 0).float())

        return reachable > 0

    else:
        # For larger datasets, use scipy sparse
        if verbose:
            print(f"  [Graph] k-hop adjacency: k_hops={k_hops}, n={n_samples}, method=sparse_scipy")
        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = knn_indices.flatten()
        data = np.ones(len(row_indices), dtype=np.float32)

        adj_sparse = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        # Make symmetric
        adj_sparse = adj_sparse.maximum(adj_sparse.T)

        # Compute k-hop reachability using sparse matrix powers: A + A² + ... + Aᵏ
        # This avoids scipy.shortest_path's 'limit' param (requires scipy>=1.8)
        reachable = adj_sparse.astype(np.float32)
        current = reachable.copy()

        for _ in range(k_hops - 1):
            current = current @ adj_sparse
            current.data[:] = 1.0  # Binarize to prevent numerical growth
            reachable = reachable + current

        # Convert to dense, binarize, and exclude self-connections
        reachable_dense = reachable.toarray()
        np.fill_diagonal(reachable_dense, 0)
        return torch.from_numpy(reachable_dense > 0).to(device)


def build_adjacency_matrix(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_samples: int,
    symmetric: bool = True,
    similarity_weights: bool = True,
    verbose: bool = False
) -> csr_matrix:
    """
    Build sparse adjacency matrix from k-NN graph.

    Args:
        knn_indices: (n, k) array of neighbor indices
        knn_distances: (n, k) array of squared L2 distances on unit vectors
        n_samples: Total number of samples
        symmetric: If True, make undirected (max of both directions)
        similarity_weights: If True, use similarity (1 - normalized_dist) as weights
        verbose: Print debug info

    Returns:
        Sparse adjacency matrix (n, n)
    """
    n, k = knn_indices.shape
    if verbose:
        print(f"  [Graph] Adjacency matrix: n={n_samples}, k={k}, symmetric={symmetric}")
    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = knn_indices.flatten()

    if similarity_weights:
        # Convert squared L2 distance to cosine similarity.
        # For unit vectors: ||a-b||² = 2(1-cos(a,b)), so cos_sim = 1 - dist²/2.
        # Clamp to [0, 1] since k-NN neighbors should have non-negative similarity.
        similarities = np.clip(1.0 - knn_distances.flatten() / 2.0, 0.0, 1.0)
        data = similarities.astype(np.float32)
    else:
        data = np.ones(len(row_indices), dtype=np.float32)

    adj = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_samples, n_samples)
    )

    if symmetric:
        adj = adj.maximum(adj.T)

    return adj


def propagate_importance(
    importance: np.ndarray,
    graph: csr_matrix,
    gamma: float = 0.85,
    num_iterations: int = 3,
    walk_length: int = 5,
    damping: float = 0.5,
    convergence_threshold: float = 1e-6,
    verbose: bool = False
) -> np.ndarray:
    """
    Propagate importance through graph using random walk / Katz centrality.

    High-importance nodes spread their importance to neighbors. A node near
    many high-importance nodes becomes important itself.

    Uses matrix formulation: reachability = I + γP + γ²P² + ... + γ^L P^L
    where P is the row-stochastic transition matrix.

    Final importance = damping * initial + (1 - damping) * propagated

    Args:
        importance: (n,) initial importance scores
        graph: Sparse adjacency matrix (n, n). Can be weighted.
        gamma: Geometric decay per hop (0-1). Lower = faster decay.
        num_iterations: Number of propagation iterations (compounds effect)
        walk_length: Steps per iteration (controls spread distance)
        damping: Balance between initial and propagated (0-1).
                 Higher = more weight on initial importance.
        convergence_threshold: Stop if max change < threshold
        verbose: Print convergence info

    Returns:
        (n,) propagated importance scores
    """
    from scipy.sparse import diags, identity

    n = graph.shape[0]

    # Convert to transition matrix (row-stochastic)
    row_sums = np.array(graph.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero for isolated nodes
    D_inv = diags(1.0 / row_sums)
    P = D_inv @ graph  # P[i,j] = prob of transitioning i -> j

    # Compute multi-step reachability with geometric decay
    # reachability[i,j] = expected discounted visits to j starting from i
    I = identity(n, format='csr')
    reachability = I.tocsr()
    P_power = I.tocsr()
    decay_factor = 1.0

    for step in range(1, walk_length + 1):
        P_power = P_power @ P
        decay_factor *= gamma
        reachability = reachability + decay_factor * P_power

    # Normalize by geometric sum
    geometric_sum = (1 - gamma ** (walk_length + 1)) / (1 - gamma) if gamma < 1.0 else (walk_length + 1)
    reachability = reachability / geometric_sum

    # Propagate importance
    current = importance.copy()

    for iteration in range(num_iterations):
        # Propagate: each node receives importance from nodes that can reach it
        propagated = reachability.T @ current

        # Blend with initial importance
        new_importance = damping * importance + (1 - damping) * propagated

        # Check convergence
        change = np.abs(new_importance - current).max()
        current = new_importance

        if verbose:
            print(f"Propagation iter {iteration + 1}/{num_iterations}: max_change={change:.2e}")

        if change < convergence_threshold:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break

    return current


def compute_reachability_matrix(
    graph: csr_matrix,
    walk_length: int = 5,
    gamma: float = 0.85,
    verbose: bool = False
) -> csr_matrix:
    """
    Compute multi-step reachability matrix with geometric decay.

    reachability[i,j] = expected discounted visits to j starting from i
                      = Σ_{t=0}^{L} γ^t P^t[i,j]

    This is used for importance-weighted facility location.

    Args:
        graph: Sparse adjacency matrix
        walk_length: Number of steps
        gamma: Geometric decay per step
        verbose: Print computation details

    Returns:
        Sparse reachability matrix (n, n)
    """
    from scipy.sparse import diags, identity

    n = graph.shape[0]

    if verbose:
        graph_nnz = graph.nnz
        graph_sparsity = 1.0 - graph_nnz / (n * n)
        print(f"  [Reachability] Computing: n={n}, walk_length={walk_length}, gamma={gamma}")
        print(f"  [Reachability] Input graph: nnz={graph_nnz}, sparsity={graph_sparsity:.4f}")

    # Convert to transition matrix
    row_sums = np.array(graph.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = diags(1.0 / row_sums)
    P = D_inv @ graph

    # Compute reachability with decay
    I = identity(n, format='csr')
    reachability = I.tocsr()
    P_power = I.tocsr()
    decay_factor = 1.0

    for step in range(1, walk_length + 1):
        P_power = P_power @ P
        decay_factor *= gamma
        reachability = reachability + decay_factor * P_power

        if verbose:
            step_nnz = P_power.nnz
            step_sparsity = 1.0 - step_nnz / (n * n)
            print(f"    [Reachability] step={step}: P^{step} nnz={step_nnz}, "
                  f"sparsity={step_sparsity:.4f}, decay={decay_factor:.4f}")

    if verbose:
        final_nnz = reachability.nnz
        final_sparsity = 1.0 - final_nnz / (n * n)
        reach_data = reachability.data
        print(f"  [Reachability] Final: nnz={final_nnz}, sparsity={final_sparsity:.4f}")
        print(f"  [Reachability] Values: min={reach_data.min():.4f}, "
              f"max={reach_data.max():.4f}, mean={reach_data.mean():.4f}")

    return reachability


def get_graph_backend_info() -> dict:
    """Get information about available graph computation backends."""
    return {
        'faiss_available': FAISS_AVAILABLE,
        'faiss_gpu_available': FAISS_GPU_AVAILABLE,
        'faiss_num_gpus': faiss.get_num_gpus() if FAISS_AVAILABLE else 0,
    }
