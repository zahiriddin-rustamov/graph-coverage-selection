"""
MICCAI Experiment Runner

Structured experiment framework for sample selection methods.

Usage:
    python -m miccai.run --datasets organsmnist --methods fps --embeddings uni --ratios 0.02

    # With importance weighting (optional modifier)
    python -m miccai.run --datasets organsmnist --embeddings uni \
        --methods facility --importance test_attention --ratios 0.02
"""

__version__ = "0.1.0"

from .importance import (
    compute_importance,
    get_importance_requirements,
    get_available_importance_methods,
    IMPORTANCE_METHODS,
)

from .selection import (
    select,
    get_available_methods,
    get_method_requirements,
    get_method_importance_mode,
    METHODS,
)

from .graph import (
    build_knn_graph,
    build_adjacency_matrix,
    compute_k_hop_adjacency,
    propagate_importance,
    compute_reachability_matrix,
    get_graph_backend_info,
)

__all__ = [
    # Importance
    'compute_importance',
    'get_importance_requirements',
    'get_available_importance_methods',
    'IMPORTANCE_METHODS',
    # Selection
    'select',
    'get_available_methods',
    'get_method_requirements',
    'get_method_importance_mode',
    'METHODS',
    # Graph
    'build_knn_graph',
    'build_adjacency_matrix',
    'compute_k_hop_adjacency',
    'propagate_importance',
    'compute_reachability_matrix',
    'get_graph_backend_info',
]
