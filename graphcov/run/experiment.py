"""
Experiment orchestration.

Handles running experiments across multiple configurations,
saving results, and tracking run history.
"""

import sys
import json
import time

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import load_dataset, get_labels, get_dataset_info
from .embeddings import load_or_compute_embeddings, get_available_sources, load_or_compute_eva, load_or_compute_raw_dynamics, extract_embeddings_with_model
from .selection import (
    select, get_method_requirements, get_available_methods,
    get_method_importance_mode, METHODS
)
from .eva import get_optimal_windows, derive_eva_scores
from .importance import (
    compute_importance, get_importance_requirements,
    get_available_importance_methods, IMPORTANCE_METHODS
)
from .graph import build_knn_graph, build_adjacency_matrix, propagate_importance
from .evaluation import evaluate_selection, evaluate_linear_probe, set_seed
from .results import (
    generate_run_id, create_run_dir, save_config as _save_config,
    save_summary as _save_summary, get_git_commit,
    append_to_csv as _append_to_csv_shared,
)

# Default directories
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'results'
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / 'cache' / 'embeddings'

# Methods that don't need embeddings at all
# Training dynamics methods train their own model
EMBEDDING_FREE_METHODS = {'eva', 'aum', 'forgetting', 'el2n_top', 'el2n_bottom', 'el2n_mid', 'random', 'random_proportional'}


def needs_per_embedding_run(method: str, importance_methods: Optional[List[str]]) -> bool:
    """
    Check if a method needs to run separately for each embedding.

    A method needs per-embedding runs if:
    1. It uses embeddings directly (not in EMBEDDING_FREE_METHODS), OR
    2. It's embedding-free but uses importance weights (which depend on embeddings)

    Methods that are embedding-free AND ignore importance only need to run once.
    """
    if method not in EMBEDDING_FREE_METHODS:
        return True
    # Embedding-free method - check if it actually uses importance
    if importance_methods and get_method_importance_mode(method) != 'ignore':
        return True
    return False



# get_git_commit imported from .results


def resolve_dependencies(methods: List[str], embeddings: List[str]) -> List[str]:
    """
    Ensure required embeddings are included based on method requirements.

    If any method requires variance_stats, 'trained' must be included.
    EL2N methods no longer require 'trained' — they derive scores from training dynamics.
    """
    needs_trained = False
    for method in methods:
        reqs = get_method_requirements(method)
        if reqs & {'variance_stats'}:
            needs_trained = True
            break

    if needs_trained and 'trained' not in embeddings:
        print("Note: Adding 'trained' embedding (required for variance methods)")
        embeddings = ['trained'] + list(embeddings)

    return list(embeddings)


def resolve_importance_dependencies(
    methods: List[str],
    importance_methods: Optional[List[str]],
    embeddings: List[str]
) -> List[str]:
    """
    Ensure required embeddings are included based on importance method requirements.

    Also checks if any method requires importance but no importance is specified.
    """
    if not importance_methods:
        return list(embeddings)

    needs_trained = False
    for imp_method in importance_methods:
        reqs = get_importance_requirements(imp_method)
        if reqs & {'el2n_scores', 'variance_stats'}:
            needs_trained = True
            break

    if needs_trained and 'trained' not in embeddings:
        print("Note: Adding 'trained' embedding (required for importance method)")
        embeddings = ['trained'] + list(embeddings)

    return list(embeddings)


def filter_valid_methods(methods: List[str], embedding_source: str) -> List[str]:
    """
    Filter methods to only those valid for the given embedding source.

    - Training dynamics methods (eva, forgetting, el2n_*) work with any embedding source.
    - Variance methods only work with 'trained' embeddings.
    - 'none' embedding source only works with embedding-free methods.
    """
    valid = []
    for method in methods:
        reqs = get_method_requirements(method)

        # Handle 'none' embedding source (for embedding-free methods)
        if embedding_source == 'none':
            if method in EMBEDDING_FREE_METHODS:
                valid.append(method)
            continue

        # If method needs variance, only valid with 'trained'
        if reqs & {'variance_stats'}:
            if embedding_source == 'trained':
                valid.append(method)
        else:
            valid.append(method)

    return valid


def append_to_csv(path: Path, row: Dict):
    """Append a single row to CSV file, creating headers if needed."""
    _append_to_csv_shared(path, row)


def run_single(
    dataset_name: str,
    embedding_source: str,
    method: str,
    ratio: float,
    trial: int,
    embedding_cache: Dict,
    config: Dict,
    importance_method: Optional[str] = None,
    selection_cache: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment configuration.

    Args:
        dataset_name: Name of dataset
        embedding_source: Embedding source to use
        method: Selection method
        ratio: Selection ratio
        trial: Trial number (0-indexed)
        embedding_cache: Shared cache for embeddings
        config: Full experiment config
        importance_method: Optional importance method to compute and use

    Returns:
        Result dict with metrics and history
    """
    seed = config['seed'] + trial
    verbose = config.get('verbose', False)
    _verbose_level = config.get('_verbose_level', 1)

    # Load dataset
    train_dataset, info = load_dataset(dataset_name, 'train', config.get('size', 224), verbose=(verbose and _verbose_level >= 2))
    test_dataset, _ = load_dataset(dataset_name, 'test', config.get('size', 224), verbose=False)  # Less verbose for test

    num_classes = len(info['label'])
    in_channels = info['n_channels']
    labels = get_labels(train_dataset)

    if verbose:
        n_train = len(train_dataset)
        n_test = len(test_dataset)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        class_imbalance = label_counts.max() / label_counts.min()
        print(f"  [Dataset] {dataset_name}: train={n_train}, test={n_test}, "
              f"classes={num_classes}, channels={in_channels}")
        print(f"  [Dataset] Label distribution: min={label_counts.min()}, max={label_counts.max()}, "
              f"imbalance_ratio={class_imbalance:.2f}")

    # Get or compute embeddings (skip for embedding-free methods)
    emb_data = {}
    embeddings = None
    trained_model = None  # For reuse when extracting val embeddings
    if method not in EMBEDDING_FREE_METHODS:
        cache_key = f"{dataset_name}_{embedding_source}"
        # Request model if we'll need it for val embeddings (trained + test_attention)
        need_model_for_val = (embedding_source == 'trained' and importance_method == 'test_attention')

        if cache_key not in embedding_cache or need_model_for_val:
            embedding_data = load_or_compute_embeddings(
                dataset_name=dataset_name,
                split='train',
                source=embedding_source,
                dataset=train_dataset,
                num_classes=num_classes,
                in_channels=in_channels,
                size=config.get('size', 224),
                seed=config['seed'],  # Use base seed for embeddings
                trained_epochs=config.get('trained_epochs', 200),
                cache_dir=config.get('cache_dir'),
                force_recompute=config.get('force_recompute', False),
                verbose=verbose,
                return_model=need_model_for_val
            )
            embedding_cache[cache_key] = embedding_data
            if need_model_for_val:
                trained_model = embedding_data.get('model')

        emb_data = embedding_cache[cache_key]
        embeddings = emb_data.get('embeddings')

    # Calculate budget
    n_total = len(train_dataset)
    n_select = int(n_total * ratio)
    budget_per_class = n_select // num_classes

    if verbose:
        actual_select = budget_per_class * num_classes
        print(f"  [Budget] ratio={ratio}, n_total={n_total}, n_select={n_select}, "
              f"budget_per_class={budget_per_class}, actual_select={actual_select}")

    # Handle training dynamics methods (EVA, AUM, Forgetting, EL2N)
    # Raw dynamics (200-epoch training) are ratio-independent — train once, reuse
    TRAINING_DYNAMICS_METHODS = {'eva', 'aum', 'forgetting', 'el2n_top', 'el2n_bottom', 'el2n_mid'}
    eva_scores = None
    aum_scores = None
    forgetting_scores = None
    el2n_scores = None

    if method in TRAINING_DYNAMICS_METHODS:
        # Load/compute raw dynamics once per dataset (ratio-independent)
        # Use eva_size (default 28) for scoring — matches EVA paper's 28×28 protocol.
        # Evaluation still uses the main --size (e.g. 224).
        eva_size = config.get('eva_size', 28)
        raw_dynamics_key = f"{dataset_name}_raw_dynamics_s{eva_size}"
        if raw_dynamics_key not in embedding_cache:
            # Load a separate dataset at eva_size for dynamics computation
            eva_dataset, _ = load_dataset(dataset_name, 'train', eva_size, verbose=False)
            if verbose:
                print(f"  [Dynamics] Using {eva_size}x{eva_size} images for training dynamics scoring "
                      f"(evaluation uses {config.get('size', 224)}x{config.get('size', 224)})")
            raw_data = load_or_compute_raw_dynamics(
                dataset_name=dataset_name,
                split='train',
                dataset=eva_dataset,
                num_classes=num_classes,
                in_channels=in_channels,
                size=eva_size,
                seed=config['seed'],
                eva_epochs=config.get('eva_epochs', 200),
                window_size=config.get('eva_window_size', 10),
                cache_dir=config.get('cache_dir'),
                force_recompute=config.get('force_recompute', False),
                verbose=verbose
            )
            embedding_cache[raw_dynamics_key] = raw_data

        raw_data = embedding_cache[raw_dynamics_key]
        aum_scores = raw_data.get('aum_scores')
        forgetting_scores = raw_data.get('forgetting_scores')

        # EVA scores depend on ratio (different window positions) — derive on the fly
        if method == 'eva':
            early_start, late_start = get_optimal_windows(
                dataset_name, ratio,
                eva_epochs=config.get('eva_epochs', 200),
                verbose=verbose,
            )
            eva_scores = derive_eva_scores(
                raw_data['all_l2_scores'],
                window_size=config.get('eva_window_size', 10),
                early_window_start=early_start,
                late_window_start=late_start,
                verbose=verbose,
            )[0]

        # EL2N: average L2 error over first 20 epochs of the same training run
        if method.startswith('el2n_'):
            all_l2 = raw_data['all_l2_scores']  # (epochs, n)
            el2n_end = min(20, all_l2.shape[0])
            el2n_scores = all_l2[:el2n_end].mean(axis=0)
            if verbose:
                print(f"  [EL2N] Derived from training dynamics epochs 1-{el2n_end}, "
                      f"range=[{el2n_scores.min():.4f}, {el2n_scores.max():.4f}]")

    # Compute importance if specified
    importance = None
    if importance_method:
        # Check method's importance mode
        importance_mode = get_method_importance_mode(method)

        if importance_mode == 'ignore':
            # Method doesn't use importance - skip computation but allow for analysis
            if verbose:
                print(f"  Note: Method '{method}' ignores importance, "
                      f"but computing '{importance_method}' for logging")

        # Get validation embeddings for test_attention importance
        test_embeddings = None
        if importance_method == 'test_attention':
            val_emb_key = f"{dataset_name}_{embedding_source}_val"
            if val_emb_key not in embedding_cache:
                val_dataset, _ = load_dataset(dataset_name, 'val', config.get('size', 224))

                # If we have a trained model from 'trained' source, reuse it for val embeddings
                if trained_model is not None:
                    if verbose:
                        print(f"  [Optimization] Reusing trained model for val embeddings (no redundant training)")
                    val_embeddings = extract_embeddings_with_model(
                        model=trained_model,
                        dataset=val_dataset,
                        batch_size=128,
                        apply_imagenet_norm=False,  # 'trained' doesn't use imagenet norm
                        verbose=verbose
                    )
                    embedding_cache[val_emb_key] = {'embeddings': val_embeddings}
                else:
                    # Fall back to full computation (for non-trained sources)
                    val_emb_data = load_or_compute_embeddings(
                        dataset_name=dataset_name,
                        split='val',
                        source=embedding_source,
                        dataset=val_dataset,
                        num_classes=num_classes,
                        in_channels=in_channels,
                        size=config.get('size', 224),
                        seed=config['seed'],
                        trained_epochs=config.get('trained_epochs', 200),
                        cache_dir=config.get('cache_dir'),
                        force_recompute=config.get('force_recompute', False),
                        verbose=verbose
                    )
                    embedding_cache[val_emb_key] = val_emb_data
            test_embeddings = embedding_cache[val_emb_key].get('embeddings')

        # Build graph if needed for propagation or centrality
        k_neighbors = config.get('k_neighbors', 10)
        graph = None
        if config.get('propagate', False) or importance_method == 'centrality':
            if embeddings is not None:
                knn_idx, knn_dist = build_knn_graph(embeddings, k_neighbors, verbose=verbose)
                graph = build_adjacency_matrix(knn_idx, knn_dist, len(embeddings), verbose=verbose)

        # Compute importance
        importance_result = compute_importance(
            method=importance_method,
            labels=labels,
            embeddings=embeddings,
            el2n_scores=emb_data.get('el2n_scores'),
            variance_stats=(emb_data.get('conf_variance'), emb_data.get('conf_mean'))
            if 'conf_variance' in emb_data else None,
            test_embeddings=test_embeddings,
            graph=graph,
            temperature=config.get('temperature', 0.1),
            k_neighbors=k_neighbors,
            verbose=verbose,
        )
        importance = importance_result.scores

        # Propagate importance if requested
        if config.get('propagate', False) and graph is not None:
            importance = propagate_importance(
                importance=importance,
                graph=graph,
                gamma=config.get('gamma', 0.85),
                num_iterations=config.get('prop_iterations', 3),
                walk_length=config.get('walk_length', 5),
                damping=config.get('damping', 0.5),
                verbose=verbose,
            )

    # Select samples using unified interface (with caching for deterministic methods)
    NONDETERMINISTIC_METHODS = {'random', 'random_proportional', 'fps', 'graph_fps'}
    is_deterministic = method not in NONDETERMINISTIC_METHODS

    # Build cache key: includes all parameters that affect selection output
    sel_cache_key = None
    if selection_cache is not None and is_deterministic:
        sel_cache_key = (
            dataset_name, embedding_source, method, ratio,
            importance_method,
            config.get('k_neighbors', 10),
            config.get('k_hops'),  # None = per-method default
            config.get('walk_length', 5),
            config.get('gamma', 0.85),
            config.get('coverage_mode', 'prob'),
            config.get('global_selection', False),
            config.get('propagate', False),
            config.get('size', 224),
        )

    if sel_cache_key is not None and sel_cache_key in selection_cache:
        selected, selection_time = selection_cache[sel_cache_key]
        if verbose:
            print(f"  [Selection] Cache hit for {method} — reusing {len(selected)} indices "
                  f"(original selection took {selection_time:.2f}s)")
    else:
        t_select_start = time.time()
        # Build method overrides — only include params that were explicitly set
        # so that per-method defaults in spec.kwargs are preserved
        method_overrides = {
            'k_neighbors': config.get('k_neighbors', 10),
            'walk_length': config.get('walk_length', 5),
            'gamma': config.get('gamma', 0.85),
            'coverage_mode': config.get('coverage_mode', 'prob'),
            'global_selection': config.get('global_selection', False),
            'sparse_cpu': config.get('sparse_cpu', False),
        }
        # k_hops: only override if explicitly set (None means use per-method default)
        if config.get('k_hops') is not None:
            method_overrides['k_hops'] = config['k_hops']

        selected = select(
            method=method,
            labels=labels,
            budget_per_class=budget_per_class,
            embeddings=embeddings,
            el2n_scores=el2n_scores if el2n_scores is not None else emb_data.get('el2n_scores'),
            variance_stats=(emb_data.get('conf_variance'), emb_data.get('conf_mean'))
            if 'conf_variance' in emb_data else None,
            eva_scores=eva_scores,
            aum_scores=aum_scores,
            forgetting_scores=forgetting_scores,
            importance=importance,
            importance_name=importance_method,  # For verbose logging
            seed=seed,
            verbose=verbose,
            _verbose_level=_verbose_level,
            **method_overrides,
        )
        selection_time = time.time() - t_select_start

        # Cache the result for deterministic methods
        if sel_cache_key is not None:
            selection_cache[sel_cache_key] = (selected, selection_time)

    # Linear probe evaluation (fast, no representation mismatch)
    linear_probe_acc = None
    linear_probe_bal_acc = None
    if config.get('linear_probe', False) and embeddings is not None:
        test_emb_key = f"{dataset_name}_{embedding_source}_test"
        if test_emb_key not in embedding_cache:
            test_emb_data = load_or_compute_embeddings(
                dataset_name=dataset_name,
                split='test',
                source=embedding_source,
                dataset=test_dataset,
                num_classes=num_classes,
                in_channels=in_channels,
                size=config.get('size', 224),
                seed=config['seed'],
                trained_epochs=config.get('trained_epochs', 200),
                cache_dir=config.get('cache_dir'),
                force_recompute=config.get('force_recompute', False),
                verbose=verbose,
            )
            embedding_cache[test_emb_key] = test_emb_data

        test_embeddings = embedding_cache[test_emb_key].get('embeddings')
        test_labels_arr = get_labels(test_dataset)

        linear_probe_acc, linear_probe_bal_acc = evaluate_linear_probe(
            train_embeddings=embeddings[selected],
            train_labels=labels[selected],
            test_embeddings=test_embeddings,
            test_labels=test_labels_arr,
            seed=seed,
            verbose=verbose,
        )

    training_paradigm = config.get('training_paradigm', 'epoch')

    if verbose:
        print(f"  [Training] paradigm={training_paradigm}, selected={len(selected)} samples")
        if training_paradigm == 'iteration':
            print(f"  [Training] iterations={config.get('iterations', 40000)}, "
                  f"test_interval={config.get('test_interval', 800)}, "
                  f"batch_size={config.get('batch_size', 256)}")
        else:
            print(f"  [Training] epochs={config.get('epochs', 200)}, "
                  f"batch_size={config.get('batch_size', 256)}, "
                  f"lr={config.get('lr', 0.1)}")
        print(f"  [Training] augment={config.get('augment', False)}, "
              f"deterministic={config.get('deterministic', False)}, seed={seed}")

    t_train_start = time.time()
    eval_result = evaluate_selection(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        selected_indices=selected,
        num_classes=num_classes,
        in_channels=in_channels,
        training_paradigm=training_paradigm,
        epochs=config.get('epochs', 200),
        iterations=config.get('iterations', 40000),
        test_interval=config.get('test_interval', 800),
        test_every_n_epochs=config.get('test_every_n_epochs', 10),
        batch_size=config.get('batch_size', 256),
        lr=config.get('lr', 0.1),
        momentum=config.get('momentum', 0.9),
        nesterov=config.get('nesterov', False),
        weight_decay=config.get('weight_decay', 0.0005),
        augment=config.get('augment', False),
        size=config.get('size', 224),
        seed=seed,
        return_history=True,
        verbose=verbose,
        verbose_per_class=(verbose and _verbose_level >= 2),
        deterministic=config.get('deterministic', False),
        num_workers=config.get('num_workers', 4)
    )

    training_time = time.time() - t_train_start

    # Both paradigms now return: acc, bal_acc, history, best_metrics, per_class
    acc, bal_acc, history, best_metrics, per_class = eval_result

    result = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'best_accuracy': best_metrics['best_acc'],
        'best_balanced_accuracy': best_metrics['best_bal_acc'],
        'linear_probe_accuracy': linear_probe_acc,
        'linear_probe_balanced_accuracy': linear_probe_bal_acc,
        'history': history,
        'per_class': per_class,
        'n_selected': len(selected),
        'budget_per_class': budget_per_class,
        'selection_time_s': round(selection_time, 2),
        'training_time_s': round(training_time, 2),
    }

    if training_paradigm == 'iteration':
        result['best_iteration'] = best_metrics['best_iteration']
    else:
        result['best_epoch'] = best_metrics['best_epoch']

    return result


def run_experiment(config: Dict) -> str:
    """
    Run full experiment with multiple configurations.

    Unified interface: all methods use --methods, with optional --importance modifier.

    Args:
        config: Experiment configuration with keys:
            - datasets: List of dataset names
            - embeddings: List of embedding sources
            - methods: List of selection methods
            - importance_methods: Optional list of importance methods
            - ratios: List of selection ratios
            - trials: Number of trials per configuration
            - epochs: Training epochs (default 200)
            - batch_size: Batch size (default 256)
            - lr: Learning rate (default 0.1)
            - momentum: SGD momentum (default 0.9)
            - weight_decay: Weight decay (default 0.0005)
            - seed: Base random seed (default 42)
            - output_dir: Output directory (default graphcov/results)
            - cache_dir: Cache directory (default graphcov/cache/embeddings)

    Returns:
        run_id: Identifier for this run
    """
    # Generate run ID
    run_id = generate_run_id()
    start_time = datetime.now()

    # Setup directories
    output_dir = Path(config.get('output_dir', DEFAULT_OUTPUT_DIR))
    run_dir = create_run_dir(run_id, base_dir=output_dir)

    results_csv = output_dir / 'results.csv'

    # Get methods and importance methods
    methods = config['methods']
    importance_methods = config.get('importance_methods')  # Optional, may be None

    # Split methods into embedding-free and embedding-dependent
    # Embedding-free methods run once per dataset (with embedding='none')
    # Embedding-dependent methods run per embedding
    embedding_free_methods = [m for m in methods if not needs_per_embedding_run(m, importance_methods)]
    embedding_dependent_methods = [m for m in methods if needs_per_embedding_run(m, importance_methods)]

    # Determine embeddings list
    if embedding_dependent_methods:
        # Resolve dependencies for methods that need embeddings
        embeddings = resolve_dependencies(embedding_dependent_methods, config.get('embeddings') or [])
        embeddings = resolve_importance_dependencies(embedding_dependent_methods, importance_methods, embeddings)
    else:
        # All methods are embedding-free
        embeddings = []

    config['embeddings'] = embeddings if embeddings else ['none']

    # Build methods string for display
    if importance_methods:
        methods_str = f"methods={methods}, importance={importance_methods}"
        if config.get('propagate', False):
            methods_str += "+prop"
    else:
        methods_str = str(methods)

    # Save config
    config_to_save = {
        'run_id': run_id,
        'script': 'experiment',
        'command': ' '.join(sys.argv),
        'started_at': start_time.isoformat(),
        **{k: v for k, v in config.items() if k not in ['cache_dir', 'output_dir']},
        'cache_dir': str(config.get('cache_dir', DEFAULT_CACHE_DIR)),
        'output_dir': str(output_dir),
    }
    _save_config(run_dir, config_to_save)

    print(f"\nRun ID: {run_id}")
    print(f"Output: {run_dir}")
    print(f"Datasets: {config['datasets']}")
    print(f"Embeddings: {embeddings if embeddings else ['none (embedding-free methods only)']}")
    print(f"Methods: {methods_str}")
    print(f"Ratios: {config['ratios']}")
    print(f"Trials: {config['trials']}")

    # Track progress
    all_history = []
    all_results = []
    all_per_class = []
    completed = 0
    failed = 0
    skipped = 0

    # Embedding cache (shared across configs to avoid recomputation)
    embedding_cache = {}

    # Selection cache: deterministic methods produce identical subsets across trials,
    # so we cache selected indices to avoid redundant recomputation.
    # Key: (dataset, embedding, method, ratio, importance, k, k_hops, walk_length, gamma, coverage_mode, global, propagate, size)
    # Value: (selected_indices, selection_time)
    selection_cache = {}

    # Calculate total configurations
    # If importance_methods is specified, we run cross-product: methods × importance
    # Otherwise, we run methods with importance=None
    n_importance = len(importance_methods) if importance_methods else 1
    n_datasets = len(config['datasets'])
    n_ratios = len(config['ratios'])
    n_trials = config['trials']

    total_configs = 0
    # Embedding-free methods: run once per dataset (not per embedding)
    total_configs += n_datasets * len(embedding_free_methods) * n_importance * n_ratios * n_trials
    # Embedding-dependent methods: run per embedding
    for dataset in config['datasets']:
        for embedding in embeddings:
            valid_methods = filter_valid_methods(embedding_dependent_methods, embedding)
            total_configs += len(valid_methods) * n_importance * n_ratios * n_trials

    print(f"Total configurations: {total_configs}")
    if embedding_free_methods:
        print(f"  Embedding-free methods (run once): {embedding_free_methods}")
    if embedding_dependent_methods:
        print(f"  Embedding-dependent methods: {embedding_dependent_methods}")
    print()

    pbar = tqdm(total=total_configs, desc='Running')

    # Build list of importance methods to iterate over
    # If importance_methods is None, use [None] to run once without importance
    importance_list = importance_methods if importance_methods else [None]

    # Helper to run a single config and record results
    def run_and_record(dataset, embedding, method, imp_method, ratio, trial):
        nonlocal completed, failed
        seed = config['seed'] + trial

        # Build method name for display
        if imp_method:
            display_name = f"{method}+{imp_method}"
            if config.get('propagate', False):
                display_name += "+prop"
        else:
            display_name = method

        # Add global suffix for methods that support it
        GLOBAL_CAPABLE = {'graph_a2', 'graph_coverage', 'heat_kernel', 'facility'}
        if config.get('global_selection', False) and method in GLOBAL_CAPABLE:
            display_name += " (global)"

        try:
            result = run_single(
                dataset_name=dataset,
                embedding_source=embedding,
                method=method,
                ratio=ratio,
                trial=trial,
                embedding_cache=embedding_cache,
                config=config,
                importance_method=imp_method,
                selection_cache=selection_cache,
            )

            # Add identifiers to history
            for h in result['history']:
                h.update({
                    'dataset': dataset,
                    'embedding': embedding,
                    'method': display_name,
                    'base_method': method,
                    'importance': imp_method,
                    'propagate': config.get('propagate', False) if imp_method else False,
                    'ratio': ratio,
                    'trial': trial + 1,
                    'seed': seed,
                })
            all_history.extend(result['history'])

            # Append to results.csv
            result_row = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset,
                'embedding': embedding,
                'method': display_name,
                'base_method': method,
                'importance': imp_method,
                'propagate': config.get('propagate', False) if imp_method else False,
                'ratio': ratio,
                'trial': trial + 1,
                'seed': seed,
                'budget_per_class': result['budget_per_class'],
                'n_selected': result['n_selected'],
                'accuracy': round(result['accuracy'], 6),
                'balanced_accuracy': round(result['balanced_accuracy'], 6),
            }
            # Add best_* fields (tracked for both epoch and iteration paradigms)
            result_row['best_accuracy'] = round(result['best_accuracy'], 6)
            result_row['best_balanced_accuracy'] = round(result['best_balanced_accuracy'], 6)
            if 'best_iteration' in result:
                result_row['best_iteration'] = result['best_iteration']
            if 'best_epoch' in result:
                result_row['best_epoch'] = result['best_epoch']
            # Add linear probe fields when available
            if result.get('linear_probe_accuracy') is not None:
                result_row['lp_accuracy'] = round(result['linear_probe_accuracy'], 6)
                result_row['lp_balanced_accuracy'] = round(result['linear_probe_balanced_accuracy'], 6)
            # Add timing
            result_row['selection_time_s'] = result['selection_time_s']
            result_row['training_time_s'] = result['training_time_s']
            append_to_csv(results_csv, result_row)
            all_results.append(result_row)

            # Collect per-class results
            for cls, cls_data in result['per_class'].items():
                all_per_class.append({
                    'dataset': dataset,
                    'embedding': embedding,
                    'method': display_name,
                    'base_method': method,
                    'importance': imp_method,
                    'ratio': ratio,
                    'trial': trial + 1,
                    'seed': seed,
                    'class': cls,
                    'class_accuracy': round(cls_data['accuracy'], 6),
                    'class_count': cls_data['count'],
                })

            completed += 1
            pbar.set_postfix({
                'done': completed,
                'fail': failed,
                'bal_acc': f"{result['balanced_accuracy']:.3f}"
            })

        except Exception as e:
            import traceback
            print(f"\nFailed: {dataset}/{embedding}/{display_name}/{ratio}/t{trial}")
            print(f"  Error: {type(e).__name__}: {e}")
            if config.get('verbose', False):
                traceback.print_exc()
            failed += 1

        pbar.update(1)

    # Main experiment loop
    for dataset in config['datasets']:
        # Phase 1: Run embedding-free methods (once per dataset, with embedding='none')
        for method in embedding_free_methods:
            for imp_method in importance_list:
                for ratio in config['ratios']:
                    for trial in range(config['trials']):
                        run_and_record(dataset, 'none', method, imp_method, ratio, trial)

        # Phase 2: Run embedding-dependent methods (per embedding)
        for embedding in embeddings:
            valid_methods = filter_valid_methods(embedding_dependent_methods, embedding)
            for method in valid_methods:
                for imp_method in importance_list:
                    for ratio in config['ratios']:
                        for trial in range(config['trials']):
                            run_and_record(dataset, embedding, method, imp_method, ratio, trial)

    pbar.close()

    # Save training history
    if all_history:
        history_df = pd.DataFrame(all_history)
        # Column order depends on training paradigm
        if config.get('training_paradigm') == 'iteration':
            cols = ['dataset', 'embedding', 'method', 'base_method', 'importance', 'ratio', 'trial', 'seed',
                    'iteration', 'train_loss', 'train_acc', 'test_acc', 'test_bal_acc', 'lr']
        else:
            cols = ['dataset', 'embedding', 'method', 'base_method', 'importance', 'ratio', 'trial', 'seed',
                    'epoch', 'train_loss', 'train_acc', 'test_acc', 'test_bal_acc', 'lr']
        # Only include columns that exist in the dataframe
        cols = [c for c in cols if c in history_df.columns]
        history_df = history_df[cols]
        history_df.to_csv(run_dir / 'training_history.csv', index=False)

    # Save per-class results
    if all_per_class:
        per_class_df = pd.DataFrame(all_per_class)
        cols = ['dataset', 'embedding', 'method', 'base_method', 'importance',
                'ratio', 'trial', 'seed', 'class', 'class_accuracy', 'class_count']
        cols = [c for c in cols if c in per_class_df.columns]
        per_class_df = per_class_df[cols]
        per_class_df.to_csv(run_dir / 'per_class_results.csv', index=False)

    # Save summary
    summary = {
        'run_id': run_id,
        'started_at': start_time.isoformat(),
        'finished_at': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds(),
        'status': 'completed' if failed == 0 else 'completed_with_errors',
        'total_configs': total_configs,
        'completed_configs': completed,
        'failed_configs': failed,
        'skipped_configs': skipped,
        'git_commit': get_git_commit(),
    }
    _save_summary(run_dir, summary)

    print(f"\n{'='*60}")
    print(f"Run {run_id} completed")
    print(f"  Completed: {completed}/{total_configs}")
    print(f"  Failed: {failed}")
    print(f"  Duration: {summary['duration_seconds']:.1f}s")
    print(f"  Results: {results_csv}")
    print(f"  History: {run_dir}/")
    print(f"{'='*60}")

    # Print summary tables
    if all_results:
        print_summary_tables(all_results, config['ratios'])

    return run_id


def list_methods():
    """Print available selection methods."""
    print("\nAvailable selection methods:")
    print("-" * 80)
    print(f"{'Method':<25} {'Importance':<12} {'Needs':<30} Description")
    print("-" * 80)
    for name in sorted(METHODS.keys()):
        spec = METHODS[name]
        needs = ', '.join(spec.needs) if spec.needs else 'none'
        print(f"  {name:<23} {spec.importance:<12} {needs:<30} {spec.description}")

    print("\nImportance modes:")
    print("  - ignore:   Method has its own selection logic, ignores --importance")
    print("  - optional: Uses importance if provided, defaults to uniform")
    print("  - required: Must have importance weights (defaults to uniform if not provided)")


def list_datasets():
    """Print available datasets."""
    from .data import DATASETS
    print("\nAvailable datasets:")
    print("-" * 60)
    for name in DATASETS:
        try:
            info = get_dataset_info(name)
            n_classes = len(info['label'])
            n_channels = info['n_channels']
            print(f"  {name:20s} classes: {n_classes:3d}  channels: {n_channels}")
        except Exception:
            print(f"  {name:20s} (info unavailable)")


def list_embeddings():
    """Print available embedding sources."""
    print("\nAvailable embedding sources:")
    print("-" * 60)
    available = get_available_sources()
    all_sources = ['random', 'imagenet', 'trained', 'uni', 'conch']
    for source in all_sources:
        status = "available" if source in available else "not installed"
        print(f"  {source:15s} {status}")


def list_importance():
    """Print available importance methods."""
    print("\nAvailable importance methods:")
    print("-" * 60)
    for method in IMPORTANCE_METHODS:
        reqs = get_importance_requirements(method)
        needs = ', '.join(reqs) if reqs else 'none'
        print(f"  {method:20s} needs: {needs}")

    print("\nMethods that use importance:")
    print("-" * 60)
    print("  Optional (work with or without --importance):")
    for name in sorted(METHODS.keys()):
        if METHODS[name].importance == 'optional':
            print(f"    {name}")
    print("  Required (best with --importance, defaults to uniform):")
    for name in sorted(METHODS.keys()):
        if METHODS[name].importance == 'required':
            print(f"    {name}")


# =============================================================================
# Summary Tables
# =============================================================================

def format_mean_std(mean: float, std: float, width: int = 12) -> str:
    """Format mean±std with fixed width."""
    if std > 0:
        return f"{mean:.3f}±{std:.3f}".rjust(width)
    else:
        return f"{mean:.3f}".rjust(width)


def print_summary_tables(results: List[Dict], ratios: Optional[List[float]] = None):
    """
    Print formatted summary tables from results.

    Embedding-free methods (with embedding='none') are included as baselines
    in every embedding's table for easy comparison.

    Args:
        results: List of result dicts with keys:
            dataset, embedding, method, ratio, balanced_accuracy
        ratios: Optional list of ratios to display (auto-detected if None)
    """
    if not results:
        print("\nNo results to summarize.")
        return

    df = pd.DataFrame(results)

    # Get unique values
    datasets = df['dataset'].unique()
    all_embeddings = df['embedding'].unique()
    if ratios is None:
        ratios = sorted(df['ratio'].unique())

    # Separate embedding-free results (baselines) from embedding-dependent results
    baseline_df = df[df['embedding'] == 'none']
    has_baselines = not baseline_df.empty

    # Get actual embeddings (excluding 'none')
    actual_embeddings = [e for e in all_embeddings if e != 'none']

    # If only embedding-free methods, show one table per dataset
    if not actual_embeddings:
        actual_embeddings = ['none']

    # Format ratio headers
    ratio_headers = [f"{r*100:.0f}%" for r in ratios]
    col_width = 12

    metrics = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced Accuracy')]
    # Add best metrics if present
    if 'best_accuracy' in df.columns and df['best_accuracy'].notna().any():
        metrics += [('best_accuracy', 'Best Accuracy'), ('best_balanced_accuracy', 'Best Balanced Accuracy')]
    # Add linear probe metrics if present in results
    if 'lp_accuracy' in df.columns and df['lp_accuracy'].notna().any():
        metrics += [('lp_accuracy', 'LP Accuracy'), ('lp_balanced_accuracy', 'LP Balanced Accuracy')]

    for dataset in datasets:
        for embedding in actual_embeddings:
            # Get embedding-dependent results for this embedding
            subset = df[(df['dataset'] == dataset) & (df['embedding'] == embedding)]

            # Include baselines (embedding='none') in this table
            if has_baselines and embedding != 'none':
                dataset_baselines = baseline_df[baseline_df['dataset'] == dataset]
                subset = pd.concat([dataset_baselines, subset], ignore_index=True)

            if subset.empty:
                continue

            # Get methods, with baselines first
            all_methods = subset['method'].unique()
            baseline_methods = baseline_df['method'].unique() if has_baselines else []
            # Sort: baselines first, then embedding-dependent methods
            methods = sorted([m for m in all_methods if m in baseline_methods])
            methods += sorted([m for m in all_methods if m not in baseline_methods])

            for metric_key, metric_label in metrics:
                if metric_key not in subset.columns:
                    continue

                agg = subset.groupby(['method', 'ratio'])[metric_key].agg(['mean', 'std']).reset_index()

                print(f"\n{'='*70}")
                if embedding == 'none':
                    print(f"Dataset: {dataset} | Embedding: none (embedding-free) | Metric: {metric_label}")
                else:
                    print(f"Dataset: {dataset} | Embedding: {embedding} | Metric: {metric_label}")
                print(f"{'='*70}")

                header = "Method".ljust(20) + "".join(h.rjust(col_width) for h in ratio_headers)
                print(header)
                print("-" * len(header))

                for method in methods:
                    # Mark baseline methods
                    if method in baseline_methods and embedding != 'none':
                        display_method = f"{method} *"
                    else:
                        display_method = method
                    row = display_method.ljust(20)
                    for ratio in ratios:
                        match = agg[(agg['method'] == method) & (agg['ratio'] == ratio)]
                        if not match.empty:
                            mean = match['mean'].values[0]
                            std = match['std'].values[0]
                            row += format_mean_std(mean, std, col_width)
                        else:
                            row += "-".rjust(col_width)
                    print(row)

                print("-" * len(header))


def analyze_run(
    run_id: str,
    output_dir: Optional[Path] = None,
    ratios: Optional[List[float]] = None
):
    """
    Analyze and print summary for a past run.

    Args:
        run_id: Run identifier (e.g., '20240110_143052')
        output_dir: Results directory (default: graphcov/results)
        ratios: Optional list of ratios to display
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    results_csv = output_dir / 'results.csv'

    if not results_csv.exists():
        print(f"Error: No results file found at {results_csv}")
        return

    # Load results
    df = pd.read_csv(results_csv)

    # Filter by run_id
    if run_id != 'all':
        if run_id not in df['run_id'].values:
            available = df['run_id'].unique()
            print(f"Error: Run '{run_id}' not found.")
            print(f"Available runs: {', '.join(available[:10])}")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
            return
        df = df[df['run_id'] == run_id]

    print(f"\nAnalyzing run: {run_id}")
    print(f"Total results: {len(df)}")

    # Load run config if available
    run_dir = output_dir / 'runs' / run_id
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Command: {config.get('command', 'N/A')}")

    # Convert to list of dicts for print_summary_tables
    results = df.to_dict('records')
    print_summary_tables(results, ratios)


def list_runs(output_dir: Optional[Path] = None, limit: int = 10):
    """List recent runs."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    results_csv = output_dir / 'results.csv'

    if not results_csv.exists():
        print("No results found.")
        return

    df = pd.read_csv(results_csv)

    # Get unique runs with counts
    run_counts = df.groupby('run_id').agg({
        'dataset': 'nunique',
        'embedding': 'nunique',
        'method': 'nunique',
        'balanced_accuracy': 'count'
    }).reset_index()
    run_counts.columns = ['run_id', 'datasets', 'embeddings', 'methods', 'total']

    # Sort by run_id (descending = most recent first)
    run_counts = run_counts.sort_values('run_id', ascending=False)

    print(f"\nRecent runs (showing {min(limit, len(run_counts))} of {len(run_counts)}):")
    print("-" * 70)
    print(f"{'Run ID':<20} {'Datasets':>10} {'Embeddings':>12} {'Methods':>10} {'Results':>10}")
    print("-" * 70)

    for _, row in run_counts.head(limit).iterrows():
        print(f"{row['run_id']:<20} {row['datasets']:>10} {row['embeddings']:>12} {row['methods']:>10} {row['total']:>10}")

    print("-" * 70)
    print(f"Use --analyze RUN_ID to see details for a specific run")
