"""
CLI entry point for miccai.run

Usage:
    # Basic selection methods
    python -m miccai.run --datasets organsmnist --methods fps --embeddings uni --ratios 0.02

    # With importance weighting (optional modifier)
    python -m miccai.run --datasets organsmnist --embeddings uni \
        --methods facility --importance test_attention --ratios 0.02

    # Compare methods with and without importance
    python -m miccai.run --datasets organsmnist --embeddings uni \
        --methods fps facility --importance test_attention uniform --ratios 0.02

    python -m miccai.run --list-methods
    python -m miccai.run --list-importance
    python -m miccai.run --list-datasets
    python -m miccai.run --list-embeddings

    python -m miccai.run --list-runs
    python -m miccai.run --analyze RUN_ID
"""

import argparse
import sys
from pathlib import Path

from .data import DATASETS
from .selection import get_available_methods, get_method_importance_mode, METHODS
from .importance import get_available_importance_methods
from .embeddings import get_available_sources
from .experiment import (
    run_experiment, list_methods, list_datasets, list_embeddings,
    list_importance, analyze_run, list_runs
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run sample selection experiments on MedMNIST datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic selection methods
  python -m miccai.run --datasets organsmnist --embeddings uni --methods fps --ratios 0.05 --trials 1

  # Multiple methods comparison
  python -m miccai.run --datasets organsmnist \
                       --embeddings uni \
                       --methods random fps facility graph_facility \
                       --ratios 0.02 0.05 0.1 \
                       --trials 3

  # With importance weighting (optional modifier for compatible methods)
  python -m miccai.run --datasets organsmnist \
                       --embeddings uni \
                       --methods facility fps \
                       --importance test_attention \
                       --ratios 0.02 0.05 \
                       --trials 3

  # Compare multiple importance methods
  python -m miccai.run --datasets organsmnist \
                       --embeddings uni \
                       --methods facility \
                       --importance uniform density test_attention \
                       --ratios 0.02 \
                       --trials 3

  # Methods that require importance (global_influence, graph_coverage)
  python -m miccai.run --datasets organsmnist \
                       --embeddings uni \
                       --methods global_influence \
                       --importance test_attention \
                       --ratios 0.02

  # With importance propagation
  python -m miccai.run --datasets organsmnist \
                       --embeddings uni \
                       --methods facility \
                       --importance test_attention \
                       --propagate --gamma 0.85 --walk-length 5 \
                       --ratios 0.02

  # EVA method (no embeddings needed)
  python -m miccai.run --datasets organsmnist --methods eva --ratios 0.02 0.05

  # List available options
  python -m miccai.run --list-methods
  python -m miccai.run --list-importance
  python -m miccai.run --list-datasets
  python -m miccai.run --list-embeddings

  # Analyze past runs
  python -m miccai.run --list-runs
  python -m miccai.run --analyze 20240110_143052
  python -m miccai.run --analyze all  # analyze all runs together
        """
    )

    # List options
    parser.add_argument('--list-methods', action='store_true',
                        help='List available selection methods')
    parser.add_argument('--list-importance', action='store_true',
                        help='List available importance methods')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    parser.add_argument('--list-embeddings', action='store_true',
                        help='List available embedding sources')

    # Analysis options
    parser.add_argument('--list-runs', action='store_true',
                        help='List recent experiment runs')
    parser.add_argument('--analyze', type=str, metavar='RUN_ID',
                        help='Analyze a past run (use "all" for all runs)')

    # Experiment configuration
    parser.add_argument('--datasets', type=str, nargs='+',
                        help=f'Datasets to use. Available: {", ".join(DATASETS[:5])}...')
    parser.add_argument('--embeddings', type=str, nargs='+',
                        help='Embedding sources: random, imagenet, trained, uni, conch')

    # Selection methods
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Selection methods (use --list-methods to see all)')

    # Optional importance modifier
    parser.add_argument('--importance', type=str, nargs='+',
                        help='Optional importance methods to weight selection '
                             '(cross-product with --methods). Methods with importance="optional" '
                             'will use these weights; methods with importance="ignore" will ignore them.')
    parser.add_argument('--propagate', action='store_true',
                        help='Propagate importance through graph before selection')

    # Importance parameters
    parser.add_argument('-t', '--temperature', type=float, default=0.1,
                        help='Temperature for test_attention (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.85,
                        help='Decay factor for importance propagation (default: 0.85)')
    parser.add_argument('--walk-length', type=int, default=5,
                        help='Walk length for propagation/coverage (default: 5)')
    parser.add_argument('--prop-iterations', type=int, default=3,
                        help='Propagation iterations (default: 3)')
    parser.add_argument('--damping', type=float, default=0.85,
                        help='Damping for propagation (default: 0.5)')

    # Common parameters
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.02, 0.05, 0.1],
                        help='Selection ratios (default: 0.02 0.05 0.1)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per config (default: 3)')

    # Training settings
    parser.add_argument('--training-paradigm', type=str, default='epoch',
                        choices=['epoch', 'iteration'],
                        help='Training paradigm: epoch-based (default) or iteration-based')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200, used when paradigm=epoch)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Total training iterations (default: 10000, used when paradigm=iteration)')
    parser.add_argument('--test-interval', type=int, default=1000,
                        help='Test every N iterations (default: 1000, used when paradigm=iteration)')
    parser.add_argument('--test-every-n-epochs', type=int, default=10,
                        help='Test every N epochs (default: 10, used when paradigm=epoch)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum (default: False)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--trained-epochs', type=int, default=200,
                        help='Training epochs for "trained" embedding source (default: 200)')

    # EVA method settings
    parser.add_argument('--eva-epochs', type=int, default=200,
                        help='EVA training epochs (default: 200)')
    parser.add_argument('--eva-window-size', type=int, default=10,
                        help='EVA window size K (default: 10)')
    parser.add_argument('--eva-size', type=int, default=28,
                        help='Image size for EVA/training-dynamics scoring (default: 28, matching EVA paper). '
                             'Scoring and evaluation use separate resolutions.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable full determinism (slower, ~10-50%% overhead)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation during training (RandomCrop + HorizontalFlip)')
    parser.add_argument('--size', type=int, default=224,
                        help='Image size (default: 224)')

    # Graph method settings
    parser.add_argument('-k', '--k-neighbors', type=int, default=10,
                        help='k for k-NN graph construction (default: 10)')
    parser.add_argument('--k-hops', type=int, default=None,
                        help='Number of hops for graph neighborhood (default: per-method, e.g. graph_a2=2)')
    parser.add_argument('--coverage-mode', type=str, default='prob',
                        choices=['prob', 'max'],
                        help='Coverage mode for global_influence: prob (probabilistic) or max (default: prob)')
    parser.add_argument('--global', dest='global_selection', action='store_true',
                        help='Build graph across all classes (global) instead of per-class. '
                             'Applies to: graph_a2, graph_coverage, heat_kernel, facility')
    parser.add_argument('--sparse-cpu', action='store_true',
                        help='Force sparse CPU path for greedy selection')
    parser.add_argument('--linear-probe', action='store_true',
                        help='Also evaluate via linear probe on embeddings (fast, no representation mismatch)')

    # I/O settings
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: miccai/results)')
    parser.add_argument('--cache', type=str, default=None,
                        help='Cache directory (default: miccai/cache/embeddings)')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of cached embeddings')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose output (show graph/selection/importance details)')
    parser.add_argument('--verbose-level', type=int, default=2, choices=[1, 2, 3],
                        help='Verbose detail level: 1=summary, 2=detailed, 3=per-iteration (default: 1)')

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    errors = []
    warnings = []

    if args.datasets:
        for ds in args.datasets:
            if ds not in DATASETS:
                errors.append(f"Unknown dataset: {ds}")

    if args.embeddings:
        available = get_available_sources()
        for emb in args.embeddings:
            if emb not in ['random', 'imagenet', 'trained', 'uni', 'conch']:
                errors.append(f"Unknown embedding source: {emb}")
            elif emb not in available:
                errors.append(f"Embedding '{emb}' not available (missing dependency)")

    if args.methods:
        available = get_available_methods()
        for method in args.methods:
            if method not in available:
                errors.append(f"Unknown method: {method}")

    if args.importance:
        available = get_available_importance_methods()
        for imp in args.importance:
            if imp not in available:
                errors.append(f"Unknown importance method: {imp}")

        # Warn if all methods ignore importance
        if args.methods:
            all_ignore = all(get_method_importance_mode(m) == 'ignore' for m in args.methods)
            if all_ignore:
                warnings.append(
                    f"All specified methods ignore importance. "
                    f"The --importance flag will have no effect on selection. "
                    f"Methods that use importance: fps, facility, graph_fps, graph_facility, "
                    f"greedy_importance, graph_coverage, global_influence"
                )

    if args.ratios:
        for r in args.ratios:
            if not 0 < r <= 1:
                errors.append(f"Invalid ratio: {r} (must be in (0, 1])")

    # Check size constraints for foundation models
    if args.embeddings and args.size:
        foundation_models = {'uni', 'conch'}
        selected_foundation = set(args.embeddings) & foundation_models
        if selected_foundation:
            # UNI/CONCH are ViT models with 16x16 patch size, minimum ~112-224 recommended
            if args.size < 112:
                errors.append(
                    f"Size {args.size} is too small for foundation models {selected_foundation}. "
                    f"ViT-based models (UNI, CONCH) require size >= 112 (224 recommended). "
                    f"Use --embeddings trained/imagenet/random for small images."
                )

    # Check for methods that require importance without --importance flag
    if args.methods and not args.importance:
        required_methods = [m for m in args.methods if get_method_importance_mode(m) == 'required']
        if required_methods:
            warnings.append(
                f"Methods {required_methods} work best with --importance. "
                f"Will use uniform importance as fallback."
            )

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def main():
    print(f"[CMD] python -u -m miccai.run {' '.join(sys.argv[1:])}")
    args = parse_args()

    # Handle list commands
    if args.list_methods:
        list_methods()
        return

    if args.list_importance:
        list_importance()
        return

    if args.list_datasets:
        list_datasets()
        return

    if args.list_embeddings:
        list_embeddings()
        return

    if args.list_runs:
        output_dir = Path(args.output) if args.output else None
        list_runs(output_dir)
        return

    if args.analyze:
        output_dir = Path(args.output) if args.output else None
        analyze_run(args.analyze, output_dir)
        return

    # Check required arguments
    if not args.datasets:
        print("Error: --datasets is required")
        print("Use --list-datasets to see available datasets")
        sys.exit(1)

    if not args.methods:
        print("Error: --methods is required")
        print("Use --list-methods to see available selection methods")
        sys.exit(1)

    # Validate first (catches unknown methods before misleading embeddings error)
    validate_args(args)

    # Methods that don't require embeddings at all
    # Training dynamics methods train their own model
    EMBEDDING_FREE_METHODS = {'eva', 'aum', 'forgetting', 'el2n_top', 'el2n_bottom', 'el2n_mid', 'random', 'random_proportional'}

    # Check if embeddings are required
    all_embedding_free = all(m in EMBEDDING_FREE_METHODS for m in args.methods)

    # If importance is specified with embedding-free methods, they'll need embeddings
    if args.importance and all_embedding_free:
        all_embedding_free = False  # Importance computation needs embeddings

    if not args.embeddings and not all_embedding_free:
        print("Error: --embeddings is required for the specified methods")
        print("Use --list-embeddings to see available sources")
        print("Note: Training dynamics methods (eva, aum, forgetting, el2n_*) and random methods can run without embeddings")
        sys.exit(1)

    # Build config
    config = {
        'datasets': args.datasets,
        'embeddings': args.embeddings,
        'methods': args.methods,
        'ratios': args.ratios,
        'trials': args.trials,
        # Training parameters
        'training_paradigm': args.training_paradigm,
        'epochs': args.epochs,
        'iterations': args.iterations,
        'test_interval': args.test_interval,
        'test_every_n_epochs': args.test_every_n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'nesterov': args.nesterov,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'trained_epochs': args.trained_epochs,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'augment': args.augment,
        'size': args.size,
        'force_recompute': args.force_recompute,
        'verbose': args.verbose,
        '_verbose_level': args.verbose_level,
        # Graph parameters
        'k_neighbors': args.k_neighbors,
        'k_hops': args.k_hops,
        'coverage_mode': args.coverage_mode,
        'global_selection': args.global_selection,
        'sparse_cpu': args.sparse_cpu,
        'linear_probe': args.linear_probe,
        # EVA parameters
        'eva_epochs': args.eva_epochs,
        'eva_window_size': args.eva_window_size,
        'eva_size': args.eva_size,
    }

    # Add importance-related config if specified
    if args.importance:
        config['importance_methods'] = args.importance
        config['propagate'] = args.propagate
        config['temperature'] = args.temperature
        config['gamma'] = args.gamma
        config['walk_length'] = args.walk_length
        config['prop_iterations'] = args.prop_iterations
        config['damping'] = args.damping

    if args.output:
        config['output_dir'] = Path(args.output)
    if args.cache:
        config['cache_dir'] = Path(args.cache)

    # Run
    run_experiment(config)


if __name__ == '__main__':
    main()
