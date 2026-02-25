"""Compare a single method across image sizes.

Usage:
    python -m graphcov.run.compare_sizes --method graph_a2 --sizes 28 64 128 224 \
        --dataset organsmnist --embedding uni --ratio 0.02

    # With linear probe evaluation
    python -m graphcov.run.compare_sizes --method graph_a2 --sizes 28 64 128 224 \
        --datasets organsmnist dermamnist --embedding uni --ratio 0.02 --linear-probe

    # Full training evaluation
    python -m graphcov.run.compare_sizes --method graph_a2 --sizes 28 128 224 \
        --dataset organsmnist --embedding uni --ratio 0.02 \
        --train --epochs 200 --trials 3
"""

import argparse
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from graphcov.run.data import load_dataset, get_labels
from graphcov.run.embeddings import load_or_compute_embeddings, extract_embeddings_with_model
from graphcov.run.selection import select
from graphcov.run.evaluation import evaluate_linear_probe, evaluate_selection
from graphcov.run.results import (
    generate_run_id, create_run_dir, save_config, save_summary,
    save_training_history, save_selections, get_git_commit,
)


def overlap_pct(a, b):
    return len(set(a) & set(b)) / len(a) * 100


def run_dataset(dataset_name, args):
    """Run size comparison for a single dataset."""
    sizes = args.sizes
    k = args.k_neighbors

    # Load data and embeddings per size (different resolutions)
    datasets_by_size = {}
    test_datasets_by_size = {}
    embeddings_by_size = {}
    models_by_size = {}
    labels = None
    num_classes = None
    in_channels = None

    for size in sizes:
        dataset, info = load_dataset(dataset_name, 'train', size=size, verbose=(size == sizes[0]))
        test_dataset, _ = load_dataset(dataset_name, 'test', size=size, verbose=False)

        if labels is None:
            labels = get_labels(dataset)
            num_classes = len(info['label'])
            in_channels = info['n_channels']

        need_model = (args.embedding == 'trained' and args.linear_probe)
        emb_data = load_or_compute_embeddings(
            dataset_name=dataset_name, split='train', source=args.embedding,
            dataset=dataset, num_classes=num_classes, in_channels=in_channels,
            size=size, seed=args.seed,
            cache_dir=Path('graphcov/cache/embeddings'), verbose=True,
            return_model=need_model)

        datasets_by_size[size] = dataset
        test_datasets_by_size[size] = test_dataset
        embeddings_by_size[size] = emb_data['embeddings']
        if emb_data.get('model') is not None:
            models_by_size[size] = emb_data['model']

    budget_per_class = max(1, int(len(labels) * args.ratio / num_classes))
    total_selected = budget_per_class * num_classes
    print(f"\nDataset: {dataset_name} | Method: {args.method} | Embedding: {args.embedding}")
    print(f"Ratio: {args.ratio} | Budget/class: {budget_per_class} | "
          f"Total: {total_selected} | k={k}")
    print(f"Sizes: {sizes}\n")

    # --- Selection for each size ---
    selections = {}
    selection_times = {}
    for size in sizes:
        t0 = time.time()
        indices = select(
            method=args.method, labels=labels, budget_per_class=budget_per_class,
            embeddings=embeddings_by_size[size], seed=args.seed, verbose=False,
            k_neighbors=k, global_selection=args.global_selection)
        sel_time = time.time() - t0
        selections[size] = indices
        selection_times[size] = round(sel_time, 2)
        print(f"  size={size:>4d}: {len(indices)} samples selected ({sel_time:.1f}s)")

    # --- Overlap table ---
    print(f"\nOverlap % (row selected ∩ col selected / row size):")
    print(f"{'size':>8s}", end='')
    for s in sizes:
        print(f"{str(s):>10s}", end='')
    print()

    for s1 in sizes:
        print(f"{str(s1):>8s}", end='')
        for s2 in sizes:
            if s1 == s2:
                print(f"{'---':>10s}", end='')
            else:
                pct = overlap_pct(selections[s1], selections[s2])
                print(f"{pct:>9.1f}%", end='')
        print()

    # --- Linear probe ---
    lp_results = {}
    if args.linear_probe:
        print(f"\nLinear Probe Results:")
        print(f"{'size':>8s}{'Accuracy':>12s}{'Bal. Acc.':>12s}")
        print("-" * 32)

        for size in sizes:
            sel = selections[size]
            emb = embeddings_by_size[size]

            # Get test embeddings
            if size in models_by_size:
                test_embeddings = extract_embeddings_with_model(
                    model=models_by_size[size], dataset=test_datasets_by_size[size],
                    batch_size=128, apply_imagenet_norm=False, verbose=False)
            else:
                test_emb_data = load_or_compute_embeddings(
                    dataset_name=dataset_name, split='test', source=args.embedding,
                    dataset=test_datasets_by_size[size], num_classes=num_classes,
                    in_channels=in_channels, size=size, seed=args.seed,
                    cache_dir=Path('graphcov/cache/embeddings'), verbose=False)
                test_embeddings = test_emb_data['embeddings']
            test_labels = get_labels(test_datasets_by_size[size])

            acc, bal_acc = evaluate_linear_probe(
                train_embeddings=emb[sel],
                train_labels=labels[sel],
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                seed=args.seed)
            lp_results[size] = (acc, bal_acc)
            print(f"{str(size):>8s}{acc:>12.4f}{bal_acc:>12.4f}")
        print("-" * 32)

    # --- Full training ---
    train_results = {}
    all_histories = {}
    all_per_class = []  # per-class accuracy across all sizes and trials
    if args.train:
        n_trials = args.trials
        print(f"\nFull Training ({args.epochs} epochs × {n_trials} trials):")
        header = (f"{'size':>8s}{'Acc (m±s)':>16s}{'Bal Acc (m±s)':>16s}"
                  f"{'Best Acc (m±s)':>16s}{'Best Bal (m±s)':>16s}{'Best @':>8s}")
        print(header)
        print("-" * len(header))

        training_times = {}
        for size in sizes:
            sel = selections[size]
            accs, bal_accs = [], []
            best_accs, best_bal_accs, best_steps = [], [], []
            size_histories = []
            size_train_times = []
            for t in range(n_trials):
                seed = args.seed + t
                t0 = time.time()
                acc, bal_acc, history, best_metrics, per_class = evaluate_selection(
                    train_dataset=datasets_by_size[size],
                    test_dataset=test_datasets_by_size[size],
                    selected_indices=sel,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    training_paradigm='epoch',
                    epochs=args.epochs,
                    test_every_n_epochs=args.test_every_n_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    size=size,
                    seed=seed,
                    return_history=True,
                    verbose=True,
                    num_workers=args.num_workers)
                train_time = time.time() - t0
                accs.append(acc)
                bal_accs.append(bal_acc)
                best_accs.append(best_metrics['best_acc'])
                best_bal_accs.append(best_metrics['best_bal_acc'])
                best_steps.append(best_metrics.get('best_epoch', 0))
                size_histories.append(history)
                size_train_times.append(round(train_time, 2))

                # Collect per-class results
                for cls, cls_data in per_class.items():
                    all_per_class.append({
                        'size': size,
                        'dataset': dataset_name,
                        'trial': t + 1,
                        'seed': seed,
                        'class': cls,
                        'class_accuracy': round(cls_data['accuracy'], 6),
                        'class_count': cls_data['count'],
                    })

            all_histories[size] = size_histories
            training_times[size] = size_train_times
            mean_acc, std_acc = np.mean(accs), np.std(accs)
            mean_bal, std_bal = np.mean(bal_accs), np.std(bal_accs)
            mean_bacc, std_bacc = np.mean(best_accs), np.std(best_accs)
            mean_bbal, std_bbal = np.mean(best_bal_accs), np.std(best_bal_accs)
            mean_step = np.mean(best_steps)
            train_results[size] = {
                'acc': (mean_acc, std_acc), 'bal_acc': (mean_bal, std_bal),
                'best_acc': (mean_bacc, std_bacc), 'best_bal_acc': (mean_bbal, std_bbal),
                'best_step': mean_step,
            }
            print(f"{str(size):>8s}"
                  f"{mean_acc:>8.4f}±{std_acc:<6.4f}"
                  f"{mean_bal:>8.4f}±{std_bal:<6.4f}"
                  f"{mean_bacc:>8.4f}±{std_bacc:<6.4f}"
                  f"{mean_bbal:>8.4f}±{std_bbal:<6.4f}"
                  f"{mean_step:>8.0f}")

        print("-" * len(header))

        # --- Milestone table ---
        total = args.epochs
        all_ms = [50, 100, 200, 300, 500, 750, 1000, 1500]
        milestones = [m for m in all_ms if m <= total]
        if len(milestones) > 1:
            print(f"\nBest Balanced Accuracy by Epoch Milestone (mean over {n_trials} trials):")
            ms_header = f"{'size':>8s}" + "".join(f"{'≤'+str(m):>10s}" for m in milestones)
            print(ms_header)
            print("-" * len(ms_header))

            for size in sizes:
                row = f"{str(size):>8s}"
                for milestone in milestones:
                    trial_bests = []
                    for hist in all_histories[size]:
                        best = 0.0
                        for entry in hist:
                            ep = entry.get('epoch', 0)
                            if ep <= milestone and 'test_bal_acc' in entry:
                                best = max(best, entry['test_bal_acc'])
                        trial_bests.append(best)
                    row += f"{np.mean(trial_bests):>10.4f}"
                print(row)

            print("-" * len(ms_header))

    # --- Save results to disk (namespaced by dataset) ---
    save_selections(args.run_dir, {s: list(v) for s, v in selections.items()},
                    filename=f'selections_{dataset_name}.json')

    if lp_results or train_results:
        summary_rows = []
        for size in sizes:
            row = {'size': size, 'dataset': dataset_name, 'selection_time_s': selection_times.get(size)}
            if size in lp_results:
                row['lp_acc'] = round(lp_results[size][0], 6)
                row['lp_bal_acc'] = round(lp_results[size][1], 6)
            if size in train_results:
                r = train_results[size]
                row['acc_mean'] = round(r['acc'][0], 6)
                row['acc_std'] = round(r['acc'][1], 6)
                row['bal_acc_mean'] = round(r['bal_acc'][0], 6)
                row['bal_acc_std'] = round(r['bal_acc'][1], 6)
                row['best_acc_mean'] = round(r['best_acc'][0], 6)
                row['best_bal_acc_mean'] = round(r['best_bal_acc'][0], 6)
                row['best_step'] = r['best_step']
                row['training_time_mean_s'] = round(np.mean(training_times[size]), 2)
            summary_rows.append(row)
        save_training_history(args.run_dir, summary_rows,
                              filename=f'summary_{dataset_name}.csv')

    if all_histories:
        history_rows = []
        for size in sizes:
            for t, hist in enumerate(all_histories[size]):
                for entry in hist:
                    row = {'size': size, 'dataset': dataset_name, 'trial': t, 'seed': args.seed + t}
                    row['epoch'] = entry.get('epoch', 0)
                    for col in ['train_loss', 'train_acc', 'test_acc', 'test_bal_acc', 'lr']:
                        if col in entry:
                            row[col] = entry[col]
                    history_rows.append(row)
        save_training_history(args.run_dir, history_rows,
                              filename=f'training_history_{dataset_name}.csv')

    if all_per_class:
        save_training_history(args.run_dir, all_per_class,
                              filename=f'per_class_{dataset_name}.csv')

    print(f"\nResults saved to: {args.run_dir}")

    return {
        'selections': selections,
        'lp_results': lp_results,
        'train_results': train_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare a method across image sizes')
    parser.add_argument('--method', required=True, help='Selection method to compare')
    parser.add_argument('--sizes', type=int, nargs='+', required=True, help='Image sizes to sweep')
    parser.add_argument('--datasets', nargs='+', default=['organsmnist'], help='Datasets')
    parser.add_argument('--embedding', default='uni', help='Embedding source')
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('--k-neighbors', type=int, default=20, help='Fixed k for graph (default: 20)')
    parser.add_argument('--global', dest='global_selection', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # Evaluation options
    parser.add_argument('--linear-probe', action='store_true', help='Run linear probe evaluation')
    parser.add_argument('--train', action='store_true', help='Run full training evaluation')
    parser.add_argument('--trials', type=int, default=3, help='Trials for training eval')

    # Training config (only used with --train)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--test-every-n-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')
    args = parser.parse_args()

    args.sizes = sorted(args.sizes)

    # Create run directory
    start_time = datetime.now()
    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)
    args.run_id = run_id
    args.run_dir = run_dir

    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")

    save_config(run_dir, {
        'run_id': run_id,
        'script': 'compare_sizes',
        'command': ' '.join(sys.argv),
        'started_at': start_time.isoformat(),
        'method': args.method,
        'sizes': args.sizes,
        'datasets': args.datasets,
        'embedding': args.embedding,
        'ratio': args.ratio,
        'k_neighbors': args.k_neighbors,
        'seed': args.seed,
        'linear_probe': args.linear_probe,
        'train': args.train,
        'trials': args.trials,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    })

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        run_dataset(dataset_name, args)
        print(f"{'='*60}")

    # Save summary
    save_summary(run_dir, {
        'run_id': run_id,
        'started_at': start_time.isoformat(),
        'finished_at': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds(),
        'status': 'completed',
        'git_commit': get_git_commit(),
    })


if __name__ == '__main__':
    main()
