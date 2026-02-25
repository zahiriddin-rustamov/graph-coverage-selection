"""Compare global vs per-class selection for a method.

Usage:
    python -m graphcov.run.compare_global --method graph_a2 \
        --dataset organsmnist --embedding uni --ratio 0.02

    # With training
    python -m graphcov.run.compare_global --method graph_a2 \
        --dataset organsmnist --embedding uni --ratio 0.02 \
        --train --epochs 200 --trials 3 --linear-probe
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

MODES = ['per-class', 'global']


def overlap_pct(a, b):
    return len(set(a) & set(b)) / len(a) * 100


def run_dataset(dataset_name, args):
    """Run global vs per-class comparison for a single dataset."""
    dataset, info = load_dataset(dataset_name, 'train', size=args.size, verbose=True)
    test_dataset, _ = load_dataset(dataset_name, 'test', size=args.size, verbose=False)
    labels = get_labels(dataset)
    num_classes = len(info['label'])
    in_channels = info['n_channels']

    need_model = (args.embedding == 'trained' and args.linear_probe)
    emb_data = load_or_compute_embeddings(
        dataset_name=dataset_name, split='train', source=args.embedding,
        dataset=dataset, num_classes=num_classes, in_channels=in_channels,
        size=args.size, seed=args.seed,
        cache_dir=Path('graphcov/cache/embeddings'), verbose=True,
        return_model=need_model)
    embeddings = emb_data['embeddings']
    trained_model = emb_data.get('model')

    budget_per_class = max(1, int(len(labels) * args.ratio / num_classes))
    total_selected = budget_per_class * num_classes
    print(f"\nDataset: {dataset_name} | Method: {args.method} | Embedding: {args.embedding}")
    print(f"Ratio: {args.ratio} | Budget/class: {budget_per_class} | "
          f"Total: {total_selected} | k={args.k_neighbors}")
    print(f"Modes: {MODES}\n")

    # --- Selection for each mode ---
    selections = {}
    selection_times = {}
    for mode in MODES:
        is_global = (mode == 'global')
        t0 = time.time()
        indices = select(
            method=args.method, labels=labels, budget_per_class=budget_per_class,
            embeddings=embeddings, seed=args.seed, verbose=False,
            k_neighbors=args.k_neighbors, global_selection=is_global,
            sparse_cpu=args.sparse_cpu)
        sel_time = time.time() - t0
        selections[mode] = indices
        selection_times[mode] = round(sel_time, 2)
        # Show class distribution
        sel_labels = labels[indices]
        counts = {c: int(np.sum(sel_labels == c)) for c in range(num_classes)}
        min_c, max_c = min(counts.values()), max(counts.values())
        print(f"  {mode:>10s}: {len(indices)} selected, "
              f"class range=[{min_c}, {max_c}], "
              f"counts={list(counts.values())}")

    # --- Overlap ---
    pct = overlap_pct(selections['per-class'], selections['global'])
    print(f"\nOverlap: {pct:.1f}% ({int(len(set(selections['per-class']) & set(selections['global'])))} / {len(selections['per-class'])})")

    # --- Linear probe ---
    lp_results = {}
    if args.linear_probe:
        if trained_model is not None:
            test_embeddings = extract_embeddings_with_model(
                model=trained_model, dataset=test_dataset,
                batch_size=128, apply_imagenet_norm=False, verbose=False)
        else:
            test_emb_data = load_or_compute_embeddings(
                dataset_name=dataset_name, split='test', source=args.embedding,
                dataset=test_dataset, num_classes=num_classes, in_channels=in_channels,
                size=args.size, seed=args.seed,
                cache_dir=Path('graphcov/cache/embeddings'), verbose=False)
            test_embeddings = test_emb_data['embeddings']
        test_labels = get_labels(test_dataset)

        print(f"\nLinear Probe Results:")
        print(f"{'mode':>12s}{'Accuracy':>12s}{'Bal. Acc.':>12s}")
        print("-" * 36)

        for mode in MODES:
            sel = selections[mode]
            acc, bal_acc = evaluate_linear_probe(
                train_embeddings=embeddings[sel],
                train_labels=labels[sel],
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                seed=args.seed)
            lp_results[mode] = (acc, bal_acc)
            print(f"{mode:>12s}{acc:>12.4f}{bal_acc:>12.4f}")
        print("-" * 36)

    # --- Full training ---
    train_results = {}
    all_histories = {}
    all_per_class = []  # per-class accuracy across all modes and trials
    if args.train:
        n_trials = args.trials
        print(f"\nFull Training ({args.epochs} epochs × {n_trials} trials):")
        header = (f"{'mode':>12s}{'Acc (m±s)':>16s}{'Bal Acc (m±s)':>16s}"
                  f"{'Best Acc (m±s)':>16s}{'Best Bal (m±s)':>16s}{'Best @':>8s}")
        print(header)
        print("-" * len(header))

        training_times = {}
        for mode in MODES:
            sel = selections[mode]
            accs, bal_accs = [], []
            best_accs, best_bal_accs, best_steps = [], [], []
            mode_histories = []
            mode_train_times = []
            for t in range(n_trials):
                seed = args.seed + t
                t0 = time.time()
                acc, bal_acc, history, best_metrics, per_class = evaluate_selection(
                    train_dataset=dataset,
                    test_dataset=test_dataset,
                    selected_indices=sel,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    training_paradigm='epoch',
                    epochs=args.epochs,
                    test_every_n_epochs=args.test_every_n_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    size=args.size,
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
                mode_histories.append(history)
                mode_train_times.append(round(train_time, 2))

                # Collect per-class results
                for cls, cls_data in per_class.items():
                    all_per_class.append({
                        'mode': mode,
                        'dataset': dataset_name,
                        'trial': t + 1,
                        'seed': seed,
                        'class': cls,
                        'class_accuracy': round(cls_data['accuracy'], 6),
                        'class_count': cls_data['count'],
                    })

            all_histories[mode] = mode_histories
            training_times[mode] = mode_train_times
            mean_acc, std_acc = np.mean(accs), np.std(accs)
            mean_bal, std_bal = np.mean(bal_accs), np.std(bal_accs)
            mean_bacc, std_bacc = np.mean(best_accs), np.std(best_accs)
            mean_bbal, std_bbal = np.mean(best_bal_accs), np.std(best_bal_accs)
            mean_step = np.mean(best_steps)
            train_results[mode] = {
                'acc': (mean_acc, std_acc), 'bal_acc': (mean_bal, std_bal),
                'best_acc': (mean_bacc, std_bacc), 'best_bal_acc': (mean_bbal, std_bbal),
                'best_step': mean_step,
            }
            print(f"{mode:>12s}"
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
            ms_header = f"{'mode':>12s}" + "".join(f"{'≤'+str(m):>10s}" for m in milestones)
            print(ms_header)
            print("-" * len(ms_header))

            for mode in MODES:
                row = f"{mode:>12s}"
                for milestone in milestones:
                    trial_bests = []
                    for hist in all_histories[mode]:
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
    save_selections(args.run_dir, {m: list(v) for m, v in selections.items()},
                    filename=f'selections_{dataset_name}.json')

    if lp_results or train_results:
        summary_rows = []
        for mode in MODES:
            row = {'mode': mode, 'dataset': dataset_name, 'selection_time_s': selection_times.get(mode)}
            if mode in lp_results:
                row['lp_acc'] = round(lp_results[mode][0], 6)
                row['lp_bal_acc'] = round(lp_results[mode][1], 6)
            if mode in train_results:
                r = train_results[mode]
                row['acc_mean'] = round(r['acc'][0], 6)
                row['acc_std'] = round(r['acc'][1], 6)
                row['bal_acc_mean'] = round(r['bal_acc'][0], 6)
                row['bal_acc_std'] = round(r['bal_acc'][1], 6)
                row['best_acc_mean'] = round(r['best_acc'][0], 6)
                row['best_bal_acc_mean'] = round(r['best_bal_acc'][0], 6)
                row['best_step'] = r['best_step']
                row['training_time_mean_s'] = round(np.mean(training_times[mode]), 2)
            summary_rows.append(row)
        save_training_history(args.run_dir, summary_rows,
                              filename=f'summary_{dataset_name}.csv')

    if all_histories:
        history_rows = []
        for mode in MODES:
            for t, hist in enumerate(all_histories[mode]):
                for entry in hist:
                    row = {'mode': mode, 'dataset': dataset_name, 'trial': t, 'seed': args.seed + t}
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
    parser = argparse.ArgumentParser(description='Compare global vs per-class selection')
    parser.add_argument('--method', required=True, help='Selection method to compare')
    parser.add_argument('--datasets', nargs='+', default=['organsmnist'], help='Datasets')
    parser.add_argument('--embedding', default='uni', help='Embedding source')
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('--k-neighbors', type=int, default=20, help='k for graph (default: 20)')
    parser.add_argument('--sparse-cpu', action='store_true',
                        help='Force sparse CPU path for greedy selection')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=224)

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
        'script': 'compare_global',
        'command': ' '.join(sys.argv),
        'started_at': start_time.isoformat(),
        'method': args.method,
        'datasets': args.datasets,
        'embedding': args.embedding,
        'ratio': args.ratio,
        'k_neighbors': args.k_neighbors,
        'size': args.size,
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
