"""
Training and evaluation utilities.

Handles training ResNet on selected subsets and computing metrics.
Returns training history for analysis.

Supports two training paradigms:
1. Epoch-based (default): Fixed number of epochs, LR scheduler steps per epoch
2. Iteration-based: Fixed number of iterations, LR scheduler steps per iteration
   - Follows D2Pruning approach for fair comparison across subset sizes
   - Includes periodic testing and best accuracy tracking
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional

from .embeddings import ResNet18WithFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate selection quality via linear probe on embeddings.

    Trains logistic regression on selected embeddings and evaluates on
    test embeddings. Measures selection quality in the same representation
    space where selection was performed (no representation mismatch).

    Returns:
        accuracy, balanced_accuracy
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
    )
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)

    acc = accuracy_score(test_labels, preds)
    bal_acc = balanced_accuracy_score(test_labels, preds)

    if verbose:
        print(f"  [LinearProbe] Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}")

    return acc, bal_acc


def set_seed(seed: int, deterministic: bool = False):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable full determinism (slower, ~10-50% overhead).
                      If False, only set seeds (faster, mostly reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # These slow down training but ensure exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        # Faster but not 100% deterministic on GPU
        torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step_scheduler_per_iteration: bool = False
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Optional LR scheduler (stepped per iteration if step_scheduler_per_iteration=True)
        step_scheduler_per_iteration: If True, step scheduler after each batch

    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.dim() > 1:
            labels = labels.squeeze(1)  # Squeeze label dim only, keep batch dim

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Step scheduler per iteration (iteration-based paradigm)
        if step_scheduler_per_iteration and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    verbose: bool = False,
    verbose_per_class: bool = False,
    return_per_class: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, Dict[int, Dict]]]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        loader: DataLoader
        verbose: Print overall metrics
        verbose_per_class: Print per-class accuracy breakdown
        return_per_class: Return per-class results dict

    Returns:
        accuracy: Standard accuracy
        balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        per_class: (if return_per_class) Dict mapping class -> {accuracy, count}
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if labels.dim() > 1:
                labels = labels.squeeze(1)  # Squeeze label dim only, keep batch dim

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    if verbose:
        print(f"  [Eval] Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}")

    if verbose_per_class or return_per_class:
        unique_classes = np.unique(all_labels)
        per_class = {}
        for c in unique_classes:
            mask = all_labels == c
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class[int(c)] = {'accuracy': float(class_acc), 'count': int(mask.sum())}

        if verbose_per_class:
            per_class_acc = {c: v['accuracy'] for c, v in per_class.items()}
            print(f"  [Eval] Per-class accuracy: {per_class_acc}")
            worst_class = min(per_class_acc, key=per_class_acc.get)
            best_class = max(per_class_acc, key=per_class_acc.get)
            print(f"  [Eval] Worst class: {worst_class} ({per_class_acc[worst_class]:.4f}), "
                  f"Best class: {best_class} ({per_class_acc[best_class]:.4f})")

        if return_per_class:
            return acc, bal_acc, per_class

    return acc, bal_acc


def train_iterations(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    total_iterations: int,
    test_interval: int = 800,
    verbose: bool = True,
    verbose_per_class: bool = False
) -> Dict:
    """
    Train for a fixed number of iterations with per-iteration LR scheduling.

    This follows the D2Pruning approach for fair comparison across different
    subset sizes. The LR scheduler steps after each gradient update.

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader for periodic evaluation
        optimizer: Optimizer
        scheduler: LR scheduler (will be stepped per iteration)
        criterion: Loss function
        total_iterations: Total number of gradient steps
        test_interval: Evaluate on test set every N iterations
        verbose: Show progress bar

    Returns:
        Dict with keys:
            - final_acc: Final test accuracy
            - final_bal_acc: Final balanced accuracy
            - best_acc: Best test accuracy achieved
            - best_bal_acc: Best balanced accuracy achieved
            - best_iteration: Iteration where best accuracy was achieved
            - history: List of dicts with iteration-level metrics
    """
    model.train()
    data_iter = iter(train_loader)

    history = []
    best_acc = 0.0
    best_bal_acc = 0.0
    best_iteration = 0

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(range(total_iterations), desc="Training", leave=False) if verbose else range(total_iterations)

    for iteration in pbar:
        # Get next batch, cycling through data if needed
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, labels = next(data_iter)

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.dim() > 1:
            labels = labels.squeeze(1)  # Squeeze label dim only, keep batch dim

        # Forward pass
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler per iteration
        scheduler.step()

        # Track running stats
        running_loss += loss.item() * len(labels)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += len(labels)

        # Periodic evaluation
        if (iteration + 1) % test_interval == 0 or iteration == total_iterations - 1:
            acc, bal_acc = evaluate_model(model, test_loader)
            model.train()  # Switch back to train mode

            # Track best
            if bal_acc > best_bal_acc:
                best_acc = acc
                best_bal_acc = bal_acc
                best_iteration = iteration + 1

            # Record history
            avg_loss = running_loss / running_total if running_total > 0 else 0
            train_acc = running_correct / running_total if running_total > 0 else 0
            history.append({
                'iteration': iteration + 1,
                'train_loss': round(avg_loss, 6),
                'train_acc': round(train_acc, 6),
                'test_acc': round(acc, 6),
                'test_bal_acc': round(bal_acc, 6),
                'lr': round(scheduler.get_last_lr()[0], 8),
            })

            # Reset running stats
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'acc': f"{acc:.3f}",
                    'bal_acc': f"{bal_acc:.3f}",
                    'best': f"{best_bal_acc:.3f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })

    # Final evaluation
    final_acc, final_bal_acc, per_class = evaluate_model(model, test_loader, verbose=verbose,
                                                          verbose_per_class=verbose_per_class, return_per_class=True)

    return {
        'final_acc': final_acc,
        'final_bal_acc': final_bal_acc,
        'best_acc': best_acc,
        'best_bal_acc': best_bal_acc,
        'best_iteration': best_iteration,
        'history': history,
        'per_class': per_class,
    }


def evaluate_selection(
    train_dataset,
    test_dataset,
    selected_indices: List[int],
    num_classes: int,
    in_channels: int,
    # Training paradigm
    training_paradigm: str = 'epoch',
    epochs: int = 200,
    iterations: int = 40000,
    test_interval: int = 800,
    test_every_n_epochs: int = 10,
    # Optimizer settings
    batch_size: int = 256,
    lr: float = 0.1,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0005,
    # Augmentation
    augment: bool = False,
    size: int = 224,
    # Other settings
    seed: int = 42,
    return_history: bool = False,
    verbose: bool = True,
    verbose_per_class: bool = False,
    deterministic: bool = False,
    num_workers: int = 4
) -> Union[Tuple[float, float], Tuple[float, float, List[Dict]], Tuple[float, float, List[Dict], Dict]]:
    """
    Train ResNet on selected subset and evaluate.

    Supports two training paradigms:
    1. 'epoch' (default): Fixed epochs, LR scheduler steps per epoch
    2. 'iteration': Fixed iterations, LR scheduler steps per iteration
       - Follows D2Pruning approach for fair comparison across subset sizes
       - Includes periodic testing and best accuracy tracking

    Training protocol:
    - SGD with momentum (optionally Nesterov)
    - Cosine annealing learning rate scheduler
    - Weight decay
    - Optional data augmentation (RandomCrop + HorizontalFlip)

    Args:
        train_dataset: Full training dataset
        test_dataset: Test dataset
        selected_indices: Indices of selected samples
        num_classes: Number of classes
        in_channels: Number of input channels
        training_paradigm: 'epoch' or 'iteration' (default: 'epoch')
        epochs: Number of training epochs (default: 200, used when paradigm='epoch')
        iterations: Total training iterations (default: 40000, used when paradigm='iteration')
        test_interval: Test every N iterations (default: 800, used when paradigm='iteration')
        test_every_n_epochs: Test every N epochs (default: 10, used when paradigm='epoch')
        batch_size: Batch size (default: 256)
        lr: Initial learning rate (default: 0.1)
        momentum: SGD momentum (default: 0.9)
        nesterov: Use Nesterov momentum (default: False)
        weight_decay: Weight decay for regularization (default: 0.0005)
        augment: Enable data augmentation (RandomCrop + HorizontalFlip) (default: False)
        size: Image size for augmentation (default: 224)
        seed: Random seed
        return_history: If True, also return training history
        verbose: If True, show progress bar
        deterministic: If True, enable full determinism (slower)

    Returns:
        If training_paradigm='epoch':
            If return_history=False: (accuracy, balanced_accuracy)
            If return_history=True: (accuracy, balanced_accuracy, history)
        If training_paradigm='iteration':
            If return_history=False: (accuracy, balanced_accuracy)
            If return_history=True: (accuracy, balanced_accuracy, history, best_metrics)
                where best_metrics = {best_acc, best_bal_acc, best_iteration}
    """
    set_seed(seed, deterministic=deterministic)

    # Create data loaders
    subset = Subset(train_dataset, selected_indices)

    # Apply augmentation if requested
    if augment:
        from .data import wrap_with_augmentation
        subset = wrap_with_augmentation(subset, in_channels=in_channels, size=size)

    train_loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    # Initialize model
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)

    # SGD with momentum and weight decay (optionally Nesterov)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )

    criterion = nn.CrossEntropyLoss()

    if training_paradigm == 'iteration':
        # Iteration-based training with per-iteration LR scheduling
        # T_max = total iterations for proper cosine decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=1e-4
        )

        result = train_iterations(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            total_iterations=iterations,
            test_interval=test_interval,
            verbose=verbose,
            verbose_per_class=verbose_per_class
        )

        acc = result['final_acc']
        bal_acc = result['final_bal_acc']

        per_class = result['per_class']

        if return_history:
            best_metrics = {
                'best_acc': result['best_acc'],
                'best_bal_acc': result['best_bal_acc'],
                'best_iteration': result['best_iteration'],
            }
            return acc, bal_acc, result['history'], best_metrics, per_class
        return acc, bal_acc

    else:
        # Epoch-based training (original behavior)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = []
        best_acc = 0.0
        best_bal_acc = 0.0
        best_epoch = 0

        # Training loop
        epoch_iter = tqdm(range(epochs), desc="Training", leave=False) if verbose else range(epochs)

        for epoch in epoch_iter:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            # Evaluate on test set every N epochs
            test_acc, test_bal_acc = None, None
            if (epoch + 1) % test_every_n_epochs == 0:
                test_acc, test_bal_acc = evaluate_model(model, test_loader)

                if test_bal_acc > best_bal_acc:
                    best_acc = test_acc
                    best_bal_acc = test_bal_acc
                    best_epoch = epoch + 1

                if verbose and hasattr(epoch_iter, 'set_postfix'):
                    epoch_iter.set_postfix({
                        'loss': f"{train_loss:.4f}",
                        'acc': f"{test_acc:.3f}",
                        'bal_acc': f"{test_bal_acc:.3f}",
                        'best': f"{best_bal_acc:.3f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}",
                    })

            if return_history:
                entry = {
                    'epoch': epoch + 1,
                    'train_loss': round(train_loss, 6),
                    'train_acc': round(train_acc, 6),
                    'lr': round(scheduler.get_last_lr()[0], 8),
                }
                if test_acc is not None:
                    entry['test_acc'] = round(test_acc, 6)
                    entry['test_bal_acc'] = round(test_bal_acc, 6)
                history.append(entry)

        # Final evaluation
        acc, bal_acc, per_class = evaluate_model(model, test_loader, verbose=verbose,
                                                  verbose_per_class=verbose_per_class, return_per_class=True)

        # Check if final epoch is the best
        if bal_acc > best_bal_acc:
            best_acc = acc
            best_bal_acc = bal_acc
            best_epoch = epochs

        if return_history:
            best_metrics = {
                'best_acc': best_acc,
                'best_bal_acc': best_bal_acc,
                'best_epoch': best_epoch,
            }
            return acc, bal_acc, history, best_metrics, per_class
        return acc, bal_acc


def quick_evaluate(
    train_dataset,
    test_dataset,
    selected_indices: List[int],
    num_classes: int,
    in_channels: int,
    epochs: int = 50,
    batch_size: int = 256,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Quick evaluation with fewer epochs for rapid prototyping.

    Uses same training protocol (SGD, cosine annealing) but fewer epochs.
    """
    return evaluate_selection(
        train_dataset,
        test_dataset,
        selected_indices,
        num_classes,
        in_channels,
        training_paradigm='epoch',
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        return_history=False,
        verbose=False
    )
