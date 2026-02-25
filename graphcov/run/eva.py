"""
Training-dynamics based coreset selection methods.

This module implements EVA, AUM, and Forgetting - all computed during a single
200-epoch training pass for efficiency.

Methods:
- EVA: Dual-window variance of L2 error vectors
- AUM: Area Under Margin - average margin between true class and max other class
- Forgetting: Count of correct→incorrect transitions during training

References:
- EVA: Evolution-aware Variance for medical image coreset selection
- AUM: Pleiss et al., "Identifying Mislabeled Data using the Area Under the Margin Ranking"
- Forgetting: Toneva et al., "An Empirical Study of Example Forgetting during Deep Neural Network Learning"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from .embeddings import ResNet18WithFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_samples: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-sample metrics for all samples at current epoch.

    Returns three arrays:
    1. L2 error scores (for EVA): ||f_θ(x_i) - y_i||_2
    2. Margin scores (for AUM): P(true_class) - max(P(other_classes))
    3. Correctness (for Forgetting): 1 if correctly classified, 0 otherwise

    Args:
        model: Model to evaluate
        loader: DataLoader (should iterate through all samples)
        num_samples: Total number of samples
        num_classes: Number of classes

    Returns:
        Tuple of (l2_scores, margins, correctness) - each (num_samples,) array
    """
    model.eval()
    l2_scores = np.zeros(num_samples)
    margins = np.zeros(num_samples)
    correctness = np.zeros(num_samples, dtype=np.float32)

    idx = 0
    with torch.no_grad():
        for imgs, labels in loader:
            batch_size = len(labels)
            imgs = imgs.to(device, non_blocking=True)
            labels_flat = labels.to(device, non_blocking=True)
            if labels_flat.dim() > 1:
                labels_flat = labels_flat.squeeze(1)  # Squeeze label dim only, keep batch dim

            # Get predicted probabilities (softmax)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes)

            # === L2 Error (for EVA) ===
            one_hot = torch.zeros(batch_size, num_classes, device=device)
            one_hot.scatter_(1, labels_flat.unsqueeze(1), 1.0)
            error = probs - one_hot
            l2 = torch.norm(error, p=2, dim=1)  # (batch_size,)

            # === Margin (for AUM) ===
            # Get probability of true class
            prob_true = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # (batch_size,)
            # Get max probability of other classes
            # Set true class prob to -inf, then take max
            probs_masked = probs.clone()
            probs_masked.scatter_(1, labels_flat.unsqueeze(1), float('-inf'))
            prob_max_other = probs_masked.max(dim=1).values  # (batch_size,)
            margin = prob_true - prob_max_other  # (batch_size,)

            # === Correctness (for Forgetting) ===
            preds = logits.argmax(dim=1)
            correct = (preds == labels_flat).float()  # (batch_size,)

            # Store results
            l2_scores[idx:idx + batch_size] = l2.cpu().numpy()
            margins[idx:idx + batch_size] = margin.cpu().numpy()
            correctness[idx:idx + batch_size] = correct.cpu().numpy()
            idx += batch_size

    return l2_scores, margins, correctness


def compute_window_variance(scores_window: np.ndarray, window_name: str = "", verbose: bool = False) -> np.ndarray:
    """
    Compute variance of scores within a window.

    V^(i) = (1/K) * sum_{k=0}^{K-1} (S_k^(i) - mean^(i))^2

    Args:
        scores_window: (K, num_samples) array of scores across K epochs
        window_name: Name for verbose output
        verbose: Print variance statistics

    Returns:
        (num_samples,) array of variance scores
    """
    # Mean across epochs for each sample
    mean_scores = scores_window.mean(axis=0)

    # Variance across epochs for each sample
    variance = ((scores_window - mean_scores) ** 2).mean(axis=0)

    if verbose:
        percentiles = np.percentile(variance, [0, 25, 50, 75, 90, 99, 100])
        print(f"    [EVA] {window_name} variance: min={variance.min():.6f}, max={variance.max():.6f}, "
              f"mean={variance.mean():.6f}")
        print(f"    [EVA] {window_name} percentiles [0,25,50,75,90,99,100]: {[f'{p:.6f}' for p in percentiles]}")

    return variance


def derive_eva_scores(
    all_l2_scores: np.ndarray,
    window_size: int = 10,
    early_window_start: int = 1,
    late_window_start: int = 190,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive EVA scores from raw per-epoch L2 scores.

    Slices the appropriate windows from the full epoch history and computes
    dual-window variance. This is cheap (no training), so it can be called
    per-ratio without redundant computation.

    Args:
        all_l2_scores: (epochs, num_samples) array of L2 error scores
        window_size: Size of each window K
        early_window_start: Start epoch of early window (1-indexed)
        late_window_start: Start epoch of late window (1-indexed)
        verbose: Print details

    Returns:
        Tuple of (eva_scores, early_variance, late_variance)
    """
    # Convert 1-indexed epoch to 0-indexed array position
    early_l2 = all_l2_scores[early_window_start - 1 : early_window_start - 1 + window_size]
    late_l2 = all_l2_scores[late_window_start - 1 : late_window_start - 1 + window_size]

    early_variance = compute_window_variance(early_l2, "Early", verbose=verbose)
    late_variance = compute_window_variance(late_l2, "Late", verbose=verbose)
    eva_scores = early_variance + late_variance

    return eva_scores, early_variance, late_variance


def compute_training_dynamics(
    train_dataset,
    num_classes: int,
    in_channels: int,
    # Training params (matching EVA paper)
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    # Window params (for EVA)
    window_size: int = 10,
    early_window_start: int = 1,
    late_window_start: int = 190,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute training dynamics metrics: EVA, AUM, and Forgetting.

    All three metrics are computed during a single 200-epoch training pass:

    1. EVA (Evolution-aware Variance):
       - Records L2 error at each epoch within dual windows
       - Final score = variance(early_window) + variance(late_window)
       - Higher score = more important sample

    2. AUM (Area Under Margin):
       - Records margin = P(true) - max(P(other)) at each epoch
       - Final score = average margin across all epochs
       - Lower score = harder/noisier sample (select low AUM for hard samples)

    3. Forgetting:
       - Counts transitions from correct → incorrect prediction
       - Higher score = more forgetting events = more important sample

    Args:
        train_dataset: Full training dataset
        num_classes: Number of classes
        in_channels: Number of input channels
        epochs: Total training epochs (default: 200)
        batch_size: Batch size (default: 256)
        lr: Initial learning rate (default: 0.1)
        momentum: SGD momentum (default: 0.9)
        weight_decay: Weight decay (default: 0.0005)
        window_size: Size of each EVA window K (default: 10)
        early_window_start: Start epoch of early window (default: 1)
        late_window_start: Start epoch of late window (default: 190)
        seed: Random seed
        verbose: Show progress bar

    Returns:
        Dict with:
            - eva_scores: (n,) EVA scores (higher = more important)
            - early_variance: (n,) variance from early window
            - late_variance: (n,) variance from late window
            - aum_scores: (n,) AUM scores (lower = harder sample)
            - forgetting_scores: (n,) forgetting event counts (higher = more important)
            - metadata: Training metadata
    """
    import random

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_samples = len(train_dataset)

    # Create loader for full dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: keep order for score assignment
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)

    # SGD with momentum and cosine annealing (as per EVA paper)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Storage for EVA - collect ALL epochs for flexible window selection
    early_window_end = early_window_start + window_size
    late_window_end = late_window_start + window_size
    all_l2_scores = []  # Will be (epochs, num_samples)

    # Storage for AUM (all epochs)
    all_margins = []  # Will be (epochs, num_samples)

    # Storage for Forgetting
    forgetting_counts = np.zeros(num_samples, dtype=np.float32)
    prev_correctness = None  # Track previous epoch's correctness

    if verbose:
        print(f"[Training Dynamics] Computing EVA, AUM, Forgetting over {epochs} epochs...")
        print(f"[EVA] Early window: epochs {early_window_start}-{early_window_end-1}")
        print(f"[EVA] Late window: epochs {late_window_start}-{late_window_end-1}")

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training Dynamics") if verbose else range(1, epochs + 1)

    for epoch in epoch_iter:
        # Training step
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if labels.dim() > 1:
                labels = labels.squeeze(1)  # Squeeze label dim only, keep batch dim

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Compute metrics for this epoch
        l2_scores, margins, correctness = compute_epoch_metrics(
            model, eval_loader, num_samples, num_classes
        )

        # Store margins for AUM (all epochs)
        all_margins.append(margins)

        # Track forgetting events (correct → incorrect transitions)
        if prev_correctness is not None:
            # Forgetting event: was correct (1) in prev epoch, now incorrect (0)
            forgetting_events = (prev_correctness > 0.5) & (correctness < 0.5)
            forgetting_counts += forgetting_events.astype(np.float32)
        prev_correctness = correctness.copy()

        # Store L2 scores for all epochs
        all_l2_scores.append(l2_scores)

        if verbose:
            epoch_iter.set_postfix({
                'lr': f"{scheduler.get_last_lr()[0]:.4f}",
                'epoch': epoch,
                'forget': int(forgetting_counts.sum())
            })

    # === Compute EVA scores ===
    all_l2_scores = np.array(all_l2_scores)  # (epochs, num_samples)
    eva_scores, early_variance, late_variance = derive_eva_scores(
        all_l2_scores, window_size, early_window_start, late_window_start, verbose=verbose
    )

    # === Compute AUM scores ===
    all_margins = np.array(all_margins)  # (epochs, num_samples)
    aum_scores = all_margins.mean(axis=0)  # Average margin across epochs

    # === Forgetting scores already computed ===
    forgetting_scores = forgetting_counts

    if verbose:
        print(f"\n[EVA] Score range: [{eva_scores.min():.6f}, {eva_scores.max():.6f}]")
        print(f"[EVA] Mean: {eva_scores.mean():.6f}, Std: {eva_scores.std():.6f}")
        eva_percentiles = np.percentile(eva_scores, [0, 25, 50, 75, 100])
        print(f"[EVA] Percentiles [0,25,50,75,100]: {[f'{p:.4f}' for p in eva_percentiles]}")

        print(f"\n[AUM] Score range: [{aum_scores.min():.6f}, {aum_scores.max():.6f}]")
        print(f"[AUM] Mean: {aum_scores.mean():.6f}, Std: {aum_scores.std():.6f}")
        aum_percentiles = np.percentile(aum_scores, [0, 25, 50, 75, 100])
        print(f"[AUM] Percentiles [0,25,50,75,100]: {[f'{p:.4f}' for p in aum_percentiles]}")
        # AUM interpretation: high = easy (always correct), low = hard/noisy
        n_negative_margin = (aum_scores < 0).sum()
        print(f"[AUM] Samples with negative avg margin (often wrong): {n_negative_margin} ({100*n_negative_margin/num_samples:.1f}%)")

        print(f"\n[Forgetting] Score range: [{forgetting_scores.min():.0f}, {forgetting_scores.max():.0f}]")
        print(f"[Forgetting] Mean: {forgetting_scores.mean():.2f}, Std: {forgetting_scores.std():.2f}")
        n_never_forgotten = (forgetting_scores == 0).sum()
        n_often_forgotten = (forgetting_scores >= 5).sum()
        print(f"[Forgetting] Never forgotten: {n_never_forgotten} ({100*n_never_forgotten/num_samples:.1f}%)")
        print(f"[Forgetting] Forgotten 5+ times: {n_often_forgotten} ({100*n_often_forgotten/num_samples:.1f}%)")

    return {
        'eva_scores': eva_scores,
        'early_variance': early_variance,
        'late_variance': late_variance,
        'aum_scores': aum_scores,
        'forgetting_scores': forgetting_scores,
        'all_l2_scores': all_l2_scores,
        'metadata': {
            'epochs': epochs,
            'window_size': window_size,
            'early_window': (early_window_start, early_window_end - 1),
            'late_window': (late_window_start, late_window_end - 1),
            'seed': seed,
        }
    }


# Backward compatibility alias
def compute_eva_scores(
    train_dataset,
    num_classes: int,
    in_channels: int,
    **kwargs
) -> Dict[str, Any]:
    """Backward-compatible wrapper for compute_training_dynamics."""
    return compute_training_dynamics(train_dataset, num_classes, in_channels, **kwargs)


def select_by_eva(
    eva_scores: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int = 42,
    verbose: bool = False
) -> List[int]:
    """
    Select samples with highest EVA scores (per-class balanced).

    Note: The EVA paper describes "global top-M" selection, but this creates
    severe class imbalance. Per-class selection is required to match their
    reported results (~97% accuracy).

    Args:
        eva_scores: (n,) array of EVA scores
        labels: (n,) array of class labels
        budget_per_class: Number of samples to select per class
        seed: Random seed (for tie-breaking)
        verbose: Print selection details

    Returns:
        List of selected indices
    """
    np.random.seed(seed)
    selected = []

    if verbose:
        print(f"  [EVA Selection] Selecting top-{budget_per_class} per class by EVA score")

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        class_scores = eva_scores[class_indices]
        k = min(budget_per_class, len(class_indices))
        top_k_local = np.argsort(class_scores)[-k:]
        selected.extend(class_indices[top_k_local])

        if verbose:
            selected_scores = class_scores[top_k_local]
            print(f"    [EVA Selection] class={c}: selected {k}/{len(class_indices)}, "
                  f"score_range=[{selected_scores.min():.6f}, {selected_scores.max():.6f}]")

    return selected


# Optimal window settings from EVA paper (Section A7: Parameter Settings)
# Format: (selection_rate, early_start, late_start)
OPTIMAL_WINDOWS_ORGANAMNIST = {
    0.02: (1, 190),
    0.05: (1, 190),
    0.10: (100, 190),
    0.20: (1, 150),
    0.30: (90, 150),
    0.50: (90, 150),
    0.70: (1, 190),
    0.90: (90, 190),
}

OPTIMAL_WINDOWS_ORGANSMNIST = {
    0.02: (90, 100),
    0.05: (150, 190),
    0.10: (100, 190),
    0.20: (100, 190),
    0.30: (170, 190),
    0.50: (150, 190),
    0.70: (90, 150),
    0.90: (1, 100),
}


def get_optimal_windows(
    dataset_name: str,
    selection_rate: float,
    default_early: int = 1,
    default_late: int = 100,
    eva_epochs: int = 200,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Get optimal window settings from EVA paper.

    For datasets without paper-specified windows, defaults to (1, 100) which
    places the late window at mid-training where cosine-annealed LR is still
    ~0.05 and the model is actively learning. The EVA paper's default of
    late=190 is broken because cosine annealing drives LR to ~0.0006 by then,
    freezing the model and producing zero variance.

    Args:
        dataset_name: Dataset name (e.g., 'organamnist', 'organsmnist')
        selection_rate: Selection ratio (e.g., 0.02, 0.05)
        default_early: Default early window start
        default_late: Default late window start (mid-training, not end)
        eva_epochs: Total EVA training epochs (for computing fallback)
        verbose: Print window selection details

    Returns:
        (early_window_start, late_window_start)
    """
    dataset_lower = dataset_name.lower()

    if 'organamnist' in dataset_lower:
        windows = OPTIMAL_WINDOWS_ORGANAMNIST
        source = 'OPTIMAL_WINDOWS_ORGANAMNIST'
    elif 'organsmnist' in dataset_lower:
        windows = OPTIMAL_WINDOWS_ORGANSMNIST
        source = 'OPTIMAL_WINDOWS_ORGANSMNIST'
    else:
        # Default for other datasets: place late window at mid-training
        # where cosine-annealed LR is still meaningful (~0.05 at epoch 100/200)
        if verbose:
            print(f"  [EVA Windows] No optimal windows for {dataset_name}, using defaults: "
                  f"early={default_early}, late={default_late} "
                  f"(mid-training, LR still active)")
        return (default_early, default_late)

    # Find closest selection rate
    closest_rate = min(windows.keys(), key=lambda x: abs(x - selection_rate))
    early, late = windows[closest_rate]

    if verbose:
        print(f"  [EVA Windows] Dataset={dataset_name}, selection_rate={selection_rate}")
        print(f"  [EVA Windows] Using {source}[{closest_rate}]: early={early}, late={late}")
        if closest_rate != selection_rate:
            print(f"  [EVA Windows] Note: Exact rate {selection_rate} not found, using closest={closest_rate}")

    return (early, late)
