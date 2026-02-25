"""
Embedding extraction with caching.

Handles multiple embedding sources:
- random: Random ResNet (untrained)
- imagenet: ImageNet-pretrained ResNet
- trained: Task-trained ResNet (200 epochs SGD, also produces EL2N/variance)
- uni: UNI foundation model (MahmoodLab)
- conch: CONCH foundation model (MahmoodLab)

EL2N/variance scores are computed separately via compute_el2n_scores() when needed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / 'cache' / 'embeddings'

# Available embedding sources (for feature extraction)
# Note: 'trained' trains a task-specific model; others use pretrained weights
EMBEDDING_SOURCES = ['random', 'imagenet', 'trained', 'uni', 'conch']

# Check available optional dependencies
TIMM_AVAILABLE = False
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    pass

CONCH_AVAILABLE = False
try:
    import timm.layers.attention_pool2d as _attn_pool
    from timm.layers import RotaryEmbedding as _RotaryEmbedding
    _attn_pool.RotaryEmbedding = _RotaryEmbedding
    from conch.open_clip_custom import create_model_from_pretrained
    CONCH_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Models
# =============================================================================

class ResNet18WithFeatures(nn.Module):
    """ResNet18 wrapper that exposes get_features method."""

    def __init__(self, num_classes: int, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        """Extract features before the final fc layer."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ViTFeatureExtractor(nn.Module):
    """Wrapper for ViT-based models (UNI) to extract features."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)
        if len(features.shape) == 3:
            features = features[:, 0, :]
        return features


class CONCHFeatureExtractor(nn.Module):
    """Wrapper for CONCH model to extract visual features."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.encode_image(x, proj_contrast=False, normalize=False)
        return features


def get_uni_model():
    """Load UNI foundation model."""
    if not TIMM_AVAILABLE:
        raise RuntimeError("timm not available for UNI")

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    model.eval()
    return ViTFeatureExtractor(model).to(device)


def get_conch_model():
    """Load CONCH foundation model."""
    if not CONCH_AVAILABLE:
        raise RuntimeError("conch package not available")

    model, _ = create_model_from_pretrained(
        'conch_ViT-B-16',
        "hf_hub:MahmoodLab/conch"
    )
    model.eval()
    return CONCHFeatureExtractor(model).to(device)


# =============================================================================
# Caching
# =============================================================================

def get_cache_path(
    dataset: str,
    split: str,
    source: str,
    size: int,
    seed: int = 42,
    trained_epochs: int = 200,
    cache_dir: Optional[Path] = None
) -> Path:
    """Generate cache file path for embeddings."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base = f"{dataset}_{split}_{source}_{size}"

    if source == 'trained':
        # Include epochs and seed since training is deterministic
        return cache_dir / f"{base}_e{trained_epochs}_s{seed}.npz"
    elif source == 'random':
        # Random depends on seed
        return cache_dir / f"{base}_s{seed}.npz"
    else:
        # Pretrained models (imagenet, uni, conch) don't depend on seed/epochs
        return cache_dir / f"{base}.npz"


def load_cached(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load cached embeddings if they exist."""
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return None


def save_cached(path: Path, data: Dict[str, np.ndarray]):
    """Save embeddings to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


# =============================================================================
# Indexed Dataset Wrapper
# =============================================================================

class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return indices and handle normalization."""

    def __init__(self, dataset, apply_imagenet_norm: bool = False):
        self.dataset = dataset
        self.apply_imagenet_norm = apply_imagenet_norm
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label_val = label.item() if hasattr(label, 'item') else int(label[0])

        if self.apply_imagenet_norm:
            # Convert to 3 channels if needed
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] != 3:
                img = img[:3, :, :]

            # Denormalize from [-1, 1] to [0, 1]
            img = img * 0.5 + 0.5
            # Apply ImageNet normalization
            img = (img - self.imagenet_mean) / self.imagenet_std

        return img, label_val, idx


# =============================================================================
# EL2N Score Computation (separate from embeddings)
# =============================================================================

def compute_el2n_scores(
    indexed_dataset,
    num_classes: int,
    in_channels: int,
    epochs_el2n: int = 10,
    epochs_var: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute EL2N scores and variance statistics via short training.

    This is a lightweight computation (20 epochs) specifically for EL2N/variance.
    NOT for embedding extraction - use 'trained' embedding source for that.

    Args:
        indexed_dataset: Dataset with (img, label, idx) returns
        num_classes: Number of classes
        in_channels: Number of input channels
        epochs_el2n: Epochs for EL2N collection (default: 10)
        epochs_var: Total epochs including variance collection (default: 20)
        batch_size: Batch size (default: 128)
        seed: Random seed
        verbose: Print progress

    Returns:
        el2n_scores: (n,) array of average EL2N scores
        variance_stats: (conf_variance, conf_mean) arrays
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        drop_last=True
    )
    n_samples = len(indexed_dataset)

    if verbose:
        print(f"  [EL2N] Starting: n={n_samples}, epochs_el2n={epochs_el2n}, epochs_var={epochs_var}")

    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Pre-allocate GPU tensor for EL2N scores
    el2n_tensor = torch.zeros(n_samples, epochs_el2n, device=device)

    for epoch in range(epochs_el2n):
        model.train()
        for imgs, labels, indices in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            indices_gpu = indices.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                y_onehot = F.one_hot(labels, num_classes=num_classes).float()
                el2n = (probs - y_onehot).norm(dim=-1)
                el2n_tensor[indices_gpu, epoch] = el2n

    # Average EL2N scores across epochs
    el2n_scores = el2n_tensor.mean(dim=1).cpu().numpy()

    if verbose:
        el2n_percentiles = np.percentile(el2n_scores, [0, 25, 50, 75, 100])
        print(f"  [EL2N] EL2N phase complete: "
              f"min={el2n_scores.min():.4f}, max={el2n_scores.max():.4f}, mean={el2n_scores.mean():.4f}")
        print(f"  [EL2N] EL2N percentiles [0,25,50,75,100]: {[f'{p:.4f}' for p in el2n_percentiles]}")

    # Continue training for variance computation
    n_var_epochs = epochs_var - epochs_el2n
    confidence_tensor = torch.zeros(n_samples, n_var_epochs, device=device)

    for epoch_idx in range(n_var_epochs):
        model.train()
        for imgs, labels, indices in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            indices_gpu = indices.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                confidence = probs[torch.arange(len(labels), device=device), labels]
                confidence_tensor[indices_gpu, epoch_idx] = confidence

    # Compute variance and mean
    conf_np = confidence_tensor.cpu().numpy()
    confidence_variance = np.var(conf_np, axis=1)
    confidence_mean = np.mean(conf_np, axis=1)

    if verbose:
        var_percentiles = np.percentile(confidence_variance, [0, 25, 50, 75, 100])
        print(f"  [EL2N] Variance phase complete: "
              f"var_range=[{confidence_variance.min():.4f}, {confidence_variance.max():.4f}], "
              f"conf_mean_range=[{confidence_mean.min():.4f}, {confidence_mean.max():.4f}]")
        print(f"  [EL2N] Variance percentiles [0,25,50,75,100]: {[f'{p:.4f}' for p in var_percentiles]}")

    return el2n_scores, (confidence_variance, confidence_mean)


def train_full_model(
    indexed_dataset,
    num_classes: int,
    in_channels: int,
    epochs: int = 200,
    el2n_epochs: int = 10,
    var_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[nn.Module, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Train a full model (matching evaluation setup) and compute EL2N/variance.

    Uses SGD with cosine annealing, same as evaluation training.
    Collects EL2N scores during early epochs and variance during mid-epochs
    as a byproduct of training (no extra cost).

    Args:
        indexed_dataset: Dataset with (img, label, idx) returns
        num_classes: Number of classes
        in_channels: Number of input channels
        epochs: Total training epochs (default: 200)
        el2n_epochs: Collect EL2N during first N epochs (default: 10)
        var_epochs: Collect variance during epochs el2n_epochs to var_epochs (default: 20)
        batch_size: Batch size (default: 256)
        lr: Initial learning rate (default: 0.1)
        momentum: SGD momentum (default: 0.9)
        weight_decay: Weight decay (default: 0.0005)
        seed: Random seed
        verbose: Print training progress

    Returns:
        model: Trained model
        el2n_scores: (n,) array of average EL2N scores from early training
        variance_stats: (conf_variance, conf_mean) arrays from mid training
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        drop_last=True
    )
    n_samples = len(indexed_dataset)

    if verbose:
        print(f"  [FullTraining] Starting: n={n_samples}, epochs={epochs}, "
              f"lr={lr}, batch_size={batch_size}")
        print(f"  [FullTraining] EL2N collection: epochs 1-{el2n_epochs}, "
              f"Variance collection: epochs {el2n_epochs+1}-{var_epochs}")

    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Pre-allocate tensors for EL2N and variance collection
    el2n_tensor = torch.zeros(n_samples, el2n_epochs, device=device)
    n_var_epochs = var_epochs - el2n_epochs
    confidence_tensor = torch.zeros(n_samples, n_var_epochs, device=device)

    pbar = tqdm(range(epochs), desc="Training", leave=False) if verbose else range(epochs)

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for imgs, labels, indices in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            indices_gpu = indices.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
            epoch_total += len(labels)

            # Collect EL2N during early epochs (epochs 0 to el2n_epochs-1)
            if epoch < el2n_epochs:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    y_onehot = F.one_hot(labels, num_classes=num_classes).float()
                    el2n = (probs - y_onehot).norm(dim=-1)
                    el2n_tensor[indices_gpu, epoch] = el2n

            # Collect variance during mid epochs (epochs el2n_epochs to var_epochs-1)
            elif epoch < var_epochs:
                var_idx = epoch - el2n_epochs
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    confidence = probs[torch.arange(len(labels), device=device), labels]
                    confidence_tensor[indices_gpu, var_idx] = confidence

        scheduler.step()

        if verbose and isinstance(pbar, tqdm):
            acc = epoch_correct / epoch_total
            pbar.set_postfix({
                'loss': f"{epoch_loss/epoch_total:.4f}",
                'acc': f"{acc:.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.4f}"
            })

    # Compute final scores
    el2n_scores = el2n_tensor.mean(dim=1).cpu().numpy()
    conf_np = confidence_tensor.cpu().numpy()
    confidence_variance = np.var(conf_np, axis=1)
    confidence_mean = np.mean(conf_np, axis=1)

    if verbose:
        print(f"  [FullTraining] Complete. Final train acc: {epoch_correct/epoch_total:.3f}")
        print(f"  [FullTraining] EL2N: [{el2n_scores.min():.4f}, {el2n_scores.max():.4f}], "
              f"mean={el2n_scores.mean():.4f}")
        print(f"  [FullTraining] Variance: [{confidence_variance.min():.4f}, {confidence_variance.max():.4f}]")

    model.eval()
    return model, el2n_scores, (confidence_variance, confidence_mean)


# =============================================================================
# Main Extraction Functions
# =============================================================================

def extract_embeddings_with_model(
    model: nn.Module,
    dataset,
    batch_size: int = 128,
    apply_imagenet_norm: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """
    Extract embeddings from a dataset using an existing model.

    This is useful for extracting embeddings from val/test sets using a model
    that was already trained on the train set (e.g., 'trained' source model).

    Args:
        model: Pre-trained model with get_features() method
        dataset: PyTorch dataset
        batch_size: Batch size for extraction
        apply_imagenet_norm: Whether to apply ImageNet normalization
        verbose: Print extraction details

    Returns:
        (n, d) array of embeddings
    """
    model.eval()
    indexed_dataset = IndexedDataset(dataset, apply_imagenet_norm=apply_imagenet_norm)
    loader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for imgs, _, indices in tqdm(loader, desc="Extracting embeddings", leave=False):
            imgs = imgs.to(device)
            features = model.get_features(imgs)
            all_embeddings.append(features.cpu().numpy())
            all_indices.extend(indices.numpy())

    embeddings = np.vstack(all_embeddings)

    # Reorder to match original indices
    all_indices_arr = np.array(all_indices)
    embeddings_ordered = np.empty((len(dataset), embeddings.shape[1]), dtype=embeddings.dtype)
    embeddings_ordered[all_indices_arr] = embeddings

    if verbose:
        print(f"  [Embeddings] Extracted using existing model: n={len(dataset)}, dim={embeddings.shape[1]}")

    return embeddings_ordered


def extract_embeddings(
    dataset,
    source: str,
    num_classes: int,
    in_channels: int,
    batch_size: int = 128,
    seed: int = 42,
    trained_epochs: int = 200,
    verbose: bool = False,
    return_model: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from a dataset.

    Args:
        dataset: PyTorch dataset
        source: Embedding source ('random', 'imagenet', 'trained', 'uni', 'conch')
        num_classes: Number of classes
        in_channels: Number of input channels
        batch_size: Batch size for extraction
        seed: Random seed
        trained_epochs: Epochs for 'trained' source (default: 200)
        verbose: Print extraction details
        return_model: If True, include the model in the returned dict (useful for reuse)

    Returns:
        Dict with keys:
            - 'embeddings': (n, d) array
            - 'el2n_scores': (n,) array (only for 'trained')
            - 'conf_variance': (n,) array (only for 'trained')
            - 'conf_mean': (n,) array (only for 'trained')
            - 'model': nn.Module (only if return_model=True)
    """
    if source not in EMBEDDING_SOURCES:
        raise ValueError(f"Unknown source: {source}. Available: {EMBEDDING_SOURCES}")

    use_imagenet_norm = source in ['imagenet', 'uni', 'conch']
    indexed_dataset = IndexedDataset(dataset, apply_imagenet_norm=use_imagenet_norm)

    result = {}
    model = None
    el2n_scores = None
    variance_stats = None

    if source == 'random':
        torch.manual_seed(seed)
        model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)
        model.eval()

    elif source == 'imagenet':
        model = ResNet18WithFeatures(num_classes, 3, pretrained=True).to(device)
        model.eval()

    elif source == 'trained':
        # Full training with SGD - produces high-quality task-specific embeddings
        # Also collects EL2N/variance as byproduct (no extra cost)
        indexed_dataset_trained = IndexedDataset(dataset, apply_imagenet_norm=False)
        model, el2n_scores, variance_stats = train_full_model(
            indexed_dataset_trained,
            num_classes,
            in_channels,
            epochs=trained_epochs,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose
        )
        indexed_dataset = indexed_dataset_trained

    elif source == 'uni':
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not available for UNI embeddings")
        model = get_uni_model()

    elif source == 'conch':
        if not CONCH_AVAILABLE:
            raise RuntimeError("conch not available for CONCH embeddings")
        model = get_conch_model()

    # Extract embeddings
    loader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for imgs, _, indices in tqdm(loader, desc=f"Extracting {source}", leave=False):
            imgs = imgs.to(device)

            if source in ['uni', 'conch']:
                features = model(imgs)
            else:
                features = model.get_features(imgs)

            all_embeddings.append(features.cpu().numpy())
            all_indices.extend(indices.numpy())

    embeddings = np.vstack(all_embeddings)

    # Reorder to match original indices
    all_indices_arr = np.array(all_indices)
    embeddings_ordered = np.empty((len(dataset), embeddings.shape[1]), dtype=embeddings.dtype)
    embeddings_ordered[all_indices_arr] = embeddings

    result['embeddings'] = embeddings_ordered

    if el2n_scores is not None:
        result['el2n_scores'] = el2n_scores
    if variance_stats is not None:
        result['conf_variance'] = variance_stats[0]
        result['conf_mean'] = variance_stats[1]
    if return_model and model is not None:
        result['model'] = model

    if verbose:
        n_samples, emb_dim = embeddings_ordered.shape
        extras = []
        if el2n_scores is not None:
            extras.append(f"el2n=[{el2n_scores.min():.3f}, {el2n_scores.max():.3f}]")
        if variance_stats is not None:
            extras.append(f"conf_var=[{variance_stats[0].min():.3f}, {variance_stats[0].max():.3f}]")
        extra_str = ", " + ", ".join(extras) if extras else ""
        print(f"  [Embeddings] {source}: n={n_samples}, dim={emb_dim}{extra_str}")

    return result


def load_or_compute_embeddings(
    dataset_name: str,
    split: str,
    source: str,
    dataset,
    num_classes: int,
    in_channels: int,
    size: int = 224,
    seed: int = 42,
    trained_epochs: int = 200,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    verbose: bool = False,
    return_model: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load embeddings from cache or compute them.

    Args:
        dataset_name: Name of dataset (for cache key)
        split: Dataset split
        source: Embedding source ('random', 'imagenet', 'trained', 'uni', 'conch')
        dataset: PyTorch dataset
        num_classes: Number of classes
        in_channels: Number of input channels
        size: Image size
        seed: Random seed
        trained_epochs: Epochs for 'trained' source (default: 200)
        cache_dir: Cache directory (default: graphcov/cache/embeddings)
        force_recompute: If True, ignore cache
        verbose: Print extraction details
        return_model: If True, include trained model in result (forces recompute if cached)

    Returns:
        Dict with embeddings and optionally el2n_scores, variance stats, model
    """
    cache_path = get_cache_path(
        dataset_name, split, source, size, seed, trained_epochs, cache_dir
    )

    # If return_model is requested, we need to compute fresh (can't get model from cache)
    use_cache = not force_recompute and not return_model

    if use_cache:
        cached = load_cached(cache_path)
        if cached is not None:
            if verbose:
                emb = cached['embeddings']
                print(f"  [Embeddings] Loaded {source} from cache: n={emb.shape[0]}, dim={emb.shape[1]}")
            else:
                print(f"  Loaded cached {source} embeddings from {cache_path.name}")
            return cached

    print(f"  Computing {source} embeddings...")
    data = extract_embeddings(
        dataset, source, num_classes, in_channels,
        seed=seed, trained_epochs=trained_epochs, verbose=verbose,
        return_model=return_model
    )

    # Don't cache the model - only cache embeddings and scores
    cache_data = {k: v for k, v in data.items() if k != 'model'}
    save_cached(cache_path, cache_data)
    if not verbose:
        print(f"  Cached to {cache_path.name}")

    return data


def get_available_sources() -> list:
    """Get list of available embedding sources based on installed packages."""
    available = ['random', 'imagenet', 'trained']
    if TIMM_AVAILABLE:
        available.append('uni')
    if CONCH_AVAILABLE:
        available.append('conch')
    return available


# =============================================================================
# EVA Score Computation
# =============================================================================

def compute_eva_embeddings(
    dataset,
    num_classes: int,
    in_channels: int,
    seed: int = 42,
    eva_epochs: int = 200,
    window_size: int = 10,
    early_window_start: int = 1,
    late_window_start: int = 190,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute training dynamics scores: EVA, AUM, and Forgetting.

    All metrics are computed during a single training pass for efficiency.

    EVA captures the evolutionary process of model training through a dual-window
    approach. AUM measures the margin between true class and competitors.
    Forgetting counts transitions from correct to incorrect predictions.

    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
        in_channels: Number of input channels
        seed: Random seed
        eva_epochs: Total training epochs (default: 200)
        window_size: Size of each window K (default: 10)
        early_window_start: Start epoch of early window (default: 1)
        late_window_start: Start epoch of late window (default: 190)
        verbose: Print progress

    Returns:
        Dict with keys:
            - 'eva_scores': (n,) array of EVA scores (higher = more important)
            - 'early_variance': (n,) variance from early window
            - 'late_variance': (n,) variance from late window
            - 'aum_scores': (n,) AUM scores (lower = harder sample)
            - 'forgetting_scores': (n,) forgetting event counts (higher = more important)
    """
    from .eva import compute_training_dynamics

    result = compute_training_dynamics(
        train_dataset=dataset,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=eva_epochs,
        batch_size=256,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        window_size=window_size,
        early_window_start=early_window_start,
        late_window_start=late_window_start,
        seed=seed,
        verbose=verbose,
    )

    return {
        'eva_scores': result['eva_scores'],
        'early_variance': result['early_variance'],
        'late_variance': result['late_variance'],
        'aum_scores': result['aum_scores'],
        'forgetting_scores': result['forgetting_scores'],
    }


def get_eva_cache_path(
    dataset: str,
    split: str,
    size: int,
    seed: int = 42,
    eva_epochs: int = 200,
    early_start: int = 1,
    late_start: int = 190,
    cache_dir: Optional[Path] = None
) -> Path:
    """Generate cache file path for EVA scores."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{dataset}_{split}_eva_{size}_e{eva_epochs}_w{early_start}_{late_start}_s{seed}.npz"


def load_or_compute_eva(
    dataset_name: str,
    split: str,
    dataset,
    num_classes: int,
    in_channels: int,
    size: int = 224,
    seed: int = 42,
    eva_epochs: int = 200,
    window_size: int = 10,
    early_window_start: int = 1,
    late_window_start: int = 190,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load EVA scores from cache or compute them.

    Note: EVA computation is expensive (requires 200-epoch training).

    Args:
        dataset_name: Name of dataset (for cache key)
        split: Dataset split
        dataset: PyTorch dataset
        num_classes: Number of classes
        in_channels: Number of input channels
        size: Image size
        seed: Random seed
        eva_epochs: Total training epochs
        window_size: Window size K
        early_window_start: Early window start epoch
        late_window_start: Late window start epoch
        cache_dir: Cache directory
        force_recompute: If True, ignore cache
        verbose: Print progress

    Returns:
        Dict with eva_scores, early_variance, late_variance, aum_scores, forgetting_scores
    """
    cache_path = get_eva_cache_path(
        dataset_name, split, size, seed, eva_epochs,
        early_window_start, late_window_start, cache_dir
    )

    if not force_recompute:
        cached = load_cached(cache_path)
        if cached is not None:
            if verbose:
                scores = cached['eva_scores']
                print(f"  [EVA] Loaded from cache: n={len(scores)}, range=[{scores.min():.6f}, {scores.max():.6f}]")
            else:
                print(f"  Loaded cached EVA scores from {cache_path.name}")
            return cached

    print(f"  Computing EVA scores (this requires {eva_epochs}-epoch training)...")
    data = compute_eva_embeddings(
        dataset, num_classes, in_channels,
        seed=seed,
        eva_epochs=eva_epochs,
        window_size=window_size,
        early_window_start=early_window_start,
        late_window_start=late_window_start,
        verbose=verbose
    )

    save_cached(cache_path, data)
    if not verbose:
        print(f"  Cached to {cache_path.name}")

    return data


def get_dynamics_cache_path(
    dataset: str,
    split: str,
    size: int,
    seed: int = 42,
    epochs: int = 200,
    cache_dir: Optional[Path] = None
) -> Path:
    """Generate cache file path for raw training dynamics (ratio-independent)."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{dataset}_{split}_dynamics_{size}_e{epochs}_s{seed}.npz"


def load_or_compute_raw_dynamics(
    dataset_name: str,
    split: str,
    dataset,
    num_classes: int,
    in_channels: int,
    size: int = 224,
    seed: int = 42,
    eva_epochs: int = 200,
    window_size: int = 10,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load or compute raw training dynamics (ratio-independent).

    Trains a model for eva_epochs and collects per-epoch L2 scores, AUM, and
    forgetting. The raw L2 scores can be used to derive EVA scores for any
    ratio's window parameters without retraining.

    Returns:
        Dict with all_l2_scores (epochs, n), aum_scores (n,), forgetting_scores (n,)
    """
    cache_path = get_dynamics_cache_path(
        dataset_name, split, size, seed, eva_epochs, cache_dir
    )

    if not force_recompute:
        cached = load_cached(cache_path)
        if cached is not None:
            if verbose:
                print(f"  [Dynamics] Loaded raw training dynamics from cache: {cache_path.name}")
            else:
                print(f"  Loaded cached training dynamics from {cache_path.name}")
            return cached

    print(f"  Computing training dynamics ({eva_epochs}-epoch training)...")
    from .eva import compute_training_dynamics

    result = compute_training_dynamics(
        train_dataset=dataset,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=eva_epochs,
        window_size=window_size,
        # Use default window positions — EVA scores will be re-derived per ratio
        early_window_start=1,
        late_window_start=eva_epochs - window_size,
        seed=seed,
        verbose=verbose,
    )

    # Cache only ratio-independent data
    raw_data = {
        'all_l2_scores': result['all_l2_scores'],
        'aum_scores': result['aum_scores'],
        'forgetting_scores': result['forgetting_scores'],
    }

    save_cached(cache_path, raw_data)
    if not verbose:
        print(f"  Cached training dynamics to {cache_path.name}")

    return raw_data
