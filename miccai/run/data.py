"""
Dataset loading utilities.

Handles MedMNIST dataset loading with consistent transforms.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO
from torchvision import transforms
from typing import Tuple, Dict, Any, Optional

# Available datasets
DATASETS = [
    'pneumoniamnist',
    'dermamnist',
    'pathmnist',
    'bloodmnist',
    'retinamnist',
    'breastmnist',
    'organamnist',
    'organcmnist',
    'organsmnist',
    'chestmnist',
    'octmnist',
    'tissuemnist',
]


def get_transform(in_channels: int, size: int = 224) -> transforms.Compose:
    """Get standard transform for MedMNIST (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
    ])


def get_train_transform(in_channels: int, size: int = 224) -> transforms.Compose:
    """
    Get training transform for MedMNIST with data augmentation.

    Augmentation strategy (matches CIFAR baseline from D2Pruning):
    - RandomCrop with padding: provides translation invariance
    - RandomHorizontalFlip: safe for most medical images

    Note: Vertical flip is NOT included as it can be problematic for
    some medical images where orientation matters (e.g., chest X-rays).

    Transform order (same as CIFAR for consistency):
    1. Spatial augmentation on PIL (RandomCrop, RandomHorizontalFlip)
    2. ToTensor
    3. Normalize
    """
    # Padding is proportional to image size (4 pixels for 32x32, scaled for other sizes)
    padding = max(2, size // 8)

    return transforms.Compose([
        # Spatial augmentation on PIL images (matches CIFAR baseline)
        transforms.RandomCrop(size, padding=padding, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        # Convert and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
    ])


class AugmentedDataset(Dataset):
    """
    Wrapper that applies augmentation transforms to an existing dataset.

    This is useful when you want to apply augmentation only during training,
    while the original dataset (used for embedding extraction) remains unchanged.
    """

    def __init__(self, dataset: Dataset, augment_transform: transforms.Compose):
        """
        Args:
            dataset: Original dataset (should return (tensor, label) tuples)
            augment_transform: Transform to apply for augmentation
        """
        self.dataset = dataset
        self.augment_transform = augment_transform

        # Get original normalization parameters from dataset transform if possible
        # We'll denormalize, augment, then renormalize
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup denormalization and renormalization transforms."""
        # Standard MedMNIST normalization
        self.mean = 0.5
        self.std = 0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Denormalize: x_orig = x * std + mean
        img_denorm = img * self.std + self.mean

        # Clamp to [0, 1] for safety
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # Convert to PIL for torchvision transforms
        # Note: img is (C, H, W), need to handle properly
        from torchvision.transforms.functional import to_pil_image

        # Convert tensor to PIL
        img_pil = to_pil_image(img_denorm)

        # Apply augmentation (which includes ToTensor and Normalize)
        img_aug = self.augment_transform(img_pil)

        return img_aug, label


def wrap_with_augmentation(dataset: Dataset, in_channels: int, size: int = 224) -> Dataset:
    """
    Wrap a dataset with augmentation transforms.

    Args:
        dataset: Original dataset (with standard transforms applied)
        in_channels: Number of input channels
        size: Image size

    Returns:
        Dataset with augmentation applied on-the-fly
    """
    aug_transform = get_train_transform(in_channels, size)
    return AugmentedDataset(dataset, aug_transform)


def load_dataset(
    name: str,
    split: str = 'train',
    size: int = 224,
    verbose: bool = False
) -> Tuple[Any, Dict]:
    """
    Load a MedMNIST dataset.

    Args:
        name: Dataset name (e.g., 'organsmnist')
        split: One of 'train', 'val', 'test'
        size: Image size (default 224 for foundation models)
        verbose: Print dataset details

    Returns:
        dataset: PyTorch dataset
        info: Dataset info dict with keys like 'n_channels', 'label', etc.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {DATASETS}")

    info = INFO[name]
    in_channels = info['n_channels']
    DataClass = getattr(medmnist, info['python_class'])

    transform = get_transform(in_channels, size)
    dataset = DataClass(split=split, transform=transform, download=True, size=size)

    if verbose:
        labels = get_labels(dataset)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        print(f"  [Data] Loaded {name}/{split}: n={len(dataset)}, size={size}x{size}, "
              f"channels={in_channels}, classes={n_classes}")
        print(f"  [Data] Class distribution: min={label_counts.min()}, max={label_counts.max()}, "
              f"mean={label_counts.mean():.1f}, std={label_counts.std():.1f}")
        if n_classes <= 15:
            print(f"  [Data] Per-class counts: {dict(zip(unique_labels, label_counts))}")

    return dataset, info


def get_labels(dataset) -> np.ndarray:
    """Extract labels as flat numpy array."""
    labels = dataset.labels
    if labels.ndim > 1:
        labels = labels.squeeze()
    return labels.astype(np.int64)


def get_dataset_info(name: str) -> Dict:
    """Get dataset info without loading data."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {DATASETS}")
    return INFO[name]
