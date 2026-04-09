"""
Generic DataLoader factories.

These functions work with any torch Dataset that:
  - Returns (image_tensor, label_tensor) from __getitem__
  - Exposes a .label_array() -> np.ndarray[int] method (required only when
    weighted_sampling=True)

Typical usage
-------------
    from src.data.loaders import build_loaders

    train_loader, val_loader = build_loaders(
        train_ds, val_ds,
        batch_size=32,
        weighted_sampling=True,   # up-samples minority class
    )
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def get_weighted_sampler(
    labels: np.ndarray,
    random_state: int = 42,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples the minority class.

    Parameters
    ----------
    labels : np.ndarray of int
        Integer class label for every sample in the dataset.
    random_state : int

    Returns
    -------
    WeightedRandomSampler
        Draws ``len(labels)`` samples with replacement, balanced across classes.
    """
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[labels]

    generator = torch.Generator().manual_seed(random_state)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def build_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    weighted_sampling: bool = True,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Parameters
    ----------
    train_ds, val_ds : torch.utils.data.Dataset
        Any dataset returning (image_tensor, label_tensor).
        If ``weighted_sampling=True``, ``train_ds`` must implement
        ``label_array() -> np.ndarray[int]``.
    batch_size : int
    num_workers : int
    weighted_sampling : bool
        If True (default), uses WeightedRandomSampler on the train loader to
        equalise class frequencies. Set to False if you handle class imbalance
        via loss weights instead.
    random_state : int
        Seed for the sampler's random generator.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    if weighted_sampling:
        sampler = get_weighted_sampler(train_ds.label_array(), random_state=random_state)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,        # mutually exclusive with shuffle=True
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
