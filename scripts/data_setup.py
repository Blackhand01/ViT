"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = os.cpu_count()
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates training and testing DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): Transformations to apply to the data.
        batch_size (int): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: Training DataLoader, testing DataLoader, class names.
    """
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
