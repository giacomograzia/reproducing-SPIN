"""
Data loading and preprocessing utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class FACTDataset(Dataset):
    """Custom dataset for FACT AI project."""
    
    def __init__(self, data_path: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file
            transform: Optional transform to apply to data
        """
        self.data_path = data_path
        self.transform = transform
        # TODO: Load and preprocess data
        self.data = None
        self.labels = None
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data) if self.data is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item by index.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (data, label)
        """
        # TODO: Implement data retrieval
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


def load_data(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Load data and create DataLoader.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = FACTDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data.
    
    Args:
        data: Raw data DataFrame
        
    Returns:
        Preprocessed data DataFrame
    """
    # TODO: Implement preprocessing
    processed_data = data.copy()
    
    # Example preprocessing steps:
    # 1. Handle missing values
    # 2. Normalize numerical features
    # 3. Encode categorical variables
    # 4. Feature engineering
    
    return processed_data


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    np.random.seed(seed)
    
    # TODO: Implement data splitting
    n = len(data)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
    test_data = data.iloc[test_idx]
    
    return train_data, val_data, test_data
