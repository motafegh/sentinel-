"""
BCCC-SCsVul-2024 Dataset Loader

Loads 111,897 Solidity contracts with 241 pre-extracted features.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class BCCCDataset(Dataset):
    """PyTorch Dataset for BCCC smart contract vulnerabilities.
    
    Args:
        csv_path: Path to BCCC-SCsVul-2024.csv
        transform: Optional feature transformation
    
    Returns:
        Tuple of (features: Tensor[241], label: int)
    """
    
    def __init__(
        self, 
        csv_path: str,
        transform: Optional[callable] = None
    ):
        """Load and prepare BCCC dataset."""
        # Load CSV
        print(f"Loading BCCC dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} contracts")
        
        # Identify feature and label columns
        self.label_cols = [col for col in self.df.columns if col.startswith('Class')]
        self.feature_cols = [
            col for col in self.df.columns 
            if not col.startswith('Class') 
            and col not in ['ID', 'Unnamed: 0']  # Exclude ID and index
        ]

        
        # Extract features as numpy array
        # Extract ONLY numeric features (filter out string columns)
        self.numeric_feature_cols = [
            col for col in self.feature_cols 
            if self.df[col].dtype in ['int64', 'float64']
        ]
        print(f"Numeric features: {len(self.numeric_feature_cols)}/{len(self.feature_cols)}")

        # Extract features as numpy array
        self.features = self.df[self.numeric_feature_cols].values.astype('float32')
        
        # Convert one-hot labels to integer labels
        self.labels = self.df[self.label_cols].values.argmax(axis=1)
        
        # Store transform
        self.transform = transform
        
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Number of classes: {len(self.label_cols)}")
    
    def __len__(self) -> int:
        """Return total number of contracts."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get contract at index idx."""
        # Get features and label
        features = self.features[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = int(label)
        
        # Apply transform if exists
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_class_distribution(self):
        """Return dictionary of class counts."""
        unique, counts = torch.tensor(self.labels).unique(return_counts=True)
        class_names = [col.split(':')[1] for col in self.label_cols]
        return {class_names[i]: counts[i].item() for i in range(len(class_names))}
