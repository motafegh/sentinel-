"""
PyTorch Dataset for SolidiFI-Benchmark
Provides lazy loading of preprocessed smart contracts with line-level vulnerability labels

Features:
- Lazy loading (memory efficient)
- Line-level bug labels from BugLog CSVs
- Stratified train/val/test splitting
- Production-ready with error handling

Example:
    >>> dataset = SolidiFIDataset("ml/data/SolidiFI-processed")
    >>> print(len(dataset))  # 350
    >>> sample = dataset[0]
    >>> print(sample['labels']['bug_count'])  # Number of bugs in contract
    
    >>> # Create splits
    >>> splits = SolidiFIDataset.create_splits()
    >>> train_indices = splits['train']
"""

from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SolidiFIDataset(Dataset):
    """
    SolidiFI-Benchmark dataset for vulnerability detection.
    
    Dataset structure:
        SolidiFI-processed/
        ├── Re-entrancy_buggy_1.sol
        ├── Re-entrancy_buggy_2.sol
        ├── Timestamp-Dependency_buggy_1.sol
        └── ... (350 preprocessed contracts)
        
        SolidiFI-benchmark/buggy_contracts/
        ├── Re-entrancy/
        │   ├── buggy_1.sol
        │   ├── BugLog_1.csv
        │   └── ...
        └── ... (7 categories)
    
    Args:
        data_dir: Path to preprocessed contracts directory
        categories: List of vulnerability types to include (default: all 7)
    
    Attributes:
        CATEGORIES: List of 7 vulnerability types
        samples: List of sample metadata dicts
    """
    
    # Define vulnerability categories
    CATEGORIES = [
        "Re-entrancy",
        "Timestamp-Dependency", 
        "Unchecked-Send",
        "Unhandled-Exceptions",
        "TOD",
        "Overflow-Underflow",
        "tx.origin"
    ]
    
    def __init__(
        self, 
        data_dir: str = "ml/data/SolidiFI-processed",
        categories: Optional[List[str]] = None
    ):
        """
        Initialize dataset by discovering all preprocessed contract files.
        
        This does NOT load contract contents - only finds file paths!
        Memory usage: ~10KB for 350 file paths (very efficient)
        
        Args:
            data_dir: Path to directory with preprocessed .sol files
            categories: Subset of categories to load (default: all)
            
        Raises:
            FileNotFoundError: If data_dir doesn't exist
        """
        self.data_dir = Path(data_dir).expanduser()
        
        # Check if directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed data directory not found: {self.data_dir}\n"
                f"Did you run ml/scripts/preprocess_solidifi.py?"
            )
        
        # Use all categories if none specified
        self.categories = categories if categories else self.CATEGORIES
        
        # Storage for dataset index
        self.samples: List[Dict] = []
        
        # Build the dataset index
        self._build_index()
        
        # Print statistics
        self._print_stats()
    
    def _build_index(self):
        """
        Discover all preprocessed contract files and build internal index.
        
        File naming convention: {category}_buggy_{num}.sol
        Example: Re-entrancy_buggy_1.sol
        
        Populates self.samples with metadata dicts.
        """
        for category in self.categories:
            # Find all files for this category
            pattern = f"{category}_buggy_*.sol"
            contract_files = sorted(self.data_dir.glob(pattern))
            
            if not contract_files:
                print(f"⚠️  Warning: No contracts found for {category}")
                continue
            
            for contract_path in contract_files:
                # Extract contract number: Re-entrancy_buggy_26.sol → 26
                # Split by underscore, take last part, remove .sol
                parts = contract_path.stem.split("_")
                contract_num = parts[-1]  # "26"
                
                # Store metadata (NOT the file contents!)
                self.samples.append({
                    "contract_path": contract_path,
                    "category": category,
                    "contract_num": contract_num
                })
    
    def _print_stats(self):
        """Print dataset statistics for verification."""
        print("=" * 80)
        print("SOLIDIFI DATASET INITIALIZED")
        print("=" * 80)
        print(f"Total contracts: {len(self.samples)}")
        print(f"Data directory:  {self.data_dir}")
        print(f"\n📊 Distribution by category:")
        
        # Count contracts per category
        category_counts = Counter(sample["category"] for sample in self.samples)
        
        for category in self.CATEGORIES:
            count = category_counts.get(category, 0)
            if count > 0:
                percentage = (count / len(self.samples)) * 100
                print(f"  {category:25s}: {count:3d} ({percentage:.1f}%)")
        print("=" * 80)
    
    def _parse_buglog(self, category: str, contract_num: str) -> Dict:
        """
        Parse BugLog CSV file for vulnerability labels.
        
        BugLog location: ml/data/SolidiFI-benchmark/buggy_contracts/{category}/BugLog_{num}.csv
        BugLog format:
            loc,length,bug type,approach
            170,9,Re-entrancy,code snippet injection
            161,7,Re-entrancy,code snippet injection
        
        Args:
            category: Vulnerability category (e.g., "Re-entrancy")
            contract_num: Contract number (e.g., "26")
            
        Returns:
            Dictionary containing:
                - bug_locations: List of (line_number, length) tuples
                - bug_types: List of bug type strings
                - bug_count: Total number of bugs
                - has_bugs: Boolean
        """
        # Construct BugLog path
        buglog_dir = Path("ml/data/SolidiFI-benchmark/buggy_contracts") / category
        buglog_path = buglog_dir / f"BugLog_{contract_num}.csv"
        
        # Default return value (no bugs found)
        default_result = {
            "bug_locations": [],
            "bug_types": [],
            "bug_count": 0,
            "has_bugs": False
        }
        
        # Check if BugLog exists
        if not buglog_path.exists():
            print(f"⚠️  BugLog not found: {buglog_path}")
            return default_result
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(buglog_path)
            
            # Check if empty
            if df.empty:
                return default_result
            
            # Extract bug information
            bug_locations = []
            bug_types = []
            
            for _, row in df.iterrows():
                loc = int(row['loc'])
                length = int(row['length'])
                bug_type = str(row['bug type']).strip()
                
                bug_locations.append((loc, length))
                bug_types.append(bug_type)
            
            return {
                "bug_locations": bug_locations,
                "bug_types": bug_types,
                "bug_count": len(bug_locations),
                "has_bugs": len(bug_locations) > 0
            }
            
        except Exception as e:
            print(f"❌ Error parsing {buglog_path}: {e}")
            return default_result
    
    def __len__(self) -> int:
        """
        Return total number of samples.
        
        Called by DataLoader to determine batch count.
        
        Returns:
            Number of contracts in dataset (350 for full dataset)
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load ONE sample (contract + labels).
        
        This is where lazy loading happens!
        Called by DataLoader during training (batch_size times per batch).
        
        Args:
            idx: Index from 0 to len(dataset)-1
            
        Returns:
            Dictionary containing:
                - code: Preprocessed contract source code (string)
                - category: Vulnerability type (string)
                - contract_num: Contract number (string)
                - path: Full path to contract file (string)
                - labels: Parsed bug information (dict)
                    - bug_locations: List of (line, length) tuples
                    - bug_types: List of bug type strings
                    - bug_count: Integer count of bugs
                    - has_bugs: Boolean
        
        Example:
            >>> sample = dataset[0]
            >>> print(sample['labels']['bug_count'])  # 3
            >>> print(sample['labels']['bug_locations'])  # [(170, 9), (161, 7), ...]
        """
        # Get metadata for this sample
        sample = self.samples[idx]
        
        # Load preprocessed contract (LAZY LOADING - only when requested)
        try:
            code = sample["contract_path"].read_text(encoding="utf-8")
        except Exception as e:
            print(f"❌ Error loading {sample['contract_path']}: {e}")
            code = ""  # Return empty string on error
        
        # Parse BugLog labels
        labels = self._parse_buglog(
            category=sample["category"],
            contract_num=sample["contract_num"]
        )
        
        # Return structured data
        return {
            "code": code,
            "category": sample["category"],
            "contract_num": sample["contract_num"],
            "path": str(sample["contract_path"]),
            "labels": labels
        }
    
    def get_category_counts(self) -> Dict[str, int]:
        """
        Get contract counts per category.
        
        Returns:
            Dictionary mapping category name to count
            
        Example:
            >>> counts = dataset.get_category_counts()
            >>> print(counts['Re-entrancy'])  # 50
        """
        return dict(Counter(sample["category"] for sample in self.samples))
    
    def get_sample_by_category(self, category: str, num: int) -> Optional[Dict]:
        """
        Get specific contract by category and number.
        
        Args:
            category: Vulnerability type (e.g., "Re-entrancy")
            num: Contract number (1-50)
            
        Returns:
            Sample dict or None if not found
            
        Example:
            >>> sample = dataset.get_sample_by_category("Re-entrancy", 5)
            >>> print(sample['code'][:100])
        """
        for idx, sample in enumerate(self.samples):
            if sample["category"] == category and sample["contract_num"] == str(num):
                return self.__getitem__(idx)
        return None
    
    @staticmethod
    def create_splits(
        data_dir: str = "ml/data/SolidiFI-processed",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        save_path: str = "ml/data/solidifi_splits.json"
    ) -> Dict[str, List[int]]:
        """
        Create stratified train/val/test splits and save to JSON.
        
        Stratification ensures each split has balanced vulnerability categories.
        Uses scikit-learn's train_test_split with stratification.
        
        Args:
            data_dir: Path to preprocessed data
            train_ratio: Proportion for training (default: 0.7 = 70%)
            val_ratio: Proportion for validation (default: 0.15 = 15%)
            test_ratio: Proportion for test (default: 0.15 = 15%)
            random_seed: Random seed for reproducibility (default: 42)
            save_path: Where to save split indices JSON
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing index lists
            
        Example:
            >>> splits = SolidiFIDataset.create_splits()
            >>> print(f"Train: {len(splits['train'])}")  # 245
            >>> print(f"Val: {len(splits['val'])}")      # 53
            >>> print(f"Test: {len(splits['test'])}")    # 52
        """
        # Create temporary dataset to get indices
        dataset = SolidiFIDataset(data_dir=data_dir)
        
        # Get categories for stratification
        indices = list(range(len(dataset)))
        categories = [dataset.samples[i]["category"] for i in indices]
        
        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=categories,
            random_state=random_seed
        )
        
        # Get categories for temp split
        temp_categories = [categories[i] for i in temp_indices]
        
        # Second split: val vs test
        # Adjust ratio: val_ratio / (val_ratio + test_ratio)
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            stratify=temp_categories,
            random_state=random_seed
        )
        
        splits = {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices)
        }
        
        # Print statistics
        print("\n" + "="*80)
        print("TRAIN/VAL/TEST SPLIT CREATED")
        print("="*80)
        print(f"Total samples: {len(dataset)}")
        print(f"Random seed: {random_seed}")
        print(f"\nSplit sizes:")
        print(f"  Train: {len(splits['train']):3d} ({len(splits['train'])/len(dataset)*100:.1f}%)")
        print(f"  Val:   {len(splits['val']):3d} ({len(splits['val'])/len(dataset)*100:.1f}%)")
        print(f"  Test:  {len(splits['test']):3d} ({len(splits['test'])/len(dataset)*100:.1f}%)")
        
        # Check stratification
        print(f"\n📊 Category distribution per split:")
        for split_name, split_indices in splits.items():
            split_categories = [categories[i] for i in split_indices]
            category_counts = Counter(split_categories)
            print(f"\n  {split_name.upper()}:")
            for cat in dataset.CATEGORIES:
                count = category_counts.get(cat, 0)
                if count > 0:
                    print(f"    {cat:25s}: {count:2d}")
        
        # Save to JSON
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"\n✅ Splits saved to: {save_path}")
        print("="*80)
        
        return splits
    
    @staticmethod
    def load_splits(split_path: str = "ml/data/solidifi_splits.json") -> Dict[str, List[int]]:
        """
        Load previously saved splits from JSON.
        
        Args:
            split_path: Path to splits JSON file
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing index lists
            
        Raises:
            FileNotFoundError: If split file doesn't exist
            
        Example:
            >>> splits = SolidiFIDataset.load_splits()
            >>> train_indices = splits['train']
            >>> from torch.utils.data import Subset
            >>> train_dataset = Subset(dataset, train_indices)
        """
        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Run SolidiFIDataset.create_splits() first"
            )
        
        with open(split_path, 'r') as f:
            return json.load(f)
