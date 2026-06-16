"""
Data Quality Validation for SENTINEL Dataset

Purpose: Comprehensive validation of graph-token paired data
Author: Ali Motafegh
Date: February 16, 2026
"""

import torch
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

class DataQualityValidator:
    """
    Validates structural, statistical, and semantic quality of dataset.
    """
    
    def __init__(self, graph_dir: str, token_dir: str, metadata_path: str):
        self.graph_dir = Path(graph_dir)
        self.token_dir = Path(token_dir)
        self.metadata = pd.read_parquet(metadata_path)
        
        print("=" * 70)
        print("DATA QUALITY VALIDATOR - INITIALIZED")
        print("=" * 70)
        print(f"Graph directory: {self.graph_dir}")
        print(f"Token directory: {self.token_dir}")
        print(f"Metadata: {len(self.metadata)} contracts")
        print()
    
    def run_all_validations(self) -> Dict:
        """Run complete validation suite."""
        report = {}
        
        print("Starting validation suite...\n")
        
        # Phase 1: Structural
        print("Phase 1: Structural Validation")
        print("-" * 70)
        report['structural'] = self.validate_structure()
        
        # Phase 2: Statistical
        print("\nPhase 2: Statistical Analysis")
        print("-" * 70)
        report['statistical'] = self.validate_statistics()
        
        return report
    
    def validate_structure(self) -> Dict:
        """Phase 1: Structural validation."""
        results = {}
        
        # Check 1: File counts
        print("\n1. FILE INTEGRITY CHECK")
        graph_files = list(self.graph_dir.glob("*.pt"))
        token_files = list(self.token_dir.glob("*.pt"))
        
        results['file_counts'] = {
            'graphs': len(graph_files),
            'tokens': len(token_files),
            'expected': len(self.metadata[self.metadata['success'] == True])
        }
        
        print(f"   Graphs found: {len(graph_files):,}")
        print(f"   Tokens found: {len(token_files):,}")
        print(f"   Expected: {results['file_counts']['expected']:,}")
        
        # Check 2: Hash pairing
        print("\n2. HASH PAIRING CHECK")
        graph_hashes = {f.stem for f in graph_files}
        token_hashes = {f.stem for f in token_files}
        intersection = graph_hashes & token_hashes
        
        results['pairing'] = {
            'paired': len(intersection),
            'only_graphs': len(graph_hashes - token_hashes),
            'only_tokens': len(token_hashes - graph_hashes),
            'pairing_rate': len(intersection) / max(len(graph_hashes), len(token_hashes)) * 100
        }
        
        print(f"   Paired files: {len(intersection):,}")
        print(f"   Only graphs: {len(graph_hashes - token_hashes):,}")
        print(f"   Only tokens: {len(token_hashes - graph_hashes):,}")
        print(f"   Pairing rate: {results['pairing']['pairing_rate']:.2f}%")
        
        # Check 3: Schema validation (sample)
        print("\n3. SCHEMA VALIDATION (sampling 100 files)")
        
        sample_hashes = list(intersection)[:100]
        schema_issues = {
            'graph_issues': [],
            'token_issues': [],
            'pairing_issues': []
        }
        
        for hash_id in tqdm(sample_hashes, desc="   Validating schemas"):
            # Load files
            try:
                graph = torch.load(
                    self.graph_dir / f"{hash_id}.pt",
                    weights_only=False
                )
                token = torch.load(
                    self.token_dir / f"{hash_id}.pt",
                    weights_only=False
                )
                
                # Validate graph schema
                issues = self._validate_graph_schema(graph, hash_id)
                schema_issues['graph_issues'].extend(issues)
                
                # Validate token schema
                issues = self._validate_token_schema(token, hash_id)
                schema_issues['token_issues'].extend(issues)
                
                # Validate pairing
                if graph.contract_hash != token['contract_hash']:
                    schema_issues['pairing_issues'].append(
                        f"{hash_id}: Hash mismatch"
                    )
                if graph.contract_path != token['contract_path']:
                    schema_issues['pairing_issues'].append(
                        f"{hash_id}: Path mismatch"
                    )
                    
            except Exception as e:
                schema_issues['graph_issues'].append(f"{hash_id}: Load error - {e}")
        
        results['schema'] = {
            'samples_checked': len(sample_hashes),
            'graph_issues': len(schema_issues['graph_issues']),
            'token_issues': len(schema_issues['token_issues']),
            'pairing_issues': len(schema_issues['pairing_issues']),
            'details': schema_issues
        }
        
        print(f"\n   Samples checked: {len(sample_hashes)}")
        print(f"   Graph issues: {len(schema_issues['graph_issues'])}")
        print(f"   Token issues: {len(schema_issues['token_issues'])}")
        print(f"   Pairing issues: {len(schema_issues['pairing_issues'])}")
        
        return results
    
    def validate_statistics(self) -> Dict:
        """Phase 2: Statistical analysis."""
        results = {}
        
        # Get paired hashes
        graph_files = list(self.graph_dir.glob("*.pt"))
        token_files = list(self.token_dir.glob("*.pt"))
        graph_hashes = {f.stem for f in graph_files}
        token_hashes = {f.stem for f in token_files}
        paired_hashes = list(graph_hashes & token_hashes)
        
        # Sample for analysis (1000 samples for better statistics)
        sample_size = min(1000, len(paired_hashes))
        np.random.seed(42)  # Reproducibility
        sample_hashes = np.random.choice(paired_hashes, sample_size, replace=False)
        
        print(f"\nAnalyzing {sample_size:,} randomly sampled contracts...")
        print()
        
        # Collect statistics
        graph_stats = {
            'num_nodes': [],
            'num_edges': [],
            'labels': [],
            'node_to_edge_ratio': []
        }
        
        token_stats = {
            'num_tokens': [],
            'truncated': [],
            'padding_ratio': []
        }
        
        print("1. COLLECTING DATA STATISTICS")
        for hash_id in tqdm(sample_hashes, desc="   Processing samples"):
            try:
                # Load files
                graph = torch.load(
                    self.graph_dir / f"{hash_id}.pt",
                    weights_only=False
                )
                token = torch.load(
                    self.token_dir / f"{hash_id}.pt",
                    weights_only=False
                )
                
                # Graph statistics
                graph_stats['num_nodes'].append(graph.num_nodes)
                graph_stats['num_edges'].append(graph.num_edges)
                graph_stats['labels'].append(graph.y.item())
                
                # Node-to-edge ratio (complexity metric)
                if graph.num_nodes > 0:
                    ratio = graph.num_edges / graph.num_nodes
                    graph_stats['node_to_edge_ratio'].append(ratio)
                
                # Token statistics
                token_stats['num_tokens'].append(token['num_tokens'])
                token_stats['truncated'].append(token['truncated'])
                
                # Padding ratio (how much is real vs padding)
                real_tokens = token['attention_mask'].sum().item()
                padding_ratio = (512 - real_tokens) / 512
                token_stats['padding_ratio'].append(padding_ratio)
                
            except Exception as e:
                continue
        
        # Analyze label distribution
        print("\n2. LABEL DISTRIBUTION")
        labels = np.array(graph_stats['labels'])
        label_counts = {
            'safe': int(np.sum(labels == 0)),
            'vulnerable': int(np.sum(labels == 1))
        }
        total = len(labels)
        
        results['labels'] = {
            'safe_count': label_counts['safe'],
            'vulnerable_count': label_counts['vulnerable'],
            'safe_percentage': label_counts['safe'] / total * 100,
            'vulnerable_percentage': label_counts['vulnerable'] / total * 100,
            'balance_ratio': min(label_counts['safe'], label_counts['vulnerable']) / 
                           max(label_counts['safe'], label_counts['vulnerable'])
        }
        
        print(f"   Safe contracts: {label_counts['safe']:,} ({results['labels']['safe_percentage']:.1f}%)")
        print(f"   Vulnerable: {label_counts['vulnerable']:,} ({results['labels']['vulnerable_percentage']:.1f}%)")
        print(f"   Balance ratio: {results['labels']['balance_ratio']:.3f}")
        
        # Interpret balance
        if results['labels']['balance_ratio'] >= 0.8:
            print(f"   ✅ Well balanced!")
        elif results['labels']['balance_ratio'] >= 0.5:
            print(f"   ⚠️  Moderate imbalance (manageable with class weights)")
        else:
            print(f"   ❌ Severe imbalance (needs special handling)")
        
        # Analyze graph statistics
        print("\n3. GRAPH STATISTICS")
        nodes = np.array(graph_stats['num_nodes'])
        edges = np.array(graph_stats['num_edges'])
        ratios = np.array(graph_stats['node_to_edge_ratio'])
        
        results['graphs'] = {
            'nodes': {
                'mean': float(np.mean(nodes)),
                'median': float(np.median(nodes)),
                'std': float(np.std(nodes)),
                'min': int(np.min(nodes)),
                'max': int(np.max(nodes)),
                'p25': float(np.percentile(nodes, 25)),
                'p75': float(np.percentile(nodes, 75)),
                'p95': float(np.percentile(nodes, 95)),
                'p99': float(np.percentile(nodes, 99))
            },
            'edges': {
                'mean': float(np.mean(edges)),
                'median': float(np.median(edges)),
                'std': float(np.std(edges)),
                'min': int(np.min(edges)),
                'max': int(np.max(edges)),
                'p95': float(np.percentile(edges, 95)),
                'p99': float(np.percentile(edges, 99))
            },
            'node_to_edge_ratio': {
                'mean': float(np.mean(ratios)),
                'median': float(np.median(ratios))
            }
        }
        
        print(f"   Nodes per graph:")
        print(f"      Mean: {results['graphs']['nodes']['mean']:.1f}")
        print(f"      Median: {results['graphs']['nodes']['median']:.1f}")
        print(f"      Range: {results['graphs']['nodes']['min']} - {results['graphs']['nodes']['max']}")
        print(f"      95th percentile: {results['graphs']['nodes']['p95']:.1f}")
        
        print(f"\n   Edges per graph:")
        print(f"      Mean: {results['graphs']['edges']['mean']:.1f}")
        print(f"      Median: {results['graphs']['edges']['median']:.1f}")
        print(f"      Range: {results['graphs']['edges']['min']} - {results['graphs']['edges']['max']}")
        
        print(f"\n   Node-to-Edge ratio: {results['graphs']['node_to_edge_ratio']['mean']:.2f}")
        print(f"      (Higher = more connections per node)")
        
        # Analyze token statistics
        print("\n4. TOKEN STATISTICS")
        num_tokens = np.array(token_stats['num_tokens'])
        truncated = np.array(token_stats['truncated'])
        padding = np.array(token_stats['padding_ratio'])
        
        results['tokens'] = {
            'num_tokens': {
                'mean': float(np.mean(num_tokens)),
                'median': float(np.median(num_tokens)),
                'min': int(np.min(num_tokens)),
                'max': int(np.max(num_tokens)),
                'p95': float(np.percentile(num_tokens, 95))
            },
            'truncation': {
                'rate': float(np.mean(truncated)),
                'count': int(np.sum(truncated))
            },
            'padding': {
                'mean_ratio': float(np.mean(padding)),
                'median_ratio': float(np.median(padding))
            }
        }
        
        print(f"   Real tokens (before padding):")
        print(f"      Mean: {results['tokens']['num_tokens']['mean']:.1f}")
        print(f"      Median: {results['tokens']['num_tokens']['median']:.1f}")
        print(f"      Max: {results['tokens']['num_tokens']['max']}")
        
        print(f"\n   Truncation:")
        print(f"      Rate: {results['tokens']['truncation']['rate']*100:.1f}%")
        print(f"      ({results['tokens']['truncation']['count']:,} contracts truncated)")
        
        print(f"\n   Padding:")
        print(f"      Mean padding ratio: {results['tokens']['padding']['mean_ratio']*100:.1f}%")
        print(f"      (Lower = more real content)")
        
        # Outlier detection
        print("\n5. OUTLIER DETECTION")
        outliers = {
            'tiny_graphs': int(np.sum(nodes < 3)),
            'huge_graphs': int(np.sum(nodes > results['graphs']['nodes']['p99'])),
            'disconnected': int(np.sum(edges == 0)),
            'very_sparse': int(np.sum(ratios < 0.5))
        }
        
        results['outliers'] = outliers
        
        print(f"   Tiny graphs (<3 nodes): {outliers['tiny_graphs']}")
        print(f"   Huge graphs (>99th %ile): {outliers['huge_graphs']}")
        print(f"   Disconnected (0 edges): {outliers['disconnected']}")
        print(f"   Very sparse (<0.5 edges/node): {outliers['very_sparse']}")
        
        if sum(outliers.values()) < sample_size * 0.05:
            print(f"   ✅ <5% outliers (acceptable)")
        else:
            print(f"   ⚠️  >5% outliers (may need handling)")
        
        return results
    
    def _validate_graph_schema(self, graph, hash_id: str) -> List[str]:
        """Validate graph data structure."""
        issues = []
        
        # Check required attributes
        required = ['x', 'edge_index', 'edge_attr', 'y', 
                   'contract_hash', 'contract_path']
        for attr in required:
            if not hasattr(graph, attr):
                issues.append(f"{hash_id}: Missing attribute '{attr}'")
        
        if hasattr(graph, 'x'):
            # Check feature dimensions
            if graph.x.dim() != 2:
                issues.append(f"{hash_id}: x should be 2D, got {graph.x.dim()}D")
            elif graph.x.shape[1] != 8:
                issues.append(f"{hash_id}: x should have 8 features, got {graph.x.shape[1]}")
        
        if hasattr(graph, 'edge_index'):
            # Check edge format
            if graph.edge_index.dim() != 2:
                issues.append(f"{hash_id}: edge_index should be 2D")
            elif graph.edge_index.shape[0] != 2:
                issues.append(f"{hash_id}: edge_index should be [2, num_edges]")
        
        if hasattr(graph, 'y'):
            # Check label
            if graph.y.item() not in [0, 1]:
                issues.append(f"{hash_id}: y should be 0 or 1, got {graph.y.item()}")
        
        return issues
    
    def _validate_token_schema(self, token: Dict, hash_id: str) -> List[str]:
        """Validate token data structure."""
        issues = []
        
        # Check required keys
        required = ['input_ids', 'attention_mask', 'contract_hash', 
                   'num_tokens', 'truncated']
        for key in required:
            if key not in token:
                issues.append(f"{hash_id}: Missing key '{key}'")
        
        if 'input_ids' in token:
            # Check token length
            if token['input_ids'].shape[0] != 512:
                issues.append(f"{hash_id}: input_ids should be 512, got {token['input_ids'].shape[0]}")
        
        if 'attention_mask' in token:
            # Check mask values
            unique_vals = torch.unique(token['attention_mask'])
            if not all(v in [0, 1] for v in unique_vals):
                issues.append(f"{hash_id}: attention_mask should only have 0s and 1s")
        
        return issues
    
    def save_report(self, report: Dict, output_path: str = 'ml/analysis/validation_report.json'):
        """Save validation report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        json_report = self._make_json_serializable(report)
        
        with open(output_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print(f"Report saved to: {output_path}")
        print(f"{'=' * 70}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy/torch types to JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj


# Main execution
if __name__ == "__main__":
    validator = DataQualityValidator(
        graph_dir='ml/data/graphs',
        token_dir='ml/data/tokens',
        metadata_path='ml/data/processed/contracts_metadata.parquet'
    )
    
    report = validator.run_all_validations()
    validator.save_report(report)
