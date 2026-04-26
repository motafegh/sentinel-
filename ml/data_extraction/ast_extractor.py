#!/usr/bin/env python
"""
AST Extractor V4.2 – PRODUCTION (Bug Fixes)

Fixes:
- Conditional --allow-paths (only for solc >= 0.5.0)
- Safe attribute access for events_emitted
- Better error handling for older Slither APIs

Author: Ali
Date: Feb 15, 2026
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import partial
import multiprocessing as mp
import warnings
import json
from datetime import datetime
import hashlib
import re

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from tqdm import tqdm
# Our hash utility (shared with tokenizer)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.hash_utils import get_contract_hash, get_filename_from_hash

try:
    from slither import Slither
    from slither.core.declarations import Contract, Function, Modifier, Event
except ImportError as e:
    raise ImportError("Slither not installed. Run: pip install slither-analyzer") from e

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("torch-geometric not installed. Run: pip install torch-geometric")


# # ============================================================================
# # HASH UTILITIES
# # ============================================================================

# def get_contract_hash(contract_path: str) -> str:
#     """Generate MD5 hash for contract identification."""
#     abs_path = str(Path(contract_path).resolve())
#     return hashlib.md5(abs_path.encode('utf-8')).hexdigest()


# def get_filename_from_hash(hash_str: str) -> str:
#     """Generate .pt filename from hash."""
#     return f"{hash_str}.pt"


# ============================================================================
# CONFIGURATION
# ============================================================================

NODE_TYPES = {
    'STATE_VAR': 0,
    'FUNCTION': 1,
    'MODIFIER': 2,
    'EVENT': 3,
    'FALLBACK': 4,
    'RECEIVE': 5,
    'CONSTRUCTOR': 6,
    'CONTRACT': 7
}

EDGE_TYPES = {
    'CALLS': 0,
    'READS': 1,
    'WRITES': 2,
    'EMITS': 3,
    'INHERITS': 4
}

VISIBILITY_MAP = {
    "public": 0,
    "external": 0,
    "internal": 1,
    "private": 2
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_solc_version(version: str) -> tuple:
    """
    Parse solc version string to (major, minor, patch) tuple.
    
    Examples:
        '0.4.26' -> (0, 4, 26)
        '0.5.0' -> (0, 5, 0)
        '0.8.19' -> (0, 8, 19)
    """
    try:
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
        if match:
            return tuple(int(x) for x in match.groups())
    except:
        pass
    return (0, 0, 0)


def solc_supports_allow_paths(version: str) -> bool:
    """
    Check if solc version supports --allow-paths flag.
    
    --allow-paths was introduced in solc 0.5.0
    """
    major, minor, patch = parse_solc_version(version)
    return (major, minor) >= (0, 5)


def get_solc_binary(version: str) -> Optional[str]:
    """Resolve solc binary path in Poetry virtualenv."""
    if not version:
        return None
    
    venv_path = Path.cwd() / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    candidates = [
        venv_path / f"solc-{version}",
        venv_path / "solc"
    ]
    
    for p in candidates:
        if p.exists():
            return str(p)
    
    return None


def get_project_root() -> Path:
    """Dynamically compute project root."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent.parent


def node_features(obj: Any, node_type_id: int) -> List[float]:
    """Extract 8-dimensional node features."""
    type_id = float(node_type_id)
    visibility = 0.0
    pure = 0.0
    view = 0.0
    payable = 0.0
    reentrant = 0.0
    complexity = 0.0
    loc = 0.0
    
    if hasattr(obj, "visibility") and obj.visibility:
        visibility = float(VISIBILITY_MAP.get(str(obj.visibility), 0))
    
    if hasattr(obj, "source_mapping") and obj.source_mapping:
        lines_attr = getattr(obj.source_mapping, 'lines', None)
        if lines_attr:
            loc = float(len(lines_attr)) if isinstance(lines_attr, list) else float(lines_attr)
    
    if isinstance(obj, Function):
        pure = 1.0 if obj.pure else 0.0
        view = 1.0 if obj.view else 0.0
        payable = 1.0 if obj.payable else 0.0
        reentrant = 1.0 if getattr(obj, 'is_reentrant', False) else 0.0
        
        try:
            complexity = float(len(obj.nodes)) if obj.nodes else 0.0
        except:
            complexity = 0.0
        
        if obj.is_constructor:
            type_id = float(NODE_TYPES['CONSTRUCTOR'])
        elif obj.is_fallback:
            type_id = float(NODE_TYPES['FALLBACK'])
        elif obj.is_receive:
            type_id = float(NODE_TYPES['RECEIVE'])
    
    return [type_id, visibility, pure, view, payable, reentrant, complexity, loc]


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class ASTExtractorV4:
    """Production-grade graph extractor with MD5 naming."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = get_project_root()
    
    def _get_slither_instance(
        self,
        contract_path: str,
        solc_binary: Optional[str] = None,
        solc_version: Optional[str] = None
    ) -> Optional[Slither]:
        """
        Create Slither instance with proper import resolution.
        
        🔧 FIX: Only use --allow-paths for solc >= 0.5.0
        """
        try:
            # Check if this solc version supports --allow-paths
            solc_args = None
            if solc_version and solc_supports_allow_paths(solc_version):
                allow_paths = f"--allow-paths .,{self.project_root}"
                solc_args = allow_paths
            
            sl = Slither(
                str(contract_path),
                solc=str(solc_binary) if solc_binary else None,
                solc_args=solc_args,
                detectors_to_run=[],
            )
            return sl
            
        except Exception as e:
            if self.verbose:
                print(f"  Slither failed for {Path(contract_path).name}: {e}")
            return None
    
    def contract_to_pyg(
        self,
        contract_path: str,
        solc_binary: Optional[str] = None,
        solc_version: Optional[str] = None,
        label: int = 0
    ) -> Optional[Data]:
        """
        Convert Solidity contract to PyTorch Geometric Data object.
        
        🔧 FIX: Safe attribute access for Slither API compatibility
        """
        # Get Slither instance
        sl = self._get_slither_instance(contract_path, solc_binary, solc_version)
        if sl is None or not sl.contracts:
            return None
        
        contracts = [c for c in sl.contracts if not c.is_from_dependency()]
        if not contracts:
            return None
        
        contract = contracts[0]
        
        # ====================================================================
        # BUILD NODES
        # ====================================================================
        
        node_features_list = []
        node_map = {}
        
        def add_node(obj: Any, node_type_id: int) -> None:
            """Add node to graph if not already present."""
            if isinstance(obj, Contract):
                name = obj.name
            else:
                name = obj.canonical_name
            
            if name in node_map:
                return
            
            node_map[name] = len(node_features_list)
            node_features_list.append(node_features(obj, node_type_id))
        
        # Add contract node
        add_node(contract, NODE_TYPES['CONTRACT'])
        
        # Add state variables
        for var in contract.state_variables:
            add_node(var, NODE_TYPES['STATE_VAR'])
        
        # Add functions
        for func in contract.functions:
            if func.is_constructor:
                add_node(func, NODE_TYPES['CONSTRUCTOR'])
            elif func.is_fallback:
                add_node(func, NODE_TYPES['FALLBACK'])
            elif func.is_receive:
                add_node(func, NODE_TYPES['RECEIVE'])
            else:
                add_node(func, NODE_TYPES['FUNCTION'])
        
        # Add modifiers
        for mod in contract.modifiers:
            add_node(mod, NODE_TYPES['MODIFIER'])
        
        # Add events
        for event in contract.events:
            add_node(event, NODE_TYPES['EVENT'])
        
        # ====================================================================
        # BUILD EDGES
        # ====================================================================
        
        edges = []
        edge_types = []
        
        def add_edge(src_name: str, dst_name: str, edge_type: int) -> None:
            """Add edge if both nodes exist."""
            src_idx = node_map.get(src_name)
            dst_idx = node_map.get(dst_name)
            
            if src_idx is not None and dst_idx is not None:
                edges.append([src_idx, dst_idx])
                edge_types.append(edge_type)
        
        # Function edges
        for func in contract.functions:
            func_name = func.canonical_name
            
            # CALLS edges (internal calls)
            for call in func.internal_calls:
                if hasattr(call, 'canonical_name'):
                    add_edge(func_name, call.canonical_name, EDGE_TYPES['CALLS'])
            
            # READS edges
            for var in func.state_variables_read:
                add_edge(func_name, var.canonical_name, EDGE_TYPES['READS'])
            
            # WRITES edges
            for var in func.state_variables_written:
                add_edge(func_name, var.canonical_name, EDGE_TYPES['WRITES'])
            
            # 🔧 FIX: EMITS edges (safe attribute access)
            # Some Slither versions don't have events_emitted
            if hasattr(func, 'events_emitted'):
                try:
                    for evt in func.events_emitted:
                        add_edge(func_name, evt.canonical_name, EDGE_TYPES['EMITS'])
                except:
                    pass  # Skip if any error
        
        # INHERITS edges
        try:
            for parent in contract.inheritance:
                add_edge(contract.name, parent.name, EDGE_TYPES['INHERITS'])
        except:
            pass  # Skip if inheritance not available
        
        # ====================================================================
        # CREATE DATA OBJECT
        # ====================================================================
        
        if len(node_features_list) == 0:
            return None
        
        x = torch.tensor(node_features_list, dtype=torch.float)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.long)
        
        y = torch.tensor([label], dtype=torch.long)
        contract_hash = get_contract_hash(contract_path)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            contract_hash=contract_hash,
            contract_path=str(contract_path),
            contract_name=contract.name,
            num_nodes=len(node_features_list),
            num_edges=len(edges)
        )
        
        return data
    
    def extract_batch_with_checkpoint(
        self,
        df: pd.DataFrame,
        n_workers: int = 11,
        chunksize: int = 50,
        output_dir: Path = Path("ml/data/graphs"),
        checkpoint_every: int = 500
    ) -> List[Data]:
        """Parallel extraction with checkpoint system."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = output_dir / "checkpoint.json"
        
        # Load checkpoint
        processed_hashes = set()
        if checkpoint_file.exists():
            print("📂 Loading checkpoint...")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_hashes = set(checkpoint.get('processed', []))
            print(f"   Found {len(processed_hashes):,} already processed contracts")
        
        # Filter already processed
        if processed_hashes:
            df = df.copy()
            df['_temp_hash'] = df['contract_path'].apply(get_contract_hash)
            df = df[~df['_temp_hash'].isin(processed_hashes)]
            df = df.drop(columns=['_temp_hash'])
            print(f"   Remaining: {len(df):,} contracts")
        
        # Resolve solc binaries
        if "solc_binary" not in df.columns:
            print("🔧 Resolving solc binaries...")
            df = df.copy()
            df["solc_binary"] = df["detected_version"].apply(get_solc_binary)
            found = df["solc_binary"].notna().sum()
            print(f"   Binary found for {found:,}/{len(df):,} contracts")
        
        df = df[df["solc_binary"].notna()].copy()
        
        # Group by version
        groups = df.groupby("detected_version")
        all_data = []
        total_processed = len(processed_hashes)
        
        print(f"\n🚀 Processing {len(groups)} version groups...\n")
        
        for version, group in tqdm(groups, desc="Version groups"):
            solc_bin = group.iloc[0]["solc_binary"]
            
            # 🔧 FIX: Pass solc_version to worker
            worker = partial(
                self.contract_to_pyg,
                solc_binary=solc_bin,
                solc_version=version,
                label=0
            )
            
            with mp.Pool(processes=n_workers) as pool:
                results = []
                for i, result in enumerate(tqdm(
                    pool.imap(worker, group["contract_path"].tolist(), chunksize=chunksize),
                    total=len(group),
                    desc=f"  v{version}",
                    leave=False
                )):
                    if result is not None:
                        results.append(result)
                        
                        filename = get_filename_from_hash(result.contract_hash)
                        graph_file = output_dir / filename
                        torch.save(result, graph_file)
                        
                        processed_hashes.add(result.contract_hash)
                        total_processed += 1
                        
                        if total_processed % checkpoint_every == 0:
                            with open(checkpoint_file, 'w') as f:
                                json.dump({
                                    'processed': list(processed_hashes),
                                    'total': total_processed,
                                    'timestamp': datetime.now().isoformat(),
                                    'completed': False
                                }, f, indent=2)
                            if self.verbose:
                                print(f"💾 Checkpoint: {total_processed:,} contracts")
                
                all_data.extend(results)
        
        # Final checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed': list(processed_hashes),
                'total': total_processed,
                'timestamp': datetime.now().isoformat(),
                'completed': True
            }, f, indent=2)
        
        print(f"\n✅ Successfully processed {len(all_data):,} NEW graphs")
        print(f"📁 Total graphs now: {total_processed:,}")
        
        return all_data


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AST Extractor V4.2")
    parser.add_argument("--input", default="ml/data/processed/contracts_metadata.parquet")
    parser.add_argument("--output", default="ml/data/graphs")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 AST Extractor V4.2 - PRODUCTION (Bug Fixes)")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔐 Hash: MD5 (guaranteed uniqueness)")
    print(f"📂 Output: {args.output}")
    print(f"⚙️  Workers: {args.workers}")
    print("="*70)
    print()
    
    df = pd.read_parquet(args.input)
    df = df[df["success"] == True].copy()
    print(f"✅ Loaded {len(df):,} successful contracts")
    
    if args.test:
        df = df.head(100)
        print(f"🧪 TEST MODE: Processing {len(df)} contracts")
    
    if args.resume:
        print("🔄 RESUME MODE: Will skip already processed contracts")
    
    print()
    
    extractor = ASTExtractorV4(verbose=args.verbose)
    
    print(f"💾 Checkpoints: Every {args.checkpoint_every} contracts")
    print(f"⚠️  Press Ctrl+C to stop (safe - can resume with --resume)")
    print()
    
    graphs = extractor.extract_batch_with_checkpoint(
        df,
        n_workers=args.workers,
        output_dir=Path(args.output),
        checkpoint_every=args.checkpoint_every
    )
    
    print()
    print("="*70)
    print("✅ EXTRACTION COMPLETE")
    print("="*70)
    print(f"Graphs created: {len(graphs):,}")
    print(f"Output directory: {args.output}")
    print("="*70)
