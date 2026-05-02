#!/usr/bin/env python
"""
CodeBERT Tokenizer V1 - Production Pipeline
============================================

Converts Solidity smart contracts to CodeBERT token sequences for
the Dual-Path GNN Transformer architecture.

Features:
- MD5 hash naming (matches graph files)
- Checkpoint/resume system
- Multiprocessing (11 workers)
- Batch processing for speed
- Error handling and logging

Author: Ali
Date: February 15, 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import partial
import multiprocessing as mp
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")  # Suppress HuggingFace warnings

# Data processing
import pandas as pd

# PyTorch
import torch

# Progress bar
from tqdm import tqdm

# Transformers (HuggingFace)
try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError("transformers not installed. Run: poetry add transformers") from e

# Our hash utility + schema version for metadata tracking
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.hash_utils import get_contract_hash, get_filename_from_hash
from src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION
# ============================================================================
# CONFIGURATION
# ============================================================================

# CodeBERT model name
TOKENIZER_MODEL = "microsoft/codebert-base"

# Tokenization parameters
MAX_LENGTH = 512          # CodeBERT's maximum sequence length
PADDING = "max_length"    # Always pad to 512
TRUNCATION = True         # Cut long contracts at 512

# Processing parameters
DEFAULT_WORKERS = 11      # Number of parallel workers
DEFAULT_CHUNK_SIZE = 50   # Contracts per worker batch
CHECKPOINT_INTERVAL = 500 # Save progress every N contracts

# Directories
DEFAULT_INPUT = "ml/data/processed/contracts_metadata.parquet"
DEFAULT_OUTPUT = "ml/data/tokens"
# ============================================================================
# GLOBAL TOKENIZER (Worker-level)
# ============================================================================

# This will be initialized ONCE per worker process
tokenizer = None


def init_worker():
    """
    Initialize worker process with tokenizer.
    
    Called ONCE when each worker starts (not per contract).
    Loads the 500MB CodeBERT tokenizer into worker's memory.
    
    Why this matters:
    - Without: Load 500MB model 68,568 times = DISASTER (hours)
    - With: Load 500MB model 11 times = Fast (seconds)
    
    This is a standard multiprocessing pattern for expensive initialization.
    """
    global tokenizer
    
    print(f"  Worker {mp.current_process().name}: Loading CodeBERT tokenizer...")
    
    # Load tokenizer (500MB download first time, then cached)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        cache_dir=".cache/huggingface",  # Cache locally
        use_fast=True  # Use fast Rust-based tokenizer (10× faster)
    )
    
    print(f"  Worker {mp.current_process().name}: Ready!")
# ============================================================================
# TOKENIZATION FUNCTIONS
# ============================================================================

def tokenize_single_contract(contract_path: str) -> Optional[Dict[str, Any]]:
    """
    Tokenize a single Solidity contract.
    
    This function runs in worker processes. Uses the global 'tokenizer'
    initialized by init_worker().
    
    Args:
        contract_path: Path to .sol file
        
    Returns:
        Dictionary with tokens and metadata, or None if failed
        
    Process:
        1. Read source code
        2. Tokenize with CodeBERT
        3. Generate MD5 hash
        4. Package as dictionary
    """
    global tokenizer
    
    try:
        # Step 1: Read source code
        contract_path = Path(contract_path)
        
        # Read with error handling for bad encoding
        try:
            code = contract_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback: Replace bad characters with �
            code = contract_path.read_text(encoding='utf-8', errors='replace')
        
        # Skip empty files
        if len(code.strip()) == 0:
            return None
        
        # Step 2: Tokenize
        # This is the core CodeBERT processing
        encoded = tokenizer(
            code,
            max_length=MAX_LENGTH,       # Truncate at 512
            padding=PADDING,              # Pad to 512
            truncation=TRUNCATION,        # Enable truncation
            return_tensors='pt'           # Return PyTorch tensors
        )
        
        # Step 3: Extract tensors
        # encoded is a dict: {'input_ids': tensor, 'attention_mask': tensor}
        input_ids = encoded['input_ids'].squeeze(0)        # (1, 512) → (512,)
        attention_mask = encoded['attention_mask'].squeeze(0)  # (1, 512) → (512,)
        
        # Step 4: Calculate metadata
        # How many real tokens (before padding)?
        num_real_tokens = attention_mask.sum().item()  # Count 1s in mask
        
        # Was it truncated? (more than 512 tokens originally)
        # We can't know for sure without re-tokenizing without truncation,
        # but if all 512 positions are real tokens (no padding), it was likely truncated
        truncated = num_real_tokens >= (MAX_LENGTH - 2)  # -2 for [CLS] and [SEP]
        
        # Step 5: Generate hash (same as graph extractor!)
        contract_hash = get_contract_hash(contract_path)
        
        # Step 6: Package result
        result = {
            'input_ids': input_ids,                            # (512,) tensor
            'attention_mask': attention_mask,                  # (512,) tensor
            'contract_hash': contract_hash,                    # MD5 hash
            'contract_path': str(contract_path),               # Source file
            'num_tokens': num_real_tokens,                     # Actual tokens
            'truncated': truncated,                            # Was it cut?
            'tokenizer_name': TOKENIZER_MODEL,                 # Reproducibility
            'max_length': MAX_LENGTH,                          # Config
            'feature_schema_version': FEATURE_SCHEMA_VERSION,  # Graph schema version at extraction time
        }
        
        return result
        
    except Exception as e:
        # Something went wrong - skip this contract
        # Main process will log it
        return None
def save_token_file(token_data: Dict[str, Any], output_dir: Path) -> bool:
    """
    Save tokenized contract to .pt file.
    
    Args:
        token_data: Dictionary from tokenize_single_contract()
        output_dir: Directory to save to (ml/data/tokens/)
        
    Returns:
        True if saved successfully, False otherwise
        
    File naming:
        Uses MD5 hash from token_data['contract_hash']
        Matches graph file naming: {hash}.pt
    """
    try:
        # Generate filename from hash
        contract_hash = token_data['contract_hash']
        filename = get_filename_from_hash(contract_hash)
        output_path = output_dir / filename
        
        # Save with torch.save
        # Note: Contains both tensors and metadata
        torch.save(token_data, output_path)
        
        return True
        
    except Exception as e:
        return False
# ============================================================================
# BATCH PROCESSING WITH CHECKPOINT
# ============================================================================

def process_batch_with_checkpoint(
    contracts_df: pd.DataFrame,
    output_dir: Path,
    n_workers: int = DEFAULT_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    checkpoint_every: int = CHECKPOINT_INTERVAL
) -> Dict[str, Any]:
    """
    Process all contracts with multiprocessing and checkpointing.
    
    This is the main processing function. It:
    1. Loads checkpoint if exists (for resume)
    2. Filters out already-processed contracts
    3. Distributes work to worker pool
    4. Saves checkpoints periodically
    5. Collects statistics
    
    Args:
        contracts_df: DataFrame with 'contract_path' column
        output_dir: Where to save token files
        n_workers: Number of parallel workers
        chunk_size: Contracts per worker batch
        checkpoint_every: Save progress every N contracts
        
    Returns:
        Dictionary with statistics (total, successful, failed, etc.)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_dir / "checkpoint.json"
    failed_file = output_dir / "failed_contracts.json"
    
    # ========================================================================
    # LOAD CHECKPOINT
    # ========================================================================
    
    processed_hashes = set()
    failed_hashes = []
    
    if checkpoint_file.exists():
        print("📂 Loading checkpoint...")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_hashes = set(checkpoint.get('processed', []))
            failed_hashes = checkpoint.get('failed', [])
        print(f"   Found {len(processed_hashes):,} processed, {len(failed_hashes):,} failed")
    
    # ========================================================================
    # FILTER ALREADY PROCESSED
    # ========================================================================
    
    if processed_hashes:
        contracts_df = contracts_df.copy()
        contracts_df['_temp_hash'] = contracts_df['contract_path'].apply(get_contract_hash)
        contracts_df = contracts_df[~contracts_df['_temp_hash'].isin(processed_hashes)]
        contracts_df = contracts_df.drop(columns=['_temp_hash'])
        print(f"   Remaining: {len(contracts_df):,} contracts")
    
    if len(contracts_df) == 0:
        print("✅ All contracts already processed!")
        return {
            'total': len(processed_hashes),
            'successful': len(processed_hashes),
            'failed': len(failed_hashes),
            'new': 0
        }
    
    # ========================================================================
    # PREPARE CONTRACT PATHS
    # ========================================================================
    
    contract_paths = contracts_df['contract_path'].tolist()
    total_to_process = len(contract_paths)
    
    print(f"\n🚀 Starting tokenization...")
    print(f"   Contracts: {total_to_process:,}")
    print(f"   Workers:   {n_workers}")
    print(f"   Chunk:     {chunk_size}")
    print()
    
    # ========================================================================
    # MULTIPROCESSING POOL
    # ========================================================================
    
    stats = {
        'successful': 0,
        'failed': 0,
        'truncated': 0
    }
    
    # Create worker pool with initializer
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        
        # Process contracts in parallel
        # imap returns results as they complete (good for progress bar)
        results_iter = pool.imap(
            tokenize_single_contract,
            contract_paths,
            chunksize=chunk_size
        )
        
        # Progress bar
        for i, result in enumerate(tqdm(results_iter, total=total_to_process, desc="Tokenizing")):
            
            if result is not None:
                # Success! Save the token file
                if save_token_file(result, output_dir):
                    processed_hashes.add(result['contract_hash'])
                    stats['successful'] += 1
                    
                    if result['truncated']:
                        stats['truncated'] += 1
                else:
                    # Save failed but tokenization succeeded
                    failed_hashes.append(result['contract_hash'])
                    stats['failed'] += 1
            else:
                # Tokenization failed
                # Get the hash of the failed contract
                try:
                    failed_hash = get_contract_hash(contract_paths[i])
                    failed_hashes.append(failed_hash)
                except:
                    pass
                stats['failed'] += 1
            
            # ================================================================
            # CHECKPOINT SAVING
            # ================================================================
            
            current_total = len(processed_hashes)
            if current_total % checkpoint_every == 0 and current_total > 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed': list(processed_hashes),
                        'failed': failed_hashes,
                        'total': current_total,
                        'timestamp': datetime.now().isoformat(),
                        'completed': False
                    }, f, indent=2)
                
                # Also save failed list
                with open(failed_file, 'w') as f:
                    json.dump(failed_hashes, f, indent=2)
    
    # ========================================================================
    # FINAL CHECKPOINT
    # ========================================================================
    
    final_total = len(processed_hashes)
    
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed': list(processed_hashes),
            'failed': failed_hashes,
            'total': final_total,
            'timestamp': datetime.now().isoformat(),
            'completed': True
        }, f, indent=2)
    
    with open(failed_file, 'w') as f:
        json.dump(failed_hashes, f, indent=2)
    
    # ========================================================================
    # RETURN STATISTICS
    # ========================================================================
    
    return {
        'total': final_total,
        'successful': stats['successful'],
        'failed': stats['failed'],
        'truncated': stats['truncated'],
        'truncation_rate': stats['truncated'] / stats['successful'] if stats['successful'] > 0 else 0
    }
# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CodeBERT Tokenizer V1 - Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python tokenizer_v1_production.py
  
  # Test mode (100 contracts)
  python tokenizer_v1_production.py --test
  
  # Resume after interruption
  python tokenizer_v1_production.py --resume
  
  # Custom configuration
  python tokenizer_v1_production.py --workers 8 --checkpoint-every 1000
        """
    )
    
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of workers (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Contracts per chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N contracts (default: {CHECKPOINT_INTERVAL})"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 100 contracts"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skips processed contracts)"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    print("=" * 70)
    print("🚀 CodeBERT Tokenizer V1 - Production Pipeline")
    print("=" * 70)
    print(f"📅 Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model:     {TOKENIZER_MODEL}")
    print(f"📏 Max Length: {MAX_LENGTH} tokens")
    print(f"📂 Output:    {args.output}")
    print(f"⚙️  Workers:   {args.workers}")
    print("=" * 70)
    print()
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("📂 Loading contract metadata...")
    df = pd.read_parquet(args.input)
    
    # Filter successful contracts only
    df = df[df['success'] == True].copy()
    print(f"✅ Loaded {len(df):,} successful contracts")
    
    # Test mode
    if args.test:
        df = df.head(100)
        print(f"🧪 TEST MODE: Processing {len(df)} contracts")
    
    # Resume mode
    if args.resume:
        print("🔄 RESUME MODE: Will skip already processed contracts")
    
    print()
    
    # ========================================================================
    # PROCESS
    # ========================================================================
    
    print(f"💾 Checkpoints: Every {args.checkpoint_every} contracts")
    print(f"⚠️  Press Ctrl+C to stop (safe - can resume with --resume)")
    print()
    
    stats = process_batch_with_checkpoint(
        contracts_df=df,
        output_dir=Path(args.output),
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        checkpoint_every=args.checkpoint_every
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print()
    print("=" * 70)
    print("✅ TOKENIZATION COMPLETE")
    print("=" * 70)
    print(f"Total processed:  {stats['total']:,}")
    print(f"Successful:       {stats['successful']:,}")
    print(f"Failed:           {stats['failed']:,}")
    print(f"Truncated:        {stats['truncated']:,} ({stats['truncation_rate']:.1%})")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    
    # Success rate check
    if stats['successful'] > 0:
        success_rate = stats['successful'] / (stats['successful'] + stats['failed'])
        if success_rate < 0.95:
            print()
            print("⚠️  WARNING: Success rate < 95%")
            print(f"   Check failed_contracts.json for details")
        else:
            print()
            print(f"🎉 Success rate: {success_rate:.1%} - Excellent!")
