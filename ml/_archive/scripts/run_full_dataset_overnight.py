#!/usr/bin/env python3
"""
SENTINEL - Full BCCC Dataset Analysis (Overnight Run)

Features:
- Processes all 111,897 contracts
- Chunks of 5,000 contracts (saves after each chunk)
- Resume from checkpoint if interrupted
- Detailed logging to file
- Final comprehensive report
- ETA tracking

Run with: python3 run_full_dataset_overnight.py
"""

from ml.src.tools.slither_wrapper import SlitherWrapper
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import logging
import sys

# Setup logging
log_file = Path("ml/logs/overnight_analysis.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 5000  # Process 5K contracts at a time (~15-20 min per chunk)
CHECKPOINT_FILE = Path("ml/data/overnight_checkpoint.json")
RESULTS_FILE = Path("ml/data/processed/bccc_full_dataset_results.json")
TIMEOUT = 20
MAX_WORKERS = 20

def load_checkpoint():
    """Load existing checkpoint if available."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"📂 Checkpoint found: {len(checkpoint['completed_contracts'])} contracts already analyzed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return None

def save_checkpoint(checkpoint_data):
    """Save checkpoint to disk."""
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"💾 Checkpoint saved: {len(checkpoint_data['completed_contracts'])} total analyzed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def save_results(all_results):
    """Save final results to JSON with COMPLETE data."""
    try:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert SlitherResult objects to dicts (FIXED VERSION)
        results_data = []
        for r in all_results:
            results_data.append({
                'contract_path': r.contract_path,
                'success': r.success,
                'error_type': r.error_type,
                'error_message': r.error_message,
                'detected_version': r.detected_version,
                'analysis_time': r.analysis_time,
                'timestamp': r.timestamp,
                
                # Vulnerability counts (ADDED MEDIUM)
                'high_impact_count': r.high_impact_count(),
                'medium_impact_count': r.medium_impact_count(),  # 🔧 FIXED
                
                # Simple type list (existing)
                'vulnerability_types': r.vulnerability_types(),
                
                # Full findings with impact levels (NEW - for ML features)
                'findings': [
                    {
                        'check': f.check,
                        'impact': f.impact.value,
                        'confidence': f.confidence.value,
                        'description': f.description,
                        'lines': f.lines
                    }
                    for f in r.findings
                ]
            })
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Results saved: {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_statistics(all_results, duration_seconds):
    """Print comprehensive statistics."""
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    
    print("\n" + "="*80)
    print("📊 FULL DATASET ANALYSIS - FINAL STATISTICS")
    print("="*80)
    
    # Basic stats
    print(f"\n🎯 Execution Summary:")
    print(f"   Total contracts: {len(all_results):,}")
    print(f"   Successful: {len(successful):,} ({len(successful)/len(all_results)*100:.1f}%)")
    print(f"   Failed: {len(failed):,} ({len(failed)/len(all_results)*100:.1f}%)")
    print(f"   Duration: {duration_seconds/3600:.1f} hours ({duration_seconds/60:.0f} minutes)")
    print(f"   Throughput: {len(all_results)/duration_seconds:.2f} contracts/second")
    
    # Vulnerability stats
    total_vulns = sum(r.high_impact_count() for r in successful)
    contracts_with_vulns = sum(1 for r in successful if r.high_impact_count() > 0)
    
    print(f"\n🐛 Vulnerability Detection:")
    print(f"   Total high-impact vulnerabilities: {total_vulns:,}")
    print(f"   Contracts with vulnerabilities: {contracts_with_vulns:,}/{len(successful):,} ({contracts_with_vulns/len(successful)*100:.1f}%)")
    print(f"   Average vulns per vulnerable contract: {total_vulns/contracts_with_vulns:.2f}" if contracts_with_vulns > 0 else "   No vulnerabilities found")
    
    # Top vulnerability types
    vuln_types = Counter()
    for r in successful:
        for vuln in r.vulnerability_types():
            vuln_types[vuln] += 1
    
    if vuln_types:
        print(f"\n🔥 Top 15 Vulnerability Types:")
        for i, (vuln, count) in enumerate(vuln_types.most_common(15), 1):
            print(f"   {i:2d}. {vuln}: {count:,}")
    
    # Version distribution
    version_counter = Counter(r.detected_version for r in all_results if r.detected_version)
    print(f"\n📋 Top 15 Solidity Versions:")
    for i, (version, count) in enumerate(version_counter.most_common(15), 1):
        success = sum(1 for r in all_results if r.detected_version == version and r.success)
        print(f"   {i:2d}. v{version}: {success:,}/{count:,} ({success/count*100:.0f}% success)")
    
    # Error breakdown
    if failed:
        error_types = Counter(r.error_type for r in failed if r.error_type)
        print(f"\n❌ Failure Breakdown:")
        for error_type, count in error_types.most_common(10):
            print(f"   {error_type}: {count:,} ({count/len(failed)*100:.1f}%)")
    
    print("\n" + "="*80)

def main():
    """Main overnight analysis function."""
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("🌙 SENTINEL - OVERNIGHT FULL DATASET ANALYSIS")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Load all contract paths
    logger.info("📦 Loading contract paths...")
    bccc_dir = Path("BCCC-SCsVul-2024/SourceCodes")
    
    if not bccc_dir.exists():
        logger.error(f"❌ Dataset not found: {bccc_dir}")
        return
    
    all_contracts = sorted([str(c) for c in bccc_dir.rglob("*.sol")])
    logger.info(f"✅ Found {len(all_contracts):,} contracts")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_contracts = set()
    all_results = []
    
    if checkpoint:
        completed_contracts = set(checkpoint['completed_contracts'])
        all_results = checkpoint.get('results_summary', [])
        logger.info(f"🔄 Resuming from checkpoint: {len(completed_contracts):,} contracts already done")
    
    # Filter out completed contracts
    remaining_contracts = [c for c in all_contracts if c not in completed_contracts]
    
    if not remaining_contracts:
        logger.info("✅ All contracts already analyzed!")
        return
    
    logger.info(f"📊 Remaining contracts: {len(remaining_contracts):,}")
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   - Chunk size: {CHUNK_SIZE:,} contracts")
    logger.info(f"   - Workers: {MAX_WORKERS}")
    logger.info(f"   - Timeout: {TIMEOUT}s per contract")
    logger.info(f"   - Total chunks: {(len(remaining_contracts) + CHUNK_SIZE - 1) // CHUNK_SIZE}")
    
    # Estimate time
    estimated_seconds = len(remaining_contracts) * 0.13  # ~7-8 contracts/sec
    estimated_hours = estimated_seconds / 3600
    eta = start_time + timedelta(seconds=estimated_seconds)
    
    logger.info(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
    logger.info(f"🎯 ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize wrapper
    wrapper = SlitherWrapper(
        use_docker=False,
        verbose=False,
        timeout=TIMEOUT,
        min_impact="Medium"
    )
    
    # Process in chunks
    total_chunks = (len(remaining_contracts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in range(0, len(remaining_contracts), CHUNK_SIZE):
        chunk_num = chunk_idx // CHUNK_SIZE + 1
        chunk = remaining_contracts[chunk_idx:chunk_idx + CHUNK_SIZE]
        
        logger.info("="*80)
        logger.info(f"📦 CHUNK {chunk_num}/{total_chunks} - Processing {len(chunk):,} contracts")
        logger.info("="*80)
        
        chunk_start = datetime.now()
        
        try:
            # Process chunk
            chunk_results = wrapper.run_batch(chunk, max_workers=MAX_WORKERS)
            all_results.extend(chunk_results)
            
            # Update completed contracts
            completed_contracts.update(chunk)
            
            chunk_duration = (datetime.now() - chunk_start).total_seconds()
            chunk_successful = sum(1 for r in chunk_results if r.success)
            
            logger.info(f"✅ Chunk {chunk_num} complete:")
            logger.info(f"   - Success: {chunk_successful}/{len(chunk)} ({chunk_successful/len(chunk)*100:.1f}%)")
            logger.info(f"   - Duration: {chunk_duration/60:.1f} minutes")
            logger.info(f"   - Throughput: {len(chunk)/chunk_duration:.2f} contracts/sec")
            
            # Save checkpoint after each chunk
            checkpoint_data = {
                'last_updated': datetime.now().isoformat(),
                'completed_contracts': list(completed_contracts),
                'total_analyzed': len(all_results),
                'chunks_completed': chunk_num,
                'total_chunks': total_chunks
            }
            save_checkpoint(checkpoint_data)
            
            # Update ETA
            total_duration = (datetime.now() - start_time).total_seconds()
            avg_throughput = len(all_results) / total_duration
            remaining = len(all_contracts) - len(all_results)
            remaining_seconds = remaining / avg_throughput
            new_eta = datetime.now() + timedelta(seconds=remaining_seconds)
            
            logger.info(f"📊 Progress: {len(all_results):,}/{len(all_contracts):,} ({len(all_results)/len(all_contracts)*100:.1f}%)")
            logger.info(f"⏱️  Updated ETA: {new_eta.strftime('%H:%M:%S')}\n")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupted by user! Saving progress...")
            checkpoint_data = {
                'last_updated': datetime.now().isoformat(),
                'completed_contracts': list(completed_contracts),
                'total_analyzed': len(all_results),
                'chunks_completed': chunk_num - 1,
                'total_chunks': total_chunks
            }
            save_checkpoint(checkpoint_data)
            logger.info("💾 Progress saved. Run again to resume.")
            return
            
        except Exception as e:
            logger.error(f"❌ Error in chunk {chunk_num}: {e}")
            logger.info("💾 Saving progress before continuing...")
            checkpoint_data = {
                'last_updated': datetime.now().isoformat(),
                'completed_contracts': list(completed_contracts),
                'total_analyzed': len(all_results),
                'chunks_completed': chunk_num - 1,
                'total_chunks': total_chunks
            }
            save_checkpoint(checkpoint_data)
            continue
    
    # Final statistics
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {total_duration/3600:.2f} hours")
    
    # Print and save final statistics
    print_statistics(all_results, total_duration)
    save_results(all_results)
    
    # Clean up checkpoint file
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("🧹 Checkpoint file removed (analysis complete)")
    
    logger.info(f"\n📁 Log file: {log_file}")
    logger.info(f"📁 Results file: {RESULTS_FILE}")
    logger.info("\n✨ Ready for Module 4 feature engineering!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)

