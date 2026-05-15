"""
Quick validation script for SmartBugs Wild dataset.
Checks for critical issues before starting ML training.
"""

import pandas as pd
from pathlib import Path
import sys


def validate_smartbugs_dataset(data_dir: str = "../data/smartbugs-wild"):
    """Validate SmartBugs Wild dataset for ML training."""

    data_path = Path(data_dir)
    contracts_dir = data_path / "contracts"
    csv_path = data_path / "contracts.csv"

    print("="*80)
    print("SMARTBUGS WILD DATASET VALIDATION")
    print("="*80)

    # Check 1: Directory structure
    print("\n1. DIRECTORY STRUCTURE:")
    if not data_path.exists():
        print(f"   ❌ Dataset directory not found: {data_path}")
        return False
    print(f"   ✅ Dataset directory found: {data_path}")

    if not contracts_dir.exists():
        print(f"   ❌ Contracts directory not found: {contracts_dir}")
        return False
    print(f"   ✅ Contracts directory found")

    if not csv_path.exists():
        print(f"   ❌ Metadata CSV not found: {csv_path}")
        return False
    print(f"   ✅ Metadata CSV found")

    # Check 2: Contract files
    print("\n2. CONTRACT FILES:")
    contract_files = list(contracts_dir.glob("*.sol"))
    print(f"   ✅ Found {len(contract_files):,} Solidity files")

    if len(contract_files) == 0:
        print("   ❌ No contract files found!")
        return False

    # Sample contracts
    sample_sizes = []
    for f in contract_files[:10]:
        sample_sizes.append(f.stat().st_size)
    avg_size = sum(sample_sizes) / len(sample_sizes)
    print(f"   ✅ Average contract size (sample): {avg_size/1024:.2f} KB")

    # Check 3: Metadata
    print("\n3. METADATA:")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ CSV loaded: {len(df):,} records")
        print(f"   ✅ Columns: {', '.join(df.columns.tolist())}")

        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"   ⚠️  Found {missing} missing values")
        else:
            print(f"   ✅ No missing values")

    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return False

    # Check 4: CRITICAL - Labels
    print("\n4. ⚠️  VULNERABILITY LABELS (CRITICAL CHECK):")
    vuln_keywords = ['vuln', 'bug', 'issue', 'security', 'exploit',
                     'attack', 'reentrancy', 'overflow', 'label']
    vuln_columns = [col for col in df.columns
                   if any(keyword in col.lower() for keyword in vuln_keywords)]

    if len(vuln_columns) == 0:
        print("   ❌ NO VULNERABILITY LABELS FOUND!")
        print("   ❌ This dataset is UNLABELED - cannot use for supervised learning directly")
        print("\n   💡 This dataset contains:")
        print("      • 47,398 real-world smart contracts")
        print("      • Metadata (transactions, compiler, dates)")
        print("      • NO vulnerability labels")
        print("\n   🎯 RECOMMENDED ACTIONS:")
        print("      1. Download SmartBugs Curated (labeled subset)")
        print("      2. Download SmartBugs Results (analysis outputs)")
        print("      3. Use SolidiFI dataset (has labels)")
        print("      4. Run analysis tools to generate labels")
        print("\n   ℹ️  For your learning journey:")
        print("      • Use this for: pre-training, unsupervised learning")
        print("      • Need labeled data for: supervised vulnerability detection")
        has_labels = False
    else:
        print(f"   ✅ Found {len(vuln_columns)} label columns:")
        for col in vuln_columns:
            print(f"      • {col}")
        has_labels = True

    # Check 5: Data quality
    print("\n5. DATA QUALITY:")

    # Compiler versions
    if 'compiler_version' in df.columns:
        n_compilers = df['compiler_version'].nunique()
        print(f"   ✅ Compiler versions: {n_compilers} unique")

    # Transaction counts
    if 'nb_transaction' in df.columns:
        avg_tx = df['nb_transaction'].mean()
        median_tx = df['nb_transaction'].median()
        print(f"   ✅ Avg transactions: {avg_tx:.0f} (median: {median_tx:.0f})")

    # Date range
    if 'creation_date' in df.columns:
        min_date = df['creation_date'].min()
        max_date = df['creation_date'].max()
        print(f"   ✅ Date range: {min_date} to {max_date}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)

    if has_labels:
        print("✅ Dataset is ready for supervised learning")
        print("✅ Proceed with preprocessing and model training")
    else:
        print("⚠️  Dataset is UNLABELED - not ready for supervised learning")
        print("⚠️  You need to obtain labeled data first!")
        print("\n📋 NEXT STEPS:")
        print("1. Clone SmartBugs Curated repository:")
        print("   git clone https://github.com/smartbugs/smartbugs-curated.git")
        print("\n2. Or download SolidiFI dataset:")
        print("   git clone https://github.com/DependableSystemsLab/SolidiFI.git")
        print("\n3. Or get SmartBugs Results:")
        print("   git clone https://github.com/smartbugs/smartbugs-results.git")

    print("="*80)

    return has_labels


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/smartbugs-wild"
    has_labels = validate_smartbugs_dataset(data_dir)
    sys.exit(0 if has_labels else 1)
