"""
SENTINEL - Full Dataset Pydantic Validation
Week 1 Day 2 - Final Production Test

Tests complete Pydantic schema on 101,897 contract records.
Validates:
  - Type correctness
  - Field constraints
  - Semantic consistency (counts match findings)
  - Error field consistency

Author: Ali Motafegh
Date: February 11, 2026
"""

from models_v2 import validate_batch, ContractResult
import json
from pathlib import Path
from datetime import datetime
from collections import Counter


def main():
    print("="*80)
    print("🔍 SENTINEL - PYDANTIC VALIDATION: Full Production Dataset")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    data_file = Path("../../data/processed/bccc_full_dataset_results.json")
    
    print(f"\n📂 Loading dataset...")
    print(f"   File: {data_file}")
    
    if not data_file.exists():
        print(f"❌ ERROR: File not found: {data_file}")
        return
    
    file_size_mb = data_file.stat().st_size / 1024 / 1024
    print(f"   Size: {file_size_mb:.1f} MB")
    
    with open(data_file) as f:
        data = json.load(f)
    
    total_records = len(data)
    print(f"   Records: {total_records:,}")
    
    # ========================================================================
    # STEP 2: RUN PYDANTIC VALIDATION
    # ========================================================================
    
    print(f"\n⚙️  Running Pydantic validation...")
    print(f"   Validating type, field, and semantic constraints...")
    
    start_time = datetime.now()
    valid, invalid = validate_batch(data)
    duration = (datetime.now() - start_time).total_seconds()
    
    throughput = total_records / duration if duration > 0 else 0
    
    # ========================================================================
    # STEP 3: VALIDATION RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 VALIDATION RESULTS")
    print("="*80)
    
    valid_count = len(valid)
    invalid_count = len(invalid)
    valid_pct = (valid_count / total_records * 100) if total_records > 0 else 0
    invalid_pct = (invalid_count / total_records * 100) if total_records > 0 else 0
    
    print(f"\n✅ Valid records:   {valid_count:>7,} ({valid_pct:>6.2f}%)")
    print(f"❌ Invalid records: {invalid_count:>7,} ({invalid_pct:>6.2f}%)")
    print(f"⏱️  Duration:        {duration:>7.1f}s")
    print(f"⚡ Throughput:      {throughput:>7,.0f} records/sec")
    
    # ========================================================================
    # STEP 4: ANALYZE VALIDATION FAILURES (if any)
    # ========================================================================
    
    if invalid:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILURES ANALYSIS")
        print("="*80)
        
        # Group errors by type
        error_types = Counter()
        error_fields = Counter()
        
        for item in invalid:
            error_msg = str(item['error'])
            
            # Count validation error count (e.g., "1 validation error", "2 validation errors")
            if "validation error" in error_msg:
                first_line = error_msg.split('\n')[0]
                error_types[first_line] += 1
            
            # Extract field names
            lines = error_msg.split('\n')
            for line in lines:
                if line and not line.startswith(' ') and 'validation error' not in line.lower():
                    error_fields[line.strip()] += 1
        
        print(f"\n🔍 Error Distribution:")
        for error, count in error_types.most_common(5):
            print(f"   {count:>5,}x  {error}")
        
        if error_fields:
            print(f"\n📋 Top Failed Fields:")
            for field, count in error_fields.most_common(10):
                print(f"   {count:>5,}x  {field}")
        
        # Sample errors
        print(f"\n📄 Sample Errors (first 3):")
        for i, item in enumerate(invalid[:3], 1):
            contract = item['data'].get('contract_path', 'unknown')
            contract_short = contract.split('/')[-1][:50] if '/' in contract else contract[:50]
            error = str(item['error'])[:250]
            
            print(f"\n   {i}. Contract: {contract_short}")
            print(f"      Error: {error}")
    
    # ========================================================================
    # STEP 5: ANALYZE VALID DATA
    # ========================================================================
    
    if valid:
        print("\n" + "="*80)
        print("✅ VALID DATA ANALYSIS")
        print("="*80)
        
        # Split by analysis status
        successful = [r for r in valid if r.success]
        failed_analysis = [r for r in valid if not r.success]
        
        print(f"\n📊 Analysis Status:")
        print(f"   ✅ Successful analysis: {len(successful):>7,} ({len(successful)/len(valid)*100:>5.1f}%)")
        print(f"   ❌ Failed analysis:     {len(failed_analysis):>7,} ({len(failed_analysis)/len(valid)*100:>5.1f}%)")
        
        # ML-ready subset (successful with version detected)
        ml_ready = [r for r in successful if r.detected_version is not None]
        
        print(f"\n🎯 ML-Ready Dataset:")
        print(f"   Records with detected version: {len(ml_ready):>7,}")
        print(f"   Records without version:       {len(successful) - len(ml_ready):>7,}")
        
        # Vulnerability statistics
        total_high = sum(r.high_impact_count for r in successful)
        total_medium = sum(r.medium_impact_count for r in successful)
        total_findings = total_high + total_medium
        
        with_vulns = [r for r in successful if r.high_impact_count > 0 or r.medium_impact_count > 0]
        no_vulns = len(successful) - len(with_vulns)
        
        print(f"\n🐛 Vulnerability Statistics:")
        print(f"   Total findings validated: {total_findings:>7,}")
        print(f"   ├─ High impact:           {total_high:>7,} ({total_high/total_findings*100:>5.1f}%)")
        print(f"   └─ Medium impact:         {total_medium:>7,} ({total_medium/total_findings*100:>5.1f}%)")
        print(f"\n   Vulnerable contracts:     {len(with_vulns):>7,} ({len(with_vulns)/len(successful)*100:>5.1f}%)")
        print(f"   Non-vulnerable contracts: {no_vulns:>7,} ({no_vulns/len(successful)*100:>5.1f}%)")
        
        avg_findings = total_findings / len(with_vulns) if with_vulns else 0
        print(f"   Avg findings per vuln contract: {avg_findings:.2f}")
        
        # Top vulnerability types
        vuln_types = Counter()
        for r in successful:
            for vtype in r.vulnerability_types:
                vuln_types[vtype] += 1
        
        if vuln_types:
            print(f"\n📋 Top 10 Vulnerability Types:")
            for i, (vtype, count) in enumerate(vuln_types.most_common(10), 1):
                print(f"   {i:>2}. {vtype:<30} {count:>6,}")
        
        # Semantic validation confirmation
        print(f"\n✅ Semantic Validation Results:")
        print(f"   ✓ All high_impact_count fields match findings")
        print(f"   ✓ All medium_impact_count fields match findings")
        print(f"   ✓ All vulnerability_types consistent with findings")
        print(f"   ✓ All error fields consistent with success status")
        
        # Failed analysis breakdown
        if failed_analysis:
            error_type_dist = Counter(r.error_type for r in failed_analysis)
            
            print(f"\n📊 Failed Analysis Breakdown:")
            for error_type, count in error_type_dist.most_common(10):
                print(f"   {error_type:<30} {count:>6,} ({count/len(failed_analysis)*100:>5.1f}%)")
    
    # ========================================================================
    # STEP 6: FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)
    
    if invalid_count == 0:
        print("\n🎉 SUCCESS! All records passed Pydantic validation!")
        print("\n✅ Dataset Quality:")
        print("   • Type safety:        100% validated")
        print("   • Field constraints:  100% satisfied")
        print("   • Semantic rules:     100% consistent")
        print("   • Production-ready:   YES")
        
        if valid:
            ml_ready_count = len([r for r in valid if r.success and r.detected_version])
            print(f"\n🚀 Ready for Next Steps:")
            print(f"   • ML-ready records:   {ml_ready_count:,}")
            print(f"   • Week 1 Day 3:       Statistical validation (outliers)")
            print(f"   • Week 1 Day 4:       Duplicate detection")
            print(f"   • Week 1 Day 5:       Quality report generation")
    else:
        print(f"\n⚠️  ATTENTION: {invalid_count:,} records failed validation")
        print(f"\n📋 Next Steps:")
        print(f"   1. Review error messages above")
        print(f"   2. Fix schema or data issues")
        print(f"   3. Re-run validation")
        print(f"\n   Most likely fixes:")
        print(f"   • Update schema to handle edge cases")
        print(f"   • Clean data to match schema constraints")
    
    print("\n" + "="*80)
    
    # ========================================================================
    # STEP 7: SAVE VALIDATION REPORT
    # ========================================================================
    
    report_file = Path("../../data/reports/pydantic_validation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": total_records,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validation_duration_seconds": duration,
        "throughput_records_per_sec": throughput,
        "ml_ready_count": len([r for r in valid if r.success and r.detected_version]) if valid else 0,
        "successful_analysis_count": len([r for r in valid if r.success]) if valid else 0,
        "failed_analysis_count": len([r for r in valid if not r.success]) if valid else 0,
        "total_high_impact": sum(r.high_impact_count for r in valid if r.success) if valid else 0,
        "total_medium_impact": sum(r.medium_impact_count for r in valid if r.success) if valid else 0
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Validation report saved: {report_file}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
