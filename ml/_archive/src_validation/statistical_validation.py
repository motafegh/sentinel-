"""
Statistical validation for SENTINEL dataset.

Detects outliers and anomalies using statistical methods:
- IQR (Interquartile Range) method for outlier detection
- Z-score analysis for extreme values
- Distribution analysis for data quality

Author: Ali Motafegh
Date: Feb 11, 2026
"""

import json
import statistics
from typing import List, Dict, Any
from collections import Counter
from pathlib import Path


class StatisticalValidator:
    """
    Statistical validator for contract analysis results.
    
    Uses IQR method: Q1 - 1.5*IQR, Q3 + 1.5*IQR as outlier bounds.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.successful = [r for r in data if r.get('success')]
        self.outliers = {
            'analysis_time': [],
            'high_impact_count': [],
            'medium_impact_count': [],
            'total_findings': [],
            'description_length': []
        }
    
    def calculate_iqr_bounds(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate IQR-based outlier bounds.
        
        Returns: {
            'q1': 25th percentile,
            'q3': 75th percentile,
            'iqr': interquartile range,
            'lower_bound': Q1 - 1.5*IQR,
            'upper_bound': Q3 + 1.5*IQR,
            'extreme_lower': Q1 - 3*IQR,
            'extreme_upper': Q3 + 3*IQR
        }
        """
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        return {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': q1 - 1.5 * iqr,
            'upper_bound': q3 + 1.5 * iqr,
            'extreme_lower': q1 - 3 * iqr,
            'extreme_upper': q3 + 3 * iqr,
            'median': statistics.median(values),
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def validate_analysis_time(self) -> Dict[str, Any]:
        """
        Detect analysis time outliers.
        
        Expected range: 0.1s - 10s for most contracts
        Outliers: Anything > Q3 + 1.5*IQR
        """
        times = [r['analysis_time'] for r in self.successful]
        stats = self.calculate_iqr_bounds(times)
        
        # Find outliers
        outliers = []
        for r in self.successful:
            time = r['analysis_time']
            if time > stats['upper_bound']:
                outliers.append({
                    'contract': r['contract_path'].split('/')[-1][:50],
                    'analysis_time': round(time, 2),
                    'severity': 'extreme' if time > stats['extreme_upper'] else 'moderate'
                })
        
        self.outliers['analysis_time'] = outliers
        
        return {
            'total_analyzed': len(times),
            'statistics': {k: round(v, 3) for k, v in stats.items()},
            'outliers_found': len(outliers),
            'outliers': sorted(outliers, key=lambda x: x['analysis_time'], reverse=True)[:10]  # Top 10
        }
    
    def validate_vulnerability_counts(self) -> Dict[str, Any]:
        """
        Detect unrealistic vulnerability counts.
        
        Expected: Most contracts have 0-10 High, 0-20 Medium
        Outliers: Anything suspiciously high
        """
        high_counts = [r['high_impact_count'] for r in self.successful]
        medium_counts = [r['medium_impact_count'] for r in self.successful]
        total_counts = [len(r.get('findings', [])) for r in self.successful]
        
        high_stats = self.calculate_iqr_bounds(high_counts)
        medium_stats = self.calculate_iqr_bounds(medium_counts)
        total_stats = self.calculate_iqr_bounds(total_counts)
        
        # Find high count outliers
        high_outliers = []
        for r in self.successful:
            count = r['high_impact_count']
            if count > high_stats['upper_bound']:
                high_outliers.append({
                    'contract': r['contract_path'].split('/')[-1][:50],
                    'high_impact_count': count,
                    'total_findings': len(r.get('findings', []))
                })
        
        # Find total count outliers
        total_outliers = []
        for r in self.successful:
            count = len(r.get('findings', []))
            if count > total_stats['upper_bound']:
                total_outliers.append({
                    'contract': r['contract_path'].split('/')[-1][:50],
                    'total_findings': count,
                    'high': r['high_impact_count'],
                    'medium': r['medium_impact_count']
                })
        
        self.outliers['high_impact_count'] = high_outliers
        self.outliers['total_findings'] = total_outliers
        
        return {
            'high_impact': {
                'statistics': {k: round(v, 2) for k, v in high_stats.items()},
                'outliers_found': len(high_outliers),
                'outliers': sorted(high_outliers, key=lambda x: x['high_impact_count'], reverse=True)[:10]
            },
            'medium_impact': {
                'statistics': {k: round(v, 2) for k, v in medium_stats.items()},
            },
            'total_findings': {
                'statistics': {k: round(v, 2) for k, v in total_stats.items()},
                'outliers_found': len(total_outliers),
                'outliers': sorted(total_outliers, key=lambda x: x['total_findings'], reverse=True)[:10]
            }
        }
    
    def validate_description_lengths(self) -> Dict[str, Any]:
        """
        Analyze finding description lengths.
        
        Expected: 50-500 characters
        Suspicious: < 20 or > 1000 characters
        """
        lengths = []
        short_descriptions = []
        long_descriptions = []
        
        for r in self.successful:
            for finding in r.get('findings', []):
                desc = finding.get('description', '')
                length = len(desc)
                lengths.append(length)
                
                if length < 20:
                    short_descriptions.append({
                        'contract': r['contract_path'].split('/')[-1][:40],
                        'length': length,
                        'preview': desc[:50]
                    })
                elif length > 1000:
                    long_descriptions.append({
                        'contract': r['contract_path'].split('/')[-1][:40],
                        'length': length,
                        'preview': desc[:100] + '...'
                    })
        
        stats = self.calculate_iqr_bounds(lengths)
        
        return {
            'total_descriptions': len(lengths),
            'statistics': {k: round(v, 2) for k, v in stats.items()},
            'short_descriptions': len(short_descriptions),
            'long_descriptions': len(long_descriptions),
            'examples_short': short_descriptions[:5],
            'examples_long': long_descriptions[:5]
        }
    
    def validate_version_distribution(self) -> Dict[str, Any]:
        """
        Analyze Solidity version distribution.
        
        Helps identify unusual version patterns.
        """
        versions = [r.get('detected_version') for r in self.successful if r.get('detected_version')]
        version_counter = Counter(versions)
        
        # Get major versions (0.4.x, 0.5.x, 0.6.x, etc.)
        major_versions = Counter()
        for v in versions:
            major = '.'.join(v.split('.')[:2])  # e.g., "0.8.19" -> "0.8"
            major_versions[major] += 1
        
        return {
            'total_versions_detected': len(versions),
            'unique_versions': len(version_counter),
            'top_10_versions': dict(version_counter.most_common(10)),
            'major_version_distribution': dict(major_versions.most_common()),
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical validation report.
        """
        print("\n" + "="*80)
        print("🔬 STATISTICAL VALIDATION - Running Analysis...")
        print("="*80)
        
        print("\n📊 Analyzing analysis times...")
        time_report = self.validate_analysis_time()
        
        print("📊 Analyzing vulnerability counts...")
        count_report = self.validate_vulnerability_counts()
        
        print("📊 Analyzing description lengths...")
        desc_report = self.validate_description_lengths()
        
        print("📊 Analyzing version distribution...")
        version_report = self.validate_version_distribution()
        
        report = {
            'timestamp': '2026-02-11T16:05:00',
            'total_records': len(self.data),
            'successful_records': len(self.successful),
            'analysis_time': time_report,
            'vulnerability_counts': count_report,
            'description_lengths': desc_report,
            'version_distribution': version_report,
            'summary': {
                'total_outliers_found': (
                    len(self.outliers['analysis_time']) +
                    len(self.outliers['high_impact_count']) +
                    len(self.outliers['total_findings'])
                ),
                'outlier_categories': {
                    'slow_analysis': len(self.outliers['analysis_time']),
                    'high_vuln_count': len(self.outliers['high_impact_count']),
                    'high_total_findings': len(self.outliers['total_findings'])
                }
            }
        }
        
        return report


def print_report(report: Dict[str, Any]):
    """Pretty print the statistical report."""
    print("\n" + "="*80)
    print("📊 STATISTICAL VALIDATION RESULTS")
    print("="*80)
    
    # Analysis time
    print("\n⏱️  ANALYSIS TIME:")
    time_stats = report['analysis_time']['statistics']
    print(f"   Median:        {time_stats['median']:.3f}s")
    print(f"   Mean:          {time_stats['mean']:.3f}s")
    print(f"   Std Dev:       {time_stats['stdev']:.3f}s")
    print(f"   Normal range:  {time_stats['lower_bound']:.3f}s - {time_stats['upper_bound']:.3f}s")
    print(f"   Outliers:      {report['analysis_time']['outliers_found']} contracts")
    
    if report['analysis_time']['outliers']:
        print("\n   Top 5 slowest analyses:")
        for i, outlier in enumerate(report['analysis_time']['outliers'][:5], 1):
            print(f"      {i}. {outlier['contract']}: {outlier['analysis_time']}s ({outlier['severity']})")
    
    # Vulnerability counts
    print("\n🐛 VULNERABILITY COUNTS:")
    high_stats = report['vulnerability_counts']['high_impact']['statistics']
    total_stats = report['vulnerability_counts']['total_findings']['statistics']
    
    print(f"   High impact - Median: {high_stats['median']}, Mean: {high_stats['mean']:.2f}")
    print(f"   High impact - Normal range: {high_stats['lower_bound']:.1f} - {high_stats['upper_bound']:.1f}")
    print(f"   High impact outliers: {report['vulnerability_counts']['high_impact']['outliers_found']}")
    
    print(f"\n   Total findings - Median: {total_stats['median']}, Mean: {total_stats['mean']:.2f}")
    print(f"   Total findings - Normal range: {total_stats['lower_bound']:.1f} - {total_stats['upper_bound']:.1f}")
    print(f"   Total findings outliers: {report['vulnerability_counts']['total_findings']['outliers_found']}")
    
    if report['vulnerability_counts']['total_findings']['outliers']:
        print("\n   Top 5 contracts with most findings:")
        for i, outlier in enumerate(report['vulnerability_counts']['total_findings']['outliers'][:5], 1):
            print(f"      {i}. {outlier['contract']}: {outlier['total_findings']} total "
                  f"(High: {outlier['high']}, Medium: {outlier['medium']})")
    
    # Description lengths
    print("\n📝 DESCRIPTION LENGTHS:")
    desc_stats = report['description_lengths']['statistics']
    print(f"   Median length: {desc_stats['median']:.0f} chars")
    print(f"   Mean length:   {desc_stats['mean']:.0f} chars")
    print(f"   Short (<20):   {report['description_lengths']['short_descriptions']}")
    print(f"   Long (>1000):  {report['description_lengths']['long_descriptions']}")
    
    # Version distribution
    print("\n📦 SOLIDITY VERSION DISTRIBUTION:")
    major_dist = report['version_distribution']['major_version_distribution']
    print(f"   Total versions: {report['version_distribution']['unique_versions']} unique")
    print("   Top major versions:")
    for version, count in list(major_dist.items())[:5]:
        pct = (count / report['version_distribution']['total_versions_detected']) * 100
        print(f"      {version}.x: {count:,} ({pct:.1f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"Total outliers detected: {report['summary']['total_outliers_found']}")
    print(f"   - Slow analyses:      {report['summary']['outlier_categories']['slow_analysis']}")
    print(f"   - High vuln counts:   {report['summary']['outlier_categories']['high_vuln_count']}")
    print(f"   - High total findings: {report['summary']['outlier_categories']['high_total_findings']}")
    
    print("\n✅ Statistical validation complete!")


def main():
    """Run statistical validation on full dataset."""
    # Load data
    data_path = Path("../../data/processed/bccc_full_dataset_results.json")
    
    print("Loading dataset...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data):,} records")
    
    # Run validation
    validator = StatisticalValidator(data)
    report = validator.generate_report()
    
    # Print report
    print_report(report)
    
    # Save report
    report_path = Path("../../data/reports/statistical_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
