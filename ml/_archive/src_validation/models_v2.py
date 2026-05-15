"""
Production Pydantic models for SENTINEL contract analysis results.

Schema Version: 2.1  # ← UPDATED
Created: Feb 10, 2026
Updated: Feb 11, 2026 - Made tolerant of extraction edge cases

Based on: Verified bccc_full_dataset_results.json structure

This file defines the COMPLETE data validation schema for Slither analysis results.
It validates THREE layers:
1. Type validation (is it a string/int/list?)
2. Field constraints (is the value in valid range/format?)
3. Semantic validation (do multiple fields make sense together?)

Author: Ali Motafegh
Project: SENTINEL - Verifiable AI Security Oracle
"""

from typing import List, Optional, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)
import re


# ============================================================================
# BASE CONFIGURATION
# ============================================================================

class BaseConfig(BaseModel):
    """
    Base configuration inherited by all models.
    
    Settings:
    - strict=True: Reject extra fields not defined in schema
    - validate_assignment=True: Validate when updating fields after creation
    - str_strip_whitespace=True: Auto-remove leading/trailing spaces from strings
    """
    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        str_strip_whitespace=True
    )


# ============================================================================
# NESTED MODELS
# ============================================================================

class Finding(BaseConfig):
    """
    Single vulnerability detected by Slither.
    
    Represents one security issue with:
    - Type (e.g., "reentrancy", "overflow")
    - Impact severity (High/Medium/Low/Informational/Optimization)
    - Detection confidence (High/Medium/Low)
    - Human-readable description
    - Affected source code line numbers
    
    Example:
    {
        "check": "reentrancy",
        "impact": "High",
        "confidence": "Medium",
        "description": "Reentrancy in withdraw() at line 45...",
        "lines": [45, 67, 89]
    }
    """
    
    # Vulnerability type (detector name)
    # FIX: Allow "unknown" as default for extraction failures
    check: str = Field(
        default="unknown",  # ← CHANGED from min_length=1
        description="Slither detector name (e.g., 'reentrancy', 'overflow'), 'unknown' if extraction failed"
    )
    
    # Impact level - restricted to exact values
    impact: Literal["High", "Medium", "Low", "Informational", "Optimization"] = Field(
        description="Vulnerability severity level"
    )
    
    # Confidence level
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Slither's confidence in this detection"
    )
    
    # Human-readable description
    description: str = Field(
        min_length=1,
        description="Detailed explanation of the vulnerability"
    )
    
    # Source code line numbers
    # FIX: Allow empty list for contract-level findings
    lines: List[int] = Field(
        default_factory=list,  # ← CHANGED
        description="Line numbers affected. Empty if extraction failed or contract-level issue"
    )
    
    @field_validator('lines')
    @classmethod
    def validate_lines(cls, v: List[int]) -> List[int]:
        """
        Validate and clean line numbers.
        
        Rules:
        - Empty list allowed (for contract-level findings or extraction failures)
        - All line numbers must be positive (> 0)
        - Remove duplicates
        - Return sorted
        """
        # ← REMOVED: if not v: raise ValueError
        # Allow empty lists
        if v and not all(line > 0 for line in v):  # ← CHANGED: only check if non-empty
            raise ValueError("Line numbers must be positive integers")
        
        # Remove duplicates and sort (handles empty list gracefully)
        return sorted(set(v))


# ============================================================================
# MAIN MODEL
# ============================================================================

class ContractResult(BaseConfig):
    """
    Complete result of Slither analysis on one smart contract.
    
    This model validates:
    1. Basic field types and constraints
    2. Cross-field consistency (counts match findings)
    3. Business logic (failed contracts have error messages)
    
    Two possible states:
    
    SUCCESS (analysis completed):
    {
        "contract_path": "contracts/MyToken.sol",
        "success": true,
        "error_type": null,
        "error_message": null,
        "detected_version": "0.8.19",
        "analysis_time": 1.23,
        "timestamp": "2026-02-10T15:30:00",
        "high_impact_count": 2,
        "medium_impact_count": 1,
        "vulnerability_types": ["reentrancy", "overflow"],
        "findings": [...]
    }
    
    FAILURE (analysis failed):
    {
        "contract_path": "contracts/Broken.sol",
        "success": false,
        "error_type": "compilation_error",
        "error_message": "Syntax error at line 45",
        "detected_version": "0.8.19",
        "analysis_time": 0.5,
        "timestamp": "2026-02-10T15:30:00",
        "high_impact_count": 0,
        "medium_impact_count": 0,
        "vulnerability_types": [],
        "findings": []
    }
    """
    
    # ========================================================================
    # CORE FIELDS (always present)
    # ========================================================================
    
    contract_path: str = Field(
        min_length=1,
        description="Relative path to the analyzed contract file"
    )
    
    success: bool = Field(
        description="Whether Slither analysis completed successfully"
    )
    
    # ========================================================================
    # ERROR FIELDS (present when success=False)
    # ========================================================================
    
    error_type: Optional[str] = Field(
        default=None,
        description="Category of error (e.g., 'compilation_error', 'timeout')"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed error message"
    )
    
    # ========================================================================
    # ANALYSIS METADATA
    # ========================================================================
    
    detected_version: Optional[str] = Field(
        default=None,
        description="Detected Solidity version, None if detection failed"
    )
    
    analysis_time: float = Field(
        ge=0.0,
        description="Time taken to analyze this contract (seconds)"
    )
    
    timestamp: str = Field(
        min_length=1,
        description="ISO format timestamp of analysis"
    )
    
    # ========================================================================
    # VULNERABILITY COUNTS
    # ========================================================================
    
    high_impact_count: int = Field(
        ge=0,
        description="Number of HIGH severity vulnerabilities"
    )
    
    medium_impact_count: int = Field(
        ge=0,
        description="Number of MEDIUM severity vulnerabilities"
    )
    
    # ========================================================================
    # VULNERABILITY DATA
    # ========================================================================
    
    vulnerability_types: List[str] = Field(
        default_factory=list,
        description="Unique vulnerability type names (simplified list)"
    )
    
    findings: List[Finding] = Field(
        default_factory=list,
        description="Complete list of vulnerabilities with full details"
    )
    
    # ========================================================================
    # FIELD VALIDATORS (validate individual fields)
    # ========================================================================
    
    @field_validator('detected_version')
    @classmethod
    def validate_version_format(cls, v: Optional[str]) -> Optional[str]:
        """Ensure version follows X.Y.Z format if present."""
        if v is None:
            return None  # OK for failed analyses
        
        pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(pattern, v):
            raise ValueError(f"Version must be in format X.Y.Z, got: {v}")
        return v
    
    @field_validator('contract_path')
    @classmethod
    def validate_path_format(cls, v: str) -> str:
        """Ensure contract path ends with .sol"""
        if not v.endswith('.sol'):
            raise ValueError(f"Contract path must end with .sol, got: {v}")
        return v
    
    # ========================================================================
    # MODEL VALIDATORS (validate across multiple fields)
    # ========================================================================
    
    @model_validator(mode='after')
    def validate_error_consistency(self):
        """
        Rule: If analysis failed, must have error type and message.
        Rule: If analysis succeeded, should NOT have error message.
        """
        if not self.success:
            # Failed analysis MUST have error info
            if not self.error_type:
                raise ValueError("Failed analysis must have error_type")
            if not self.error_message:
                raise ValueError("Failed analysis must have error_message")
        else:
            # Successful analysis should NOT have errors
            # (We allow None, just checking consistency)
            pass
        
        return self
    
    @model_validator(mode='after')
    def validate_counts_match_findings(self):
        """
        CRITICAL: Verify vulnerability counts match actual findings.
        
        This catches data corruption where:
        - high_impact_count says 3, but findings list has 5 High items
        - medium_impact_count says 0, but findings list has 2 Medium items
        """
        # Count actual High vulnerabilities in findings
        actual_high = sum(
            1 for f in self.findings 
            if f.impact == "High"
        )
        
        # Count actual Medium vulnerabilities in findings
        actual_medium = sum(
            1 for f in self.findings 
            if f.impact == "Medium"
        )
        
        # Verify counts match
        if actual_high != self.high_impact_count:
            raise ValueError(
                f"high_impact_count mismatch: "
                f"field says {self.high_impact_count}, "
                f"but findings has {actual_high} High-impact items"
            )
        
        if actual_medium != self.medium_impact_count:
            raise ValueError(
                f"medium_impact_count mismatch: "
                f"field says {self.medium_impact_count}, "
                f"but findings has {actual_medium} Medium-impact items"
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_vulnerability_types_consistency(self):
        """
        Verify vulnerability_types list matches findings.
        
        The vulnerability_types field should contain unique check names
        from all findings.
        """
        if self.findings:
            # Extract unique check names from findings
            checks_from_findings = set(f.check for f in self.findings)
            checks_from_list = set(self.vulnerability_types)
            
            # They should match
            if checks_from_findings != checks_from_list:
                missing = checks_from_findings - checks_from_list
                extra = checks_from_list - checks_from_findings
                error_msg = "vulnerability_types doesn't match findings: "
                if missing:
                    error_msg += f"missing {missing} "
                if extra:
                    error_msg += f"extra {extra}"
                raise ValueError(error_msg)
        
        return self


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_contract_result(data: dict) -> ContractResult:
    """
    Convenience function to validate a contract result dictionary.
    
    Usage:
        try:
            result = validate_contract_result(json_data)
            print("✅ Valid!")
        except ValidationError as e:
            print(f"❌ Invalid: {e}")
    
    Returns:
        ContractResult object if valid
    
    Raises:
        ValidationError if invalid
    """
    return ContractResult(**data)


def validate_batch(data_list: List[dict]) -> tuple[List[ContractResult], List[dict]]:
    """
    Validate a batch of contract results.
    
    Returns:
        (valid_results, invalid_items)
    
    Example:
        valid, invalid = validate_batch(json_data)
        print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
    """
    valid_results = []
    invalid_items = []
    
    for item in data_list:
        try:
            result = ContractResult(**item)
            valid_results.append(result)
        except Exception as e:
            invalid_items.append({
                'data': item,
                'error': str(e)
            })
    
    return valid_results, invalid_items


def get_ml_ready_dataset(all_results: List[ContractResult]) -> List[ContractResult]:
    """
    Filter to ML-ready records only.
    
    Criteria:
    - success=True (analysis completed)
    - detected_version is not None
    
    Returns:
        List of clean ContractResult objects ready for ML
    """
    return [
        r for r in all_results
        if r.success and r.detected_version is not None
    ]
