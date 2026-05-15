"""
Pydantic models for contract result validation.
Based on actual SlitherWrapper output structure.
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ContractResult(BaseModel):
    """Schema for a single Slither analysis result"""
    
    model_config = ConfigDict(strict=True)
    
    # Path to analyzed contract (relative to project root)
    contract_path: str = Field(min_length=1)
    
    # Whether analysis completed successfully
    success: bool
    
    # Error details (only present when success=False)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Detected Solidity version (e.g., "0.8.19")
    detected_version: str = Field(min_length=1)
    
    # Analysis execution time in seconds
    analysis_time: float = Field(ge=0)
    
    # Count of HIGH impact vulnerabilities
    high_impact_count: int = Field(ge=0)
    
    # List of vulnerability type names (e.g., ["reentrancy", "overflow"])
    vulnerability_types: List[str]
