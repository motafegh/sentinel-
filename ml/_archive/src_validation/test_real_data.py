from models import ContractResult
from pydantic import ValidationError

print("="*60)
print("TEST 1: Successful Contract (From Real Data)")
print("="*60)

successful_contract = {
    "contract_path": "BCCC-SCsVul-2024/SourceCodes/ExternalBug/test.sol",
    "success": True,
    "error_type": None,
    "error_message": None,
    "detected_version": "0.4.11",
    "analysis_time": 0.834,
    "high_impact_count": 0,
    "vulnerability_types": ["shadowing-abstract"]
}

try:
    result = ContractResult(**successful_contract)
    print("✅ Valid!")
    print(f"Path: {result.contract_path}")
    print(f"Version: {result.detected_version}")
    print(f"Vulnerabilities: {result.vulnerability_types}")
    print(f"Analysis time: {result.analysis_time}s")
except ValidationError as e:
    print("❌ Failed:")
    print(e)

print("\n" + "="*60)
print("TEST 2: Failed Contract (From Real Data)")
print("="*60)

failed_contract = {
    "contract_path": "BCCC-SCsVul-2024/SourceCodes/DenialOfService/test.sol",
    "success": False,
    "error_type": "version_install_failed",
    "error_message": "Failed to install Solidity 0.3.5",
    "detected_version": "0.3.5",
    "analysis_time": 0.0,
    "high_impact_count": 0,
    "vulnerability_types": []
}

try:
    result = ContractResult(**failed_contract)
    print("✅ Valid!")
    print(f"Error type: {result.error_type}")
    print(f"Error message: {result.error_message}")
except ValidationError as e:
    print("❌ Failed:")
    print(e)
