from models import ContractResult
from pydantic import ValidationError

print("="*50)
print("TEST 1: Valid Data")
print("="*50)

valid_data = {
    "contract_path": "contract.sol",
    "success": True,
    "high_impact_count": 3
}

result = ContractResult(**valid_data)
print("✅ Valid:", result)

print("\n" + "="*50)
print("TEST 2: Invalid Type (string instead of bool)")
print("="*50)

invalid_data = {
    "contract_path": "contract.sol",
    "success": "true",  # String, not boolean
    "high_impact_count": 3
}

try:
    result = ContractResult(**invalid_data)
    print("⚠️  NO ERROR - Pydantic accepted it:", result)
except ValidationError as e:
    print("❌ Caught error:", e)

print("\n" + "="*50)
print("TEST 3: Missing Field")
print("="*50)

missing_field = {
    "contract_path": "contract.sol",
    "success": True
    # Missing: high_impact_count
}

try:
    result = ContractResult(**missing_field)
    print("⚠️  NO ERROR:", result)
except ValidationError as e:
    print("❌ Caught error:")
    print(e)
print("\n" + "="*50)
print("TEST 4: Empty String")
print("="*50)

empty_string = {
    "contract_path": "",  # Empty!
    "success": True,
    "high_impact_count": 0
}

try:
    result = ContractResult(**empty_string)
    print("⚠️  NO ERROR:", result)
except ValidationError as e:
    print("❌ Caught error:")
    print(e)

print("\n" + "="*50)
print("TEST 5: Negative Count")
print("="*50)

negative_count = {
    "contract_path": "contract.sol",
    "success": True,
    "high_impact_count": -5  # Negative!
}

try:
    result = ContractResult(**negative_count)
    print("⚠️  NO ERROR:", result)
except ValidationError as e:
    print("❌ Caught error:")
    print(e)
