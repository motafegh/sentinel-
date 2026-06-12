"""Hand-crafted trivial positive and negative Solidity contracts for the
probe dataset. One trivial positive per class — the simplest possible
example that exhibits the pattern. One trivial negative — a clean
OZ-style contract that exhibits NONE of the 10 patterns.

The trivial positive is the "did the model learn the pattern" probe.
The trivial negative is the "did the model learn to NOT over-predict"
probe. Both are required for the model interpretability suite
(ml/scripts/interpretability/).

Source: Stage 4 Task 4.6 per AUDIT_PATCHES 4-P6.
"""
from __future__ import annotations

# BCCC 12-class → SENTINEL 10-class mapping
_BCCC_TO_SENTINEL = {
    "externalbug": "ExternalBug",
    "gasexception": "GasException",
    "mishandledexception": "MishandledException",
    "timestamp": "Timestamp",
    "unusedreturn": "UnusedReturn",
    "calltounknown": "CallToUnknown",
    "denialofservice": "DenialOfService",
    "integeruo": "IntegerUO",
    "reentrancy": "Reentrancy",
    "transactionorderdependence": "TransactionOrderDependence",
}


def bccc_class_to_sentinel(bccc_class: str) -> str | None:
    """Map a BCCC review-batch class string to a SENTINEL class name."""
    return _BCCC_TO_SENTINEL.get(bccc_class.lower())


# ── Trivial positives (one per class) ─────────────────────────────────────────
# Each is the SIMPLEST possible Solidity contract that exhibits the pattern.
# If the model classifies any of these as negative, it has NOT learned the
# pattern.

TRIVIAL_POSITIVES: dict[str, str] = {
    "Reentrancy": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: classic withdraw-before-state-update (CEI violation).
// State write AFTER external call → reentrancy possible.
contract TrivialPositive_Reentrancy {
    mapping(address => uint) public balances;
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }
}
""",
    "CallToUnknown": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: raw low-level call to a dynamic address with the
// return value discarded.
contract TrivialPositive_CallToUnknown {
    function execute(address target, bytes calldata data) public {
        target.call(data);
    }
}
""",
    "Timestamp": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: block.timestamp in a security-sensitive conditional.
contract TrivialPositive_Timestamp {
    mapping(address => uint) public unlockTime;
    function withdraw() public {
        require(block.timestamp >= unlockTime[msg.sender]);
        payable(msg.sender).transfer(1 ether);
    }
}
""",
    "IntegerUO": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: arithmetic inside unchecked{} block in 0.8.x.
contract TrivialPositive_IntegerUO {
    function add(uint a, uint b) public pure returns (uint) {
        unchecked { return a + b; }
    }
}
""",
    "ExternalBug": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: tx.origin used for authentication.
contract TrivialPositive_ExternalBug {
    address public owner;
    constructor() { owner = msg.sender; }
    modifier onlyOwner {
        require(tx.origin == owner, "not owner");
        _;
    }
}
""",
    "DenialOfService": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: unbounded loop with external transfer — one revert
// from any investor DoSes the whole payout.
contract TrivialPositive_DenialOfService {
    address[] public investors;
    mapping(address => uint) public shares;
    function distribute() public {
        for (uint i = 0; i < investors.length; i++) {
            payable(investors[i]).transfer(shares[investors[i]]);
        }
    }
}
""",
    "GasException": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: unchecked send() — return bool ignored.
contract TrivialPositive_GasException {
    function payout(address payable to, uint amount) public {
        to.send(amount);
    }
}
""",
    "MishandledException": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: low-level call with discarded (bool, bytes) return.
contract TrivialPositive_MishandledException {
    function execute(address target) public {
        target.call("");
    }
}
""",
    "UnusedReturn": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: internal function call with return value discarded.
contract TrivialPositive_UnusedReturn {
    function check(uint x) internal pure returns (bool) { return x > 0; }
    function doStuff(uint x) public {
        check(x);
    }
}
""",
    "TransactionOrderDependence": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial positive: tx.origin in a permission check (front-runnable).
contract TrivialPositive_TransactionOrderDependence {
    function claim() public {
        require(tx.origin == msg.sender, "no contracts");
        payable(msg.sender).transfer(1 ether);
    }
}
""",
}


# ── Trivial negative (one for all classes) ───────────────────────────────────
# A clean OZ-style contract that exhibits NONE of the 10 patterns.
# Used to test that the model does not over-predict on safe code.

TRIVIAL_NEGATIVE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Trivial negative: a clean OZ-style contract that exhibits NONE of the
// 10 sentinel patterns.
//   - No raw low-level calls (no dot-call/delegatecall/send on dynamic targets)
//   - No reentrancy (state update before any external call)
//   - No timestamp-based gating in conditionals
//   - No origin-equals-sender is forbidden (uses sender)
//   - All arithmetic is checked (reverts on overflow in 0.8.x)
//   - No DoS pattern (single-recipient transfer, not unbounded loop)
//   - No ignored return values
contract TrivialNegative {
    mapping(address => uint) private _balances;
    mapping(address => mapping(address => uint)) private _allowances;
    uint private _totalSupply;

    event Transfer(address indexed from, address indexed to, uint amount);
    event Approval(address indexed owner, address indexed spender, uint amount);

    function balanceOf(address account) external view returns (uint) {
        return _balances[account];
    }
    function totalSupply() external view returns (uint) {
        return _totalSupply;
    }
    function allowance(address owner, address spender) external view returns (uint) {
        return _allowances[owner][spender];
    }
    function transfer(address to, uint amount) external returns (bool) {
        require(to != address(0), "ERC20: to zero");
        require(_balances[msg.sender] >= amount, "ERC20: balance");
        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    function approve(address spender, uint amount) external returns (bool) {
        require(spender != address(0), "ERC20: spender zero");
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    function transferFrom(address from, address to, uint amount) external returns (bool) {
        require(to != address(0), "ERC20: to zero");
        require(_balances[from] >= amount, "ERC20: balance");
        require(_allowances[from][msg.sender] >= amount, "ERC20: allowance");
        _allowances[from][msg.sender] -= amount;
        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }
    function _mint(address to, uint amount) internal {
        require(to != address(0), "ERC20: mint to zero");
        _totalSupply += amount;
        _balances[to] += amount;
        emit Transfer(address(0), to, amount);
    }
}
"""
