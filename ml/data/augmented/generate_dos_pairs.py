#!/usr/bin/env python3
"""
generate_dos_pairs.py — Write 30 DoS-vulnerable / 30 DoS-safe Solidity contract pairs.

Each pair is structurally identical except for the DoS mitigation:
  - Vulnerable: unbounded loop / no guard / push-payment  → DenialOfService=1
  - Safe:       bounded loop / require guard / pull-payment → all classes=0

DoS Patterns Covered (SWC-128 & SWC-113):
  1. Unbounded loop over growing array with .transfer() inside
  2. Unbounded loop with .send() inside
  3. Unexpected revert() in loop body (one revert blocks all)
  4. assert() in loop body (gas exhaustion)
  5. Block gas limit with nested loops
  6. Unbounded while loop with external call
  7. Array push inside iteration (growing while iterating)
  8. Unbounded mapping iteration via array key
  9. Large storage reads in loop
  10. Refund via push-payment (each .transfer can fail)
  11. Loop with delegatecall inside
  12. Self-destruct pattern in loop
  13. Unbounded gas-forwarding in loop
  14. Payable fallback in loop (reentrancy-gas combo)
  15. State update after unbounded loop (block gas limit)
  16. Unchecked .send() return in loop (silent failure → DoS)
  17. Dynamic array length manipulation before loop
  18. Multi-recipient payout without pull pattern
  19. Unbounded ERC20 token transfer loop
  20. Loop over unbounded external data (oracle)
  21. Gas-intensive operations in loop (SHA3/SSTORE)
  22. Recursive-style call chain via callback
  23. Unbounded staker payout
  24. DoS via require on external contract state
  25. Unbounded dividend distribution
  26. Airdrop push-payment
  27. Reward pool distribution
  28. Batch processing with revert-on-first-fail
  29. Loop with .call{} inside (gas limit)
  30. DAO proposal execution over unbounded voters

All contracts are Solidity ^0.8.0.

OUTPUT
------
  ml/data/augmented/dos_vuln_{01..30}.sol
  ml/data/augmented/dos_safe_{01..30}.sol

  Does NOT write CSV rows — that is done by inject_augmented.py after extraction.

NAMING CONVENTION
-----------------
  Files prefixed with "dos_" trigger DenialOfService=1 in inject_augmented.py.
  Files with "safe" in name get all-zeros labels.

USAGE
-----
  PYTHONPATH=. python ml/scripts/generate_dos_pairs.py
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRAGMA = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\n"


def _vuln(name: str, body: str) -> str:
    return PRAGMA + f"contract {name} {{\n{body}\n}}\n"


def _safe(name: str, body: str) -> str:
    return PRAGMA + f"contract {name} {{\n{body}\n}}\n"


PAIRS: list[tuple[str, str]] = []  # (vuln_src, safe_src)

# ─── 1 — Unbounded loop over growing array with .transfer() ────────────────────
PAIRS.append((
    _vuln("DosVuln01", """
    address[] public participants;
    uint256 public pot;
    function join() external payable {
        require(msg.value >= 0.01 ether);
        participants.push(msg.sender);
        pot += msg.value;
    }
    function distribute() external {
        uint256 share = pot / participants.length;
        for (uint256 i = 0; i < participants.length; i++) {
            payable(participants[i]).transfer(share);
        }
        pot = 0;
        delete participants;
    }"""),
    _safe("DosSafe01", """
    mapping(address => uint256) public pendingWithdrawal;
    uint256 public pot;
    function join() external payable {
        require(msg.value >= 0.01 ether);
        pendingWithdrawal[msg.sender] += msg.value;
        pot += msg.value;
    }
    function withdraw() external {
        uint256 amount = pendingWithdrawal[msg.sender];
        require(amount > 0);
        pendingWithdrawal[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
    }"""),
))

# ─── 2 — Unbounded loop with .send() inside ───────────────────────────────────
PAIRS.append((
    _vuln("DosVuln02", """
    address[] public payees;
    mapping(address => uint256) public owed;
    function addPayee(address p) external { payees.push(p); }
    function payAll() external {
        for (uint256 i = 0; i < payees.length; i++) {
            uint256 amt = owed[payees[i]];
            if (amt > 0) {
                payable(payees[i]).send(amt);
                owed[payees[i]] = 0;
            }
        }
    }"""),
    _safe("DosSafe02", """
    mapping(address => uint256) public owed;
    function addOwed(address p, uint256 amt) external { owed[p] = amt; }
    function withdraw() external {
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        owed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 3 — Unexpected revert() in loop body ─────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln03", """
    mapping(address => uint256) public balances;
    address[] public holders;
    function deposit() external payable { balances[msg.sender] += msg.value; holders.push(msg.sender); }
    function refundAll() external {
        for (uint256 i = 0; i < holders.length; i++) {
            uint256 amt = balances[holders[i]];
            if (amt == 0) revert("No balance");
            (bool ok,) = holders[i].call{value: amt}("");
            require(ok);
            balances[holders[i]] = 0;
        }
    }"""),
    _safe("DosSafe03", """
    mapping(address => uint256) public balances;
    function deposit() external payable { balances[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = balances[msg.sender];
        require(amt > 0, "No balance");
        balances[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 4 — assert() in loop body (gas exhaustion) ───────────────────────────────
PAIRS.append((
    _vuln("DosVuln04", """
    mapping(address => uint256) public stakes;
    address[] public stakers;
    function stake() external payable { stakes[msg.sender] += msg.value; stakers.push(msg.sender); }
    function distributeReward(uint256 rewardPerStaker) external {
        for (uint256 i = 0; i < stakers.length; i++) {
            assert(stakes[stakers[i]] > 0);
            (bool ok,) = stakers[i].call{value: rewardPerStaker}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe04", """
    mapping(address => uint256) public stakes;
    mapping(address => uint256) public rewards;
    function stake() external payable { stakes[msg.sender] += msg.value; }
    function accumulateReward(uint256 rewardPerStaker) external {
        if (stakes[msg.sender] > 0) {
            rewards[msg.sender] += rewardPerStaker;
        }
    }
    function claimReward() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 5 — Block gas limit with nested loops ─────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln05", """
    mapping(uint256 => address[]) public groupMembers;
    uint256 public numGroups;
    function addMember(uint256 gid, address m) external { groupMembers[gid].push(m); }
    function payoutAll(uint256 perMember) external payable {
        for (uint256 g = 0; g < numGroups; g++) {
            address[] storage members = groupMembers[g];
            for (uint256 i = 0; i < members.length; i++) {
                (bool ok,) = members[i].call{value: perMember}("");
                require(ok);
            }
        }
    }"""),
    _safe("DosSafe05", """
    mapping(address => uint256) public pendingPayout;
    function addPayout(address m, uint256 amt) external { pendingPayout[m] += amt; }
    function claim() external {
        uint256 amt = pendingPayout[msg.sender];
        require(amt > 0);
        pendingPayout[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 6 — Unbounded while loop with external call ──────────────────────────────
PAIRS.append((
    _vuln("DosVuln06", """
    mapping(address => uint256) public locked;
    function lock() external payable { locked[msg.sender] += msg.value; }
    function unlockAll(address[] calldata users) external {
        uint256 i = 0;
        while (i < users.length) {
            uint256 amt = locked[users[i]];
            if (amt > 0) {
                (bool ok,) = users[i].call{value: amt}("");
                require(ok);
                locked[users[i]] = 0;
            }
            i++;
        }
    }"""),
    _safe("DosSafe06", """
    mapping(address => uint256) public locked;
    function lock() external payable { locked[msg.sender] += msg.value; }
    function unlock() external {
        uint256 amt = locked[msg.sender];
        require(amt > 0);
        locked[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 7 — Array push inside iteration (growing while iterating) ─────────────────
PAIRS.append((
    _vuln("DosVuln07", """
    address[] public recipients;
    uint256 public amountPerRecipient;
    function addRecipient(address r) external { recipients.push(r); }
    function processAll() external {
        for (uint256 i = 0; i < recipients.length; i++) {
            (bool ok,) = recipients[i].call{value: amountPerRecipient}("");
            if (ok) {
                recipients.push(recipients[i]);
            }
            require(ok);
        }
    }"""),
    _safe("DosSafe07", """
    mapping(address => uint256) public credits;
    function addCredit(address r, uint256 amt) external { credits[r] += amt; }
    function claim() external {
        uint256 amt = credits[msg.sender];
        require(amt > 0);
        credits[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 8 — Unbounded mapping iteration via array key ────────────────────────────
PAIRS.append((
    _vuln("DosVuln08", """
    mapping(address => uint256) public dividends;
    address[] public shareholders;
    function addShareholder(address s) external { shareholders.push(s); }
    function distributeDividends() external payable {
        uint256 perShare = msg.value / shareholders.length;
        for (uint256 i = 0; i < shareholders.length; i++) {
            dividends[shareholders[i]] += perShare;
            (bool ok,) = shareholders[i].call{value: perShare}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe08", """
    mapping(address => uint256) public dividends;
    function addShareholder(address s) external pure {}
    function accrueDividend(address s, uint256 amt) external { dividends[s] += amt; }
    function claimDividend() external {
        uint256 amt = dividends[msg.sender];
        require(amt > 0);
        dividends[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 9 — Large storage reads in loop ──────────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln09", """
    mapping(uint256 => bytes32) public dataStore;
    uint256 public numEntries;
    function storeEntry(bytes32 data) external { dataStore[numEntries++] = data; }
    function processAll() external {
        for (uint256 i = 0; i < numEntries; i++) {
            bytes32 data = dataStore[i];
            if (data != bytes32(0)) {
                payable(msg.sender).transfer(0.001 ether);
            }
        }
    }"""),
    _safe("DosSafe09", """
    mapping(address => bool) public hasReward;
    function markReward(address a) external { hasReward[a] = true; }
    function claimReward() external {
        require(hasReward[msg.sender]);
        hasReward[msg.sender] = false;
        (bool ok,) = msg.sender.call{value: 0.001 ether}("");
        require(ok);
    }"""),
))

# ─── 10 — Refund via push-payment (each .transfer can fail) ───────────────────
PAIRS.append((
    _vuln("DosVuln10", """
    mapping(address => uint256) public contributions;
    address[] public contributors;
    function contribute() external payable { contributions[msg.sender] += msg.value; contributors.push(msg.sender); }
    function refundAll() external {
        for (uint256 i = 0; i < contributors.length; i++) {
            uint256 amt = contributions[contributors[i]];
            if (amt > 0) {
                payable(contributors[i]).transfer(amt);
                contributions[contributors[i]] = 0;
            }
        }
    }"""),
    _safe("DosSafe10", """
    mapping(address => uint256) public contributions;
    function contribute() external payable { contributions[msg.sender] += msg.value; }
    function withdrawContribution() external {
        uint256 amt = contributions[msg.sender];
        require(amt > 0);
        contributions[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 11 — Loop with delegatecall inside ───────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln11", """
    address[] public modules;
    function addModule(address m) external { modules.push(m); }
    function executeAll() external {
        for (uint256 i = 0; i < modules.length; i++) {
            (bool ok,) = modules[i].delegatecall(abi.encodeWithSignature("execute()"));
            require(ok);
        }
    }"""),
    _safe("DosSafe11", """
    mapping(address => bool) public isModule;
    function addModule(address m) external { isModule[m] = true; }
    function executeModule(address m) external {
        require(isModule[m]);
        (bool ok,) = m.delegatecall(abi.encodeWithSignature("execute()"));
        require(ok);
    }"""),
))

# ─── 12 — Self-destruct pattern in loop ───────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln12", """
    address[] public childContracts;
    function addChild(address c) external { childContracts.push(c); }
    function destroyAll() external {
        for (uint256 i = 0; i < childContracts.length; i++) {
            IChild(childContracts[i]).destroy();
        }
    }
    interface IChild { function destroy() external; }"""),
    _safe("DosSafe12", """
    mapping(address => bool) public isChild;
    function addChild(address c) external { isChild[c] = true; }
    function destroyChild(address c) external {
        require(isChild[c]);
        IChild(c).destroy();
    }
    interface IChild { function destroy() external; }"""),
))

# ─── 13 — Unbounded gas-forwarding in loop ────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln13", """
    address[] public delegates;
    function addDelegate(address d) external { delegates.push(d); }
    function notifyAll() external {
        for (uint256 i = 0; i < delegates.length; i++) {
            (bool ok,) = delegates[i].call{value: 0, gas: 100000}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe13", """
    mapping(address => bool) public isDelegate;
    function addDelegate(address d) external { isDelegate[d] = true; }
    function notifyDelegate(address d) external {
        require(isDelegate[d]);
        (bool ok,) = d.call{value: 0, gas: 100000}("");
        require(ok);
    }"""),
))

# ─── 14 — Payable fallback in loop (gas exhaustion from fallback) ─────────────
PAIRS.append((
    _vuln("DosVuln14", """
    address[] public recipients;
    mapping(address => uint256) public owed;
    function addRecipient(address r, uint256 amt) external payable { recipients.push(r); owed[r] = amt; }
    function payAll() external {
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 amt = owed[recipients[i]];
            if (amt > 0) {
                (bool ok,) = recipients[i].call{value: amt}("");
                require(ok);
                owed[recipients[i]] = 0;
            }
        }
    }"""),
    _safe("DosSafe14", """
    mapping(address => uint256) public owed;
    function addOwed(address r, uint256 amt) external payable { owed[r] = amt; }
    function withdrawOwed() external {
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        owed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 15 — State update after unbounded loop (block gas limit) ──────────────────
PAIRS.append((
    _vuln("DosVuln15", """
    address[] public members;
    bool public distributed;
    function join() external { members.push(msg.sender); }
    function distribute() external {
        require(!distributed);
        for (uint256 i = 0; i < members.length; i++) {
            payable(members[i]).transfer(0.01 ether);
        }
        distributed = true;
    }"""),
    _safe("DosSafe15", """
    mapping(address => bool) public isMember;
    mapping(address => bool) public claimed;
    function join() external { isMember[msg.sender] = true; }
    function claim() external {
        require(isMember[msg.sender] && !claimed[msg.sender]);
        claimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: 0.01 ether}("");
        require(ok);
    }"""),
))

# ─── 16 — Unchecked .send() return in loop (silent failure → DoS) ─────────────
PAIRS.append((
    _vuln("DosVuln16", """
    address[] public investors;
    mapping(address => uint256) public invested;
    function invest() external payable { investors.push(msg.sender); invested[msg.sender] += msg.value; }
    function returnCapital() external {
        for (uint256 i = 0; i < investors.length; i++) {
            uint256 amt = invested[investors[i]];
            if (amt > 0) {
                payable(investors[i]).send(amt);
                invested[investors[i]] = 0;
            }
        }
    }"""),
    _safe("DosSafe16", """
    mapping(address => uint256) public invested;
    function invest() external payable { invested[msg.sender] += msg.value; }
    function withdrawCapital() external {
        uint256 amt = invested[msg.sender];
        require(amt > 0);
        invested[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 17 — Dynamic array length manipulation before loop ───────────────────────
PAIRS.append((
    _vuln("DosVuln17", """
    address[] public whitelist;
    function addToWhitelist(address a) external { whitelist.push(a); }
    function batchTransfer() external payable {
        uint256 perAddr = msg.value / whitelist.length;
        for (uint256 i = 0; i < whitelist.length; i++) {
            (bool ok,) = whitelist[i].call{value: perAddr}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe17", """
    mapping(address => bool) public whitelisted;
    mapping(address => uint256) public allocation;
    function addToWhitelist(address a) external { whitelisted[a] = true; }
    function setAllocation(address a, uint256 amt) external payable { allocation[a] = amt; }
    function claim() external {
        require(whitelisted[msg.sender]);
        uint256 amt = allocation[msg.sender];
        require(amt > 0);
        allocation[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 18 — Multi-recipient payout without pull pattern ─────────────────────────
PAIRS.append((
    _vuln("DosVuln18", """
    address payable[] public recipients;
    function addRecipient(address payable r) external { recipients.push(r); }
    function splitEven() external payable {
        uint256 share = msg.value / recipients.length;
        for (uint256 i = 0; i < recipients.length; i++) {
            recipients[i].transfer(share);
        }
    }"""),
    _safe("DosSafe18", """
    mapping(address => uint256) public credits;
    function addCredit(address r, uint256 amt) external payable { credits[r] += amt; }
    function withdrawCredit() external {
        uint256 amt = credits[msg.sender];
        require(amt > 0);
        credits[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 19 — Unbounded ERC20 token transfer loop ─────────────────────────────────
PAIRS.append((
    _vuln("DosVuln19", """
    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    address[] public holders;
    constructor(address t) { token = IERC20(t); }
    function addHolder(address h) external { holders.push(h); }
    function airdrop(uint256 amt) external {
        for (uint256 i = 0; i < holders.length; i++) {
            token.transfer(holders[i], amt);
        }
    }"""),
    _safe("DosSafe19", """
    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    mapping(address => uint256) public pendingAirdrop;
    constructor(address t) { token = IERC20(t); }
    function registerAirdrop(address h, uint256 amt) external { pendingAirdrop[h] = amt; }
    function claimAirdrop() external {
        uint256 amt = pendingAirdrop[msg.sender];
        require(amt > 0);
        pendingAirdrop[msg.sender] = 0;
        token.transfer(msg.sender, amt);
    }"""),
))

# ─── 20 — Loop over unbounded external data (oracle) ──────────────────────────
PAIRS.append((
    _vuln("DosVuln20", """
    interface IOracle { function getFeedCount() external view returns (uint256); function getFeed(uint256 i) external view returns (address); }
    IOracle public oracle;
    constructor(address o) { oracle = IOracle(o); }
    function payoutFeeds() external payable {
        uint256 count = oracle.getFeedCount();
        uint256 perFeed = msg.value / count;
        for (uint256 i = 0; i < count; i++) {
            address feed = oracle.getFeed(i);
            payable(feed).transfer(perFeed);
        }
    }"""),
    _safe("DosSafe20", """
    mapping(address => uint256) public feedCredits;
    function creditFeed(address f, uint256 amt) external payable { feedCredits[f] += amt; }
    function withdrawFeedCredit() external {
        uint256 amt = feedCredits[msg.sender];
        require(amt > 0);
        feedCredits[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 21 — Gas-intensive operations in loop (SHA3/SSTORE) ─────────────────────
PAIRS.append((
    _vuln("DosVuln21", """
    mapping(uint256 => bytes32) public hashes;
    uint256 public count;
    function addEntry(bytes32 h) external { hashes[count++] = h; }
    function rehashAll() external {
        for (uint256 i = 0; i < count; i++) {
            hashes[i] = keccak256(abi.encodePacked(hashes[i]));
        }
    }"""),
    _safe("DosSafe21", """
    mapping(uint256 => bytes32) public hashes;
    uint256 public count;
    function addEntry(bytes32 h) external { hashes[count++] = h; }
    function rehashEntry(uint256 i) external {
        require(i < count);
        hashes[i] = keccak256(abi.encodePacked(hashes[i]));
    }"""),
))

# ─── 22 — Recursive-style call chain via callback ─────────────────────────────
PAIRS.append((
    _vuln("DosVuln22", """
    address[] public processors;
    function addProcessor(address p) external { processors.push(p); }
    function processChain() external {
        for (uint256 i = 0; i < processors.length; i++) {
            (bool ok,) = processors[i].call(abi.encodeWithSignature("onProcess()"));
            require(ok);
        }
    }"""),
    _safe("DosSafe22", """
    mapping(address => bool) public isProcessor;
    function addProcessor(address p) external { isProcessor[p] = true; }
    function callProcessor(address p) external {
        require(isProcessor[p]);
        (bool ok,) = p.call(abi.encodeWithSignature("onProcess()"));
        require(ok);
    }"""),
))

# ─── 23 — Unbounded staker payout ─────────────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln23", """
    address[] public stakers;
    mapping(address => uint256) public staked;
    function stake() external payable { stakers.push(msg.sender); staked[msg.sender] += msg.value; }
    function distributeRewards() external payable {
        uint256 totalStaked;
        for (uint256 i = 0; i < stakers.length; i++) { totalStaked += staked[stakers[i]]; }
        for (uint256 i = 0; i < stakers.length; i++) {
            uint256 reward = msg.value * staked[stakers[i]] / totalStaked;
            (bool ok,) = stakers[i].call{value: reward}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe23", """
    mapping(address => uint256) public staked;
    mapping(address => uint256) public rewards;
    uint256 public totalStaked;
    function stake() external payable { staked[msg.sender] += msg.value; totalStaked += msg.value; }
    function accumulateRewards() external payable {
        if (totalStaked > 0 && staked[msg.sender] > 0) {
            rewards[msg.sender] += msg.value * staked[msg.sender] / totalStaked;
        }
    }
    function claimRewards() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 24 — DoS via require on external contract state ──────────────────────────
PAIRS.append((
    _vuln("DosVuln24", """
    interface IRegistry { function isRegistered(address) external view returns (bool); }
    IRegistry public registry;
    address[] public applicants;
    constructor(address r) { registry = IRegistry(r); }
    function applyFor(address a) external { applicants.push(a); }
    function processApplications() external {
        for (uint256 i = 0; i < applicants.length; i++) {
            require(registry.isRegistered(applicants[i]));
            payable(applicants[i]).transfer(0.1 ether);
        }
    }"""),
    _safe("DosSafe24", """
    mapping(address => bool) public approved;
    function approve(address a) external { approved[a] = true; }
    function claimApproval() external {
        require(approved[msg.sender]);
        approved[msg.sender] = false;
        (bool ok,) = msg.sender.call{value: 0.1 ether}("");
        require(ok);
    }"""),
))

# ─── 25 — Unbounded dividend distribution ─────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln25", """
    address[] public tokenHolders;
    mapping(address => uint256) public holdings;
    function addHolder(address h, uint256 tokens) external { tokenHolders.push(h); holdings[h] = tokens; }
    function payDividends() external payable {
        for (uint256 i = 0; i < tokenHolders.length; i++) {
            uint256 div = msg.value * holdings[tokenHolders[i]] / 10000;
            payable(tokenHolders[i]).transfer(div);
        }
    }"""),
    _safe("DosSafe25", """
    mapping(address => uint256) public holdings;
    mapping(address => uint256) public dividendBalance;
    function addHolder(address h, uint256 tokens) external { holdings[h] = tokens; }
    function accrueDividend(address h, uint256 amt) external payable { dividendBalance[h] += amt; }
    function withdrawDividend() external {
        uint256 amt = dividendBalance[msg.sender];
        require(amt > 0);
        dividendBalance[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 26 — Airdrop push-payment ────────────────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln26", """
    address[] public recipients;
    mapping(address => uint256) public airdropAmount;
    function register(address r, uint256 amt) external { recipients.push(r); airdropAmount[r] = amt; }
    function executeAirdrop() external payable {
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 amt = airdropAmount[recipients[i]];
            if (amt > 0) {
                (bool ok,) = recipients[i].call{value: amt}("");
                require(ok);
                airdropAmount[recipients[i]] = 0;
            }
        }
    }"""),
    _safe("DosSafe26", """
    mapping(address => uint256) public airdropAmount;
    function register(address r, uint256 amt) external { airdropAmount[r] = amt; }
    function claimAirdrop() external {
        uint256 amt = airdropAmount[msg.sender];
        require(amt > 0);
        airdropAmount[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 27 — Reward pool distribution ───────────────────────────────────────────
PAIRS.append((
    _vuln("DosVuln27", """
    address[] public miners;
    mapping(address => uint256) public rewards;
    function registerMiner(address m) external { miners.push(m); }
    function distributePool() external payable {
        uint256 perMiner = msg.value / miners.length;
        for (uint256 i = 0; i < miners.length; i++) {
            rewards[miners[i]] += perMiner;
            (bool ok,) = miners[i].call{value: perMiner}("");
            require(ok);
        }
    }"""),
    _safe("DosSafe27", """
    mapping(address => uint256) public rewards;
    function registerMiner(address m) external pure {}
    function accrueReward(address m, uint256 amt) external { rewards[m] += amt; }
    function claimReward() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 28 — Batch processing with revert-on-first-fail ──────────────────────────
PAIRS.append((
    _vuln("DosVuln28", """
    struct Order { address buyer; uint256 amount; bool refunded; }
    Order[] public orders;
    function placeOrder() external payable { orders.push(Order(msg.sender, msg.value, false)); }
    function refundAll() external {
        for (uint256 i = 0; i < orders.length; i++) {
            if (!orders[i].refunded) {
                (bool ok,) = orders[i].buyer.call{value: orders[i].amount}("");
                require(ok, "Refund failed");
                orders[i].refunded = true;
            }
        }
    }"""),
    _safe("DosSafe28", """
    mapping(address => uint256) public orderAmounts;
    function placeOrder() external payable { orderAmounts[msg.sender] += msg.value; }
    function refundOrder() external {
        uint256 amt = orderAmounts[msg.sender];
        require(amt > 0);
        orderAmounts[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# ─── 29 — Loop with .call{} inside (gas limit) ────────────────────────────────
PAIRS.append((
    _vuln("DosVuln29", """
    address[] public validators;
    function addValidator(address v) external { validators.push(v); }
    function notifyValidators(bytes calldata data) external {
        for (uint256 i = 0; i < validators.length; i++) {
            (bool ok,) = validators[i].call(data);
            require(ok);
        }
    }"""),
    _safe("DosSafe29", """
    mapping(address => bool) public isValidator;
    function addValidator(address v) external { isValidator[v] = true; }
    function notifyValidator(address v, bytes calldata data) external {
        require(isValidator[v]);
        (bool ok,) = v.call(data);
        require(ok);
    }"""),
))

# ─── 30 — DAO proposal execution over unbounded voters ────────────────────────
PAIRS.append((
    _vuln("DosVuln30", """
    struct Proposal { address payable recipient; uint256 amount; bool executed; uint256 voteCount; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    function createProposal(address payable r, uint256 a) external returns (uint256) {
        proposals.push(Proposal(r, a, false, 0));
        return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!hasVoted[id][msg.sender]); hasVoted[id][msg.sender] = true; proposals[id].voteCount++; }
    function executeProposal(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed);
        require(p.voteCount >= 3);
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
        p.executed = true;
    }"""),
    _safe("DosSafe30", """
    struct Proposal { address payable recipient; uint256 amount; bool executed; uint256 voteCount; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    function createProposal(address payable r, uint256 a) external returns (uint256) {
        proposals.push(Proposal(r, a, false, 0));
        return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!hasVoted[id][msg.sender]); hasVoted[id][msg.sender] = true; proposals[id].voteCount++; }
    function executeProposal(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed);
        require(p.voteCount >= 3);
        p.executed = true;
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
    }"""),
))


# ── Write files ────────────────────────────────────────────────────────────────

assert len(PAIRS) == 30, f"Expected 30 pairs, got {len(PAIRS)}"

written = []
for i, (vuln_src, safe_src) in enumerate(PAIRS, start=1):
    vpath = OUT_DIR / f"dos_vuln_{i:02d}.sol"
    spath = OUT_DIR / f"dos_safe_{i:02d}.sol"
    vpath.write_text(vuln_src)
    spath.write_text(safe_src)
    written.append(vpath)
    written.append(spath)

print(f"Written {len(written)} contracts to {OUT_DIR}")
for p in sorted(written):
    print(f"  {p.name}")
