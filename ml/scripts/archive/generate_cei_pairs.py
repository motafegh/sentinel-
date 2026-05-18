#!/usr/bin/env python3
"""
generate_cei_pairs.py — Write 25 CEI-vulnerable / 25 CEI-safe Solidity contracts.

Each pair is structurally identical except for the ordering of the external call
relative to the state update:
  - Vulnerable: external call BEFORE state write  → Reentrancy=1
  - Safe:       state write  BEFORE external call → Reentrancy=0

All contracts are Solidity ^0.8.0 (uses solc-0.8.31 in reextract).

OUTPUT
──────
  ml/data/augmented/cei_vuln_{01..25}.sol
  ml/data/augmented/cei_safe_{01..25}.sol

  Does NOT write CSV rows — that is done by inject_augmented.py after extraction.

USAGE
─────
  PYTHONPATH=. python ml/scripts/generate_cei_pairs.py
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "ml" / "data" / "augmented"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRAGMA = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\n"

# ── Template factory ──────────────────────────────────────────────────────────

def _vuln(name: str, body: str) -> str:
    return PRAGMA + f"contract {name} {{\n{body}\n}}\n"

def _safe(name: str, body: str) -> str:
    return PRAGMA + f"contract {name} {{\n{body}\n}}\n"


# ── 25 pairs ──────────────────────────────────────────────────────────────────

PAIRS: list[tuple[str, str]] = []   # (vuln_src, safe_src)

# 1 — Simple ETH withdraw
PAIRS.append((
    _vuln("WithdrawVuln01", """
    mapping(address => uint256) public balances;
    function deposit() external payable { balances[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = balances[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");  // call before update
        require(ok);
        balances[msg.sender] = 0;
    }"""),
    _safe("WithdrawSafe01", """
    mapping(address => uint256) public balances;
    function deposit() external payable { balances[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = balances[msg.sender];
        require(amt > 0);
        balances[msg.sender] = 0;                      // update before call
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 2 — ERC20 reward claim
PAIRS.append((
    _vuln("RewardVuln02", """
    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    mapping(address => uint256) public pending;
    constructor(address t) { token = IERC20(t); }
    function claim() external {
        uint256 amt = pending[msg.sender];
        require(amt > 0);
        token.transfer(msg.sender, amt);   // external call before state clear
        pending[msg.sender] = 0;
    }"""),
    _safe("RewardSafe02", """
    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    mapping(address => uint256) public pending;
    constructor(address t) { token = IERC20(t); }
    function claim() external {
        uint256 amt = pending[msg.sender];
        require(amt > 0);
        pending[msg.sender] = 0;           // clear before external call
        token.transfer(msg.sender, amt);
    }"""),
))

# 3 — Staking withdraw
PAIRS.append((
    _vuln("StakeVuln03", """
    mapping(address => uint256) public staked;
    function stake() external payable { staked[msg.sender] += msg.value; }
    function unstake(uint256 amt) external {
        require(staked[msg.sender] >= amt);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        staked[msg.sender] -= amt;
    }"""),
    _safe("StakeSafe03", """
    mapping(address => uint256) public staked;
    function stake() external payable { staked[msg.sender] += msg.value; }
    function unstake(uint256 amt) external {
        require(staked[msg.sender] >= amt);
        staked[msg.sender] -= amt;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 4 — Escrow release
PAIRS.append((
    _vuln("EscrowVuln04", """
    mapping(bytes32 => uint256) public deposits;
    mapping(bytes32 => address) public beneficiary;
    function deposit(bytes32 id, address ben) external payable {
        deposits[id] = msg.value; beneficiary[id] = ben;
    }
    function release(bytes32 id) external {
        uint256 amt = deposits[id];
        require(amt > 0);
        (bool ok,) = beneficiary[id].call{value: amt}("");
        require(ok);
        deposits[id] = 0;
    }"""),
    _safe("EscrowSafe04", """
    mapping(bytes32 => uint256) public deposits;
    mapping(bytes32 => address) public beneficiary;
    function deposit(bytes32 id, address ben) external payable {
        deposits[id] = msg.value; beneficiary[id] = ben;
    }
    function release(bytes32 id) external {
        uint256 amt = deposits[id];
        require(amt > 0);
        deposits[id] = 0;
        (bool ok,) = beneficiary[id].call{value: amt}("");
        require(ok);
    }"""),
))

# 5 — Crowdfund refund
PAIRS.append((
    _vuln("CrowdVuln05", """
    mapping(address => uint256) public contributions;
    bool public failed;
    function contribute() external payable { contributions[msg.sender] += msg.value; }
    function refund() external {
        require(failed);
        uint256 amt = contributions[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        contributions[msg.sender] = 0;
    }"""),
    _safe("CrowdSafe05", """
    mapping(address => uint256) public contributions;
    bool public failed;
    function contribute() external payable { contributions[msg.sender] += msg.value; }
    function refund() external {
        require(failed);
        uint256 amt = contributions[msg.sender];
        require(amt > 0);
        contributions[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 6 — Vault shares
PAIRS.append((
    _vuln("VaultVuln06", """
    mapping(address => uint256) public shares;
    uint256 public totalShares;
    function deposit() external payable { shares[msg.sender] += msg.value; totalShares += msg.value; }
    function redeem(uint256 s) external {
        require(shares[msg.sender] >= s);
        uint256 amt = s * address(this).balance / totalShares;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        shares[msg.sender] -= s;
        totalShares -= s;
    }"""),
    _safe("VaultSafe06", """
    mapping(address => uint256) public shares;
    uint256 public totalShares;
    function deposit() external payable { shares[msg.sender] += msg.value; totalShares += msg.value; }
    function redeem(uint256 s) external {
        require(shares[msg.sender] >= s);
        uint256 amt = s * address(this).balance / totalShares;
        shares[msg.sender] -= s;
        totalShares -= s;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 7 — Auction winner payout
PAIRS.append((
    _vuln("AuctionVuln07", """
    address public winner;
    uint256 public highBid;
    mapping(address => uint256) public bids;
    function bid() external payable {
        require(msg.value > highBid);
        highBid = msg.value; winner = msg.sender;
    }
    function claimRefund() external {
        require(msg.sender != winner);
        uint256 amt = bids[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        bids[msg.sender] = 0;
    }"""),
    _safe("AuctionSafe07", """
    address public winner;
    uint256 public highBid;
    mapping(address => uint256) public bids;
    function bid() external payable {
        require(msg.value > highBid);
        highBid = msg.value; winner = msg.sender;
    }
    function claimRefund() external {
        require(msg.sender != winner);
        uint256 amt = bids[msg.sender];
        require(amt > 0);
        bids[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 8 — Lending repay and withdraw
PAIRS.append((
    _vuln("LendVuln08", """
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public borrowed;
    function depositCollateral() external payable { collateral[msg.sender] += msg.value; }
    function repayAndWithdraw() external payable {
        require(msg.value >= borrowed[msg.sender]);
        (bool ok,) = msg.sender.call{value: collateral[msg.sender]}("");
        require(ok);
        collateral[msg.sender] = 0;
        borrowed[msg.sender] = 0;
    }"""),
    _safe("LendSafe08", """
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public borrowed;
    function depositCollateral() external payable { collateral[msg.sender] += msg.value; }
    function repayAndWithdraw() external payable {
        require(msg.value >= borrowed[msg.sender]);
        uint256 col = collateral[msg.sender];
        collateral[msg.sender] = 0;
        borrowed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: col}("");
        require(ok);
    }"""),
))

# 9 — Subscription cancel refund
PAIRS.append((
    _vuln("SubVuln09", """
    mapping(address => uint256) public expiry;
    mapping(address => uint256) public paid;
    function subscribe(uint256 dur) external payable {
        expiry[msg.sender] = block.timestamp + dur;
        paid[msg.sender] = msg.value;
    }
    function cancelRefund() external {
        require(block.timestamp < expiry[msg.sender]);
        uint256 refund = paid[msg.sender];
        require(refund > 0);
        (bool ok,) = msg.sender.call{value: refund}("");
        require(ok);
        paid[msg.sender] = 0;
        expiry[msg.sender] = 0;
    }"""),
    _safe("SubSafe09", """
    mapping(address => uint256) public expiry;
    mapping(address => uint256) public paid;
    function subscribe(uint256 dur) external payable {
        expiry[msg.sender] = block.timestamp + dur;
        paid[msg.sender] = msg.value;
    }
    function cancelRefund() external {
        require(block.timestamp < expiry[msg.sender]);
        uint256 refund = paid[msg.sender];
        require(refund > 0);
        paid[msg.sender] = 0;
        expiry[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: refund}("");
        require(ok);
    }"""),
))

# 10 — DAO proposal payout
PAIRS.append((
    _vuln("DaoVuln10", """
    struct Proposal { address payable recipient; uint256 amount; bool executed; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public voted;
    mapping(uint256 => uint256) public voteCount;
    function propose(address payable r, uint256 a) external payable returns (uint256) {
        proposals.push(Proposal(r, a, false)); return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!voted[id][msg.sender]); voted[id][msg.sender] = true; voteCount[id]++; }
    function execute(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed && voteCount[id] >= 2);
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
        p.executed = true;
    }"""),
    _safe("DaoSafe10", """
    struct Proposal { address payable recipient; uint256 amount; bool executed; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public voted;
    mapping(uint256 => uint256) public voteCount;
    function propose(address payable r, uint256 a) external payable returns (uint256) {
        proposals.push(Proposal(r, a, false)); return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!voted[id][msg.sender]); voted[id][msg.sender] = true; voteCount[id]++; }
    function execute(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed && voteCount[id] >= 2);
        p.executed = true;
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
    }"""),
))

# 11 — Flash loan callback
PAIRS.append((
    _vuln("FlashVuln11", """
    interface IFlashReceiver { function onFlash(uint256 amt) external; }
    mapping(address => uint256) public reserves;
    function flashLoan(uint256 amt) external {
        uint256 before = address(this).balance;
        IFlashReceiver(msg.sender).onFlash(amt);   // callback before balance check
        reserves[msg.sender] -= amt;
        require(address(this).balance >= before);
    }"""),
    _safe("FlashSafe11", """
    interface IFlashReceiver { function onFlash(uint256 amt) external; }
    mapping(address => uint256) public reserves;
    function flashLoan(uint256 amt) external {
        uint256 before = address(this).balance;
        reserves[msg.sender] -= amt;               // state before callback
        IFlashReceiver(msg.sender).onFlash(amt);
        require(address(this).balance >= before);
    }"""),
))

# 12 — NFT mint refund
PAIRS.append((
    _vuln("NftVuln12", """
    mapping(address => uint256) public minted;
    uint256 public price = 0.01 ether;
    function mint() external payable {
        require(msg.value >= price);
        uint256 excess = msg.value - price;
        if (excess > 0) {
            (bool ok,) = msg.sender.call{value: excess}("");  // refund before state
            require(ok);
        }
        minted[msg.sender]++;
    }"""),
    _safe("NftSafe12", """
    mapping(address => uint256) public minted;
    uint256 public price = 0.01 ether;
    function mint() external payable {
        require(msg.value >= price);
        uint256 excess = msg.value - price;
        minted[msg.sender]++;                               // state before refund
        if (excess > 0) {
            (bool ok,) = msg.sender.call{value: excess}("");
            require(ok);
        }
    }"""),
))

# 13 — Savings account
PAIRS.append((
    _vuln("SavingsVuln13", """
    mapping(address => uint256) public savings;
    mapping(address => uint256) public lastWithdraw;
    uint256 constant COOLDOWN = 1 days;
    function save() external payable { savings[msg.sender] += msg.value; }
    function withdraw(uint256 amt) external {
        require(savings[msg.sender] >= amt);
        require(block.timestamp >= lastWithdraw[msg.sender] + COOLDOWN);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        savings[msg.sender] -= amt;
        lastWithdraw[msg.sender] = block.timestamp;
    }"""),
    _safe("SavingsSafe13", """
    mapping(address => uint256) public savings;
    mapping(address => uint256) public lastWithdraw;
    uint256 constant COOLDOWN = 1 days;
    function save() external payable { savings[msg.sender] += msg.value; }
    function withdraw(uint256 amt) external {
        require(savings[msg.sender] >= amt);
        require(block.timestamp >= lastWithdraw[msg.sender] + COOLDOWN);
        savings[msg.sender] -= amt;
        lastWithdraw[msg.sender] = block.timestamp;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 14 — Insurance claim
PAIRS.append((
    _vuln("InsureVuln14", """
    mapping(address => uint256) public coverage;
    mapping(address => bool) public claimed;
    function insure() external payable { coverage[msg.sender] = msg.value * 2; }
    function claim() external {
        require(!claimed[msg.sender]);
        uint256 payout = coverage[msg.sender];
        require(payout > 0);
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
        claimed[msg.sender] = true;
    }"""),
    _safe("InsureSafe14", """
    mapping(address => uint256) public coverage;
    mapping(address => bool) public claimed;
    function insure() external payable { coverage[msg.sender] = msg.value * 2; }
    function claim() external {
        require(!claimed[msg.sender]);
        uint256 payout = coverage[msg.sender];
        require(payout > 0);
        claimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
    }"""),
))

# 15 — Referral reward
PAIRS.append((
    _vuln("ReferVuln15", """
    mapping(address => uint256) public referralRewards;
    function addReward(address ref, uint256 amt) external payable {
        referralRewards[ref] += amt;
    }
    function collectReward() external {
        uint256 reward = referralRewards[msg.sender];
        require(reward > 0);
        (bool ok,) = msg.sender.call{value: reward}("");
        require(ok);
        referralRewards[msg.sender] = 0;
    }"""),
    _safe("ReferSafe15", """
    mapping(address => uint256) public referralRewards;
    function addReward(address ref, uint256 amt) external payable {
        referralRewards[ref] += amt;
    }
    function collectReward() external {
        uint256 reward = referralRewards[msg.sender];
        require(reward > 0);
        referralRewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: reward}("");
        require(ok);
    }"""),
))

# 16 — Yield farm harvest
PAIRS.append((
    _vuln("YieldVuln16", """
    interface IERC20 { function transfer(address, uint256) external returns (bool); }
    IERC20 public rewardToken;
    mapping(address => uint256) public depositTime;
    mapping(address => uint256) public deposited;
    constructor(address t) { rewardToken = IERC20(t); }
    function deposit() external payable { depositTime[msg.sender] = block.timestamp; deposited[msg.sender] = msg.value; }
    function harvest() external {
        uint256 elapsed = block.timestamp - depositTime[msg.sender];
        uint256 reward = deposited[msg.sender] * elapsed / 365 days;
        rewardToken.transfer(msg.sender, reward);   // external before state
        depositTime[msg.sender] = block.timestamp;
    }"""),
    _safe("YieldSafe16", """
    interface IERC20 { function transfer(address, uint256) external returns (bool); }
    IERC20 public rewardToken;
    mapping(address => uint256) public depositTime;
    mapping(address => uint256) public deposited;
    constructor(address t) { rewardToken = IERC20(t); }
    function deposit() external payable { depositTime[msg.sender] = block.timestamp; deposited[msg.sender] = msg.value; }
    function harvest() external {
        uint256 elapsed = block.timestamp - depositTime[msg.sender];
        uint256 reward = deposited[msg.sender] * elapsed / 365 days;
        depositTime[msg.sender] = block.timestamp;   // state before external
        rewardToken.transfer(msg.sender, reward);
    }"""),
))

# 17 — Multi-recipient airdrop
PAIRS.append((
    _vuln("AirdropVuln17", """
    mapping(address => bool) public hasClaimed;
    mapping(address => uint256) public allocation;
    function setAllocation(address a, uint256 amt) external payable { allocation[a] = amt; }
    function claimAirdrop() external {
        require(!hasClaimed[msg.sender]);
        uint256 amt = allocation[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        hasClaimed[msg.sender] = true;
    }"""),
    _safe("AirdropSafe17", """
    mapping(address => bool) public hasClaimed;
    mapping(address => uint256) public allocation;
    function setAllocation(address a, uint256 amt) external payable { allocation[a] = amt; }
    function claimAirdrop() external {
        require(!hasClaimed[msg.sender]);
        uint256 amt = allocation[msg.sender];
        require(amt > 0);
        hasClaimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 18 — Bond redemption
PAIRS.append((
    _vuln("BondVuln18", """
    struct Bond { uint256 principal; uint256 maturity; bool redeemed; }
    mapping(address => Bond) public bonds;
    function issueBond(uint256 mat) external payable {
        bonds[msg.sender] = Bond(msg.value, block.timestamp + mat, false);
    }
    function redeem() external {
        Bond storage b = bonds[msg.sender];
        require(!b.redeemed && block.timestamp >= b.maturity);
        uint256 payout = b.principal + b.principal / 10;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
        b.redeemed = true;
    }"""),
    _safe("BondSafe18", """
    struct Bond { uint256 principal; uint256 maturity; bool redeemed; }
    mapping(address => Bond) public bonds;
    function issueBond(uint256 mat) external payable {
        bonds[msg.sender] = Bond(msg.value, block.timestamp + mat, false);
    }
    function redeem() external {
        Bond storage b = bonds[msg.sender];
        require(!b.redeemed && block.timestamp >= b.maturity);
        uint256 payout = b.principal + b.principal / 10;
        b.redeemed = true;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
    }"""),
))

# 19 — Split payment
PAIRS.append((
    _vuln("SplitVuln19", """
    address payable public partyA;
    address payable public partyB;
    mapping(address => uint256) public owed;
    constructor(address payable a, address payable b) { partyA = a; partyB = b; }
    function fund() external payable { owed[partyA] += msg.value / 2; owed[partyB] += msg.value / 2; }
    function withdraw() external {
        require(msg.sender == partyA || msg.sender == partyB);
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        (bool ok,) = payable(msg.sender).call{value: amt}("");
        require(ok);
        owed[msg.sender] = 0;
    }"""),
    _safe("SplitSafe19", """
    address payable public partyA;
    address payable public partyB;
    mapping(address => uint256) public owed;
    constructor(address payable a, address payable b) { partyA = a; partyB = b; }
    function fund() external payable { owed[partyA] += msg.value / 2; owed[partyB] += msg.value / 2; }
    function withdraw() external {
        require(msg.sender == partyA || msg.sender == partyB);
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        owed[msg.sender] = 0;
        (bool ok,) = payable(msg.sender).call{value: amt}("");
        require(ok);
    }"""),
))

# 20 — Option exercise
PAIRS.append((
    _vuln("OptionVuln20", """
    mapping(address => uint256) public optionSize;
    mapping(address => uint256) public strikePrice;
    mapping(address => bool) public exercised;
    function writeOption(address buyer, uint256 size, uint256 strike) external payable {
        optionSize[buyer] = size; strikePrice[buyer] = strike;
    }
    function exercise() external payable {
        require(!exercised[msg.sender]);
        require(msg.value >= strikePrice[msg.sender]);
        uint256 payout = optionSize[msg.sender];
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
        exercised[msg.sender] = true;
    }"""),
    _safe("OptionSafe20", """
    mapping(address => uint256) public optionSize;
    mapping(address => uint256) public strikePrice;
    mapping(address => bool) public exercised;
    function writeOption(address buyer, uint256 size, uint256 strike) external payable {
        optionSize[buyer] = size; strikePrice[buyer] = strike;
    }
    function exercise() external payable {
        require(!exercised[msg.sender]);
        require(msg.value >= strikePrice[msg.sender]);
        uint256 payout = optionSize[msg.sender];
        exercised[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
    }"""),
))

# 21 — DAO grant
PAIRS.append((
    _vuln("GrantVuln21", """
    mapping(address => uint256) public grantAmount;
    mapping(address => bool) public grantClaimed;
    address public owner;
    constructor() { owner = msg.sender; }
    function allocate(address grantee, uint256 amt) external payable { grantAmount[grantee] = amt; }
    function claimGrant() external {
        require(!grantClaimed[msg.sender]);
        uint256 amt = grantAmount[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        grantClaimed[msg.sender] = true;
    }"""),
    _safe("GrantSafe21", """
    mapping(address => uint256) public grantAmount;
    mapping(address => bool) public grantClaimed;
    address public owner;
    constructor() { owner = msg.sender; }
    function allocate(address grantee, uint256 amt) external payable { grantAmount[grantee] = amt; }
    function claimGrant() external {
        require(!grantClaimed[msg.sender]);
        uint256 amt = grantAmount[msg.sender];
        require(amt > 0);
        grantClaimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 22 — Lottery payout
PAIRS.append((
    _vuln("LotteryVuln22", """
    address[] public tickets;
    uint256 public ticketPrice = 0.01 ether;
    bool public drawn;
    address public winner;
    function buyTicket() external payable { require(msg.value == ticketPrice); tickets.push(msg.sender); }
    function draw() external {
        require(!drawn);
        drawn = true;
        winner = tickets[block.timestamp % tickets.length];
    }
    function claimPrize() external {
        require(drawn && msg.sender == winner);
        uint256 prize = address(this).balance;
        (bool ok,) = msg.sender.call{value: prize}("");
        require(ok);
        winner = address(0);     // clear after call
    }"""),
    _safe("LotterySafe22", """
    address[] public tickets;
    uint256 public ticketPrice = 0.01 ether;
    bool public drawn;
    address public winner;
    function buyTicket() external payable { require(msg.value == ticketPrice); tickets.push(msg.sender); }
    function draw() external {
        require(!drawn);
        drawn = true;
        winner = tickets[block.timestamp % tickets.length];
    }
    function claimPrize() external {
        require(drawn && msg.sender == winner);
        uint256 prize = address(this).balance;
        winner = address(0);     // clear before call
        (bool ok,) = msg.sender.call{value: prize}("");
        require(ok);
    }"""),
))

# 23 — Bridge unlock
PAIRS.append((
    _vuln("BridgeVuln23", """
    mapping(bytes32 => bool) public processed;
    mapping(bytes32 => address) public recipient;
    mapping(bytes32 => uint256) public amount;
    function lock(bytes32 id, address to) external payable { recipient[id] = to; amount[id] = msg.value; }
    function unlock(bytes32 id) external {
        require(!processed[id]);
        (bool ok,) = recipient[id].call{value: amount[id]}("");
        require(ok);
        processed[id] = true;
    }"""),
    _safe("BridgeSafe23", """
    mapping(bytes32 => bool) public processed;
    mapping(bytes32 => address) public recipient;
    mapping(bytes32 => uint256) public amount;
    function lock(bytes32 id, address to) external payable { recipient[id] = to; amount[id] = msg.value; }
    function unlock(bytes32 id) external {
        require(!processed[id]);
        processed[id] = true;
        (bool ok,) = recipient[id].call{value: amount[id]}("");
        require(ok);
    }"""),
))

# 24 — Governance treasury
PAIRS.append((
    _vuln("GovTreasuryVuln24", """
    mapping(address => uint256) public allocations;
    mapping(address => bool) public disbursed;
    address public governor;
    constructor() { governor = msg.sender; }
    function allocate(address payee, uint256 amt) external payable {
        require(msg.sender == governor); allocations[payee] = amt;
    }
    function disburse() external {
        require(!disbursed[msg.sender]);
        uint256 amt = allocations[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        disbursed[msg.sender] = true;
    }"""),
    _safe("GovTreasurySafe24", """
    mapping(address => uint256) public allocations;
    mapping(address => bool) public disbursed;
    address public governor;
    constructor() { governor = msg.sender; }
    function allocate(address payee, uint256 amt) external payable {
        require(msg.sender == governor); allocations[payee] = amt;
    }
    function disburse() external {
        require(!disbursed[msg.sender]);
        uint256 amt = allocations[msg.sender];
        require(amt > 0);
        disbursed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }"""),
))

# 25 — Margin call
PAIRS.append((
    _vuln("MarginVuln25", """
    mapping(address => uint256) public margin;
    mapping(address => uint256) public position;
    function openPosition() external payable { margin[msg.sender] = msg.value; position[msg.sender] = msg.value * 5; }
    function closePosition() external {
        uint256 pnl = margin[msg.sender];
        require(pnl > 0);
        (bool ok,) = msg.sender.call{value: pnl}("");
        require(ok);
        margin[msg.sender] = 0;
        position[msg.sender] = 0;
    }"""),
    _safe("MarginSafe25", """
    mapping(address => uint256) public margin;
    mapping(address => uint256) public position;
    function openPosition() external payable { margin[msg.sender] = msg.value; position[msg.sender] = msg.value * 5; }
    function closePosition() external {
        uint256 pnl = margin[msg.sender];
        require(pnl > 0);
        margin[msg.sender] = 0;
        position[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: pnl}("");
        require(ok);
    }"""),
))

# ── Write files ───────────────────────────────────────────────────────────────

assert len(PAIRS) == 25, f"Expected 25 pairs, got {len(PAIRS)}"

written = []
for i, (vuln_src, safe_src) in enumerate(PAIRS, start=1):
    vpath = OUT_DIR / f"cei_vuln_{i:02d}.sol"
    spath = OUT_DIR / f"cei_safe_{i:02d}.sol"
    vpath.write_text(vuln_src)
    spath.write_text(safe_src)
    written.append(vpath)
    written.append(spath)

print(f"Written {len(written)} contracts to {OUT_DIR}")
for p in sorted(written):
    print(f"  {p.name}")
