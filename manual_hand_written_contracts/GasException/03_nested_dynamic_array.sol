// expect: GasException
// Contract that maintains a nested dynamic array structure for user inventories.
// Each user can have multiple inventory slots, each with multiple items.
// Computing user totals requires nested loops over all inventories.
// Deeply nested storage reads and memory allocations cause gas spikes.
// The push/pop pattern on storage arrays causes SSTORE costs to compound.
pragma solidity ^0.8.0;

contract InventoryManager {
    struct Item {
        uint256 id;
        uint256 amount;
        string name;
    }

    struct InventorySlot {
        uint256 slotId;
        Item[] items;
    }

    struct Player {
        address addr;
        string username;
        InventorySlot[] slots;
    }

    address public owner;
    Player[] public players;
    mapping(address => uint256) public playerIndex;
    mapping(uint256 => bool) public usedItemIds;

    event PlayerRegistered(address indexed player, string username);
    event ItemAdded(address indexed player, uint256 slotId, uint256 itemId, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerPlayer(string calldata username) external {
        require(playerIndex[msg.sender] == 0, "already registered");
        uint256 idx = players.length;
        players.push();
        Player storage p = players[idx];
        p.addr = msg.sender;
        p.username = username;
        playerIndex[msg.sender] = idx + 1;
        emit PlayerRegistered(msg.sender, username);
    }

    function addSlot(uint256 slotId) external {
        uint256 idx = playerIndex[msg.sender];
        require(idx > 0, "not registered");
        Player storage p = players[idx - 1];
        uint256 slotCount = p.slots.length;
        p.slots.push();
        InventorySlot storage slot = p.slots[slotCount];
        slot.slotId = slotId;
    }

    function addItem(uint256 slotIndex, uint256 itemId, uint256 amount, string calldata name) external {
        require(!usedItemIds[itemId], "item id already used");
        uint256 idx = playerIndex[msg.sender];
        require(idx > 0, "not registered");
        Player storage p = players[idx - 1];
        require(slotIndex < p.slots.length, "invalid slot");
        InventorySlot storage slot = p.slots[slotIndex];
        slot.items.push(Item(itemId, amount, name));
        usedItemIds[itemId] = true;
        emit ItemAdded(msg.sender, slotIndex, itemId, amount);
    }

    function computeTotalItems(address player) external view returns (uint256 totalItems, uint256 totalAmount) {
        uint256 idx = playerIndex[player];
        require(idx > 0, "not registered");
        Player storage p = players[idx - 1];
        for (uint256 i = 0; i < p.slots.length; i++) {
            totalItems += p.slots[i].items.length;
            for (uint256 j = 0; j < p.slots[i].items.length; j++) {
                totalAmount += p.slots[i].items[j].amount;
            }
        }
    }

    function computeGlobalStats() external view returns (uint256 totalPlayers, uint256 totalSlots, uint256 totalItems, uint256 totalAmount) {
        totalPlayers = players.length;
        for (uint256 i = 0; i < players.length; i++) {
            totalSlots += players[i].slots.length;
            for (uint256 j = 0; j < players[i].slots.length; j++) {
                totalItems += players[i].slots[j].items.length;
                for (uint256 k = 0; k < players[i].slots[j].items.length; k++) {
                    totalAmount += players[i].slots[j].items[k].amount;
                }
            }
        }
    }

    function batchAddItems(address[] calldata users, uint256[] calldata slotIndices, uint256[] calldata itemIds, uint256[] calldata amounts, string[] calldata names) external onlyOwner {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 idx = playerIndex[users[i]];
            require(idx > 0, "not registered");
            Player storage p = players[idx - 1];
            require(slotIndices[i] < p.slots.length, "invalid slot");
            p.slots[slotIndices[i]].items.push(Item(itemIds[i], amounts[i], names[i]));
            usedItemIds[itemIds[i]] = true;
        }
    }
}