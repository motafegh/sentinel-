// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln28 {

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
    }
}
