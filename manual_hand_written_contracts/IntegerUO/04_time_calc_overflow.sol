// expect: IntegerUO
// Time-locked vesting contract that uses unchecked arithmetic for duration
// and amount calculations. Multiplications of large numbers (timestamps ×
// amounts) can overflow uint256. The lockPeriod * amount calculation
// when computing vested amounts wraps silently, potentially allowing
// early withdrawal of more tokens than deposited.
pragma solidity ^0.8.0;

contract TimeLockVesting {
    struct VestingSchedule {
        address beneficiary;
        uint256 totalAmount;
        uint256 startTime;
        uint256 duration;
        uint256 withdrawn;
        bool revocable;
    }

    address public owner;
    VestingSchedule[] public schedules;
    mapping(address => uint256[]) public userScheduleIds;
    uint256 public totalVested;

    event ScheduleCreated(uint256 indexed id, address indexed beneficiary, uint256 amount);
    event TokensReleased(uint256 indexed id, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function createSchedule(address beneficiary, uint256 totalAmount, uint256 duration, bool revocable) external onlyOwner returns (uint256) {
        uint256 id = schedules.length;
        schedules.push(VestingSchedule(beneficiary, totalAmount, block.timestamp, duration, 0, revocable));
        userScheduleIds[beneficiary].push(id);
        unchecked {
            totalVested += totalAmount;
        }
        emit ScheduleCreated(id, beneficiary, totalAmount);
        return id;
    }

    function release(uint256 scheduleId) external {
        VestingSchedule storage s = schedules[scheduleId];
        require(msg.sender == s.beneficiary || msg.sender == owner, "not authorized");
        unchecked {
            uint256 elapsed = block.timestamp - s.startTime;
            uint256 vested = 0;
            if (elapsed >= s.duration) {
                vested = s.totalAmount;
            } else {
                vested = (s.totalAmount * elapsed) / s.duration;
            }
            uint256 releasable = vested - s.withdrawn;
            require(releasable > 0, "nothing to release");
            s.withdrawn += releasable;
            (bool ok, ) = s.beneficiary.call{value: releasable}("");
            require(ok, "transfer failed");
            emit TokensReleased(scheduleId, releasable);
        }
    }

    function batchRelease(address user) external {
        uint256[] storage ids = userScheduleIds[user];
        for (uint256 i = 0; i < ids.length; i++) {
            VestingSchedule storage s = schedules[ids[i]];
            unchecked {
                uint256 elapsed = block.timestamp - s.startTime;
                if (elapsed > 0) {
                    uint256 vested = (s.totalAmount * elapsed) / s.duration;
                    uint256 releasable = vested - s.withdrawn;
                    if (releasable > 0) {
                        s.withdrawn += releasable;
                        (bool ok, ) = user.call{value: releasable}("");
                        require(ok, "batch release failed");
                        emit TokensReleased(ids[i], releasable);
                    }
                }
            }
        }
    }

    function revoke(uint256 scheduleId) external onlyOwner {
        VestingSchedule storage s = schedules[scheduleId];
        require(s.revocable, "not revocable");
        unchecked {
            uint256 elapsed = block.timestamp - s.startTime;
            uint256 vested = (s.totalAmount * elapsed) / s.duration;
            uint256 released = s.withdrawn;
            uint256 remaining = s.totalAmount - released;
            s.totalAmount = vested;
            (bool ok, ) = owner.call{value: remaining}("");
            require(ok, "revoke transfer failed");
        }
    }

    function createBatchSchedules(address[] calldata beneficiaries, uint256[] calldata amounts, uint256 duration, bool revocable) external onlyOwner {
        require(beneficiaries.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < beneficiaries.length; i++) {
            uint256 id = schedules.length;
            schedules.push(VestingSchedule(beneficiaries[i], amounts[i], block.timestamp, duration, 0, revocable));
            userScheduleIds[beneficiaries[i]].push(id);
            unchecked { totalVested += amounts[i]; }
            emit ScheduleCreated(id, beneficiaries[i], amounts[i]);
        }
    }
}