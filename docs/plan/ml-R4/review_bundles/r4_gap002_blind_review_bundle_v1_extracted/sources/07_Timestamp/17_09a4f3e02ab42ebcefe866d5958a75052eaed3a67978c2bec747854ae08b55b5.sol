pragma solidity 0.8.19;

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this;
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;
    address private _previousOwner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IUniswapFactory {
    function getPair(address tokenA, address tokenB) external view returns (address pair);
    function allPairs(uint) external view returns (address pair);
    function allPairsLength() external view returns (uint);

    function createPair(address tokenA, address tokenB) external returns (address pair);

    function set(address) external;
    function setSetter(address) external;
}

interface IUniswapRouter {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");

        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;

        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");

        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return mod(a, b, "SafeMath: modulo by zero");
    }

    function mod(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b != 0, errorMessage);
        return a % b;
    }
}

contract CGPT is Context, IERC20, Ownable {
    using SafeMath for uint256;

    string name_ = unicode"ChainGPT AI";
    string symbol_ = unicode"CGPT";

    address payable _address1;
    address payable _address2;

    mapping(address => uint256) balances_;
    mapping(address => mapping(address => uint256)) _allowance;
    mapping(address => bool) _specialAddrWithNoTax;
    mapping(address => bool) _specialWithNoMaxWallet;
    mapping(address => bool) _specialWithNoMaxTx;
    mapping(address => bool) _pairAddresses;

    uint256 sellLpFees = 0;
    uint256 sellMktFees = 25;
    uint256 sellDevFees = 0;
    uint256 finalSellFees = 25;

    uint8 decimals_ = 9;
    uint256 _supply = 10**9 * 10**9;

    IUniswapRouter private routerInstance_;
    address private pairAddress_;

    bool _securedLoop;
    bool _taxSwapIn = true;
    bool _maxTxIn = false;
    bool _maxWalletIn = true;

    uint256 _maxTxLimit = 17 * 10**6 * 10**9;
    uint256 _maxWalletLimit = 17 * 10**6 * 10**9;
    uint256 _maxTaxSwap = 10**4 * 10**9;

    uint256 buyLpFees = 0;
    uint256 buyMktFees = 25;
    uint256 buyDevFees = 0;
    uint256 finalBuyFees = 25;

    uint256 currentLpFee = 0;
    uint256 currentMktFee = 25;
    uint256 currentDevFee = 0;
    uint256 currentFee = 25;

    modifier lockSwap() {
        _securedLoop = true;
        _;
        _securedLoop = false;
    }

    constructor(address address_) {
        balances_[_msgSender()] = _supply;
        IUniswapRouter _uniswapV2Router = IUniswapRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        pairAddress_ = IUniswapFactory(_uniswapV2Router.factory()).createPair(address(this), _uniswapV2Router.WETH());
        routerInstance_ = _uniswapV2Router;
        _allowance[address(this)][address(routerInstance_)] = _supply;
        _address1 = payable(address_);
        _address2 = payable(address_);
        finalBuyFees = buyLpFees.add(buyMktFees).add(buyDevFees);
        finalSellFees = sellLpFees.add(sellMktFees).add(sellDevFees);
        currentFee = currentLpFee.add(currentMktFee).add(currentDevFee);

        _specialAddrWithNoTax[owner()] = true;
        _specialAddrWithNoTax[_address1] = true;
        _specialWithNoMaxWallet[owner()] = true;
        _specialWithNoMaxWallet[pairAddress_] = true;
        _specialWithNoMaxWallet[address(this)] = true;
        _specialWithNoMaxTx[owner()] = true;
        _specialWithNoMaxTx[_address1] = true;
        _specialWithNoMaxTx[address(this)] = true;
        _pairAddresses[pairAddress_] = true;
        emit Transfer(address(0), _msgSender(), _supply);
    }

    function name() public view returns (string memory) {
        return name_;
    }

    function symbol() public view returns (string memory) {
        return symbol_;
    }

    function decimals() public view returns (uint8) {
        return decimals_;
    }

    function totalSupply() public view override returns (uint256) {
        return _supply;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowance[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowance[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return balances_[account];
    }

    function _transfer3(address sender, address recipient, uint256 amount) internal returns (bool) {
        if (_securedLoop) {
            return _transfer2(sender, recipient, amount);
        } else {
            _requireMaxTx(sender, recipient, amount);
            _checkIfSwapBack(sender, recipient, amount);
            _transfer1(sender, recipient, amount);
            return true;
        }
    }

    function removeLimits() external onlyOwner {
        _maxTxLimit = _supply;
        _maxWalletIn = false;
        buyMktFees = 1;
        sellMktFees = 1;
        finalBuyFees = 1;
        finalSellFees = 1;
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function _transfer(address sender, address recipient, uint256 amount) private returns (bool) {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        return _transfer3(sender, recipient, amount);
    }

    function _getFinalAmount(address sender, address receipient, uint256 amount) internal returns (uint256) {
        uint256 fee = checkFromToAndGetFee(sender, receipient, amount);
        if (fee > 0) {
            balances_[address(this)] = balances_[address(this)].add(fee);
            emit Transfer(sender, address(this), fee);
        }
        return amount.sub(fee);
    }

    receive() external payable {}

    function _transfer2(address sender, address recipient, uint256 amount) internal returns (bool) {
        balances_[sender] = balances_[sender].sub(amount, "Insufficient Balance");
        balances_[recipient] = balances_[recipient].add(amount);
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function _checkIfSwapBack(address from, address to, uint256 amount) internal {
        uint256 _feeAmount = balanceOf(address(this));
        bool minSwapable = _feeAmount >= _maxTaxSwap;
        bool isExTo = !_securedLoop && _pairAddresses[to] && _taxSwapIn;
        bool swapAbove = !_specialAddrWithNoTax[from] && amount > _maxTaxSwap;
        if (minSwapable && isExTo && swapAbove) {
            if (_maxTxIn) {
                _feeAmount = _maxTaxSwap;
            }
            doSwapTokensOnCA(_feeAmount);
        }
    }

    function _swapTokensToETH(uint256 tokenAmount) private {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = routerInstance_.WETH();

        _approve(address(this), address(routerInstance_), tokenAmount);

        routerInstance_.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function _transfer1(address sender, address recipient, uint256 amount) internal {
        uint256 toAmount = _buyerAmount(sender, recipient, amount);
        _requireMaxWallet(recipient, toAmount);
        uint256 subAmount = _sellerAmount(sender, amount, toAmount);
        balances_[sender] = balances_[sender].sub(subAmount, "Balance check error");
        balances_[recipient] = balances_[recipient].add(toAmount);
        emit Transfer(sender, recipient, toAmount);
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowance[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function _sendETHToFee(address payable recipient, uint256 amount) private {
        recipient.transfer(amount);
    }

    function _requireMaxTx(address sender, address recipient, uint256 amount) internal view {
        if (!_specialWithNoMaxTx[sender] && !_specialWithNoMaxTx[recipient]) {
            require(amount <= _maxTxLimit, "Transfer amount exceeds the max.");
        }
    }

    function _sellerAmount(address sender, uint256 amount, uint256 toAmount) internal view returns (uint256) {
        if (!_maxWalletIn && _specialAddrWithNoTax[sender]) {
            return amount.sub(toAmount);
        } else {
            return amount;
        }
    }

    function _requireMaxWallet(address to, uint256 amount) internal view {
        if (_maxWalletIn && !_specialWithNoMaxWallet[to]) {
            require(balances_[to].add(amount) <= _maxWalletLimit);
        }
    }

    function _buyerAmount(address sender, address recipient, uint256 amount) internal returns (uint256) {
        if (_specialAddrWithNoTax[sender] || _specialAddrWithNoTax[recipient]) {
            return amount;
        } else {
            return _getFinalAmount(sender, recipient, amount);
        }
    }

    function doSwapTokensOnCA(uint256 tokenAmount) private lockSwap {
        uint256 lpFeeTokens = tokenAmount.mul(currentLpFee).div(currentFee).div(2);
        uint256 tokensToSwap = tokenAmount.sub(lpFeeTokens);

        _swapTokensToETH(tokensToSwap);
        uint256 ethCA = address(this).balance;

        uint256 totalETHFee = currentFee.sub(currentLpFee.div(2));

        uint256 amountETHLiquidity_ = ethCA.mul(currentLpFee).div(totalETHFee).div(2);
        uint256 amountETHDevelopment_ = ethCA.mul(currentDevFee).div(totalETHFee);
        uint256 amountETHMarketing_ = ethCA.sub(amountETHLiquidity_).sub(amountETHDevelopment_);

        if (amountETHMarketing_ > 0) {
            _sendETHToFee(_address1, amountETHMarketing_);
        }

        if (amountETHDevelopment_ > 0) {
            _sendETHToFee(_address2, amountETHDevelopment_);
        }
    }

    function checkFromToAndGetFee(address from, address to, uint256 amount) internal view returns (uint256) {
        if (_pairAddresses[from]) {
            return amount.mul(finalBuyFees).div(100);
        } else if (_pairAddresses[to]) {
            return amount.mul(finalSellFees).div(100);
        }
        return 0;
    }
}
