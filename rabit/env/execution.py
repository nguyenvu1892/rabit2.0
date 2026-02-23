class ExecutionEngine:
    def __init__(self, slippage_points: float = 0.0):
        self.slippage_points = slippage_points

    def market_fill(self, direction: int, close_price: float, spread: int):
        """
        XAU:
        1 point = 0.01
        spread in raw MT5 points
        """
        spread_price = spread * 0.01

        if direction == 1:
            # Buy at Ask
            return close_price + spread_price / 2 + self.slippage_points
        else:
            # Sell at Bid
            return close_price - spread_price / 2 - self.slippage_points