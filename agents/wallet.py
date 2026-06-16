"""
Wallet agent — the capital gatekeeper.

Tracks free cash, capital deployed in open positions, and realised P&L. Every
new trade must pass `can_afford` (cost basis + entry charges), and every exit
credits proceeds minus charges back. No trade is opened that the wallet can't
fund — position sizing is bounded by real money, not wishful thinking.
"""


class WalletAgent:
    def __init__(self, starting_cash=100_000):
        self.start = float(starting_cash)
        self.cash = float(starting_cash)      # free, un-deployed
        self.deployed = 0.0                   # cost basis tied up in open positions
        self.realized_pnl = 0.0
        self.fees_paid = 0.0

    def can_afford(self, amount):
        return amount <= self.cash + 1e-6

    def open_cost(self, cost_basis, fees):
        """Deduct cost basis + entry fees when a position opens."""
        total = cost_basis + fees
        if not self.can_afford(total):
            return False
        self.cash -= total
        self.deployed += cost_basis
        self.fees_paid += fees
        return True

    def close_proceeds(self, cost_basis_released, proceeds, fees, realized):
        """Return proceeds − fees to cash, release deployed capital, book P&L."""
        self.cash += proceeds - fees
        self.deployed -= cost_basis_released
        self.fees_paid += fees
        self.realized_pnl += realized

    def equity(self, open_mtm=0.0):
        """Total equity = free cash + current marked value of open positions."""
        return self.cash + open_mtm

    def snapshot(self, open_mtm=0.0):
        eq = self.equity(open_mtm)
        return {
            "starting_cash": round(self.start),
            "free_cash": round(self.cash, 2),
            "deployed": round(self.deployed, 2),
            "open_value": round(open_mtm, 2),
            "equity": round(eq, 2),
            "total_return_pct": round((eq / self.start - 1) * 100, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "fees_paid": round(self.fees_paid, 2),
        }
