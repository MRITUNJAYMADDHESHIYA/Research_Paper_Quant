from collections import deque
import pandas as pd


class TradeAnalyzer:
    def __init__(self, broker):
        self.broker = broker

    def get_trades(self):
        fills = self.broker.fill_history

        if not fills:
            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "pnl",
                    "return"
                ]
            )

        open_lots = deque()
        trades    = []

        for fill in fills:
            side         = (fill.side.value if hasattr(fill.side, "value") else str(fill.side)).upper()
            signed_qty   = (fill.quantity if side == "BUY" else -fill.quantity)
            remaining    = abs(signed_qty)
            direction    = 1 if signed_qty > 0 else -1
            fee_per_unit = (fill.commission / fill.quantity)

            ###### Match against opposite open lots
            while (remaining > 1e-12 and open_lots and open_lots[0]["direction"] != direction):
                lot = open_lots[0]
                matched = min(remaining, lot["quantity"])

                if lot["direction"] == 1:
                    gross_pnl = (fill.price - lot["price"]) * matched
                else:
                    gross_pnl = (lot["price"] - fill.price) * matched

                entry_fee      = (lot["fee_per_unit"] * matched)
                exit_fee       = (fee_per_unit * matched)
                net_pnl        = (gross_pnl - entry_fee - exit_fee)
                entry_notional = (lot["price"] * matched)
                trade_return   = (net_pnl / entry_notional if entry_notional > 0 else 0.0)

                trades.append({
                    "entry_time": lot["timestamp"],
                    "exit_time": fill.timestamp,
                    "entry_price": lot["price"],
                    "exit_price": fill.price,
                    "quantity": matched,
                    "direction": ("LONG" if lot["direction"] == 1 else "SHORT"),
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "pnl": net_pnl,
                    "return": trade_return
                })

                lot["quantity"] -= matched
                remaining -= matched

                if lot["quantity"] <= 1e-12:
                    open_lots.popleft()

            # Any unmatched quantity opens a new lot
            if remaining > 1e-12:
                open_lots.append({
                    "timestamp": fill.timestamp,
                    "price": fill.price,
                    "quantity": remaining,
                    "direction": direction,
                    "fee_per_unit": fee_per_unit
                })

        return pd.DataFrame(trades,
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "direction",
                "gross_pnl",
                "entry_fee",
                "exit_fee",
                "pnl",
                "return"
            ]
        )


