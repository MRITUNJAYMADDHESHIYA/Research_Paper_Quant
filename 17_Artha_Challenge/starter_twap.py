"""
Artha TWAP baseline.

Run:
    artha run starter_twap.py
"""

from artha import Bot


class MyBot(Bot):

    def on_bar(self, state):

        orders = {}

        for sym in state.instruments:

            mandate = state.mandate.get(sym, 0.0)
            remaining = state.remaining.get(sym, 0.0)

            # Zero mandate means we are not allowed to trade it.
            if mandate == 0:
                continue

            # Already completely executed.
            if abs(remaining) <= 1e-12:
                continue

            # Equal quantity across remaining bars.
            qty = abs(remaining) / max(state.bars_left, 1)

            if remaining > 0:
                orders[sym] = qty
            else:
                orders[sym] = -qty

        return orders