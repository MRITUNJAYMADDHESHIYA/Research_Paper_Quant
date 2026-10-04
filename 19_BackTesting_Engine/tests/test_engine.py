import pytest

from datetime import datetime
from broker.broker import Broker
from data.market_data import Bar

def make_bar(price):
    return Bar(
        timestamp=datetime(2026, 1, 1),
        open = price,
        high = price,
        low  = price,
        close= price,
        volume=1000
    )

def test_buy_and_sell():
    broker = Broker(initial_cash=10000, commission=0, slippage=0)
    broker.buy(quantity=10)
    broker.execute_orders(make_bar(100))

    assert broker.cash == pytest.approx(9000)  ### assert:- immediately stops the program if something wrong
    assert broker.position == 10

    broker.sell(quantity=10)
    broker.execute_orders(make_bar(110))

    assert broker.cash == pytest.approx(10100)
    assert broker.position == 0
    assert broker.realized_pnl == pytest.approx(100)

def test_reject_insufficient_cash():
    broker = Broker(initial_cash=100)
    broker.buy(quantity=10)
    broker.execute_orders(make_bar(100))

    assert broker.position == 0
    assert broker.cash == 100