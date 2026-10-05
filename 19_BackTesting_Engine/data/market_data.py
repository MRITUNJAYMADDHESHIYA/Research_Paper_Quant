from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True) ### immutability
class Bar:
    datetime: datetime
    open: float
    high: float
    low:  float
    close: float
    volume: float

    quote_asset_volume: float = 0.0
    num_trades: int = 0

    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0

    bid_volume: float = 0.0
    ask_volume: float = 0.0
    total_volume: float = 0.0

