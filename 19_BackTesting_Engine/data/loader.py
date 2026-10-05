import pandas as pd
from data.market_data import Bar

class CSVLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        df.columns = df.columns.str.lower().str.strip()

        required = ["date", "open", "high", "low", "close", "volume"]
        for column in required:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        # Convert timestamp
        df["date"] = pd.to_datetime(df["date"])

        # Sort chronologically
        df = df.sort_values("date")

        # Remove duplicate candles
        df = df.drop_duplicates(subset=["date"], keep="first")

        # Basic validation
        if df[required].isnull().any().any():
            raise ValueError("Missing OHLCV values found")
        
        bars = []
        for _, row in df.iterrows():
            bar = Bar(
                datetime=row["date"],
                open    = float(row["open"]),
                high    = float(row["high"]),
                low     = float(row["low"]),
                close   = float(row["close"]),
                volume  = float(row["volume"]),

                quote_asset_volume  = float(row.get("quote_asset_volume", 0)),
                num_trades          = int(row.get("num_trades", 0)),
                taker_buy_base      = float(row.get("taker_buy_base", 0)),
                taker_buy_quote     = float(row.get("taker_buy_quote", 0)),
                bid_volume          = float(row.get("bid_volume", 0)),
                ask_volume          = float(row.get("ask_volume", 0)),
                total_volume        = float(row.get("total_volume", row["volume"]))
            )

            bars.append(bar)

        print("CSV loaded successfully")
        print(f"Total candles : {len(bars)}")
        print(f"Start         : {bars[0].datetime}")
        print(f"End           : {bars[-1].datetime}")

        return bars
