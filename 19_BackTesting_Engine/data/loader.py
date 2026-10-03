import pandas as pd
from data.market_data import Bar

class CSVLoader:
    def __init__(self, filepath, column_map=None):
        self.filepath = filepath
        self.column_map = column_map or {}

    def load(self):
        df = pd.read_csv(self.filepath)
        df = df.rename(columns=self.column_map)
        df.columns = df.columns.str.lower().str.strip()

        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing  = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        ["datetime"] = pd.to_datetime(df["datetime"], errors="raise")

        for col in required[1:]: #### list/array/tuple skip first one
            df[col] = pd.to_numeric(df[col], errors = "raise")

        if df[required].isna().any().any():
            raise ValueError("CSV contains missing values")
        if df["datetime"].duplicated().any():
            raise ValueError("Duplicate timestamps found")

        if((df[["open", "high", "low", "close"]] <= 0).any().any()):
            raise ValueError("Invalid OHLC prices")

        if(df["volume"] < 0).any():
            raise ValueError("Negative volume found")

        if((df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or
           (df["low"] > df[["open", "close", "high"]].min(axis=1)).any()):
            raise ValueError("Invalid OHLC realtionships")


        df = df.sort_values("datetime")
        bars = []
        for row in df.itertuples(index=False):
            bars.append(Bar(timestamp= row.datetime,
                            open= float(row.open),
                            high=float(row.high),
                            low=float(row.low),
                            close=float(row.close),
                            volume=float(row.volume)))
        return bars
