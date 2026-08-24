import pandas as pd
import ta

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        df.columns = [col.strip() for col in df.columns]
        return df

    def prepare(self, df):
        df["Date"]    = pd.to_datetime(df["Date"], format="%d-%m-%Y",errors="coerce")
        df            = df.rename(columns={"Price": "Close", "Vol.": "Volume"})
        price_columns = ["Open", "High", "Low", "Close"]

        for col in price_columns:
            df[col] = (df[col].astype(str).str.replace(",","",regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")


        def convert_volume(value):
            if pd.isna(value):
                return None
            value = str(value).strip()
            if value.endswith("K"):
                return (float(value[:-1]) * 1_000)
            elif value.endswith("M"):
                return (float(value[:-1]) * 1_000_000)
            elif value.endswith("B"):
                return (float(value[:-1]) * 1_000_000_000)

            return float(value)

        df["Volume"] = (df["Volume"].apply(convert_volume))
        if "Change %" in df.columns:
            df["Change %"] = (df["Change %"].astype(str).str.replace("%","",regex=False))
            df["Change %"] = pd.to_numeric(df["Change %"],errors="coerce")

        df = df.sort_values("Date")
        df = df.drop_duplicates(subset=["Date"])

        ############# Feature Engineering #################
        df["return_1"]      = (df["Close"].pct_change(1))
        df["return_5"]      = (df["Close"].pct_change(5))
        df["return_10"]     = (df["Close"].pct_change(10))
        rsi                 = ta.momentum.RSIIndicator(close = df["Close"], window=14)
        macd                = ta.trend.MACD(close=df["Close"])
        df["macd"]          = macd.macd()
        df["rsi_14"]        = rsi.rsi()
        df["volatility_20"] = (df["return_1"].rolling(20).std())
        df["ma20"]          = (df["Close"].rolling(20).mean())
        df["ma50"]          = (df["Close"].rolling(50).mean())
        df["dist_ma20"]     = (df["Close"] / df["ma20"] - 1)
        df["dist_ma50"]     = (df["Close"] / df["ma50"] - 1)
        df["momentum20"]    = (df["Close"] .pct_change(20))
        df["volume_change"] = (df["Volume"].pct_change())
        df["Target"]        = (df["Close"].shift(-5) / df["Close"] - 1)
        
        print("\nMissing values before cleaning:")
        feature_columns = ["return_1","return_5", "return_10", "rsi_14", "macd","volatility_20", "dist_ma20", "dist_ma50", "momentum20", "volume_change", "Target"]
        print(df[feature_columns].isna().sum())

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "return_1","return_5", "return_10", "rsi_14", "macd", "volatility_20", "dist_ma20", "dist_ma50", "momentum20", "volume_change", "Target"])
        df = df.reset_index(drop=True)
        print("\nRows after cleaning:", len(df))
        print("\nCleaned data:")
        print(df.head())

        return df


    