import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("Order_Book\orderbook.csv", parse_dates=["time"])

# Shift for previous event values
df["bid_price_prev"] = df["bid_price"].shift(1)
df["bid_size_prev"] = df["bid_size"].shift(1)
df["ask_price_prev"] = df["ask_price"].shift(1)
df["ask_size_prev"] = df["ask_size"].shift(1)

# Compute en
df["en"] = (
    df["bid_size"] * (df["bid_price"] >= df["bid_price_prev"])
    - df["bid_size_prev"] * (df["bid_price"] <= df["bid_price_prev"])
    - df["ask_size"] * (df["ask_price"] <= df["ask_price_prev"])
    + df["ask_size_prev"] * (df["ask_price"] >= df["ask_price_prev"])
)

# Now compute Dt per 1-minute interval
def compute_Dt(sub):
    if len(sub) < 2:
        return np.nan
    
    # Bid contribution
    bid_num = ((sub["bid_size"] * (sub["bid_price"] < sub["bid_price_prev"]))
              + (sub["bid_size_prev"] * (sub["bid_price"] > sub["bid_price_prev"]))).sum()
    bid_den = (sub["bid_price"] != sub["bid_price_prev"]).sum()
    bid_term = bid_num / bid_den if bid_den > 0 else 0
    
    # Ask contribution
    ask_num = ((sub["ask_size"] * (sub["ask_price"] > sub["ask_price_prev"]))
              + (sub["ask_size_prev"] * (sub["ask_price"] < sub["ask_price_prev"]))).sum()
    ask_den = (sub["ask_price"] != sub["ask_price_prev"]).sum()
    ask_term = ask_num / ask_den if ask_den > 0 else 0
    
    return 0.5 * (bid_term + ask_term)

# Resample to 1-minute intervals
df.set_index("time", inplace=True)
Dt_series = df.groupby(pd.Grouper(freq="1min")).apply(compute_Dt)

print(df.head(20))
print(Dt_series.head(20))
