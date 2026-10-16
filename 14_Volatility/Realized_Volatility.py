################### Return Calculation ###################
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv")

###################### Simple Price #######################
def return_calculation(df: pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    price = df['Close'].astype(float)
    previous_price = price.shift(1)

    ##### Absolute Return ######
    df["absolute_return"] = price - previous_price

    ####### Log return #######
    df["log_return"] = np.log(price / previous_price)

    ###### arithmetic return ##########
    df["arithmetic_return"] = (price / previous_price) - 1

    return df


# data = return_calculation(df=df)
# data["volatililty_20"] = (data["log_return"].rolling(window=20).std())

# print(data)

##################### Dividends ####################
## st-->current stock price, st-1--->previous stock price, dt--->dividend per share associated with period t,zero when no dividend occurs
def dividend_return(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    price = df["Close"].astype(float)
    previous_price = price.shift(1)

    df["absolute_return"]   = (price + df["dividend"] - previous_price)
    df["log_return"]        = np.log((price + df["dividend"]) / previous_price - 1)
    df["arithmetic_return"] = ((price + df["dividend"])/ previous_price - 1)

    return df


##################### Stock split ######################
def split_return(df:pd.DataFrame, price_col:str = "Close", dividend_col:str = "Dividends", split_col:str = "Split_Ratio")->pd.DataFrame:
    df = df.copy()
    price = df[price_col].astype(float)
    previous_price = price.shift(1)

    #### Dividend per post-split share
    if dividend_col in df.columns:
        dividend = df[dividend_col].fillna(0).astype(float)
    else:
        dividend = pd.Series(0.0, index=df.index)
    
    ###### split factor: x/y (2:1 = 2.0)
    if split_col in df.columns:
        split_ratio = df[split_col].fillna(1).astype(float)
    else:
        split_ratio = pd.Series(1.0, index=df.index)

    ######### current value
    adjusted_value = split_ratio * (price + dividend)

    ######### Absolute Return
    df["absolute_return"] = (adjusted_value - previous_price)

    ######### logrithimc return
    df["log_return"]     = np.log(adjusted_value / previous_price)

    ######### arithmetic return
    df["arithmetic_return"] = (adjusted_value / previous_price - 1)
    return df

################### Sample Partitioning #############################
###### Sample partitioning is allows us to divide historical market data into smaller periods for
###### volatility estimation, signal generation and strategy rebalancing without using future information
def sample_partition(df:pd.DataFrame, window_size:int=20)->list[pd.DataFrame]:
    partitions = []
    for end in range(window_size, len(df), window_size):
        start = end - window_size
        ##### include both end points
        sample = df.iloc[start:end + 1].copy()
        partitions.append(sample)

    return partitions

################### Realized Volatility ###########################
def realized_volatility(df:pd.DataFrame, window:int = 20, periods_per_year: int = 365*24*60)->pd.DataFrame:
    df = df.copy()
    ########### log return if not available
    if "log_return" not in df.columns:
        price = df["Close"].astype(float)
        df["log_return"] = np.log(price / price.shift(1))

    previous_log = df["log_return"].shift(1)
    df["mean_return"] = (previous_log.rolling(window=window, min_periods=window).mean())

    df["realized_variance"]     = (previous_log.rolling(window=window, min_periods=window).var(ddof=1))
    df["realized_volatility"]   = np.sqrt(df["realized_variance"])
    df["annualized_variance"]   = (df["realized_variance"]*periods_per_year)
    df["annualized_volatility"] = (df["realized_variance"] * np.sqrt(periods_per_year))

    return df


                                          
################ Discrete vs Continuous Volatility and Estimation Methods
### 1.Instantaneous variance:-
### 2.Quadratic variation — the accumulated variation of a continuous-time price process.
### 3.Realized variance:-  an estimate of quadratic variation using observed discrete returns
### 4.EWMA(exponentially weighted moving average):- more weight to recent observations
###          decay = 0.94 --->6% weight on new squared return 

def discrete_volatility(df:pd.DataFrame, window:int = 20, decay:float = 0.94)->pd.DataFrame:
    df = df.copy()

    price = df["Close"].astype(float)

    #### log return
    df["log_return"] = np.log(price / price.shift(1))
    #### squared log returns
    df["squared_return"] = df["log_return"] ** 2

    #### Realized variance over the last N returns
    df["realized_variance"] = (df["squared_return"].rolling(window=window).sum())
    #### Realized volatility over the window
    df["realized_volatility"] = np.sqrt(df["realized_variance"])
    #### Rolling sample variance of individual returns
    df["sample_variance"]  = (df["log_return"].rolling(window=window).var(ddof=1))
    #### variance rate per 1-m interval
    df["variance_rate"]  = (df["realized_variance"] / window)

    #### ewma conditional variance
    df["ewma_variance"] = (df["squared_return"].shift(1).ewm(alpha=1 - decay, adjust=False)).mean()
    #### ewma volatility
    df["ewma_volatility"] = np.sqrt(df["ewma_variance"])

    return df