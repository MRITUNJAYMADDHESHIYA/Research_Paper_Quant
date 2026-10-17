##### 1.close to close
##### 2.Parkinson
##### 3.Garman-Klass
##### 4.Rogers-Satchell
##### 5.Yang-Zhang

# Estimator	        Uses	                Drift handling	            Gaps
# Close-to-Close	Close	                Mean-adjusted	            Included
# Parkinson	        High, Low           	Assumes negligible drift	Not separately
# Garman–Klass	    OHLC	                Sensitive to drift	        Not separately
# Rogers–Satchell	OHLC	                Allows drift	            Not separately
# Yang–Zhang	    OHLC + previous close	Allows drift	            Explicitly included

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_volatility_estimators(df:pd.DataFrame, window:int =20, annualized:bool =False, periods_per_year:int = 365*24*60)->pd.DataFrame:
    df = df.copy()
    O = df["Open"].astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    C = df["Close"].astype(float)

    #### 1.close-to-close
    log_return = np.log(C / C.shift(1))
    cc_var = (log_return.rolling(window).var(ddof=1))

    #### 2.Parkinson
    hl = np.log(H/L)
    parkinson_bar = hl**2 / (4 * np.log(2))
    parkinson_var = (parkinson_bar.rolling(window).mean())

    #### 3.German-Klass
    co = np.log(C/O)
    gk_bar = (0.5 * hl**2 - (2*np.log(2)-1)*co**2 )
    gk_var = (gk_bar.rolling(window).mean())

    #### 4.Rogers-satchell
    rs_bar = (np.log(H/C) * np.log(H/O) + np.log(C/L) * np.log(O/L))
    rs_var = (rs_bar.rolling(window).mean())

    #### 5.Yang-zhang
    overnight     = np.log(O/C.shift(1))
    open_close    = np.log(C/O)
    overnight_var = overnight.rolling(window).var(ddof=1)
    open_close_var = (open_close.rolling(window).var(ddof=1))
    k       = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var  = (overnight_var + k*open_close_var + (1-k)*rs_var)

    ##### volatility estimates
    variances = {
        "close_to_close": cc_var,
        "parkinson": parkinson_var,
        "garman_klass": gk_var,
        "rogers_satchell": rs_var,
        "yang_zhang": yz_var
    }

    factor  = (periods_per_year if annualized else 1)
    for name, variance in variances.items():
        df[f"{name}_variance"] = variance
        df[f"{name}_volatility"] = np.sqrt(variance.clip(lower=0) *factor)

    return df



df = pd.read_csv("C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv")

data = calculate_volatility_estimators(df, window=20, annualized=False)
columns = [
    "close_to_close_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility"
]

print(data[columns].tail(20))

data[columns].plot(figsize=(14, 7))
plt.title("Comparison of Volatility Estimators")
plt.xlabel("Candle Index")
plt.ylabel("Volatility per 1-minute return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()