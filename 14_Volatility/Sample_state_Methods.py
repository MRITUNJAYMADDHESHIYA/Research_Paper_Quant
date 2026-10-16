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
    hl = 