import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

################################## Data ############################################################################
class DataHandler:
    def __init__(self, path="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv", n_points=2880):
        self.path     = path
        self.n_points = n_points
        self.df       = None

    def load_or_simulate(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path, parse_dates=["timestamp"])
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)
            print(f"Loaded data from {self.path}")
        else:
            print("Data is incomplete")
            return None
        
        self.df = self.df.set_index("timestamp")
        self.df = self.df.asfreq("H")
        return self.df


#################################### ADF and KPSS (testing for stationary method) ###################################
class StationarityTester:
    @staticmethod
    def run_adf(x):
        res = adfuller(x, autolag="AIC")
        return {"adf_stat": res[0], "pvalue": res[1], "crit": res[4]}

    @staticmethod
    def run_kpss(x, regression="c"):
        res = kpss(x, regression=regression, nlags="auto")
        return {"kpss_stat": res[0], "pvalue": res[1], "crit": res[3]}

    def choose_series(self, series):
        log_price = np.log(series)
        log_returns = 100*np.log(series).diff().dropna()

        adf_lp = self.run_adf(log_price.dropna())
        kpss_lp = self.run_kpss(log_price.dropna(), regression="ct")
        adf_lr = self.run_adf(log_returns)
        kpss_lr = self.run_kpss(log_returns, regression="c")

        print("\nADF & KPSS on log price (trend test):", adf_lp, kpss_lp)
        print("ADF & KPSS on log returns (level test):", adf_lr, kpss_lr)

        if (adf_lp["pvalue"] < 0.05) and (kpss_lp["pvalue"] > 0.05):
            return log_price.dropna(), 0, "log_price (stationary)"
        elif (adf_lr["pvalue"] < 0.05) and (kpss_lr["pvalue"] > 0.05):
            return log_returns, 0, "log_returns (stationary)"
        else:
            return log_price.diff().dropna(), 1, "log_price differenced once (d=1)"


# data_handler = DataHandler(path="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv", n_points=2880)
# df = data_handler.load_or_simulate()
# tester = StationarityTester()
# stationary_series, d, desc = tester.choose_series(df["close"])

# print("\nChosen series description:", desc)
# print("Differencing order (d):", d)
# print("First 5 rows of stationary series:\n", stationary_series.head(30))


############################################## ARIMA Forecasting #############################################
class ARIMAForecaster:
    def __init__(self, series, differencing=0, out_path="SOL_ARIMA_Forecasts.csv"):
        self.series       = series
        self.differencing = differencing
        self.out_path     = out_path
        self.best_model   = None
        self.best_order   = None

    def grid_search(self, p_range=range(0, 3), q_range=range(0, 3)):
        best_aic = np.inf
        for p in p_range:
            for q in q_range:
                try:
                    model = ARIMA(endog=self.series, order=(p, 0, q))
                    res = model.fit(method_kwargs={"warn_convergence": False})
                    if res.aic < best_aic:
                        best_aic = res.aic
                        self.best_order = (p, self.differencing, q)
                        self.best_model = res
                except Exception:
                    continue
        print(f"\nBest ARIMA order: {self.best_order}, AIC={best_aic}")
        return self.best_model

    def residual_diagnostics(self):
        resid = self.best_model.resid.dropna()
        plt.figure(figsize=(10, 3))
        plt.plot(resid)
        plt.title("Residuals")
        plt.show()
        plot_acf(resid, lags=40, alpha=0.05)
        plt.show()
        lb = acorr_ljungbox(resid, lags=[12], return_df=True)
        print("\nLjung-Box test:", lb)
        return resid

    def walk_forward_forecast(self, steps=200):
        preds, idxs = [], []
        start_idx = int(len(self.series) * 0.7)
        train_series = self.series.iloc[:start_idx].copy()

        for i in range(min(steps, len(self.series) - start_idx - 1)):
            try:
                m = ARIMA(train_series, order=(self.best_order[0], 0, self.best_order[2]))
                r = m.fit(method_kwargs={"warn_convergence": False})
                f = r.get_forecast(steps=1)
                preds.append(f.predicted_mean.iloc[0])
                idxs.append(self.series.index[start_idx + i])
            except Exception:
                preds.append(np.nan)
                idxs.append(self.series.index[start_idx + i])
            train_series = self.series.iloc[:start_idx + i + 1].copy()

        preds_series = pd.Series(preds, index=idxs)
        plt.figure(figsize=(10, 3))
        plt.plot(self.series, label="Observed")
        plt.plot(preds_series, label="Forecasts")
        plt.legend()
        plt.show()
        return preds_series

    def save_forecasts(self, forecast_series):
        out_df = pd.DataFrame({
            "forecast_datetime": forecast_series.index,
            "forecast_value": forecast_series.values
        })
        out_df.to_csv(self.out_path, index=False)
        print(f"Forecasts saved to {self.out_path}")

################################## GARCH Model ##################################
class GARCHModel:
    def __init__(self, residuals):
        self.residuals = residuals
        self.model     = None
        self.results   = None

    def fit_garch(self, p=1, q=1):
        self.model = arch_model(self.residuals, vol="GARCH", p=p, q=q, dist="normal")
        self.results = self.model.fit(disp="off")
        print("\nGARCH Model Summary:")
        print(self.results.summary())
        return self.results

    def plot_volatility(self):
        if self.results is None:
            raise ValueError("Fit GARCH first.")
        cond_vol = self.results.conditional_volatility
        plt.figure(figsize=(10, 3))
        plt.plot(cond_vol)
        plt.title("Conditional Volatility (GARCH)")
        plt.show()


################################################## Main Function ###################################################
def main():
    data_handler = DataHandler(path="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv")
    df = data_handler.load_or_simulate()
    series = df["close"].astype(float)

    # Choose stationary series
    tester = StationarityTester()
    use_series, differencing, chosen = tester.choose_series(series)
    print("Chosen for modeling:", chosen)

    # Fit ARIMA
    forecaster = ARIMAForecaster(use_series, differencing)
    forecaster.grid_search()
    forecaster.residual_diagnostics()

    # Walk-forward forecast
    preds_series = forecaster.walk_forward_forecast(steps=200)
    forecaster.save_forecasts(preds_series)

    # Fit GARCH on ARIMA residuals
    resid = forecaster.residual_diagnostics()
    garch = GARCHModel(resid)
    garch.fit_garch(p=1, q=1)
    garch.plot_volatility()


if __name__ == "__main__":
    main()
