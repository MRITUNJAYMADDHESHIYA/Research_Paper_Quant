"""
Volatility-based trading strategy with ARIMA + GARCH pipeline, hyperparameter tuning,
faster refit option, Student-t GARCH, slippage/min-lot handling, and regime stats.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from main import DataHandler, StationarityTester, ARIMAForecaster


################################## performance + plotting + regime stats ################################################
def calc_perf_from_pnl(pnl_series, steps_per_day):
    pnl = pnl_series.fillna(0)
    returns = pnl  # pnl here are fractional returns per step (wealth multiplicative)
    wealth = (1 + returns).cumprod()
    total_return = wealth.iloc[-1] - 1.0
    mean_r = returns.mean()
    std_r = returns.std()
    ann_return = mean_r * steps_per_day * 252
    ann_vol = std_r * np.sqrt(steps_per_day * 252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()
    return {
        "total_return_%": float(total_return * 100),
        "annual_return_%": float(ann_return * 100),
        "annual_vol_%": float(ann_vol * 100),
        "sharpe": float(sharpe),
        "max_drawdown_%": float(max_dd * 100)
    }, wealth


def regime_stats(sigma_series, wealth_series, q_low=0.33, q_high=0.67):
    s = sigma_series.dropna()
    if len(s) == 0:
        return {}
    low = s.quantile(q_low)
    high = s.quantile(q_high)
    regimes = {
        "low_vol": s[s <= low].index,
        "mid_vol": s[(s > low) & (s <= high)].index,
        "high_vol": s[s > high].index
    }
    stats = {}
    for k, idxs in regimes.items():
        if len(idxs) == 0:
            stats[k] = None
            continue
        w = wealth_series.loc[wealth_series.index.intersection(idxs)]
        if len(w) == 0:
            stats[k] = None
            continue
        stats[k] = {
            "period_points": int(len(w)),
            "start": str(w.index[0]),
            "end": str(w.index[-1]),
            "return_%": float((w.iloc[-1] / w.iloc[0] - 1) * 100)
        }
    return stats

################################################### Backtesting ######################################
def backtest_full(df,
                  price_col="close",
                  in_sample_frac=0.6,
                  arima_order=(1,0,0),
                  garch_p=1, garch_q=1,
                  garch_dist="t",           # 'normal' or 't'
                  refit_every=10,          # refit models only every K steps (speedup)
                  z_entry=1.5,
                  base_frac=0.01,          # base capital fraction per trade
                  capital=100000,
                  tc_per_unit=0.0002,      # transaction cost fraction per turnover
                  slippage=0.0001,         # additional slippage fraction on trades
                  min_lot_value=100.0):    # minimum dollar exposure per trade
    """
    df: DataFrame with datetime index and price column
    refit_every: integer. If 1 => refit every step (slow). If >1 => reuse last model for next K steps.
    garch_dist: 't' uses Student's t for residuals (better for fat tails).
    """

    # -- prepare series --
    tester = StationarityTester()
    price = df[price_col].astype(float).copy()
    series, d, desc = tester.choose_series(price)
    print("Chosen series for modeling:", desc)

    returns = series.copy()  # log-returns or differenced logs expected
    n = len(returns)
    start_idx = int(in_sample_frac * n)

    # arrays to store outputs
    sigma_fore = np.zeros(n) * np.nan
    signals = np.zeros(n)
    pos_frac = np.zeros(n)
    pnl = np.zeros(n)

    # initial fit on in-sample window
    # fit ARIMA then GARCH on residuals
    def fit_models(u_returns):
        # ARIMA mean model (simple constant or ARIMA)
        try:
            ar = ARIMA(u_returns, order=(arima_order[0], 0, arima_order[2])).fit(method_kwargs={"warn_convergence": False})
            resid = ar.resid.dropna()
        except Exception:
            resid = (u_returns - np.nanmean(u_returns)).dropna()
            ar = None
        # GARCH
        try:
            am = arch_model(resid, vol="GARCH", p=garch_p, q=garch_q, dist=garch_dist, rescale=False)
            g_res = am.fit(disp="off")
        except Exception:
            # fallback to simple variance estimate
            g_res = None
        return ar, g_res

    ar_model, garch_res = fit_models(returns.iloc[:start_idx])
    last_refit_at = start_idx

    # baseline median sigma for sizing fallback
    if garch_res is not None:
        median_sigma = float(np.median(garch_res.conditional_volatility))
    else:
        median_sigma = float(returns.iloc[:start_idx].std())

    # walk-forward
    for i in range(start_idx, n-1):
        # refit condition
        if (i == start_idx) or ((i - last_refit_at) >= refit_every):
            ar_model, garch_res = fit_models(returns.iloc[:i+1])
            last_refit_at = i

        # forecast 1-step ahead volatility from current garch_res
        if garch_res is not None:
            try:
                g_fore = garch_res.forecast(horizon=1, reindex=False, method='analytic')
                var_fore = float(g_fore.variance.values[-1, 0])
                s_fore = np.sqrt(max(var_fore, 1e-12))
            except Exception:
                s_fore = median_sigma
        else:
            s_fore = median_sigma

        sigma_fore[i+1] = s_fore

        # compute standardized return for next step (we use realized next return to generate signal here as a simple strategy)
        # Note: in a realistic trading system you'd use forecast of return or other predictor instead of realized next return.
        ret_next = returns.iloc[i+1]
        z = ret_next / s_fore if s_fore > 0 else 0.0

        # signal: simple mean-revert on z
        if z > z_entry:
            signal = -1
        elif z < -z_entry:
            signal = 1
        else:
            signal = 0
        signals[i+1] = signal

        # sizing: inverse volatility scaling relative to median_sigma
        size_scale = median_sigma / s_fore if s_fore > 0 else 1.0
        size_scale = np.clip(size_scale, 0.1, 5.0)
        fraction = base_frac * size_scale * signal  # signed fraction
        # ensure minimum lot (in dollars)
        pos_value = abs(fraction) * capital
        if (pos_value > 0) and (pos_value < min_lot_value):
            # bump to min_lot_value in same sign
            fraction = np.sign(fraction) * (min_lot_value / capital)

        # compute transaction costs when position changes
        prev_frac = pos_frac[i] if i >= 0 else 0.0
        turnover = abs(fraction - prev_frac)
        tc_cost = turnover * tc_per_unit
        slp_cost = turnover * slippage

        # PnL approximation: fractional pnl = position_fraction * realized return
        pnl_frac = fraction * ret_next - tc_cost - slp_cost

        pos_frac[i+1] = fraction
        pnl[i+1] = pnl_frac

    pnl_series = pd.Series(pnl, index=returns.index).fillna(0.0)
    sigma_series = pd.Series(sigma_fore, index=returns.index)
    pos_series = pd.Series(pos_frac, index=returns.index)
    signal_series = pd.Series(signals, index=returns.index)

    # compute steps_per_day from index freq
    freq = returns.index.freqstr or pd.infer_freq(returns.index)
    if freq is None:
        steps_per_day = 24*60  # default
    elif 'T' in freq or 'min' in freq:
        steps_per_day = 24*60
    elif 'H' in freq:
        steps_per_day = 24
    elif 'D' in freq:
        steps_per_day = 1
    else:
        steps_per_day = 24*60

    perf, wealth = calc_perf_from_pnl(pnl_series, steps_per_day)
    regimes = regime_stats(sigma_series, wealth)

    results = {
        "pnl": pnl_series,
        "sigma_fore": sigma_series,
        "positions": pos_series,
        "signals": signal_series,
        "perf": perf,
        "wealth": wealth,
        "regimes": regimes
    }
    return results


####################################################### Hyperparameter tuning (simple grid on in-sample)################################################
def tune_parameters(df, price_col="close", z_grid=None, base_frac_grid=None,
                    in_sample_frac=0.6, **bt_kwargs):
    if z_grid is None:
        z_grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    if base_frac_grid is None:
        base_frac_grid = [0.005, 0.01, 0.02]

    # split data
    price = df[price_col].astype(float).copy()
    tester = StationarityTester()
    series, d, desc = tester.choose_series(price)
    returns = series
    n = len(returns)
    start_idx = int(in_sample_frac * n)

    best = None
    best_score = -np.inf
    best_params = None

    for z in z_grid:
        for bf in base_frac_grid:
            # run backtest but only on in-sample window to evaluate params quickly (no refit_every tuning here)
            sub_df = df.loc[:returns.index[start_idx-1]]  # up to in-sample end
            res = backtest_full(sub_df, price_col=price_col,
                                in_sample_frac=0.5,  # not used when df is already in-sample
                                z_entry=z,
                                base_frac=bf,
                                refit_every=50,     # coarser refit for speed
                                **bt_kwargs)
            perf = res["perf"]
            score = perf["sharpe"]
            if np.isnan(score):
                score = -np.inf
            if score > best_score:
                best_score = score
                best = res
                best_params = {"z_entry": z, "base_frac": bf, "perf": perf}

    print("Tuning complete. Best params:", best_params)
    return best_params


############################################## Main ######################################################
if __name__ == "__main__":
    PATH = r"C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv"
    if not os.path.exists(PATH):
        raise FileNotFoundError(PATH)

    # load data using your DataHandler (so it preserves your parsing logic)
    dh = DataHandler(path=PATH)
    df = dh.load_or_simulate() if hasattr(dh, "load_or_simulate") else dh.load()
    print("Loaded data. Index range:", df.index.min(), "->", df.index.max())

    # quick tune (optional) - comment out if not needed
    print("Running quick hyperparameter tuning on small grid (this may take time)...")
    best_params = tune_parameters(df, price_col="close",
                                  z_grid=[1.0, 1.5, 2.0],
                                  base_frac_grid=[0.005, 0.01],
                                  arima_order=(1,0,0),
                                  garch_p=1, garch_q=1,
                                  garch_dist="t")

    # run full backtest using best params
    chosen_z = best_params["z_entry"] if best_params is not None else 1.5
    chosen_base_frac = best_params["base_frac"] if best_params is not None else 0.01

    print(f"Running final backtest with z_entry={chosen_z}, base_frac={chosen_base_frac}")
    results = backtest_full(df,
                            price_col="close",
                            in_sample_frac=0.6,
                            arima_order=(1,0,0),
                            garch_p=1, garch_q=1,
                            garch_dist="t",
                            refit_every=20,     # refit every 20 steps to speed up
                            z_entry=chosen_z,
                            base_frac=chosen_base_frac,
                            capital=100000,
                            tc_per_unit=0.0002,
                            slippage=0.0001,
                            min_lot_value=100.0)

    print("Performance:", results["perf"])
    print("Regime stats:", results["regimes"])

    # save results
    out_df = pd.DataFrame({
        "timestamp": results["pnl"].index,
        "pnl": results["pnl"].values,
        "wealth": results["wealth"].values,
        "position_frac": results["positions"].values,
        "signal": results["signals"].values,
        "sigma_fore": results["sigma_fore"].values
    })
    out_df.to_csv("vol_strategy_full_results.csv", index=False)
    print("Saved vol_strategy_full_results.csv")

    # quick plots
    plt.figure(figsize=(12,5))
    ax1 = plt.gca()
    df["close"].loc[results["wealth"].index].plot(ax=ax1, label="price")
    ax2 = ax1.twinx()
    results["wealth"].plot(ax=ax2, color="C1", label="wealth")
    ax1.set_title("Price (left) and Strategy Wealth (right)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()

    plt.figure(figsize=(10,3))
    results["sigma_fore"].plot(title="Forecasted sigma (1-step)")
    plt.show()

    plt.figure(figsize=(10,2))
    results["signals"].replace({-1:-1, 0:0, 1:1}).plot(drawstyle="steps-post", title="Signals")
    plt.show()
