##1.Focus on Daily, weekly and monthly data
########## You can see that how you can find the Noise area use some mathematics 
##2.going long if the price is above the Noise Area and short if it is below.

import requests
import pandas as pd
from   datetime import datetime, timedelta,time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from   matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
import pytz

from dotenv import load_dotenv
import os
load_dotenv()


############################# Data Function ############################
api_key = os.getenv("api_key")
api_secret = os.getenv("api_secreat")


def fetch_alpaca_data(symbol, timeframe, start_date, end_date):

    url = 'https://data.alpaca.markets/v2/stocks/bars'
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    params = {
        'symbols': symbol,
        'timeframe': timeframe,
        'start': datetime.strptime(start_date, "%Y-%m-%d").isoformat() + 'Z',
        'end': datetime.strptime(end_date, "%Y-%m-%d").isoformat() + 'Z',
        'limit': 10000,
        'adjustment': 'raw',
        'feed': 'sip'
    }

    data_list = []
    eastern = pytz.timezone('America/New_York')  ##new
    utc = pytz.utc  

    market_open  = time(9, 30)  # Market opens at 9:30 AM
    market_close = time(15, 59)  # Market closes just before 4:00 PM

    print("Starting data fetch...")
    while True:
        print(f"Fetching data for symbols: {symbol} from {start_date} to {end_date}")
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching data with status code {response.status_code}: {response.text}")
            break

        data = response.json()

        bars = data.get('bars')

        for symbol, entries in bars.items():
            print(f"Processing {len(entries)} entries for symbol: {symbol}")
            for entry in entries:
                try:
                    utc_time = datetime.fromisoformat(entry['t'].rstrip('Z')).replace(tzinfo=utc)
                    eastern_time = utc_time.astimezone(eastern)

                    # Apply market hours filter for '1Min' timeframe
                    if timeframe == '1Min' and not (market_open <= eastern_time.time() <= market_close):
                        continue  # Skip entries outside market hours

                    data_entry = {
                        'volume': entry['v'],
                        'open': entry['o'],
                        'high': entry['h'],
                        'low': entry['l'],
                        'close': entry['c'],
                        'caldt': eastern_time  
                    }
                    data_list.append(data_entry)
                    print(f"Appended data for {symbol} at {eastern_time}")
                except Exception as e:
                    print(f"Error processing entry: {entry}, {e}")
                    continue

        if 'next_page_token' in data and data['next_page_token']:
            params['page_token'] = data['next_page_token']
            print("Fetching next page...")
        else:
            print("No more pages to fetch.")
            break

    df = pd.DataFrame(data_list)
    print("Data fetching complete.")
    return df

def fetch_alpaca_dividends(symbol, start_date, end_date):
    """
    Fetch dividend announcements from Alpaca for a specified symbol between two dates.
    This function splits the request into manageable 90-day segments to comply with API constraints.
    """

    url_base = "https://paper-api.alpaca.markets/v2/corporate_actions/announcements"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    dividends_list = []
    current_start = start_date

    while current_start < end_date:
 

        current_end = min(current_start + timedelta(days=89), end_date)
                
        url = f"{url_base}?ca_types=Dividend&since={current_start}&until={current_end}&symbol={symbol}"
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for entry in data:
                dividends_list.append({
                    'caldt': datetime.strptime(entry['ex_date'], '%Y-%m-%d'),
                    'dividend': float(entry['cash'])
                })
        else:
            print(f"Failed to fetch data for period {current_start} to {current_end}: {response.text}")
        
        current_start = current_end + timedelta(days=1)

    return pd.DataFrame(dividends_list)


############################### Function ################
symbol = 'SPY'
start_date     = '2023-06-05'
end_date       = '2024-06-05'   
spy_intra_data = fetch_alpaca_data(symbol, '1Min', start_date, end_date)
spy_daily_data = fetch_alpaca_data(symbol, '1D', start_date, end_date)
dividends      = fetch_alpaca_dividends(symbol,start_date,end_date)


######################################### Mathematics Variable ############
df = pd.DataFrame(spy_intra_data)
df['day'] = pd.to_datetime(df['caldt']).dt.date #convert caldt to day
df.set_index('caldt', inplace=True)             #set as a index

daily_groups = df.groupby('day')
all_days = df['day'].unique()

df['move_open'] = np.nan
df['vwap'] = np.nan
df['spy_dvol'] = np.nan

spy_ret = pd.Series(index=all_days, dtype=float) #daily return

for d in range(1, len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d-1]

    current_day_data = daily_groups.get_group(current_day)
    prev_day_data    = daily_groups.get_group(prev_day)

    #avg. h/l/c price
    hlc = (current_day_data['high'] + current_day_data['low'] + current_day_data['close'])/3

    #volume-weighted
    vol_x_hlc     = current_day_data['volume']*hlc
    cum_vol_x_hlc = vol_x_hlc.cumsum()
    cum_volume    = current_day_data['volume'].cumsum()

    df.loc[current_day_data.index, 'vwap'] = cum_vol_x_hlc / cum_volume

    #%change from opening price
    open_price = current_day_data['open'].iloc[0]
    df.loc[current_day_data.index, 'move_open'] = (current_day_data['close'] / open_price -1).abs()

    spy_ret.loc[current_day] = current_day_data['close'].iloc[-1] / current_day_data['close'].iloc[-1] - 1

    if d > 14:
        df.loc[current_day_data.index, 'spy_dvol'] = spy_ret.iloc[d - 15:d-1].std(skipna=False)


# Calculate the minutes from market open and determine the minute of the day for each timestamp.
df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
df['minute_of_day'] = df['min_from_open'].round().astype(int)

# Group data by 'minute_of_day' for minute-level calculations.
minute_groups = df.groupby('minute_of_day')

# Calculate rolling mean and delayed sigma for each minute of the trading day.
df['move_open_rolling_mean'] = minute_groups['move_open'].transform(lambda x: x.rolling(window=14, min_periods=13).mean())
df['sigma_open'] = minute_groups['move_open_rolling_mean'].transform(lambda x: x.shift(1))

# Convert dividend dates to datetime and merge dividend data based on trading days.
dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
df = df.merge(dividends[['day', 'dividend']], on='day', how='left')
df['dividend'] = df['dividend'].fillna(0)  # Fill missing dividend data with 0.


############################# Backtesting
# Constants and settings
AUM_0 = 100000.0
commission = 0.0035
min_comm_per_order = 0.35
band_mult = 1
trade_freq = 30
sizing_type = "vol_target"
target_vol = 0.02
max_leverage = 4
overnight_threshold = 0.02

# Group data by day for faster access
daily_groups = df.groupby('day')

# Initialize strategy DataFrame using unique days
strat = pd.DataFrame(index=all_days)
strat['ret'] = np.nan
strat['AUM'] = AUM_0
strat['ret_spy'] = np.nan

# Calculate daily returns for SPY using the closing prices
df_daily = pd.DataFrame(spy_daily_data)
df_daily['caldt'] = pd.to_datetime(df_daily['caldt']).dt.date
df_daily.set_index('caldt', inplace=True)  # Set the datetime column as the DataFrame index for easy time series manipulation.

df_daily['ret'] = df_daily['close'].diff() / df_daily['close'].shift()


# Loop through all days, starting from the second day
for d in range(1, len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d-1]

    if prev_day in daily_groups.groups and current_day in daily_groups.groups:
        prev_day_data = daily_groups.get_group(prev_day)
        current_day_data = daily_groups.get_group(current_day)

        if 'sigma_open' in current_day_data.columns and current_day_data['sigma_open'].isna().all():
            continue

        prev_close_adjusted = prev_day_data['close'].iloc[-1] - df.loc[current_day_data.index, 'dividend'].iloc[-1]

        open_price = current_day_data['open'].iloc[0]
        current_close_prices = current_day_data['close']
        spx_vol = current_day_data['spy_dvol'].iloc[0]
        vwap = current_day_data['vwap']

        sigma_open = current_day_data['sigma_open']
        UB = max(open_price, prev_close_adjusted) * (1 + band_mult * sigma_open)
        LB = min(open_price, prev_close_adjusted) * (1 - band_mult * sigma_open)

        # Determine trading signals
        signals = np.zeros_like(current_close_prices)
        signals[(current_close_prices > UB) & (current_close_prices > vwap)] = 1
        signals[(current_close_prices < LB) & (current_close_prices < vwap)] = -1


        # Position sizing
        previous_aum = strat.loc[prev_day, 'AUM']

        if sizing_type == "vol_target":
            if math.isnan(spx_vol):
                shares = round(previous_aum / open_price * max_leverage)
            else:
                if open_price == 0 or spx_vol == 0:
                    print(f"Warning: open_price={open_price}, spx_vol={spx_vol} — skipping this step.")
                    shares = 0
                else:
                    shares = round(previous_aum / open_price * min(target_vol / spx_vol, max_leverage))

        elif sizing_type == "full_notional":
            shares = round(previous_aum / open_price)

        # Apply trading signals at trade frequencies
        trade_indices = np.where(current_day_data["min_from_open"] % trade_freq == 0)[0]
        exposure = np.full(len(current_day_data), np.nan)  # Start with NaNs
        exposure[trade_indices] = signals[trade_indices]  # Apply signals at trade times

        # Custom forward-fill that stops at zeros
        last_valid = np.nan  # Initialize last valid value as NaN
        filled_values = []   # List to hold the forward-filled values
        for value in exposure:
            if not np.isnan(value):  # If current value is not NaN, update last valid value
                last_valid = value
            if last_valid == 0:  # Reset if last valid value is zero
                last_valid = np.nan
            filled_values.append(last_valid)

        exposure = pd.Series(filled_values, index=current_day_data.index).shift(1).fillna(0).values  

        # Calculate trades count based on changes in exposure
        trades_count = np.sum(np.abs(np.diff(np.append(exposure, 0))))

        overnight_move = (open_price / prev_close_adjusted - 1)
        open_trade_signal = -np.sign(overnight_move) * (abs(overnight_move) > overnight_threshold)

        trade_time_row = current_day_data[current_day_data['min_from_open'] == trade_freq]
        exit_price_minute_version_trade = trade_time_row['close'].iloc[0]

        # Calculate PnL of Mean-Reversion Portfolio (MRP)
        pnl_mean_reversion_trade = open_trade_signal * shares * (exit_price_minute_version_trade - open_price)
        comm_mean_reversion_trade = 2 * max(min_comm_per_order, commission * shares) * abs(open_trade_signal)
        net_pnl_mean_reversion = pnl_mean_reversion_trade - comm_mean_reversion_trade

        # Calculate PnL of Intraday Momentum Portfolio (IMP)
        change_1m = current_close_prices.diff()
        gross_pnl = np.sum(exposure * change_1m) * shares
        commission_paid = trades_count * max(min_comm_per_order, commission * shares)
        net_pnl_mom = gross_pnl - commission_paid

        # Calculate Total PNL
        net_pnl = net_pnl_mom + net_pnl_mean_reversion

        # Update the daily return and new AUM
        strat.loc[current_day, 'AUM'] = previous_aum + net_pnl
        strat.loc[current_day, 'ret'] = net_pnl / previous_aum

        # Save the passive Buy&Hold daily return for SPY
        strat.loc[current_day, 'ret_spy'] = df_daily.loc[df_daily.index == current_day, 'ret'].values[0]




############################## Result ###########################################
# Calculate cumulative products for AUM calculations
strat['AUM_SPX'] = AUM_0 * (1 + strat['ret_spy']).cumprod(skipna=True)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plotting the AUM of the strategy and the passive S&P 500 exposure
ax.plot(strat.index, strat['AUM'], label='Momentum', linewidth=2, color='k')
ax.plot(strat.index, strat['AUM_SPX'], label='S&P 500', linewidth=1, color='r')

# Formatting the plot
ax.grid(True, linestyle=':')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_ylabel('AUM ($)')
plt.legend(loc='upper left')
plt.title('Intraday Momentum Strategy', fontsize=12, fontweight='bold')
plt.suptitle(f'Commission = ${commission}/share', fontsize=9, verticalalignment='top')

# Show the plot
plt.show()

# Calculate additional stats and display them
stats = {
    'Total Return (%)': round((np.prod(1 + strat['ret'].dropna()) - 1) * 100, 0),
    'Annualized Return (%)': round((np.prod(1 + strat['ret']) ** (252 / len(strat['ret'])) - 1) * 100, 1),
    'Annualized Volatility (%)': round(strat['ret'].dropna().std() * np.sqrt(252) * 100, 1),
    'Sharpe Ratio': round(strat['ret'].dropna().mean() / strat['ret'].dropna().std() * np.sqrt(252), 2),
    'Hit Ratio (%)': round((strat['ret'] > 0).sum() / (strat['ret'].abs() > 0).sum() * 100, 0),
    'Maximum Drawdown (%)': round(strat['AUM'].div(strat['AUM'].cummax()).sub(1).min() * -100, 0)
}


Y = strat['ret'].dropna()
X = sm.add_constant(strat['ret_spy'].dropna())
model = sm.OLS(Y, X).fit()
stats['Alpha (%)'] = round(model.params.const * 100 * 252, 2)
stats['Beta'] = round(model.params['ret_spy'], 2)

print(stats)
