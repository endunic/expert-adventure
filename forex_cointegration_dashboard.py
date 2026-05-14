import sys
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import csv
import requests
from typing import Any, cast

# ========================= CONFIGURATION =========================
forex_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X',
    'EURGBP=X', 'EURJPY=X', 'EURCHF=X', 'EURAUD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPJPY=X', 'GBPCHF=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X',
    'AUDJPY=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X',
    'NZDJPY=X', 'NZDCAD=X', 'NZDCHF=X', 'CADJPY=X', 'CADCHF=X', 'CHFJPY=X'
]

# Dynamic start date: 729 days ago (safe limit for yfinance intraday data)
start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')

# --- DEFAULT CONFIG (Now editable in UI) ---
DEFAULT_INTERVAL = '1d'
DEFAULT_CORR = 0.5
DEFAULT_PVAL = 0.05
DEFAULT_WINDOW = 30
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_STOP = 4.0
DEFAULT_Z_EXIT = 0.2
DEFAULT_COST = 0.15
DEFAULT_ACCOUNT = 100000
DEFAULT_RISK = 0.01
SIGNAL_HISTORY_FILE = 'dashboard_output/signal_history.csv'
SIGNAL_HISTORY_COLUMNS = ['Timestamp', 'Pair', 'Signal', 'Z-Score', 'Duration (Periods)']

# Runtime settings derived from defaults
account_size = DEFAULT_ACCOUNT
risk_per_trade = DEFAULT_RISK
transaction_cost_z = DEFAULT_COST
interval = DEFAULT_INTERVAL
corr_threshold = DEFAULT_CORR
p_value_threshold = DEFAULT_PVAL
z_score_window = DEFAULT_WINDOW
z_entry = DEFAULT_Z_ENTRY
z_exit = DEFAULT_Z_EXIT
z_stop = DEFAULT_Z_STOP

# Telegram Alerts (Get these from @BotFather and @userinfobot)
use_telegram = False           # Set to True once you have your credentials
telegram_token = "YOUR_BOT_TOKEN"
telegram_chat_id = "YOUR_CHAT_ID"

# Global cache to prevent redundant API calls
_cached_close_prices = None
_last_refresh = None
IS_STREAMLIT = "streamlit" in sys.modules or "streamlit.runtime" in sys.modules

os.makedirs('dashboard_output', exist_ok=True)

# ========================= CORE FUNCTIONS =========================

@st.cache_data(ttl=900) # Cache data for 15 minutes
def download_data(interval):
    global _cached_close_prices, _last_refresh
    
    # Cache logic: Only download if cache is empty or older than 15 minutes
#    if not force and _cached_close_prices is not None:
#        if _last_refresh and (datetime.now() - _last_refresh).total_seconds() < 900:
#            return _cached_close_prices

    print(f"Downloading latest market data (Interval: {interval})...")
    data_any: Any = yf.download(forex_pairs, start=start_date, interval=interval, 
                               group_by='ticker', auto_adjust=True, progress=False)
    data = cast(pd.DataFrame, data_any)
    
    if data.empty or data.shape[1] == 0:
        raise ValueError("No data downloaded. Please check your internet connection or ticker symbols.")

    def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            # Try common field names first
            for field in ("Close", "Adj Close"):
                if field in df.columns.get_level_values(-1):
                    try:
                        return cast(pd.DataFrame, df.xs(field, level=-1, axis=1))
                    except Exception:
                        pass

            # Search levels for any OHLC-like label
            level_values = [list(df.columns.get_level_values(i).unique()) for i in range(df.columns.nlevels)]
            candidates = {"Close", "Adj Close", "close", "adj close"}
            for level_idx, values in enumerate(level_values):
                if any(v in candidates for v in values):
                    try:
                        field = next(v for v in values if v in candidates)
                        return cast(pd.DataFrame, df.xs(field, level=level_idx, axis=1))
                    except Exception:
                        pass

            raise KeyError(
                "Could not extract 'Close'/'Adj Close' from yfinance MultiIndex columns."
            )

        if "Close" in df.columns:
            out = cast(pd.DataFrame, df["Close"])
            return out.to_frame() if isinstance(out, pd.Series) else out

        if "Adj Close" in df.columns:
            out = cast(pd.DataFrame, df["Adj Close"])
            return out.to_frame() if isinstance(out, pd.Series) else out

        return cast(pd.DataFrame, df)

    close = extract_close_prices(data)

    close = close.ffill().dropna(how='all')
    close.to_csv('dashboard_output/close_prices.csv')
    
    _cached_close_prices = close
    _last_refresh = datetime.now()
    return close

@st.cache_data(ttl=900)
def scan_cointegration(close_prices, corr_threshold, p_value_threshold):
    print("Scanning for cointegrated pairs...")
    log_prices = np.log(close_prices).dropna(how='any')
    returns = log_prices.pct_change().dropna()
    corr_matrix = returns.corr()
    
    results = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            p1 = corr_matrix.columns[i]
            p2 = corr_matrix.columns[j]
            # Coerce to float to resolve Pylance typing issues with round()
            corr = float(corr_matrix.iloc[i, j])
            
            if abs(corr) > corr_threshold:
                try:
                    X = sm.add_constant(log_prices[p2])
                    model = sm.OLS(log_prices[p1], X).fit()
                    hedge = float(model.params.iloc[1])
                    spread = log_prices[p1] - hedge * log_prices[p2]
                    pval = float(adfuller(spread, autolag='AIC')[1])
                    
                    if pval < p_value_threshold:
                        # Define trade actions based on hedge ratio direction
                        side2 = "Sell" if hedge > 0 else "Buy"
                        opp_side2 = "Buy" if hedge > 0 else "Sell"
                        
                        results.append({
                            'Pair1': p1, 'Pair2': p2,
                            'Hedge_Ratio': round(hedge, 4),
                            'p_value': round(pval, 5),
                            'Correlation': round(corr, 3),
                            'Long_Action': f"Buy {p1} | {side2} {abs(hedge):.4f} {p2}",
                            'Short_Action': f"Sell {p1} | {opp_side2} {abs(hedge):.4f} {p2}"
                        })
                except (ValueError, np.linalg.LinAlgError, KeyError) as e:
                    # Log specific errors for debugging, but continue scanning
                    print(f"Skipping pair {p1}-{p2} due to error: {e}")
                    continue
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('p_value').reset_index(drop=True)
    return df

def calculate_signals(pair1, pair2, hedge_ratio, close_prices, z_score_window, z_entry, z_exit, z_stop, interval):
    log_p = np.log(close_prices[[pair1, pair2]]).dropna()
    spread = log_p[pair1] - hedge_ratio * log_p[pair2]
    
    # Calculate Half-life of Mean Reversion (Ornstein-Uhlenbeck process)
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    # Regress spread change against lag to find the rate of reversion
    X_hl = sm.add_constant(spread_lag)
    model_hl = sm.OLS(spread_diff, X_hl).fit()
    lambda_val = model_hl.params.iloc[1]
    # Half-life = -log(2) / lambda (avoid division by zero)
    half_life = round(-np.log(2) / lambda_val, 1) if lambda_val < 0 else 999

    # Adjust half-life if using intraday intervals
    if interval == '4h': half_life = round(half_life / 6, 1)
    if interval == '1h': half_life = round(half_life / 24, 1)

    mean = spread.rolling(z_score_window).mean()
    std = spread.rolling(z_score_window).std()
    z_score = (spread - mean) / std

    # Regime Detection: Slope of the Moving Average
    # Detects if the 'fair value' itself is trending (risky for mean reversion)
    regime_slope = (mean.iloc[-1] - mean.iloc[-10]) / 10 if len(mean) > 10 else 0
    regime = "STABLE" if abs(regime_slope) < 0.0005 else "DRIFTING"
    
    latest_z = z_score.iloc[-1]
    z_momentum = latest_z - z_score.iloc[-2]
    current_mean = mean.iloc[-1]
    current_std = std.iloc[-1]
    
    if abs(latest_z) >= z_stop:
        signal = "🛑 STOP LOSS"
    elif latest_z > z_entry:
        signal = "🔴 SHORT SPREAD (Sell Pair1, Buy Pair2)"
    elif latest_z < -z_entry:
        signal = "🟢 LONG SPREAD (Buy Pair1, Sell Pair2)"
    elif abs(latest_z) <= z_exit:
        signal = "✅ TAKE PROFIT (FLAT)"
    else:
        signal = "⚪ FLAT - No Signal"

    # Calculate Target Price for Pair1 (assuming Pair2 stays constant)
    # Spread = ln(P1) - h * ln(P2) => ln(P1) = Spread + h * ln(P2)
    p2_latest = close_prices[pair2].iloc[-1]
    target_ln_p1 = current_mean + hedge_ratio * np.log(p2_latest)
    target_price1 = np.exp(target_ln_p1)

    # Risk/Reward Calculation based on Z-Score distances
    # Reward = distance from entry (2.0) to exit (0.2)
    # Risk = distance from entry (2.0) to stop (4.0)
    reward_z = z_entry - z_exit
    risk_z = z_stop - z_entry
    rr_ratio = round(reward_z / risk_z, 2)
    
    return {
        'spread': spread,
        'z_score': z_score,
        'half_life': half_life,
        'latest_z': round(latest_z, 3),
        'z_momentum': round(z_momentum, 3),
        'signal': signal,
        'z_stop': z_stop,
        'target_price1': round(target_price1, 5),
        'current_price1': round(close_prices[pair1].iloc[-1], 5),
        'current_price2': round(close_prices[pair2].iloc[-1], 5),
        'rr_ratio': rr_ratio,
        'regime': regime,
        'spread_std': current_std
    }

def backtest_pair(z_score, pair_name, z_entry, z_exit, z_stop, transaction_cost_z):
    """Simple simulation: Enter at z_entry, exit at z_exit or z_stop"""
    trades = []
    equity_curve = [0]
    in_position = 0  # 1 for Long, -1 for Short, 0 for Flat
    entry_z = 0.0
    
    for i in range(len(z_score)):
        z = z_score.iloc[i]
        
        if in_position == 0:
            if z < -z_entry:
                in_position = 1 # Enter Long
                entry_z = z
            elif z > z_entry:
                in_position = -1 # Enter Short
                entry_z = z
        
        elif in_position == 1: # In Long
            if z >= -z_exit or z <= -z_stop:
                pnl = (z - entry_z) - transaction_cost_z
                trades.append(pnl)
                equity_curve.append(equity_curve[-1] + pnl)
                in_position = 0
                
        elif in_position == -1: # In Short
            if z <= z_exit or z >= z_stop:
                pnl = (entry_z - z) - transaction_cost_z
                trades.append(pnl)
                equity_curve.append(equity_curve[-1] + pnl)
                in_position = 0
                
    if not trades:
        # Keep return arity consistent with the successful path:
        # (num_trades, win_rate, total_z_pnl, max_dd, kelly, fig)
        return 0, 0, 0, 0, 0.0, None


        
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
    total_z_pnl = sum(trades)

    # Kelly Criterion: (WinProb - (LossProb / WinLossRatio))
    win_prob = win_rate / 100
    avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
    avg_loss = abs(np.mean([t for t in trades if t < 0])) if any(t < 0 for t in trades) else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    kelly = max(0, win_prob - ((1 - win_prob) / win_loss_ratio)) if win_loss_ratio != 0 else 0

    # Calculate Max Drawdown
    cum_pnl = np.cumsum(trades)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_dd = round(np.max(drawdown), 2) if len(drawdown) > 0 else 0

    # Plot Equity Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve, marker='o', linestyle='-', color='blue')
    ax.set_title(f"Backtest Equity Curve: {pair_name}")
    ax.set_ylabel("Cumulative Z-PnL")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=1)
    # plt.savefig(f'dashboard_output/{pair_name.replace("/", "_")}_backtest.png')

    return len(trades), round(win_rate, 1), round(total_z_pnl, 2), max_dd, round(kelly, 4), fig

def plot_pair(pair1, pair2, hedge_ratio, spread, z_score, z_entry, z_exit, z_stop):
    # Spread and z_score are now passed directly, avoiding re-calculation
    # log_p = np.log(close_prices[[pair1, pair2]]).dropna()
    # spread = log_p[pair1] - hedge_ratio * log_p[pair2]
    # z_score = (spread - spread.rolling(z_score_window).mean()) / spread.rolling(z_score_window).std()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    
    ax1.plot(spread, label='Spread')
    ax1.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
    ax1.set_title(f'Spread: {pair1} vs {pair2} (Hedge = {hedge_ratio})')
    ax1.legend()
    
    ax2.plot(z_score, label='Z-Score', color='purple')
    ax2.axhline(z_entry, color='orange', linestyle='--', label=f'Entry (+{z_entry})')
    ax2.axhline(-z_entry, color='orange', linestyle='--')
    ax2.axhline(z_exit, color='green', linestyle=':', label=f'Exit (+-{z_exit})')
    ax2.axhline(-z_exit, color='green', linestyle=':')
    ax2.axhline(z_stop, color='red', linestyle='-.', label=f'Stop (+-{z_stop})')
    ax2.axhline(0, color='black', alpha=0.5)
    ax2.set_title('Z-Score')
    ax2.legend()
    
    plt.tight_layout()
    # plt.savefig(f'dashboard_output/{pair1}_{pair2}_chart.png')
    return fig

def calculate_portfolio_exposure(active_list):
    """Aggregates net currency exposure to prevent over-concentration."""
    exposure = {}
    for item in active_list:
        # Extract currency codes, e.g., 'EUR' from 'EURUSD=X'
        p1_raw, p2_raw = item['Pair'].split('/')
        c1, c2 = p1_raw[:3], p2_raw[:3]
        
        # Long Spread: Buy P1, Sell P2 | Short Spread: Sell P1, Buy P2
        mult = 1 if "LONG" in item['Status'] else -1
        
        exposure[c1] = exposure.get(c1, 0) + mult
        exposure[c2] = exposure.get(c2, 0) - mult
    
    # Filter out zero exposures
    return {k: v for k, v in exposure.items() if v != 0}

def analyze_diversification(active_list, close_prices):
    """Calculates correlation between active spreads to identify redundant trades."""
    if len(active_list) < 2: return
    
    active_pairs = [item['Pair'] for item in active_list]
    print("\nINTER-SIGNAL CORRELATION (Diversification Check):")
    
    # Extract the returns for the base pairs as a proxy for signal correlation
    # A high correlation (>0.8) means you are doubling up on the same risk.
    tickers = []
    for p in active_pairs:
        tickers.extend(p.split('/'))
    
    corr_sub = close_prices[list(set(tickers))].pct_change().corr()
    print(corr_sub.round(2))
    if (corr_sub.values > 0.85).sum() > len(corr_sub.columns):
        print("⚠️  WARNING: High correlation between some active signals!")

def send_telegram_msg(message):
    """Sends a notification to your Telegram phone app."""
    if not use_telegram: return
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")


def load_signal_history():
    """Load and normalize the signal history file, with corrupted-row fallback."""
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return pd.DataFrame(columns=SIGNAL_HISTORY_COLUMNS)

    try:
        df = pd.read_csv(SIGNAL_HISTORY_FILE, parse_dates=['Timestamp'])
    except Exception as exc:
        print(f"Warning: fallback parsing signal history due to parse error: {exc}")
        df = pd.read_csv(
            SIGNAL_HISTORY_FILE,
            names=SIGNAL_HISTORY_COLUMNS,
            header=0,
            parse_dates=['Timestamp'],
            engine='python',
            on_bad_lines='skip'
        )

    # Normalize to the expected schema and drop extra malformed columns.
    for col in SIGNAL_HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df.loc[:, SIGNAL_HISTORY_COLUMNS]

    # Rewrite the file if the on-disk schema differed from the normalized schema.
    if list(df.columns) != SIGNAL_HISTORY_COLUMNS:
        df.to_csv(SIGNAL_HISTORY_FILE, index=False, quoting=csv.QUOTE_MINIMAL)

    return df


def track_signal(pair_name, signal_type, z_score, signal_history_df, duration=None):
    """Logs signals to a CSV only if the signal type has changed for the pair."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry_df = pd.DataFrame([{
        'Timestamp': now,
        'Pair': pair_name,
        'Signal': signal_type,
        'Z-Score': round(z_score, 3),
        'Duration (Periods)': duration
    }])
    
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        new_entry_df.to_csv(SIGNAL_HISTORY_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        # Use the passed history DataFrame to avoid redundant disk reads
        pair_history = signal_history_df[signal_history_df['Pair'] == pair_name]
        
        # Only append if there's no history for this pair, or if the signal type has changed
        if pair_history.empty or pair_history.iloc[-1]['Signal'] != signal_type:
            new_entry_df.to_csv(
                SIGNAL_HISTORY_FILE,
                mode='a',
                header=False,
                index=False,
                quoting=csv.QUOTE_MINIMAL
            )
            
            # Send Alert if it's an entry signal
            if "FLAT" not in signal_type:
                alert_msg = f"🚀 *NEW SIGNAL*\nPair: {pair_name}\nType: {signal_type}\nZ-Score: {round(z_score, 3)}"
                send_telegram_msg(alert_msg)

# ========================= MAIN LOOP =========================
def run_dashboard():
    # Use runtime defaults from configuration constants
    account_size_local = DEFAULT_ACCOUNT
    risk_per_trade_local = DEFAULT_RISK
    transaction_cost_z_local = DEFAULT_COST
    interval_local = DEFAULT_INTERVAL
    corr_threshold_local = DEFAULT_CORR
    p_value_threshold_local = DEFAULT_PVAL
    z_score_window_local = DEFAULT_WINDOW
    z_entry_local = DEFAULT_Z_ENTRY
    z_exit_local = DEFAULT_Z_EXIT
    z_stop_local = DEFAULT_Z_STOP

    print("\n" + "="*75)
    print("   FOREX COINTEGRATION TRADING DASHBOARD")
    print("="*75)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Account: ${account_size_local:,.0f} | Risk/Trade: {risk_per_trade_local*100:.1f}%\n")
    
    # Load signal history once for this run to calculate age and track signals
    current_signal_history = load_signal_history()

    close_prices = download_data(interval_local)
    coint_pairs = scan_cointegration(close_prices, corr_threshold_local, p_value_threshold_local)
    
    if coint_pairs.empty:
        print("No cointegrated pairs found.")
        return
    
    print(f"\nFound {len(coint_pairs)} Cointegrated Pairs. Generating Signal Summary...")
    
    # === Monitoring Section: Find Active Signals ===
    active_list = []
    watchlist = []
    
    for i, (_, row) in enumerate(coint_pairs.head(20).iterrows()):
        sig_data = calculate_signals(
            row['Pair1'], row['Pair2'], row['Hedge_Ratio'], close_prices,
            z_score_window_local, z_entry_local, z_exit_local, z_stop_local, interval_local
        )
        pair_name = f"{row['Pair1']}/{row['Pair2']}"
        
        # Calculate signal age for current signal
        signal_age = 1 # Default to 1 (fresh)
        duration_to_log = None
        pair_history = current_signal_history[current_signal_history['Pair'] == pair_name].sort_values('Timestamp')
        
        if not pair_history.empty:
            last_recorded_signal = pair_history.iloc[-1]['Signal']
            last_timestamp = pd.to_datetime(pair_history.iloc[-1]['Timestamp'])
            
            # Determine interval in seconds to calculate periods passed
            period_sec = 86400 if interval_local == '1d' else (14400 if interval_local == '4h' else 3600)
            time_diff = (datetime.now() - last_timestamp).total_seconds()
            periods_passed = round(time_diff / period_sec, 1)

            if sig_data['signal'] == last_recorded_signal:
                signal_age = periods_passed + 1
            
            # Transition to FLAT: record duration of the previous active signal
            elif "FLAT" in sig_data['signal'] and "FLAT" not in last_recorded_signal:
                duration_to_log = periods_passed

        # Now, log the signal (only if it's new or changed)
        track_signal(pair_name, sig_data['signal'], sig_data['latest_z'], current_signal_history, duration_to_log)

        # Add signal_age to sig_data for display
        sig_data['signal_age'] = signal_age

        if "FLAT" not in sig_data['signal']:
            active_list.append({
                '#': i + 1,
                'Pair': pair_name,
                'Z-Score': sig_data['latest_z'],
                'Status': sig_data['signal'],
                'Age': sig_data['signal_age']
            })
        elif abs(sig_data['latest_z']) > 1.5:
            watchlist.append({
                '#': i + 1,
                'Pair': pair_name,
                'Z-Score': sig_data['latest_z'],
                'Status': "👀 APPROACHING",
                'Age': sig_data['signal_age']
            })
    
    if active_list:
        print("\n" + "!"*20 + " ACTIVE TRADING SIGNALS " + "!"*20)
        # Filter out duplicates and display
        active_df = pd.DataFrame(active_list).drop_duplicates(subset=['Pair'])
        print(active_df.to_string(index=False))
        print(f"(Age in {interval_local} periods)")
        
        # Diversification Check
        analyze_diversification(active_list, close_prices)
        
        # Show Portfolio Exposure Summary
        exposure = calculate_portfolio_exposure(active_list)
        if exposure:
            print("\nNET CURRENCY BIAS (Aggregate of Active Signals):")
            exp_str = " | ".join([f"{k}: {'+' if v > 0 else ''}{v}" for k, v in exposure.items()])
            print(f"[{exp_str}]")
            if any(abs(v) > 2 for v in exposure.values()):
                print("⚠️  WARNING: High currency concentration detected!")
    
    if watchlist:
        print("\n" + "-"*20 + " SIGNAL WATCHLIST (Z > 1.5) " + "-"*20)
        watch_df = pd.DataFrame(watchlist).drop_duplicates(subset=['Pair'])
        print(watch_df.to_string(index=False))
        print(f"(Age in {interval_local} periods)")
    else:
        if not active_list:
            print("\nNo active signals or watchlist items. Market is currently at mean.")

    print(f"\nTop 10 Cointegrated Pairs (by p-value):")
    display_df = coint_pairs[['Pair1', 'Pair2', 'Hedge_Ratio', 'p_value']].head(10)
    print(display_df.to_string())
    
    # === Choice Section ===
    while True:
        try:
            print("\nOptions: [Number] Analyze Pair | [B + Number] Backtest Pair | [0] Exit")
            choice = input("Choice: ").strip().lower()
            if choice == '0':
                break  # Exit selection loop to go back to the "Run again?" prompt
            
            idx_str = choice.replace('b', '')
            choice_idx = int(idx_str)
            
            if 1 <= choice_idx <= len(coint_pairs):
                selected = coint_pairs.iloc[choice_idx-1]
                pair_name = f"{selected['Pair1']}/{selected['Pair2']}"
                
                sig_data = calculate_signals(
                    selected['Pair1'], selected['Pair2'], selected['Hedge_Ratio'], close_prices,
                    z_score_window_local, z_entry_local, z_exit_local, z_stop_local, interval_local
                )

                # Re-calculate signal age for the selected pair to avoid KeyError
                pair_history = current_signal_history[current_signal_history['Pair'] == pair_name].sort_values('Timestamp')
                sig_age = 1
                if not pair_history.empty:
                    last_recorded_signal = pair_history.iloc[-1]['Signal']
                    last_timestamp = pd.to_datetime(pair_history.iloc[-1]['Timestamp'])
                    
                    # Determine interval in seconds
                    period_sec = 86400 if interval_local == '1d' else (14400 if interval_local == '4h' else 3600)
                    time_diff = (datetime.now() - last_timestamp).total_seconds()
                    periods_passed = round(time_diff / period_sec, 1)

                    if sig_data['signal'] == last_recorded_signal:
                        sig_age = periods_passed + 1
                sig_data['signal_age'] = sig_age
                
                if 'b' in choice:
                    num_trades, win_rate, pnl, max_dd, kelly_val, _ = backtest_pair(
                        sig_data['z_score'], pair_name,
                        z_entry_local, z_exit_local, z_stop_local, transaction_cost_z_local
                    )
                    print("\n" + "="*40)
                    print(f"BACKTEST RESULTS: {pair_name}")
                    print("="*40)
                    print(f"Sample Period   : Last 729 days")
                    print(f"Transaction Cost: {transaction_cost_z_local} Z-units/trade")
                    print(f"Total Trades    : {num_trades}")
                    print(f"Win Rate        : {win_rate}%")
                    print(f"Total Net PnL   : {pnl} Z-units")
                    print(f"Max Drawdown    : {max_dd} Z-units")
                    print(f"Expectancy      : {round(pnl/num_trades, 2) if num_trades > 0 else 0} Z/trade")
                    print(f"Sizing Advice   : Risk {round(kelly_val * 25, 2)}% of account (Quarter-Kelly)")
                    print("="*40)
                    continue

                print(f"\nAnalyzing {selected['Pair1']} vs {selected['Pair2']}...")
                signals = sig_data
                
                # Risk Calculations (using current configuration values)
                risk_amount = account_size_local * risk_per_trade_local
                
                # True Volatility-Adjusted Sizing
                # Notional = Risk_USD / (Risk_in_Z * Std_Dev_of_Spread)
                z_risk_dist = z_stop_local - z_entry_local
                vol_adj_notional = risk_amount / (z_risk_dist * signals['spread_std'])
                
                # Quality Ranking
                if selected['p_value'] < 0.01 and signals['half_life'] < 15 and signals['regime'] == "STABLE":
                    quality = "⭐⭐⭐ HIGH (Strong Statistical Reversion)"
                elif selected['p_value'] < 0.03:
                    quality = "⭐⭐ MEDIUM"
                else:
                    quality = "⭐ LOW (High risk of breakdown)"

                print("\n" + "="*60)
                print("CURRENT TRADING SIGNAL & RISK MANAGEMENT")
                print("="*60)
                print(f"Pair           : {selected['Pair1']} / {selected['Pair2']}")
                print(f"Hedge Ratio    : {selected['Hedge_Ratio']}")
                print(f"Latest Z-Score : {signals['latest_z']}")
                print(f"Z-Momentum     : {signals['z_momentum']}")
                print(f"Signal Age     : {signals['signal_age']} {interval_local} periods")
                print(f"Half-Life (Days): {signals['half_life']}")
                print(f"Market Regime  : {signals['regime']}")
                print(f"Signal Quality : {quality}")
                print(f"Signal         : {signals['signal']}")
                print(f"Stop Loss      : Exit if |Z-Score| > {signals['z_stop']}")
                print("-" * 30)
                print(f"Long Action    : {selected['Long_Action']}")
                print(f"Short Action   : {selected['Short_Action']}")
                print("-" * 30)
                print(f"Risk Amount    : ${risk_amount:,.0f} ({risk_per_trade_local*100}%)")
                print(f"Vol-Adj Notional: ${vol_adj_notional:,.0f}")
                print(f"Target Price   : {signals['target_price1']} (on {selected['Pair1']})")
                print(f"Entry {selected['Pair1']:>8} : {signals['current_price1']}")
                print(f"Entry {selected['Pair2']:>8} : {signals['current_price2']}")
                print("="*60)
                # Pass pre-calculated spread and z_score to plot_pair
                plot_pair(selected['Pair1'], selected['Pair2'], selected['Hedge_Ratio'], 
                          signals['spread'], signals['z_score'],
                          z_entry_local, z_exit_local, z_stop_local)
                
                # Re-display the list for the next selection
                print(f"\nCointegrated Pairs available (from current scan):")
                print(display_df.to_string(index=True))
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid integer number.")
        except Exception as e:
            print(f"An error occurred: {e}")

# ========================= STREAMLIT UI =========================
def run_streamlit():
    st.set_page_config(page_title="Forex Cointegration Dashboard", layout="wide")
    st.title("📊 Forex Cointegration Trading Dashboard")
    st.markdown("**Statistical Arbitrage • Pairs Trading Scanner**")

    with st.sidebar:
        st.header("⚙️ Strategy Settings")
        interval = st.selectbox("Interval", ['1d', '4h', '1h'], index=0)
        corr_threshold = st.slider("Min Correlation", 0.3, 0.9, DEFAULT_CORR, step=0.05)
        p_value_threshold = st.slider("Max p-value", 0.001, 0.1, DEFAULT_PVAL, step=0.001)
        z_window = st.slider("Z-Score Window", 20, 60, DEFAULT_WINDOW)
        z_entry = st.slider("Z Entry", 1.5, 3.0, DEFAULT_Z_ENTRY, step=0.1)
        z_exit = st.slider("Z Exit", 0.1, 1.0, DEFAULT_Z_EXIT, step=0.1)
        z_stop = st.slider("Z Stop", 3.0, 6.0, DEFAULT_Z_STOP, step=0.1)
        transaction_cost_z = st.slider("Transaction Cost (Z-units)", 0.0, 0.5, DEFAULT_COST, step=0.01)

        st.markdown("---")
        st.header("💰 Risk Management")
        account_size = st.number_input("Account Size (USD)", min_value=1000, value=DEFAULT_ACCOUNT, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, DEFAULT_RISK * 100, step=0.1) / 100

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if st.button("🔄 Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner("Downloading data and scanning..."):
            try:
                close_prices = download_data(interval)
                coint_pairs = scan_cointegration(close_prices, corr_threshold, p_value_threshold)
                
                if coint_pairs.empty:
                    st.error("No cointegrated pairs found.")
                    st.session_state.analysis_results = None
                    return

                active_list = []
                watchlist = []
                history_df = load_signal_history()
                scanned_count = 0
                
                for _, row in coint_pairs.head(20).iterrows():
                    sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'],
                                            close_prices, z_window, z_entry, z_exit, z_stop, interval)
                    pair_name = f"{row['Pair1']}/{row['Pair2']}"
                    
                    # Logic for signal tracking age
                    pair_history = history_df[history_df['Pair'] == pair_name].sort_values('Timestamp')
                    age = 1
                    if not pair_history.empty:
                        last_recorded_signal = pair_history.iloc[-1]['Signal']
                        if sig['signal'] == last_recorded_signal:
                            last_ts = pd.to_datetime(pair_history.iloc[-1]['Timestamp'])
                            period_sec = 86400 if interval == '1d' else (14400 if interval == '4h' else 3600)
                            age = round((datetime.now() - last_ts).total_seconds() / period_sec, 1) + 1

                    track_signal(pair_name, sig['signal'], sig['latest_z'], history_df)

                    if "FLAT" not in sig['signal']:
                        active_list.append({
                            'Pair': pair_name, 'Z-Score': sig['latest_z'],
                            'Status': sig['signal'], 'Age': age
                        })
                    elif abs(sig['latest_z']) > 1.5:
                        watchlist.append({
                            'Pair': pair_name, 'Z-Score': sig['latest_z'],
                            'Status': "👀 WATCH", 'Age': age
                        })
                    scanned_count += 1

                st.session_state.analysis_results = {
                    'close_prices': close_prices, 'coint_pairs': coint_pairs,
                    'active_list': active_list, 'watchlist': watchlist,
                    'scanned_count': scanned_count,
                    'settings': {
                        'interval': interval, 'z_window': z_window,
                        'z_entry': z_entry, 'z_exit': z_exit, 'z_stop': z_stop,
                        'account_size': account_size, 'risk_per_trade': risk_per_trade,
                        'transaction_cost_z': transaction_cost_z
                    }
                }
                st.success(f"Analysis complete! Found {len(coint_pairs)} pairs.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Signals", "📋 All Pairs", "🔍 Deep Dive", "📈 Backtest"])

        with tab1:
            st.info(f"Analyzed top {res['scanned_count']} cointegrated pairs from {len(res['coint_pairs'])} total matches.")
            
            if not res['active_list'] and not res['watchlist']:
                st.warning("No active signals or watchlist items. Consider lowering the 'Min Correlation' or increasing 'Max p-value' to find more pairs, or lowering 'Z Entry' to trigger more signals.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Active Signals")
                if res['active_list']:
                    st.dataframe(pd.DataFrame(res['active_list']), use_container_width=True)
                else:
                    st.info("No active signals.")
            with col2:
                st.subheader("Watchlist")
                if res['watchlist']:
                    st.dataframe(pd.DataFrame(res['watchlist']), use_container_width=True)
                else:
                    st.write("No pairs currently near entry thresholds.")
            
            exposure = calculate_portfolio_exposure(res['active_list'])
            if exposure:
                st.subheader("Currency Exposure")
                st.write(exposure)

        with tab2:
            st.subheader("Cointegrated Pairs")
            st.dataframe(res['coint_pairs'], use_container_width=True)

        with tab3:
            idx = st.selectbox("Select Pair", range(len(res['coint_pairs'])), 
                               format_func=lambda x: f"{res['coint_pairs'].iloc[x]['Pair1']}/{res['coint_pairs'].iloc[x]['Pair2']}")
            row = res['coint_pairs'].iloc[idx]
            s = res['settings']
            sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'], res['close_prices'],
                                    s['z_window'], s['z_entry'], s['z_exit'], s['z_stop'], s['interval'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Z-Score", sig['latest_z'])
            col2.metric("Signal", sig['signal'])
            col3.metric("Half-Life", f"{sig['half_life']} days")

            st.markdown("---")
            st.subheader("Risk & Sizing")
            risk_amount = s['account_size'] * s['risk_per_trade']
            z_risk_dist = s['z_stop'] - s['z_entry']
            vol_adj_notional = risk_amount / (z_risk_dist * sig['spread_std']) if sig['spread_std'] > 0 else 0
            
            st.write(f"**Risk Amount:** ${risk_amount:,.2f}")
            st.write(f"**Vol-Adjusted Notional:** ${vol_adj_notional:,.2f}")
            st.write(f"**Target Price ({row['Pair1']}):** {sig['target_price1']}")
            
            st.pyplot(plot_pair(row['Pair1'], row['Pair2'], row['Hedge_Ratio'], sig['spread'], sig['z_score'],
                                s['z_entry'], s['z_exit'], s['z_stop']))

        with tab4:
            idx_bt = st.selectbox("Select Pair for Backtest", range(len(res['coint_pairs'])), 
                                  format_func=lambda x: f"{res['coint_pairs'].iloc[x]['Pair1']}/{res['coint_pairs'].iloc[x]['Pair2']}", key="bt_sel")
            row_bt = res['coint_pairs'].iloc[idx_bt]
            s = res['settings']
            sig_bt = calculate_signals(row_bt['Pair1'], row_bt['Pair2'], row_bt['Hedge_Ratio'], res['close_prices'],
                                       s['z_window'], s['z_entry'], s['z_exit'], s['z_stop'], s['interval'])
            
            num_t, wr, pnl, mdd, k, fig = backtest_pair(sig_bt['z_score'], f"{row_bt['Pair1']}/{row_bt['Pair2']}",
                                                       s['z_entry'], s['z_exit'], s['z_stop'], s['transaction_cost_z'])
            
            if num_t > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Trades", num_t)
                col2.metric("Win Rate", f"{wr}%")
                col3.metric("PnL (Z)", pnl)
                col4.metric("Kelly", k)
                st.pyplot(fig)
            else:
                st.info("No trades in sample.")

# ========================= RUN =========================
if __name__ == "__main__":
    if IS_STREAMLIT:
        run_streamlit()
    else:
        print("Starting Forex Cointegration Dashboard...")
        while True:
            run_dashboard()
            again = input("\nRun dashboard again? (y/n): ").strip().lower()
            if again != 'y':
                print("Thank you for using the dashboard!")
                break
