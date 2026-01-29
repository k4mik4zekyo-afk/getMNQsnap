import numpy as np
import pytz
import os
import sys
from datetime import date, time
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
print("Libraries imported successfully!")

def get_data(ticker, **kwargs):
    return yf.download(ticker, **kwargs)

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename)
    print(f"Saved to {filename}")

def plot_close(df):
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["Close"])
    plt.title("Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def plot_last_candles(df, n=7):
    sub = df.tail(n)
    fig, ax = plt.subplots(figsize=(10,4))

    for date, row in sub.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([date, date], [row["Low"], row["High"]], color=color)   # wick
        ax.plot([date, date], [row["Open"], row["Close"]], linewidth=6,
                color=color)  # body

    ax.set_title(f"Last {n} Candles")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def ensure_export_dir(path="export"):
    os.makedirs(path, exist_ok=True)
    return path

def plot_candles_to_file(
    df,
    title,
    filename,
    max_bars=120,
    interval_minutes=None,
    initial_balance=None,
):
    export_dir = ensure_export_dir()
    if df.empty:
        raise ValueError("DataFrame is empty; cannot plot candles.")
    sub = df.tail(max_bars)
    fig, ax = plt.subplots(figsize=(12, 5))

    for ts, row in sub.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([ts, ts], [row["Low"], row["High"]], color=color, linewidth=1)
        ax.plot([ts, ts], [row["Open"], row["Close"]], color=color, linewidth=5)

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    if interval_minutes is not None and interval_minutes < 1440:
        session_tz = sub.index.tz
        min_ts = sub.index.min()
        max_ts = sub.index.max()
        unique_days = pd.Index(sub.index.date).unique()
        for day in unique_days:
            base_ts = pd.Timestamp(day).tz_localize(session_tz)
            open_ts = base_ts + pd.Timedelta(hours=9, minutes=30)
            close_ts = base_ts + pd.Timedelta(hours=16)
            if min_ts <= open_ts <= max_ts:
                ax.axvline(open_ts, linestyle="--", color="gray", alpha=0.6, linewidth=1)
            if min_ts <= close_ts <= max_ts:
                ax.axvline(close_ts, linestyle="--", color="gray", alpha=0.6, linewidth=1)

    if initial_balance is not None:
        ib_high = initial_balance.get("high")
        ib_low = initial_balance.get("low")
        ib_date = initial_balance.get("date")
        if ib_high is not None and ib_low is not None:
            ax.axhline(ib_high, color="blue", linestyle="--", linewidth=1.2, label="IB High")
            ax.axhline(ib_low, color="blue", linestyle="--", linewidth=1.2, label="IB Low")
            if ib_date is not None:
                ax.text(
                    0.01,
                    0.98,
                    f"IB {ib_date}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="blue",
                )
            ax.legend(loc="upper right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(export_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved candle plot to {output_path}")

def calculate_initial_balance(
    df,
    session_start=time(9, 30),
    duration_minutes=60,
):
    if df.empty:
        raise ValueError("DataFrame is empty; cannot calculate initial balance.")
    end_minutes = session_start.hour * 60 + session_start.minute + duration_minutes
    end_hour = end_minutes // 60
    end_minute = end_minutes % 60
    session_end = time(end_hour, end_minute)

    recent_dates = pd.Index(df.index.date).unique()[::-1]
    for day in recent_dates:
        start_ts = pd.Timestamp.combine(day, session_start)
        end_ts = pd.Timestamp.combine(day, session_end)
        start_ts = pd.Timestamp(start_ts, tz=df.index.tz)
        end_ts = pd.Timestamp(end_ts, tz=df.index.tz)
        window = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if not window.empty:
            return {
                "date": day,
                "high": window["High"].max(),
                "low": window["Low"].min(),
            }
    return None

def export_mnq_candle_plots(mnq05, mnq30, mnq60, mnq1d, min_hours=22):
    base_interval_minutes = 5
    target_bars = int(np.ceil((min_hours * 60) / base_interval_minutes))
    initial_balance = calculate_initial_balance(mnq05)
    print(initial_balance)

    plot_candles_to_file(
        mnq05,
        "MNQ 5m Candles",
        "mnq05_candles.png",
        max_bars=target_bars,
        interval_minutes=5,
        initial_balance=initial_balance,
    )
    plot_candles_to_file(
        mnq30,
        "MNQ 30m Candles",
        "mnq30_candles.png",
        max_bars=target_bars,
        interval_minutes=30,
    )
    plot_candles_to_file(
        mnq60,
        "MNQ 60m Candles",
        "mnq60_candles.png",
        max_bars=target_bars,
        interval_minutes=60,
    )
    plot_candles_to_file(
        mnq1d,
        "MNQ 1D Candles",
        "mnq1d_candles.png",
        max_bars=target_bars,
        interval_minutes=1440,
    )

def plot_anchored_volume_profile(df, start_time=None, end_time=None, bins=40, 
                                  value_area_pct=0.70, filename="mnq05_anchored_vp.png"):
    """
    Plot an anchored volume profile with POC, VAH, and VAL.
    
    Parameters:
    -----------
    df : DataFrame with datetime index and OHLCV columns
    start_time : str or datetime, default None (9:30 AM PST previous day)
    end_time : str or datetime, default None (5:30 AM PST current day)
    bins : int, number of price levels
    value_area_pct : float, percentage of volume for value area (default 0.70)
    filename : str, output filename
    """
    export_dir = ensure_export_dir()
    if df.empty:
        raise ValueError("DataFrame is empty; cannot plot volume profile.")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for time-based filtering.")
    
    # Set default time range if not provided
    if start_time is None or end_time is None:
        # Get current time in PST
        pst = pytz.timezone('America/Los_Angeles')
        now = pd.Timestamp.now(tz=pst)
        
        # Default: 9:30 AM PST previous day to 5:30 AM PST current day
        if start_time is None:
            start_time = now.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=9, minutes=30)
        if end_time is None:
            end_time = now.normalize() + pd.Timedelta(hours=5, minutes=30)
    
    # Convert to timezone-aware if needed
    pst = pytz.timezone('America/Los_Angeles')
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time).tz_localize(pst)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time).tz_localize(pst)
    
    # Ensure df index is timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(pst)
    else:
        df.index = df.index.tz_convert(pst)
    
    # Filter data by time range
    anchored = df.loc[start_time:end_time].copy()
    
    if anchored.empty:
        raise ValueError(f"No data found between {start_time} and {end_time}")
    
    if "Volume" not in anchored.columns:
        raise ValueError("DataFrame must include a Volume column for volume profile.")

    # Calculate typical price
    typical_price = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3
    price_min = typical_price.min()
    price_max = typical_price.max()
    if price_min == price_max:
        price_min -= 0.5
        price_max += 0.5

    # Create volume histogram
    hist, edges = np.histogram(
        typical_price,
        bins=bins,
        range=(price_min, price_max),
        weights=anchored["Volume"],
    )
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Calculate Point of Control (POC) - price level with highest volume
    poc_index = np.argmax(hist)
    poc_price = centers[poc_index]
    poc_volume = hist[poc_index]
    
    # Calculate Value Area (70% of volume)
    total_volume = hist.sum()
    target_volume = total_volume * value_area_pct
    
    # Start from POC and expand outward to capture 70% of volume
    accumulated_volume = hist[poc_index]
    lower_index = poc_index
    upper_index = poc_index
    
    while accumulated_volume < target_volume:
        # Check which side to expand (choose side with more volume)
        lower_vol = hist[lower_index - 1] if lower_index > 0 else 0
        upper_vol = hist[upper_index + 1] if upper_index < len(hist) - 1 else 0
        
        if lower_vol > upper_vol and lower_index > 0:
            lower_index -= 1
            accumulated_volume += hist[lower_index]
        elif upper_index < len(hist) - 1:
            upper_index += 1
            accumulated_volume += hist[upper_index]
        else:
            # Can't expand further
            break
    
    val_price = centers[lower_index]  # Value Area Low
    vah_price = centers[upper_index]  # Value Area High
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot volume bars
    colors = ['lightcoral' if i < lower_index or i > upper_index else 'steelblue' 
              for i in range(len(hist))]
    ax.barh(centers, hist, height=(edges[1] - edges[0]) * 0.9, color=colors, alpha=0.7)
    
    # Add POC line
    ax.axhline(y=poc_price, color='red', linestyle='--', linewidth=2, label=f'POC: {poc_price:.2f}')
    
    # Add VAH and VAL lines
    ax.axhline(y=vah_price, color='green', linestyle='--', linewidth=1.5, label=f'VAH: {vah_price:.2f}')
    ax.axhline(y=val_price, color='orange', linestyle='--', linewidth=1.5, label=f'VAL: {val_price:.2f}')
    
    # Formatting
    ax.set_title(f'Anchored Volume Profile (MNQ 5m)\n{start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")} PST')
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add text annotation with statistics
    stats_text = f'Total Volume: {total_volume:,.0f}\nValue Area: {value_area_pct*100:.0f}%\nVA Range: {vah_price - val_price:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(export_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved anchored volume profile to {output_path}")
    print(f"POC: {poc_price:.2f} (Volume: {poc_volume:,.0f})")
    print(f"VAH: {vah_price:.2f}")
    print(f"VAL: {val_price:.2f}")
    print(f"Value Area Range: {vah_price - val_price:.2f}")
    
    return {
        'poc': poc_price,
        'vah': vah_price,
        'val': val_price,
        'total_volume': total_volume,
        'value_area_volume': accumulated_volume
    }

def export_mnq05_volume_profile(mnq05, start_time=None, end_time=None, bins=40, value_area_pct=0.70):
    """
    Export MNQ 5-minute volume profile.
    
    Parameters:
    -----------
    mnq05 : DataFrame with datetime index and OHLCV columns
    start_time : str or datetime, default None (9:30 AM PST previous day)
    end_time : str or datetime, default None (5:30 AM PST current day)
    bins : int, number of price levels
    value_area_pct : float, percentage of volume for value area (default 0.70)
    """
    return plot_anchored_volume_profile(
        mnq05,
        start_time=start_time,
        end_time=end_time,
        bins=bins,
        value_area_pct=value_area_pct,
        filename="mnq05_anchored_vp.png",
    )

def what_if(ticker, buy_date, sell_date, amount_invested):
    '''
    Parameters
    ----------
    ticker : STRING
        DESCRIPTION.
    buy_date : STRING
        DESCRIPTION.
    sell_date : STRING
        DESCRIPTION.
    amount_invested : DOLLARS
        DESCRIPTION.

    Returns
    -------
    Summary:
        Buy price
        sell price
        shares purchased
        final value
        profit
        % return
    
    DEV NOTE:
        -WHAT REMAINS:
            CALCULATING the number of shares
            returning % improvement
        -Improvements:
            create a script that does this with buys
            another complication is prove/disprove that buy6-convert-buy6
    
    '''
    sell_date_mod = pd.to_datetime(sell_date).date() + timedelta(days=1)
    sell_date_str = sell_date_mod.strftime("%Y-%m-%d")
    
    df = get_data(
        ticker,
        start=buy_date,
        end=sell_date_str,
        interval="1d",
        auto_adjust=True,
    )    
    print(df.head())
    print(df.tail())
    close_date_time_index = df[("Close", ticker)]
    #get closing prices for start and end
    a = close_date_time_index.loc[pd.to_datetime(buy_date)]
    b = close_date_time_index.loc[pd.to_datetime(sell_date)]
    
    shares = round(amount_invested/a, 4)
    final_value = round(float(b*shares), 2)
    returns = round(float(b-a)*shares, 2)
    roi = round((final_value/amount_invested-1)*100, 2)
    
    print("\nProfit:", returns,
          "\nFinal value:", final_value,
          "\nShares purchased:", shares,
          "\nPercent return:", roi
          )

if __name__ == "__main__":

    # 3. Define a simple, readable labeling function
    def label_session(ts, interval):
        t = ts.timetz()  # local time-of-day
    
        if interval == "30m":
            if time(0, 0) <= t <= time(8, 0):
                return "0. overnight"
            elif time(8, 30) <= t < time(11, 31):
                return "1. morning session"
            elif time(12, 0) <= t < time(16, 31):
                return "2. afternoon session"
            elif time(18, 0) <= t <= time(23, 30):
                return "3. asia and london open"
            else:
                return "unlabeled"
        else:
            if time(0, 0) <= t <= time(8, 25):
                return "0. overnight"
            elif time(8, 30) <= t <= time(11, 55):
                return "1. morning session"
            elif time(12, 0) <= t <= time(16, 55):
                return "2. afternoon session"
            elif time(18, 0) <= t <= time(23, 55):
                return "3. asia and london open"
            else:
                return "unlabeled"
    
    def get_MNQ(time_interval, start_time="2025-12-23", end_time="2025-12-31"):
        df = get_data(
            "MNQ=F",
            start=start_time,
            end=end_time,
            interval=time_interval,
            auto_adjust=True,
        )

        # Drop ticker level
        df = df.xs("MNQ=F", axis=1, level=1)

        session = "America/New_York"

        # --- FIX: handle tz-naive vs tz-aware safely ---
        if df.index.tz is None:
            # Yahoo / daily bars are usually UTC-but-naive
            df.index = df.index.tz_localize("UTC")

        df.index = df.index.tz_convert(session)
        # ----------------------------------------------

        df["daytime"] = df.index
        df["session"] = df["daytime"].apply(label_session, interval=time_interval)

        return df
    
    #Learning when pulling 1D tf, yfinance uses a TZ naive datetime

    from datetime import datetime, timedelta
    
    #Top line needs testing
    start = datetime.now() - timedelta(minutes=1440)
    start_day = datetime.now() - timedelta(minutes=1440*14)
    recent = datetime.now() - timedelta(minutes=10)
    mnq05 = get_MNQ(time_interval="5m",
                    start_time=start,
                    end_time=recent)

    mnq30 = get_MNQ(time_interval="30m",
                        start_time=start,
                        end_time=recent)

    mnq60 = get_MNQ(time_interval="60m",
                        start_time=start,
                        end_time=recent)

    mnq1d = get_MNQ(time_interval="1d",
                        start_time=start_day,
                        end_time=recent)
    
    print(mnq1d.tail(10))

    export_mnq_candle_plots(mnq05, mnq30, mnq60, mnq1d)
    export_mnq05_volume_profile(mnq05, value_area_pct=0.68)  # 68% instead of 70%