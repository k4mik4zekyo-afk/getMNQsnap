import numpy as np
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
    fig, (ax, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

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

    if "Volume" in sub.columns:
        vol_colors = [
            "green" if row["Close"] >= row["Open"] else "red" for _, row in sub.iterrows()
        ]
        if interval_minutes is not None:
            width_days = interval_minutes / (24 * 60)
        else:
            diffs = sub.index.to_series().diff().dropna()
            width_days = diffs.median().total_seconds() / (24 * 60 * 60) if not diffs.empty else 0.003
        ax_vol.bar(sub.index, sub["Volume"], color=vol_colors, width=width_days)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.2)
    else:
        ax_vol.axis("off")
        print(f"Volume data missing for {title}; skipping volume subplot.")

    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(export_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved candle plot to {output_path}")

def _select_ib_day(df):
    day_counts = df.groupby(df.index.date).size()
    if day_counts.empty:
        return None
    return day_counts.sort_values(ascending=False).index[0]

def _calculate_value_area(hist, centers, target_pct=0.7):
    if hist.sum() == 0:
        return None, None, None
    poc_idx = int(np.argmax(hist))
    total_volume = hist.sum()
    value_volume = hist[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while value_volume / total_volume < target_pct:
        next_low = low_idx - 1
        next_high = high_idx + 1
        low_vol = hist[next_low] if next_low >= 0 else -1
        high_vol = hist[next_high] if next_high < len(hist) else -1
        if low_vol == -1 and high_vol == -1:
            break
        if high_vol >= low_vol:
            high_idx = next_high
            value_volume += high_vol
        else:
            low_idx = next_low
            value_volume += low_vol

    return centers[poc_idx], centers[high_idx], centers[low_idx]

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

    ib_day = _select_ib_day(df)
    recent_dates = pd.Index(df.index.date).unique()[::-1]
    if ib_day is not None:
        recent_dates = [ib_day] + [d for d in recent_dates if d != ib_day]

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
                "count": len(window),
            }
    return None

def export_mnq_candle_plots(mnq05, mnq30, mnq60, mnq1d, min_hours=18):
    base_interval_minutes = 5
    target_bars = int(np.ceil((min_hours * 60) / base_interval_minutes))
    initial_balance = calculate_initial_balance(mnq05)
    if initial_balance:
        print(
            "Previous day IB:",
            f"{initial_balance['date']} | High {initial_balance['high']:.2f} | Low {initial_balance['low']:.2f}",
        )

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

def plot_anchored_volume_profile(df, anchor_index=0, bins=40, filename="mnq05_anchored_vp.png"):
    export_dir = ensure_export_dir()
    if df.empty:
        raise ValueError("DataFrame is empty; cannot plot volume profile.")
    if anchor_index < 0:
        anchor_index = max(len(df) + anchor_index, 0)
    anchor_index = min(anchor_index, len(df) - 1)

    anchored = df.iloc[anchor_index:].copy()
    if "Volume" not in anchored.columns:
        raise ValueError("DataFrame must include a Volume column for volume profile.")

    typical_price = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3
    price_min = typical_price.min()
    price_max = typical_price.max()
    if price_min == price_max:
        price_min -= 0.5
        price_max += 0.5

    hist, edges = np.histogram(
        typical_price,
        bins=bins,
        range=(price_min, price_max),
        weights=anchored["Volume"],
    )
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(centers, hist, height=(edges[1] - edges[0]) * 0.9, color="steelblue")
    ax.set_title("Anchored Volume Profile (MNQ 5m)")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.grid(True, axis="x", alpha=0.3)

    poc, vah, val = _calculate_value_area(hist, centers, target_pct=0.7)
    if poc is not None:
        for level, label in [(poc, "POC"), (vah, "VAH"), (val, "VAL")]:
            ax.axhline(level, color="darkorange", linestyle="--", linewidth=1)
            ax.text(
                hist.max() * 1.01,
                level,
                f"{label}: {level:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="darkorange",
            )
    plt.tight_layout()
    output_path = os.path.join(export_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved anchored volume profile to {output_path}")

def export_mnq05_volume_profile(mnq05, anchor_index=0, bins=40):
    plot_anchored_volume_profile(
        mnq05,
        anchor_index=anchor_index,
        bins=bins,
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
    export_mnq05_volume_profile(mnq05, anchor_index=0, bins=40)
