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

def get_volume_bar_width(index):
    if len(index) < 2:
        return 0.8
    index_numbers = plt.matplotlib.dates.date2num(index.to_pydatetime())
    diffs = np.diff(index_numbers)
    median_diff = np.median(diffs) if diffs.size else 0.8
    return median_diff * 0.8

def plot_close(df):
    if df.empty:
        raise ValueError("DataFrame is empty; cannot plot close.")
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    ax_price.plot(df.index, df["Close"], color="black")
    ax_price.set_title("Close Price")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    if "Volume" in df.columns:
        colors = np.where(df["Close"] >= df["Open"], "green", "red")
        bar_width = get_volume_bar_width(df.index)
        ax_vol.bar(df.index, df["Volume"], color=colors, width=bar_width)
        ax_vol.set_ylabel("Vol")
        ax_vol.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_last_candles(df, n=7):
    sub = df.tail(n)
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )

    for date, row in sub.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax_price.plot([date, date], [row["Low"], row["High"]], color=color)   # wick
        ax_price.plot([date, date], [row["Open"], row["Close"]], linewidth=6,
                      color=color)  # body

    ax_price.set_title(f"Last {n} Candles")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    if "Volume" in sub.columns:
        colors = np.where(sub["Close"] >= sub["Open"], "green", "red")
        bar_width = get_volume_bar_width(sub.index)
        ax_vol.bar(sub.index, sub["Volume"], color=colors, width=bar_width)
        ax_vol.set_ylabel("Vol")
        ax_vol.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def ensure_export_dir(path="export"):
    os.makedirs(path, exist_ok=True)
    return path

def plot_candles_to_file(df, title, filename, max_bars=120, ib_levels=None):
    export_dir = ensure_export_dir()
    if df.empty:
        raise ValueError("DataFrame is empty; cannot plot candles.")
    sub = df.tail(max_bars)
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )

    for ts, row in sub.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax_price.plot([ts, ts], [row["Low"], row["High"]], color=color, linewidth=1)
        ax_price.plot([ts, ts], [row["Open"], row["Close"]], color=color, linewidth=5)

    if ib_levels:
        line_styles = {
            "high": {"color": "purple", "linestyle": "-", "label": "Prev IB High"},
            "low": {"color": "purple", "linestyle": "-", "label": "Prev IB Low"},
            "mid": {"color": "orange", "linestyle": "--", "label": "Prev IB Mid"},
        }
        for key, value in ib_levels.items():
            style = line_styles.get(key, {"color": "gray", "linestyle": "--"})
            ax_price.axhline(value, **style)
            ax_price.text(
                0.99,
                value,
                f"{style['label']}: {value:.2f}",
                transform=ax_price.get_yaxis_transform(),
                color=style["color"],
                fontsize=8,
                ha="right",
                va="center",
            )
        price_min, price_max = ax_price.get_ylim()
        ib_min = min(ib_levels.values())
        ib_max = max(ib_levels.values())
        ax_price.set_ylim(min(price_min, ib_min) - 1, max(price_max, ib_max) + 1)

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    if "Volume" in sub.columns:
        colors = np.where(sub["Close"] >= sub["Open"], "green", "red")
        bar_width = get_volume_bar_width(sub.index)
        ax_vol.bar(sub.index, sub["Volume"], color=colors, width=bar_width)
        ax_vol.set_ylabel("Vol")
        ax_vol.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(export_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved candle plot to {output_path}")

def calculate_previous_day_initial_balance(df, start_time=time(9, 30), end_time=time(10, 30)):
    if df.empty:
        return None
    dates = sorted({ts.date() for ts in df.index})
    if len(dates) < 2:
        return None
    prev_day = dates[-2]
    mask = (df.index.date == prev_day) & (df.index.time >= start_time) & (df.index.time <= end_time)
    ib_slice = df.loc[mask]
    if ib_slice.empty:
        return None
    ib_high = ib_slice["High"].max()
    ib_low = ib_slice["Low"].min()
    ib_mid = (ib_high + ib_low) / 2
    return {"high": ib_high, "low": ib_low, "mid": ib_mid}

def export_mnq_candle_plots(mnq05, mnq30, mnq60, mnq1d):
    ib_levels = calculate_previous_day_initial_balance(mnq05)
    plot_candles_to_file(mnq05, "MNQ 5m Candles", "mnq05_candles.png", ib_levels=ib_levels)
    plot_candles_to_file(mnq30, "MNQ 30m Candles", "mnq30_candles.png")
    plot_candles_to_file(mnq60, "MNQ 60m Candles", "mnq60_candles.png")
    plot_candles_to_file(mnq1d, "MNQ 1D Candles", "mnq1d_candles.png", max_bars=90)

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
    total_volume = hist.sum()
    if total_volume <= 0:
        raise ValueError("Volume histogram is empty; cannot plot volume profile.")

    poc_index = int(np.argmax(hist))
    sorted_indices = np.argsort(hist)[::-1]
    cumulative_volume = 0
    value_area_indices = []
    for idx in sorted_indices:
        cumulative_volume += hist[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= total_volume * 0.7:
            break
    vah = centers[max(value_area_indices)]
    val = centers[min(value_area_indices)]

    colors = ["steelblue"] * len(hist)
    colors[poc_index] = "red"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(centers, hist, height=(edges[1] - edges[0]) * 0.9, color=colors)
    ax.set_title("Anchored Volume Profile (MNQ 5m)")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.grid(True, axis="x", alpha=0.3)

    max_volume = hist.max() if hist.size else 0
    x_label_pos = max_volume * 1.05
    ax.axhline(centers[poc_index], color="red", linewidth=1.5)
    ax.text(x_label_pos, centers[poc_index], f"POC: {centers[poc_index]:.2f}", color="red", va="center")

    for label, level in [("VAH", vah), ("VAL", val)]:
        ax.axhline(level, color="gray", linestyle="--", linewidth=1)
        ax.text(x_label_pos, level, f"{label}: {level:.2f}", color="gray", va="center")

    ax.set_xlim(0, max_volume * 1.25)
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
    start = datetime.now() - timedelta(minutes=1440 * 2)
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
