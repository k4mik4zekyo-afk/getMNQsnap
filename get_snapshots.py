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