# -*- coding: utf-8 -*-
"""
Kyle Suico
Nov 19th
"""

import numpy as np
import sys
from datetime import time
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

    '''
    To do:
        URGENT | Need to fix time stamps before doing any analysis
        why? it is possible that Initial Balance and daily extremes were calc'd
        wrong.
    
    for each day do a group by and summarize what is the max and min by date,
    trade block and do a check if high and low were swept
    #00:00-8:30 overnight
    #9:00-11:30p morning session
    #12:00p-5:00p afternoon session
    #6:00p-11:30p asia and london open
    

    Old code
    # 1. Make timestamps timezone-aware (assume they are in UTC)
    df6["daytime"] = pd.to_datetime(df6["daytime"], utc=True)
    
    # 2. Convert to the timezone whose sessions you care about
    SESSION_TZ = "America/Los_Angeles"
    df6["ts_local"] = df6["daytime"].dt.tz_convert(SESSION_TZ)
    '''
    
    # 4. Apply it once
    #df6["session"] = df6["ts_local"].apply(label_session)


if __name__ == "__main__":
    df = get_data(
        "MNQ=F",
        start="2025-10-31",
        end="2025-11-18",
        interval="30m",
        auto_adjust=True,
    )
    
    # 3. Define a simple, readable labeling function
    def label_session(ts):
        t = ts.timetz()  # local time-of-day
    
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
    
    #Note by default the TZ downloaded is UTC and not ET
    df = df.xs("MNQ=F", axis=1, level=1)
    session = "America/New_York"
    df.index = df.index.tz_convert(session)
    df['daytime'] = df.index
    df['session'] = df['daytime'].apply(label_session)
    df["day"] = df.index.date

    #Create extremes of day table by entire day
    #Below is only for NY trading hours
    #df2 = df.between_time("09:30", "15:30")
    df2 = df


    df2.day = pd.to_datetime(df2.day)
    df2["dow"] = df2["day"].dt.dayofweek

    # Create a helper variable called 'session_day' so I can combine Friday and Sunday into one day
    df2["session_day"] = df2["day"]
    # If Sunday (6), subtract 2 days → Friday
    df2.loc[df2["dow"] == 6, "session_day"] = df2.loc[df2["dow"] == 6, "day"] - pd.Timedelta(days=2)


    daily_extremes = df2.groupby(df2.session_day).agg(
        daily_high=('High', 'max'),
        daily_low=('Low', 'min'),
        volume=("Volume", "sum"),
        count=("dow", "count")
    )
        
    daily_extremes.index = pd.to_datetime(daily_extremes.index)
    daily_extremes['dayname'] = daily_extremes.index.day_name()
    
    #Start of sweeping PDH/PDL processing
    daily_extremes.index = pd.to_datetime(daily_extremes.index)
    daily_extremes["day"] = pd.to_datetime(daily_extremes.index)
    daily_extremes["day_forward"] = np.where(
        daily_extremes.day.dt.dayofweek == 4,
        daily_extremes["day"] + pd.Timedelta(days=3),
        daily_extremes["day"] + pd.Timedelta(days=1)
        )
    #if it's a Friday then add 3 days otherwise add 1 day    
        
    #print("Daily extremes table:\n")
    #print(daily_extremes)
    
    #TO DO for above:
        #If Friday/Sunday then need to make this one day and report the extreme
        #previous_daily.daytime.dt.day_name() to return day names
        #In data set skip 11-2, 11-7 and 11-9 should be combined
        #logic if day is Monday then look at Fri/Sun and take those extremes instead
    
    #note somehow the Friday/Sunday did not work in the steps below...

    previous_daily = df.merge(
        daily_extremes[['day_forward', 'daily_high', 'daily_low', 'volume']],
        left_on="day",
        right_on="day_forward",
        how="left"
        )
    
    previous_daily['signal_high_swept'] = np.where(
        previous_daily["High"] >= previous_daily["daily_high"],
        True, None)
    previous_daily['signal_low_swept'] = np.where(
        previous_daily["Low"] <= previous_daily["daily_low"],
        True, None)
    previous_daily.index = previous_daily.daytime

    #Extremes swept is a table summarizing the final output of which direction was swept (meaning their high/low was exceeded)    
    extremes_swept = previous_daily.groupby(['day']).agg(
        high_swept = ('signal_high_swept', 'any'),
        high_time = ('signal_high_swept', 'idxmax'),
        low_swept = ('signal_low_swept', 'any'),
        low_time = ('signal_low_swept', 'idxmax'),
        )
    #Idxmax is useful in that it returns the highest value for booleans it will return a '1' but becareful because it is sensitive
    #to None's and can be falsely triggered
    
    print("Extremes swept:\n")
    print(extremes_swept)
    #One downside is there are Sundays in the dataset even though I am confident in the results there's a lot of follow up questions that I have...
    #i.e. why is happening on the NA days? is liquidity still wiped or just within range?

    sys.exit("After calculations for Previous Daily")
    
    #Question that needs to be answered how often does the previous daily high get swept? previous daily low?
    #Need to return when it gets swept too... to do this I should probably loop thru and see when is the first instance that happens?
    
    #learning log ended up removing the multiindex "MNQ" in order to merge back easily into top. Re-factored the old code for cleanliness
    
    
    #Create Initial Balance table
    df3 = df.between_time("09:30", "10:00")
    initial_balance = df3.groupby(df3.index.date).agg(
        IBH=(('High', 'MNQ=F'), 'max'),
        IBL=(('Low', 'MNQ=F'), 'min'),
        volume=(("Volume", "MNQ=F"), "sum")
    )
    initial_balance['range'] = initial_balance['IBH']-initial_balance['IBL']
    initial_balance["volpace"] = initial_balance["volume"] / initial_balance["volume"].shift(1)
    print("Initial balance table:\n")
    print(initial_balance)
    
    #Create Event table
    ##First merging data to do this in a vectorized format
    initial_balance.index = pd.to_datetime(initial_balance.index)
    initial_balance["day"] = initial_balance.index.date
    
    #Preparation before merging - need to flatten df
    df4["day"] = df4.index.date
    df4["daytime"] = df4.index

    #Start of IBH/IBL Analytics
    df5 = df4.merge(
        initial_balance[["day", "IBH", "IBL"]],
        on="day",
        how="left"
        )

    c1_break_IBH   = df5["High"] > df5["IBH"]
    c2_break_IBL   = df5["Low"] < df5["IBL"]
    c3_in_range = (df5['Close'] < df5["IBH"]) & (df5['Close'] > df5["IBL"])

    df5['breakIBH'] = c1_break_IBH
    df5['breakIBL'] = c2_break_IBL
    df5['inRange'] = c3_in_range

    

    '''
    done:
    do everything vectorized first and compare to table instead of looping thru.
    approach recommends group by

    create categorical column (within range, breaks above, closes above,
                               breaks below, closes below)    
    logic for checking interactions:
        If close < IBH & close > IBL then it is within range, however...
        high > IBH and close > IBH then closes above
        high > IBH then it breaks
        low < IBL and close < IBL then closes below
        low < IBL then it breaks

    do:
    
    1-Easier project - Does PDH/PDL get wiped? Think binary Y/N. End goal is present a % this happens then can do analytics in a up/down day
    
    2-Logic for directionality post initial balance:
        loop thru a dataframe
        if below 6:30 then direction is pre-market/ON
        if 6:30-7:30 then establishing range
        then after 7:30 look into if it's within a range or...
            if breaks and in range is false, then it broken the range
            if it gets back in range assign as re-test?
        DECISION: only reset if there are 1-2 hours (???) within range
        DECISION: add in direction close minus open.
            Why? using this can help discern length of reversals??
                define is trend 3 consecutive candles in a row
    
    '''
    
    
    
    
'''
next steps:
    Which way does liquidity sweep? Or is the day moving sideways?
        IDEA #1 - overall metrics around IBH/IBL
            #1.1|Look at IBH/IBL. Report back many bars are within range?
            12 bars total - 2 are used to establish range. Report back # of bars within
            Example November 3rd - 9/10 bars are within. I would consider this a sideways day

            #1.2 | If there is a failed auction in one direction,
            then what is the likely hood that PD-X is wiped out?

        IDEA#2 - Report directionality. It is possible if in 3 sessions there is only one extreme bias then it will continue into the next day.
        e.g. +, +, +, then what is the chance that the next day will end in +
            Break up into three sessions:
                overnight (00:00 on)
                After hours 1p-00:00
                Trading hours:
                    Initial Balance
                    After IB
                    Power Hour    
    
        Idea #3 - Respecting levels/indicators.
            i.e. Bounce/Penetration detection at X EMA. Anchored VWAP at XX:YY
            What X gives the best predictor for bounce (respect)?
                May want to consider in an uptrend vs. a downtrend
    
'''