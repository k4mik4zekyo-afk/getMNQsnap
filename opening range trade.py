#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:23:39 2025

@author: kylesuico
"""

import numpy as np
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

def getIB(dataframe, asset):
    if asset != "MGC=F":
        ib_sub = dataframe.between_time("09:30", "10:00")
    else:
        ib_sub = dataframe.between_time("08:20", "9:20")

    ib = ib_sub.groupby(ib_sub.index.date).agg(
        IBH= ('High', 'max'),
        IBL= ('Low', 'min'),
        volume= ('Volume', 'sum')
    )
    ib['range'] = ib['IBH']-ib['IBL']
    
    #Create volume pace by dividing the previous day's IB volume
    ib["volpace"] = ib["volume"] / ib["volume"].shift(1)
    
    #First merging data to do this in a vectorized format
    ib.index = pd.to_datetime(ib.index)
    ib["day"] = ib.index.date
    return ib

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
    
    def get_MNQ(time_interval, asset):
        #change for MNQ=F for MNQ
        df = get_data(
            asset,
            start="2025-10-27",
            end="2025-12-19",
            interval=time_interval,
            auto_adjust=True,
        )
        
        #Note by default the TZ downloaded is UTC and not ET
        df = df.xs(asset, axis=1, level=1)
        session = "America/New_York"
        df.index = df.index.tz_convert(session)
        df['daytime'] = df.index
        df['session'] = df['daytime'].apply(label_session, interval=time_interval)
        df["day"] = df.index.date
        df["daytime"] = df.index
        return df
    
    def mergeIBnAsset(ib, df):
        ib_merged = df.merge(
            ib[["day", "IBH", "IBL"]],
            on="day",
            how="left"
            )

        #Create logical vectors
            #Note: Closes vector is NOT always a subset within the break vector.
        ib_merged['breakIBH'] = (ib_merged["High"] > ib_merged["IBH"]) & (ib_merged["Low"] < ib_merged["IBH"])
        ib_merged['breakIBL'] = (ib_merged["Low"] < ib_merged["IBL"]) & (ib_merged["High"] > ib_merged["IBL"])
        ib_merged['closedAboveIBH'] = ib_merged['Close'] > ib_merged['IBH']
        ib_merged['closedBelowIBL'] = ib_merged['Close'] < ib_merged['IBL']
        ib_merged['inRange'] = (ib_merged['Close'] < ib_merged["IBH"]) & (ib_merged['Close'] > ib_merged["IBL"])
        ib_merged.index = ib_merged.daytime
        final = ib_merged
        #final = ib_merged.query('session == "1. morning session" | session == "2. afternoon session"')
        #final = final[final['daytime'].dt.time >= time(10,0)] #removed since I want the full data
        return final
    
    #Get Open-Close summary for a specified time (TIME PERIOD)
    #Use for gold
    #STARTING_WINDOW = time(8,20)
    
    #Use for equities
    STARTING_WINDOW = time(9,30)
    #TIME_PERIOD = time(15,59) #End of day
    
    #If looking at before London close
    TIME_PERIOD = time(11,15) #Before london

    
    asset_in = "MNQ=F"
    #mnq30 = get_MNQ(time_interval="30m", asset="MGC=F")
    mnq15 = get_MNQ(time_interval="15m", asset="MNQ=F")
    #mnq5 = get_MNQ(time_interval="5m", asset_in)

    
    # --- Note need to update the variable here for time frame and commodity
    initial_balance = getIB(mnq15, asset="MNQ=F")    
    print("Initial balance table:\n")
    print(initial_balance)
    print("\nEnd of Initial balance calculation\n")
    
    # --- Note need to update the variable here for time frame and commodity
    MNQ_ib_merge0 = mergeIBnAsset(initial_balance, mnq15)
    MNQ_ib_merge = MNQ_ib_merge0
    MNQ_ib_merge0 = MNQ_ib_merge.between_time(STARTING_WINDOW, TIME_PERIOD)
    # MNQ_ib_merge0 = MNQ_ib_merge0[(MNQ_ib_merge0['daytime'].dt.time >= STARTING_WINDOW) &
    #                              (MNQ_ib_merge0['daytime'].dt.time <= TIME_PERIOD)]
    openClose = MNQ_ib_merge0.groupby(MNQ_ib_merge0.day).agg(
        Open = ('Open', 'first'),
        Close = ('Close', 'last'))
    openClose['end_dir'] = openClose.Close-openClose.Open
    
    #Get summary stats for a specified opening window
    ENDING_WINDOW = time(9,44)
    
    first_candle = MNQ_ib_merge.between_time(STARTING_WINDOW, ENDING_WINDOW)
    first_candle['open_dir'] = first_candle.Close-first_candle.Open
    first_candle['doji_crit'] = 0.1*(first_candle.High-first_candle.Low)
    
    #Print parameters
    print("The time frame we are looking at is  09:30:00 to ", TIME_PERIOD, "Eastern Time")
    print("The first candle we are at is from", STARTING_WINDOW, " to ", ENDING_WINDOW, "Eastern Time\n")
    
    #How many of these are doji candles?
    first_candle['doji'] = abs(first_candle['open_dir']) <= first_candle['doji_crit']
    percent_doji = round(sum(first_candle['doji'] == True)/len(first_candle['doji']), 3)
    print('The number of dojis in the data set are: ', percent_doji, "\n")
    
    #Create a merged df of openClose and the first candle called first_vs_end
    first_vs_end = openClose.merge(
        first_candle[['day', 'open_dir', 'doji', 'Volume', 'daytime']],
        on = 'day',
        how = 'left'
        )
    
    #re-arrange data headers for read ablility
    cols_to_front = ['open_dir', 'end_dir']
    for j, c in enumerate(cols_to_front):
        first_vs_end.insert(j, c, first_vs_end.pop(c))

    # %% Summary statistics for agreement: opening range dir and end of day direction
    
    # 1) Keep only rows where both directions exist
    d = first_vs_end[['open_dir', 'end_dir']].dropna()
    
    # direction = sign (+1, -1); zeros mean "no direction"
    d['open_sign'] = np.sign(d['open_dir'])
    d['end_sign']  = np.sign(d['end_dir'])
    
    # keep only days where both have a direction (non-zero)
    d = d[(d['open_sign'] != 0) & (d['end_sign'] != 0)]
    
    # "In direction" means end-of-day sign matches open sign
    d['in_direction'] = d['open_sign'] == d['end_sign']
    
    # overall stats
    summary = pd.Series({
        "n_days": len(d),
        "pct_in_direction": d['in_direction'].mean(),
        "n_in_direction": int(d['in_direction'].sum()),
        "n_opposite": int((~d['in_direction']).sum()),
    })
    
    # optional: breakdown by open direction
    by_open = (
        d.groupby('open_sign')['in_direction']
         .agg(n_days='size', pct_in_direction='mean', n_in_direction='sum')
         .rename(index={-1: "open_down", 1: "open_up"})
    )
    print("Summary of overall results:\n")
    print(summary)
    print("Breakdown by opening direction:")
    print(by_open)
    sys.exit("End of summary stats")
    
    # %% Development: adding VWAP to the 9:30am session
    # Why? This seems like it would be a good confluence for predicting short/long
    #trade direction during opening range b/o
    # Also this would be nice to add to the plotting modules for fast analysis as well
    
    import pandas as pd

    df = mnq15.copy()

    # df has columns: datetime, Open, High, Low, Close, Volume
    df["daytime"] = pd.to_datetime(df["daytime"])
    df = df.sort_values("daytime")
    
    # --- Make sure timestamps are in US/Eastern (handles DST correctly) ---
    # If df["datetime"] is already tz-aware, use tz_convert.
    # If it's tz-naive but represents UTC, localize to UTC then convert.
    dt = df["daytime"]
    if dt.dt.tz is None:
        dt_et = dt.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        dt_et = dt.dt.tz_convert("America/New_York")
    
    # --- Session key anchored at 09:30 ET ---
    session_key = (dt_et - pd.Timedelta(hours=9, minutes=30)).dt.floor("D")
    
    # Typical Price (common VWAP convention). Or use df["Close"].
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    
    df["vwap_anchored_930"] = (tp * df["Volume"]).groupby(session_key).cumsum() / df["Volume"].groupby(session_key).cumsum()

    
    # %% Next steps
    '''
    If I bet on the opening being the same direction for an open then I'm good...
    Ideas:
    Make a backtester that will look at the first 15mins and place a trade long (step 3)
    Define a function that will do this so I can maximize the highest win rate (step 2)
        add an argument for the number of candles if using 1m/5m                                                  
    Try the 5m and 30m (step???)
    Do some analytics and understand how volume may inform a day I make the trade or not - specifically for shorts(step ???)
        50% so I can probably take advantage of this.
        
    X-Y plot
    X-opening dir
    Y-ending dir
    colored dots for above a mean or not

    '''
    
    # %% Plotting
    
    #only applicable for the equities
    #average_vol = 110779.0
    #first_vs_end["aboveMeanVol"] = first_vs_end.Volume > average_vol

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    
    # Force browser rendering
    pio.renderers.default = "browser"
    
    x = first_vs_end["open_dir"]
    y = first_vs_end["end_dir"]
    mask = first_vs_end["doji"].astype(bool)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x[mask],
        y=y[mask],
        mode="markers",
        name="doji = True",
        customdata=np.stack([first_vs_end.index[mask]], axis=-1),
        hovertemplate="open_dir=%{x}<br>end_dir=%{y}<br>index=%{customdata[0]}<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter(
        x=x[~mask],
        y=y[~mask],
        mode="markers",
        name="doji = False",
        customdata=np.stack([first_vs_end.index[~mask]], axis=-1),
        hovertemplate="open_dir=%{x}<br>end_dir=%{y}<br>index=%{customdata[0]}<extra></extra>",
    ))
    
    fig.update_layout(
        title="open_dir vs end_dir (colored by doji)",
        xaxis_title="open_dir",
        yaxis_title="end_dir",
    )
    
    fig.show()

    # %% Trade entry for shorts only
    '''
    Overall:
        look only at days with shorts
        Entry:
            open a short with no stop loss at the midpoint of the next candle
        identify max level of risk for each day
        exit by fixed time 11:00 ET
        return table of max draw down

    Work flow:
        Define variable to record drawdown (outside loop)
        Check the first bar for direction
        Start loop for the dataframe
        Get in trade near open 0.75-0 of the low
    
    Research question:
        Which entry will be best from 0.5 to 1?
        LATER | Will it be better to use a 15m period or make the decision from the 5minute

    '''    
    def quartile_zone_signal_for_bar(df_bar, p0: float, p25: float, p50: float,
                                     high0: float) -> int:
        """
        Returns:
          1 if bar overlaps zone1 [p0, p25]
          2 if bar overlaps zone2 (p25, p50]
          5 if bar overlaps both
          0 otherwise
        """
        lo = df_bar.Low.item()
        hi = df_bar.High.item()
        # normalize just in case inputs come in reversed
        #lo, hi = (df_bar.Low, high) if low <= high else (high, low)
    
        # overlap with [a,b] is: hi >= a and lo <= b
        in_z1 = (hi >= p0) and (lo <= p25)
    
        # zone2 is (p25, p50] -> exclude "touching only at p25"
        # so we require some part strictly above p25 AND still below/at p50
        in_z2 = (hi > p25) and (lo <= p50)
    
        if hi >= high0:
            return 8
        if in_z1 and in_z2:
            return 5
        if in_z1:
            return 1
        if in_z2:
            return 2
        return 0
    
    def opening_range_signal(i, df, IO, p0, p25, p50, high):
        #output will be a list: 0: return signal, 1: drawdown, 2: extension,
        #3: drawdown time, 4: extension time
        output = [0, IO[1], IO[2], IO[3], IO[4]]
        
        #DX
        #print("Data within operning range signal \n")
        #print("Low: ", df.Low)
        #print("High: ", df.High)
        
        #record max draw down
        if(df.High > IO[1]):
            output[1] = df.High
            output[3] = i
        
        #record max extension
        if(df.Low < IO[2]):
            output[2] = df.Low
            output[4] = i
        
        #compare current candle range close to % values:
        sig = quartile_zone_signal_for_bar(df, p0, p25, p50, high)
        
        if sig != 0 and i!= 0:
            output[0] = sig
            
        return output
    
    def q1q2(df):
        #Helper fx to calculate the range and 0,25,50% levels for entry
        low0 = df.iloc[0].Low
        high0 = df.iloc[0].High
        rng0 = high0 - low0
        q1 = low0 + 0.25 * rng0
        q2 = low0 + 0.50 * rng0
        return (rng0, q1, q2, low0, high0)
    
    #Filter by EACH trade day
    import datetime    
    fmnq15 = mnq15.between_time("09:30", "16:00")
    fmnq15["day"] = fmnq15.index.date
    fmnq15.day = pd.to_datetime(fmnq15.day)
    fmnq15["dow"] = fmnq15["day"].dt.dayofweek
    fmnq15 = fmnq15[fmnq15["dow"] != 6]
    uniqdays = fmnq15.day.unique()
    
    #Integration testing starts here
    #single day
    #firstday = uniqdays[37] #Filters for December 11th - very good day for short-term shorts
    master = []
    
    for day in uniqdays:
        print(day)
        ffmnq15 = fmnq15[fmnq15.day == day]
        testqq = q1q2(ffmnq15)
        #print(testqq)    
        
        #initialize parameters by day here
        drawdown = testqq[4]
        extension = testqq[3]
        drawdown_time = 0
        extension_time = 0
        out_list = [0, drawdown, extension, drawdown_time, extension_time]
        direction = ffmnq15.iloc[0].Close-ffmnq15.iloc[0].Open
    
        # %% Continuation (unit) Testing of key functions
        # testquartile_zone = quartile_zone_signal_for_bar(
        #     ffmnq15.iloc[3],
        #     testqq[0],
        #     testqq[1],
        #     testqq[2]
        #     )
        # print(testquartile_zone)
        
        # out_list = opening_range_signal(ffmnq15.iloc[3], out_list, testqq[0], testqq[1], testqq[2])
        # print(out_list)
        
        # %% LEFT OFF HERE - Loop development - "Putting it all together
        no_trades = False
        signal = [0]*len(ffmnq15)    
        risk = testqq[4]-testqq[3]
        
        #starting from the low subtract multiple for TP level
        TP1 = testqq[3] - risk
        TP2 = testqq[3] - risk*2
        TP3 = testqq[3] - risk*3
        TP4 = testqq[3] - risk*4
        
        SLhit = None
        TP1hit = None
        TP2hit = None
        TP3hit = None
        TP4hit = None
        
        for i in range(0,len(ffmnq15)):
            if direction < 0:
                out_list = opening_range_signal(i, ffmnq15.iloc[i], out_list,
                                                testqq[0], testqq[1], testqq[2],
                                                testqq[4])
                signal[i] = out_list[0]
            else:
                no_trades = True
            
            if i > 0:
                #check if SL/Targets hit
                if bool(testqq[4] <= ffmnq15.iloc[i].High and SLhit == None):
                    SLhit = ffmnq15.iloc[i].daytime
                if bool(ffmnq15.iloc[i].Low <= TP1 and TP1hit == None):
                    TP1hit = ffmnq15.iloc[i].daytime
                if bool(ffmnq15.iloc[i].Low <= TP2 and TP2hit == None):
                    TP2hit = ffmnq15.iloc[i].daytime
                if bool(ffmnq15.iloc[i].Low <= TP3 and TP3hit == None):
                    TP3hit = ffmnq15.iloc[i].daytime
                if bool(ffmnq15.iloc[i].Low <= TP4 and TP4hit == None):
                    TP4hit = ffmnq15.iloc[i].daytime
    
        ffmnq15['signal'] = signal
        if no_trades: print("not a day to short!")
        
        #create (1) data structure for data reduction
        #day, max draw down, time for max draw down, max profit, time max profit
        #Probably would need to do this with respect to candle 1.
        #Entry from low0, or Q1 or Q2? Or do all three?
        #Exit criteria: SL | When does high0 get reached?
        #Exit crtieria: TP | 2:1, 3:1, 4:1, 5:1
    
        data_red = [day, no_trades, out_list[1], out_list[3], out_list[2],
                    out_list[4], SLhit, TP1hit, TP2hit, TP3hit, TP4hit,
                    testqq[3], risk]
        master.append(data_red)

    master_df = pd.DataFrame(master, columns=["day", "no_trades",
                                              "max drawdown", "when maxDD",
                                              "max extension", "when maxE",
                                              "SL hit", "TP1 hit", "TP2 hit",
                                              "TP3 hit", "TP4 hit",
                                              "low0", "risk"])
    
    
    
    #Need to calculate:
        #where SL is earlier than TP1 -> loss
        #where TP1 is earlier than SL -> win
        #where both are NaT -> unclear (one case)
    
    # Identify days with missing timestamps (NaT) -> do not count
    def pnl(df):
        missing_mask = df["TP2 hit"].isna() | df["SL hit"].isna()
        day_series = df["day"] if "day" in df.columns else df.index
        
        if missing_mask.any():
            print("Days with NaT in TP1 hit or SL hit (excluded from counts):")
            for d in day_series[missing_mask]:
                print(d)
        
        # Only evaluate rows where both timestamps exist
        valid = df.loc[~missing_mask].copy()
        
        # TP1 before SL?
        valid["tp_before_sl"] = valid["TP2 hit"] < valid["SL hit"]
        
        # Put risk into profit or losses (per-row columns)
        valid["profit"] = valid["risk"].where(valid["tp_before_sl"], 0)
        valid["losses"] = valid["risk"].where(~valid["tp_before_sl"], 0)
        
        # Merge results back to df (optional)
        df["tp_before_sl"] = pd.NA
        df["profit"] = 0.0
        df["losses"] = 0.0
        df.loc[valid.index, ["tp_before_sl", "profit", "losses"]] = valid[["tp_before_sl", "profit", "losses"]]
        
        # Totals
        total_profit = valid["profit"].sum()
        total_losses = valid["losses"].sum()
        
        print(f"\nTotal risk added to profit: {total_profit}")
        print(f"Total risk added to losses: {total_losses}")


    def pl_from_maxdd_maxe(df,
                           risk_col="risk",
                           day_col="day",
                           maxdd_col="when maxDD",
                           maxe_col="when maxE"):
        d = df.copy()
    
        # Handle alternative spacing if present
        if maxdd_col not in d.columns and "when max DD" in d.columns:
            maxdd_col = "when max DD"
        if maxe_col not in d.columns and "when max E" in d.columns:
            maxe_col = "when max E"
    
        # Prefer numeric; otherwise parse as datetime
        def coerce_series(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().sum() >= max(1, int(0.5 * len(s))):
                return s_num
            return pd.to_datetime(s, errors="coerce")
    
        d[maxdd_col] = coerce_series(d[maxdd_col])
        d[maxe_col]  = coerce_series(d[maxe_col])
    
        # Exclude rows where either is missing
        missing_mask = d[maxdd_col].isna() | d[maxe_col].isna()
        day_series = d[day_col] if day_col in d.columns else d.index
    
        if missing_mask.any():
            print("Excluded (missing when maxDD or when maxE):")
            for day in day_series[missing_mask]:
                print(day)
    
        valid = d.loc[~missing_mask].copy()
    
        # Ignore stop-out logic when maxDD == 0
        dd_is_zero = (valid[maxdd_col] == 0)
    
        # Stopped out if maxDD happens before maxE, AND maxDD != 0
        valid["stopped_out"] = (~dd_is_zero) & (valid[maxdd_col] < valid[maxe_col])
    
        # Profit/loss buckets using risk
        valid["profit"] = valid[risk_col].where(~valid["stopped_out"], 0)
        valid["losses"] = valid[risk_col].where(valid["stopped_out"], 0)
    
        total_profit = valid["profit"].sum()
        total_losses = valid["losses"].sum()
    
        # Write back results
        d["stopped_out"] = pd.NA
        d["profit"] = 0.0
        d["losses"] = 0.0
        d.loc[valid.index, ["stopped_out", "profit", "losses"]] = valid[["stopped_out", "profit", "losses"]]
    
        print(f"\nTotal risk added to profit: {total_profit}")
        print(f"Total risk added to losses: {total_losses}")
    
        return d, total_profit, total_losses
    
    # Usage:
    # df_out, profit, losses = pl_from_maxdd_maxe(df)
    

    shorts_only = master_df[master_df['no_trades'] == False]
    pl_from_maxdd_maxe(shorts_only)
    
    '''
    NEXT STEPS:
        C-Analysis from 12-11 it looks like the low never really recovers from 9:30a EST which feels wrong...
            -FOMC day should probably be treated like the outlier.
        C-Need to understand where 25,390 comes from... this isn't a realistic number from the chart
            -There is a difference between yahoo finance and Topstep. Yahoo finance == TradingView
            -I do not know how but I fixed this.
        -Consider including an additional datapoint to set as drawdown and extension starting points. (i.e. 915a start)
        -create a data structure for storing daily results (i.e. max draw down, max profit)
        
    
    '''

# %% reference code
    
    sys.exit("End of the line, bud")

    #get RTH
    day_df = day_df.between_time("09:30", "16:00")

    #get only week days except Sunday
    mnq15["day"] = mnq15.index.date
    mnq15.day = pd.to_datetime(mnq15.day)
    mnq15["dow"] = mnq15["day"].dt.dayofweek
    mnq15_woSunday = mnq15[mnq15["dow"] != 6]
    uniqdays = mnq15_woSunday.day.unique()
