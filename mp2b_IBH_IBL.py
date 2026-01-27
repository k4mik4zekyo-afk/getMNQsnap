# -*- coding: utf-8 -*-
"""
Kyle Suico

Current rev: Jan 11, 1104am
Created lines to save data using parquet
Tested and can only load 4 weeks worth of data I guess yfinance counts weekends as a day :(

Previous rev: Jan 9th, 525pm
Could not finish because I was hanging with Victor
-WIP: modified get_data and built in code to get 1-minute data
    -Limitation: 8-days (really 5 days)
    -Need to try again maybe end of day

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
            #start="2025-12-30",
            #end="2025-12-31",
            interval=time_interval,
            auto_adjust=True,
        )
        
        #Note by default the TZ downloaded is UTC and not ET
        df = df.xs("MNQ=F", axis=1, level=1)
        session = "America/New_York"
        df.index = df.index.tz_convert(session)
        df['daytime'] = df.index
        df['session'] = df['daytime'].apply(label_session, interval=time_interval)
        
        return df
    
    # %% Objective 20260120: test if you can get latest data
    #Conclusion:
        #Learnings time zone taken in is EST, there will be a 10 minute delay in data acquired
        #Lastly don't forget the year is 2026!!!

    from datetime import datetime, timedelta
    
    #Top line needs testing
    recent = datetime.now() - timedelta(minutes=10)
    #recent = datetime(2026, 1, 26, 16, 15, 0)# - timedelta(minutes=60)    
    mnq1a = get_MNQ(time_interval="30m",
                    start_time="2026-1-19",
                    end_time=recent)
    print(mnq1a.tail(20))
    #"quick run for Obj 20260120"
    sys.exit("End of test to load a specific time for data")

    # %% Start of getting 1min data for one month
    mnq1a = get_MNQ(time_interval="1m",
                    start_time="2025-12-2",
                    end_time="2025-12-7")
    
    mnq1b = get_MNQ(time_interval="1m",
                    start_time="2025-12-13",
                    end_time="2025-12-15")
    
    mnq1c = get_MNQ(time_interval="1m",
                    start_time="2025-12-15",
                    end_time="2025-12-23")
    
    mnq1d = get_MNQ(time_interval="1m",
                    start_time="2025-12-23",
                    end_time="2025-12-31")
    
    mnq1e = get_MNQ(time_interval="1m",
                    start_time="2025-12-31",
                    end_time="2026-1-7")
    
    mnq1f = get_MNQ(time_interval="1m",
                    start_time="2026-1-7",
                    end_time="2026-1-10")
    
    #
    #Note: Only 8 days worth of granularity is allowed probably using calendar days?
    #Only Dec 31 to Jan 8th works
    #Jan 8th isn't inclusive in my dataset
        
    df = get_MNQ(time_interval="30m")
    mnq15 = get_MNQ(time_interval="15m")
    mnq5 = get_MNQ(time_interval="5m")

    
    # %% Start of Q2 - Load Initial Balance table
    def getIB(dataframe):
        ib_sub = dataframe.between_time("09:30", "10:00")
    
        initial_balance = ib_sub.groupby(ib_sub.index.date).agg(
            IBH= ('High', 'max'),
            IBL= ('Low', 'min'),
            volume= ('Volume', 'sum')
        )
        initial_balance['range'] = initial_balance['IBH']-initial_balance['IBL']
        
        #Create volume pace by dividing the previous day's IB volume
        initial_balance["volpace"] = initial_balance["volume"] / initial_balance["volume"].shift(1)
        
        #First merging data to do this in a vectorized format
        initial_balance.index = pd.to_datetime(initial_balance.index)
        initial_balance["day"] = initial_balance.index.date
        return initial_balance
    
    # initial_balance = getIB(df)    
    # print("Initial balance table:\n")
    # print(initial_balance)
    # print("\nEnd of Initial balance calculation\n")
    
    #Simple test for getting IB
    initial_balance = getIB(mnq1f)    
    print("Initial balance table:\n")
    print(initial_balance)
    print("\nEnd of Initial balance calculation\n")
    
    #Preparation before merging - need to flatten df
    df["day"] = df.index.date
    df["daytime"] = df.index

    #Combine all into one object
    combined=pd.concat([mnq1b, mnq1c, mnq1d, mnq1e, mnq1f], axis=0)
    combined.to_parquet("Dec132025-Jan92026.parquet")

    sys.exit("This is the end of the IB line, bud")


    #Start of IBH/IBL Analytics
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
        final = ib_merged.query('session == "1. morning session" | session == "2. afternoon session"')
        #final = final[final['daytime'].dt.time >= time(10,0)] #removed since I want the full data
        return final
    
    #Get summary statistics for each day
    MNQ_ib_merge0 = mergeIBnAsset(initial_balance, df)
    MNQ_ib_merge = MNQ_ib_merge0
    MNQ_ib_merge0 = MNQ_ib_merge0[(MNQ_ib_merge0['daytime'].dt.time >= time(9,30)) &
                                 (MNQ_ib_merge0['daytime'].dt.time <= time(15,59))]
    openClose = MNQ_ib_merge0.groupby(MNQ_ib_merge0.day).agg(
        Open = ('Open', 'first'),
        Close = ('Close', 'last'))
    openClose['direction'] = openClose.Close-openClose.Open
    #print(openClose)

    # %%% Q2.1 IB Touch Detection - Leave commented out because the next section
    # the entirety of metrics
    '''
    a = MNQ_ib_merge.groupby(MNQ_ib_merge.day).agg(
            touchesIBH = ('breakIBH', 'any'),
            whenTouchesIBH = ('breakIBH', 'idxmax'),
            countTouchesIBH = ('breakIBH', 'sum'),
            touchesIBL = ('breakIBL', 'any'),
            whenTouchesIBL = ('breakIBL', 'idxmax'),
            countTouchesIBL = ('breakIBL', 'sum'),
            countInRange = ('inRange', 'sum')
        )
    #print(a)
    '''

    # %%% Q2.2 Break vs. Hold
    '''
    Goal: Did price actually break the IB?
    Distinguish between probing vs breaking.
    Dec 10th note: Commented out to suppress variables AFAIK this code is not used.
    Do not comment this out need to untangle what is causing all these messages.
    '''
    
    MNQ_ib_merge = MNQ_ib_merge[MNQ_ib_merge['daytime'].dt.time >= time(10,1)]
    b1 = MNQ_ib_merge
    b2 = b1.groupby(b1.day).agg(
            countInRange = ('inRange', 'sum'),
            touchesIBH = ('breakIBH', 'any'),
            whenTouchesIBH = ('breakIBH', 'idxmax'),
            countTouchesIBH = ('breakIBH', 'sum'),
            closesIBH = ('closedAboveIBH', 'any'),
            countClosesAboveIBH = ('closedAboveIBH', 'sum'),
            touchesIBL = ('breakIBL', 'any'),
            whenTouchesIBL = ('breakIBL', 'idxmax'),
            countTouchesIBL = ('breakIBL', 'sum'),
            closesIBL = ('closedBelowIBL', 'any'),
            countClosesBelowIBL = ('closedBelowIBL', 'sum')
        )
    b2.whenTouchesIBH[b2['touchesIBH'] == False] = None
    b2.whenTouchesIBL[b2['touchesIBL'] == False] = None
    #Continues below and gets merged with submerge

    # %%% Q2.3 Close-based Confirmation
    '''
    Goal: Did price ACCEPT beyond IB? - development code for the interactions of bars
            
    '''
    s1 = MNQ_ib_merge.High < MNQ_ib_merge.IBH
    s2 = MNQ_ib_merge.Low > MNQ_ib_merge.IBL
    #(s1) & (s2)
    s3= (MNQ_ib_merge.IBL > MNQ_ib_merge.High) | (MNQ_ib_merge.IBH < MNQ_ib_merge.Low)
    s4 = (MNQ_ib_merge.Close > MNQ_ib_merge.IBH) | (MNQ_ib_merge.Close < MNQ_ib_merge.IBL)
    
    
    conditions = [(s1) & (s2), s3, s4]
    choices = ["in range", "outside", "touching-outside"]
    default_choice = "touching-in range"
    result = np.select(conditions, choices, default=default_choice)
    #print(result)
    MNQ_ib_merge['bar_interaction'] = result
    
    
    # %%% Q2.4 Re-entry & Failure detection
    '''
    Goal: Did the break fail?
    This is classic failed auction logic.
    
    Milestones:
    Break IB → return inside IB
    Time between break and re-entry
    Side that ultimately wins
    How far price traveled before failing
    
    Outputs:
    failed_IBH
    failed_IBL
    reentry_time
    failure_distance
    
    ✅ This sets up Quarter 3 classification cleanly
    '''
    
    def whichBreakside(day_df):
        t_ibh = day_df.loc[day_df["breakIBH"]].index.min()
        t_ibl = day_df.loc[day_df["breakIBL"]].index.min()
    
        if pd.isna(t_ibh) and pd.isna(t_ibl):
            return "None"
        if pd.isna(t_ibl) or t_ibh < t_ibl:
            return "IBH"
        return "IBL"

    #test_breakside = MNQ_ib_merge.groupby("day").apply(whichBreakside)
    #print(test_breakside) #IBL, None, IBL, IBH...
    #sys.exit()
    
    '''
    def isFailedAuction(after, break_side):
        # To do:
        #     -Determine if failed auction:
        #         shallow failed break and stays within range
        #         deep failed break (full mean reversion to the other extreme)
        #     -If no failed auction, then does price break out of range and continue to run?
            
        
        # Parameters
        # ----------
        # after : Subset dataframe (MNQ_ib_merge) for a single data
        
        # Returns
        # -------
        # Series of True/False for failure
        
        if break_side == "IBH":
            back_inside = after["Low"] <= after["IBH"]
            back_outside = after["Close"] > after["IBH"]
        else:
            back_inside = after["High"] >= after["IBL"]
            back_outside = after["Close"] < after["IBL"]

        print("\nBreakside:", break_side)
        print("\nback_inside:", back_inside)
        print("\nback_outside:", back_outside)
        
        if back_inside.any():
            t_inside = after.loc[back_inside].index.min()
            truly_failed = not back_outside.loc[t_inside:].any()
        else:
            truly_failed = False
        
        print("\nAny():")
        print(back_inside.any())
        return(truly_failed)
    
    
   
    commented out - because I am working on the output table - likely need to revisit in Q3
    #z0 = date(2025, 11, 12)
    #case study for a break that stayed within range - need to validate with images.
    #11-12 is a case where price breaks but bounces of at least 2x but ultimately stays within range by EOD
    
    #case study for today Dec 2 - price breaks top then bottom in one bar.
        #is whichBreakside would see the top break first... this is a corner case that I did not consider though...
    #z = MNQ_ib_merge[MNQ_ib_merge.day == z0]
    #zz = first_break[first_break.index == z0].item()

    #zzz = isFailedAuction(z, zz)
    
    #print("Did the auction fail?")
    #print(zzz)
    
    summary = MNQ_ib_merge.groupby("day").agg(
        touchesIBH = ('breakIBH', 'any'),
        closesIBH = ('closedAboveIBH', 'any'),
        touchesIBL = ('breakIBL', 'any'),
        closesIBL = ('closedBelowIBL', 'any')        
        )
    
    #print(summary)
    '''
    
    #Create a subset called submerge of summary stats of each day
    submerge = MNQ_ib_merge.filter(items=["IBH", "IBL"])
    submerge = submerge.drop_duplicates()
    
    submerge['IBmid'] = submerge.mean(axis=1)
    submerge['day'] = submerge.index.date
    
    summary_df = openClose.merge(submerge, on="day")
    
    #merge b2 (interaction metrics) with day metrics
    summary_df = summary_df.merge(b2,
                   on="day")
    summary_df.index = summary_df.day
    
    #Create series called breakside to join with submerge
    whichBreaksideSeries = MNQ_ib_merge.groupby("day").apply(whichBreakside)
    whichBreaksideSeries.name = "breakside"
    
    #Problem: Need to understand why whichBreaksideSeries if None then it doesn't join to summary table
    
    #Original:
    summary_df2 = summary_df.join(whichBreaksideSeries,
                                 how = 'outer')
    print(summary_df2)

    '''
    Extension Outside IB (Initiative Pressure)
            Column	Type	Short description
        max_extension_outside	float	Max distance price moved beyond IB
            High-IBH
            IBL-Low
        time_outside_IB	float	Total minutes price spent outside IB
            Count: total close outside
        closed_outside_IB	bool	End-of-day close outside IB
            get end of day close - did it stay outside IB?
    '''
    
    #Filter for specific times in MNQ_ib_merge df
    MNQ_ib_merge_afterIB_beforeclose = MNQ_ib_merge[MNQ_ib_merge['daytime'].dt.time <= time(15,59)]

    #Need day in date time objectpd to date
    #pd.to_datetime(MNQ_ib_merge_afterIB_beforeclose.day).dt.date
    MNQ_ib_merge_afterIB_beforeclose.day = pd.to_datetime(MNQ_ib_merge_afterIB_beforeclose.day).dt.day

    def calcDistanceFromIB(df, day, initial_breakside):
        #Max penetration definition needs to be smarter. It must take into account the time in when IBL/IBH
        #is broken then look for highs from there (e.g. 12-8 maxP is 42 and not 67)
        #filter df down by the date interested
        temp_df = df[df.daytime.dt.date == day]
        
        #Create distanceIB which will be used to return maxE/maxP
        dict = {'High.diff': temp_df.High - temp_df.IBH,
                'HL.diff': temp_df.Low - temp_df.IBH,
                'Low.diff': temp_df.IBL - temp_df.Low,
                'LH.diff':temp_df.High - temp_df.IBL
                }
        
        distanceIB = pd.DataFrame(data = dict, columns = ['High.diff',
                                                 'HL.diff',
                                                 'Low.diff',
                                                 'LH.diff'
                                                 ])
        
        if initial_breakside == "IBH":
            #distanceIB = calcDistanceFromIB(MNQ_ib_merge_afterIB_beforeclose)
            dist_table = distanceIB.groupby(distanceIB.index.date).agg(
                maxExtension = ('High.diff', 'max'),
                maxPenetration = ('HL.diff', 'max')
                )
        elif initial_breakside == "IBL":
            dist_table = distanceIB.groupby(distanceIB.index.date).agg(
                maxExtension = ('Low.diff', 'max'),
                maxPenetration = ('LH.diff', 'max')
                )
        else:
            dist_table = None   
            
        return(dist_table)

    whichBreaksideSeries.index = pd.to_datetime(whichBreaksideSeries.index)
    
    def createExtPen_df(input_df, break_direction):
        '''
            Loop day in summary table
            get breakside value pop into calcDistanceFromIB
                -need to modify so it does the calculation for a given day
            spit out maxE/maxP
            populate a dataframe
        '''
        #Create a dataframe to populate extension and penetration
        frames = []
        for day in break_direction.index:
            maxEP = calcDistanceFromIB(input_df,
                                    day.date(),
                                    break_direction[break_direction.index.date == day.date()].item()
                                    )
            frames.append(maxEP)
        
        new_df = pd.concat(frames, ignore_index=False)
        return new_df
    
    #print("\nThis is the max Extension and Penetration table:\n")
    ep_df = createExtPen_df(MNQ_ib_merge_afterIB_beforeclose, whichBreaksideSeries)
    ep_df["day"] = ep_df.index
    #print(ep_df)
    
    #Merge ep_df and summary table
    summary_df = summary_df.drop(columns=["day"])
    summary_df = summary_df.reset_index()
    
    summary_df = summary_df.merge(ep_df,
                     on="day")
   
    print("\nSummary table for Quarter 2:\n")
    print(summary_df)
    
    
    #Time spent outside of IB (incl. candles that touch IB)
    #print("\n# of minutes closing outside IB:\n")
    minutes = 30
    def timeOutsideIB(df):
        lv = df.bar_interaction.str.contains('outside')
        count_table = df[lv].groupby(df.daytime.dt.date).agg(
            count=('day', 'count'))
        return count_table
    
    #Uncomment if you want to know the number of minutes
    #print(timeOutsideIB(MNQ_ib_merge_afterIB_beforeclose)*minutes)
    candlesOutsideIB = timeOutsideIB(MNQ_ib_merge_afterIB_beforeclose)
    summary_df["candlesOutsideIB"] = candlesOutsideIB

    #Add logic check for whether day closed outside IB to the summary_df    
    #Does day close outside IB?
    closed_outside_IB = (summary_df.IBL > summary_df.Close) | (summary_df.IBH < summary_df.Close)
    summary_df["closedOutsideIB"] = closed_outside_IB

    #re-arrange columns of summary df
    new_order = ['day', 'Open', 'Close', 'direction', 'breakside',
                 'IBH', 'IBL', 'IBmid',
                 'maxExtension',
                 #'maxPenetration',
                 'countInRange', 'candlesOutsideIB', 'closedOutsideIB',
                 'touchesIBH', 'whenTouchesIBH', 'countTouchesIBH',
                 'closesIBH', 'countClosesAboveIBH', 'touchesIBL', 'whenTouchesIBL',
                 'countTouchesIBL', 'closesIBL', 'countClosesBelowIBL'
                 ]
    summary_df = summary_df.reindex(columns=new_order)

    dataExport = False
    
    if dataExport:
        import os
        vars_list = list(locals().keys())
    
        #Clean up script for saving variables
        a = ['df', 'mnq5', 'mnq15', 'initial_balance', 'summary_df2',
             'whichBreaksideSeries', 'MNQ_ib_merge']
        keep = set(a) | {"a"}  # also keep the list 'a' itself (optional)
        
        for name in list(globals().keys()):
            if name not in keep and not name.startswith("__"):
                del globals()[name]

    import sys
    sys.exit("End of Quarter 2")


    #Need to have the algo detect for failure e.g. if IBH is the break side, then
    #look for close above IBH, otherwise if 'touching' then check for re-entry,
    #if remaining bars are within range then market is looking for balance,
    #if bars end up in other direction then this is a reversal
    #if bars end up 'outside' and in the same direction of break then a market event occurred and there's imbalance
    
    # %%% Q2.5 Automated Data analysis
    
    # counts by direction sign (pos/neg) and breakside (IBL/IBH)
    out = (
        summary_df2
        .loc[summary_df2["direction"].ne(0) & summary_df2["breakside"].isin(["IBL", "IBH"])]
        .assign(direction_sign=lambda d: d["direction"].gt(0).map({True: "positive", False: "negative"}))
        .groupby(["direction_sign", "breakside"])
        .size()
        .unstack("breakside", fill_value=0)
    )
    
    out_counts = out.copy()
    out_counts["Total"] = out_counts.sum(axis=1)
    
    out_pct = out.div(out.sum(axis=1), axis=0).mul(100).round(1)
    out_pct["Total"] = 100.0
    
    print("Totals:\n")
    print(out_counts, sep="\n")
    print("Percentages:\n")
    print(out_pct)
    
    # counts of days where both touch vs neither touches
    both_touch = (summary_df2["touchesIBH"] & summary_df2["touchesIBL"]).sum()
    both_mask = summary_df2["touchesIBH"] & summary_df2["touchesIBL"]
    both_days = summary_df2.loc[both_mask]
    #both_days

    
    neither_touch = (~summary_df2["touchesIBH"] & ~summary_df2["touchesIBL"]).sum()
    neither_mask = ~summary_df2["touchesIBH"] & ~summary_df2["touchesIBL"]
    neither_days = summary_df2.loc[neither_mask]
    #neither_days

    print("How many days both extremes touch:", both_touch)
    print("How many days neither extreme touches:", neither_touch)

    #Interactions test
    import pandas as pd

    touch_states = {"touching-outside", "touching-in range"}
    
    df = MNQ_ib_merge.copy()
    
    # get a datetime series for the index named "daytime" (works for DatetimeIndex or MultiIndex)
    dt = df.index if isinstance(df.index, pd.DatetimeIndex) else df.index.get_level_values("daytime")
    df["day"] = pd.to_datetime(dt).normalize()
    
    outside = df["bar_interaction"].eq("outside")
    touch   = df["bar_interaction"].isin(touch_states)
    
    # True only BEFORE the first "outside" of that day (the "outside" bar itself is excluded)
    before_first_outside = outside.groupby(df["day"]).cumsum().eq(0)
    touches_before_outside_by_day = (touch & before_first_outside).groupby(df["day"]).sum()
    #touches_before_outside_by_day

    touches2 = touches_before_outside_by_day.copy()
    whichBreaksideSeries.index = pd.to_datetime(whichBreaksideSeries.index).date
    touches2.index = pd.to_datetime(touches_before_outside_by_day.index).date

    df3 = pd.concat([whichBreaksideSeries, touches2], axis=1)
    #df3 = pd.concat([whichBreaksideSeries, touches_before_outside_by_day], axis=1)
    df3.columns = ["Breakside", "Number of touches before outside"]           # optional: name the columns
    print("Number of touches before breaking out:")
    print(df3)

    
    # %% Q3 - Planning
    '''
    Tech debt from Q2:
        Need to fix Max Penetration definition and candles outisde IB column
    
    Other to do:
        Done-Need to lock variables from Q2 so there is minimal re-factoring
        Re-factoring before going on? Clean up intermediate variables.
        Done-Need to create a helper function to get lower time frames if 30m then get the 15m/5m
        
    
    '''
    
    # %%% Q3 - Part 1
    '''
    ✅ Q3.1 — Structural Acceptance vs Rejection
        Goal: Decide whether the auction accepted value away from IB.
        
        Planning:
        Go down a single time frame
        Go down to any time frame (stretch)
        

    '''
    # ---- Initialize structures ----
    daily_record = []

    #initial_balance.day = pd.to_datetime(initial_balance.day)
    #initial_balance.IBH.item()
    #date = '2025-11-21 00:00:00'
    #initial_balance.day.dt.date == date.date()

    #--- Original function ---
    def analyzeCandles (df, IB, date):
        
        #Initialize variables and flags
        probe_H_active = False
        break_H_active = False
        probe_L_active = False
        break_L_active = False
        index_of_ext = None
        record = {
                        "date": None,
                        "IBH": {
                            "probe_indices": [],
                            "break_indices": [],
                            "extension_index": None,
                            "extension_size": None
                        },
                        "IBL": {
                            "probe_indices": [],
                            "break_indices": [],
                            "extension_index": None,
                            "extension_size": None
                        },
                        "events": [],
                        "overall": {
                            "extension_side": None,
                            "extension_index": None,
                            "extension_size": None
                        }
                    }
        
        #Filter for date and look only at RTH
        a = df[df.day == date]
        a = a.between_time("09:30", "16:00")
        
        #assign IB-X to a single var.
        IBH = IB[IB.day == date.date()].IBH.item()
        IBL = IB[IB.day == date.date()].IBL.item()
        record["date"] = date
        
        # ---- Determine largest extension direction ----
        up_extension  = max(a.High) - IBH
        down_extension = IBL - min(a.Low)
        
        if up_extension > down_extension:
            extreme = max(a.High)
        else:
            extreme = min(a.Low)
        
        #define convenience variables
        for i in a.index:
            high = a.High[i]
            low = a.Low[i]
            close = a.Close[i]
            
            # ---- IBH events ----        
            # -- Price Wicks
            if high > IBH and close < IBH:
                # --- Price wicks ABOVE IBH
                if probe_H_active == False:
                    # New distinct probe event
                    record["IBH"]["probe_indices"].append(i)
                    probe_H_active = True
                # --- Price back INSIDE range, so reset flag
                else:
                    probe_H_active = False
            else:
                probe_H_active = False
            
            if close > IBH:
                if break_H_active == False:
                    # New distinct break event
                    record["IBH"]["break_indices"].append(i)
                    break_H_active = True
                else:
                    break_H_active = False
            else:
                break_H_active = False
            
            # ---- IBL events ----
            # --- Price wicks BELOW IBL
            if low < IBL and close > IBL:
                if probe_L_active == False:
                    record["IBL"]["probe_indices"].append(i)
                    probe_L_active = True
                else:
                    probe_L_active = False
            else:
                probe_L_active = False
                    
            if close < IBL:
                if break_L_active == False:
                    record["IBL"]["break_indices"].append(i)
                    break_L_active = True
                else:
                    break_L_active = False
            else:
                break_L_active = False
                
            #print("\nFinished with candle #: ", i)
            #does high == max(a.High) or low == min(a.Low)
            if high == extreme or low == extreme:
                index_of_ext = i
        
        if up_extension > down_extension:
            record["overall"]["extension_side"] = "up"
            record["overall"]["extension_size"] = up_extension
            record["overall"]["extension_index"] = index_of_ext
            #record["IBH"]["extension_index"] = index_of_max_high
            #record["IBH"]["extension_size"] = up_extension
        else:
            record["overall"]["extension_side"] = "down"
            record["overall"]["extension_size"] = down_extension
            record["overall"]["extension_index"] = index_of_ext
            #record["IBL"]["extension_index"] = index_of_min_low
            #record["IBL"]["extension_size"] = down_extension
        
        
        
        return record
     

    # %% START OF State Machine
    
    #Define helper functions
    def init_record(date):
        return {
            "date": date,
            "overall": {"extension_side": None, "extension_index": None, "extension_size": None},
            "events": []
        }
    
    def init_state():
        return {
            "probe_active": {"IBH": False, "IBL": False},
            "break_seen": {"IBH": False, "IBL": False},
            "accept_pending": {"IBH": False, "IBL": False}
        }

    def emit_event(record, event_type, side, i, price):
        '''This function will save events to record'''
        record["events"].append({
            "type": event_type,
            "side": side,
            "index": i,
            "price": price
        })

    #The real meat of the state machine (analyzeCandles)
    def update_state(state, record, ib, bar, i):
        #Check IBH for probing and breaking
        upper = detect_upper(bar,
                             ib,
                             state)
        if upper[0] and upper[1] != "accept pending":
            if upper[1] == "probe":
                emit_event(record, "probe", "IBH", i, bar.Close)
                state["probe_active"]["IBH"] = True
            elif upper[1] == "break":
                emit_event(record, "break", "IBH", i, bar.Close)
                state["break_seen"]["IBH"] = True
        elif not upper[0] and upper[1] != "monitoring":
            #This condition would only be met if is within range or on the other side
            state["probe_active"]["IBH"] = False
            state["break_seen"]["IBH"] = False
        elif upper[0] and upper[1] == "accept pending" and not state["accept_pending"]["IBH"]:
            emit_event(record, "Accept/Reject?", "IBH", i, bar.Close)
            state["accept_pending"]["IBH"] = True
        
        #Check IBL for probing and breaking
        lower = detect_lower(bar,
                             ib,
                             state)
        if lower[0] and lower[1] != "accept pending":
            if lower[1] == "probe":
                emit_event(record, "probe", "IBL", i, bar.Close)
                state["probe_active"]["IBL"] = True
            elif lower[1] == "break":
                emit_event(record, "break", "IBL", i, bar.Close)
                state["break_seen"]["IBL"] = True
        elif not lower[0] and lower[1] != "monitoring":
            #This condition would only be met if is within range or on the other side
            state["probe_active"]["IBL"] = False
            state["break_seen"]["IBL"] = False
        elif lower[0] and lower[1] == "accept pending" and not state["accept_pending"]["IBL"]:
            emit_event(record, "Accept/Reject?", "IBL", i, bar.Close)
            state["accept_pending"]["IBL"] = True
        
        # --- Acceptance/rejection detection ---
            #starts when probing/break occurs
            #check if next bar closes by checking break seen
            #count closes outside
            #calculate extension (if it crosses threshold)
            #acceptance check (also rejection check)
        return state

    def detect_lower(bar, ib, state):
        if (state["probe_active"]["IBL"] and state["break_seen"]["IBL"]):
            return (True, "accept pending")
        elif bar.Low < ib.IBL.item() and bar.Close > ib.IBL.item() and not state["probe_active"]["IBL"]:
            return (True, "probe")
        elif bar.Close < ib.IBL.item() and not state["break_seen"]["IBL"]:
            return (True, "break")
        elif (state["probe_active"]["IBL"] or state["break_seen"]["IBL"]):
            return (False, "monitoring")
        else:
            return (False, "in range")

    def detect_upper(bar, ib, state):
        if (state["probe_active"]["IBH"] and state["break_seen"]["IBH"]):
            return (True, "accept pending")
        elif bar.High > ib.IBH.item() and bar.Close < ib.IBH.item() and not state["probe_active"]["IBH"]:
            return (True, "probe")
        elif bar.Close > ib.IBH.item() and not state["break_seen"]["IBH"]:
            return (True, "break")
        elif (state["probe_active"]["IBH"] or state["break_seen"]["IBH"]):
            return (False, "monitoring")
        else:
            return (False, "in range")

    #My loop but put into a function
    def process_day(day_df, date):
        global initial_balance
        record = init_record(date)
        state = init_state()
        
        ib = initial_balance.loc[date]
        #print(ib)
        day_df = day_df[day_df.day == date] 
        day_df = day_df.between_time("09:30", "16:00")
        #print(day_df)
        for i in day_df.index:
            bar = day_df.loc[i]
            state = update_state(state, record, ib, bar, i)
    
        # determine extensions here
        return record

    #test of process day
    # date = pd.Timestamp(2025, 12, 3)
    # zzz = process_day(mnq15_woSunday, date)
    # print(zzz)

#Just for reference:
    #define convenience variables
    for i in a.index:
        high = a.High[i]
        low = a.Low[i]
        close = a.Close[i]

    #Set-up before the loop:    
    mnq15["day"] = mnq15.index.date
    mnq15.day = pd.to_datetime(mnq15.day)
    mnq15["dow"] = mnq15["day"].dt.dayofweek
    mnq15_woSunday = mnq15[mnq15["dow"] != 6]
    uniqdays = mnq15_woSunday.day.unique()
    
    for date in uniqdays:
        record = analyzeCandles(mnq15_woSunday, initial_balance, date)
        daily_record.append(record)
        
    #print(len(daily_record))
    print(daily_record[14]["overall"])
    
    '''
        Code to add:
            
            1-Filter for only looking at the NY session
            
            2-Finish the loop by appending record to daily_records
            # ---- Save daily record ----
            daily_records.append(record)
    
        An upgrade of this would be to make a state machine:
            function update_state(state, bar, IBH, IBL):

        # 1. Handle IBH probe and break
        check_probe_H()
        check_break_H()
        check_accept_H()
        check_reject_H()
    
        # 2. Handle IBL probe and break
        check_probe_L()
        check_break_L()
        check_accept_L()
        check_reject_L()
    
        # 3. Optional: check rotations or extensions

    
        '''

# %% Old code
    '''
    TO DO: Need to research how to add in arguments into '.agg'
    
    daily_extremes = df2.groupby(df2.session_day).agg(
        daily_high=('High', 'max'),
        daily_low=('Low', 'min'),
        volume=("Volume", "sum"),
        count=("dow", "count")
    )
    MNQ_ib_merge.groupby(MNQ_ib_merge.day).agg(
        touchesIBH=('breakIBH', 'idxmax'))
    '''

    sys.exit("End of code")

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