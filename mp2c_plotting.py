#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:18:11 2025

@author: kylesuico

Note requires mnq15 to be loaded

"""

import mplfinance as mpf

date = pd.Timestamp(2025, 12, 8)
mnq15.daytime = pd.to_datetime(mnq15.daytime.dt.date)
day_data = mnq15[mnq15.daytime == date]

#Level one plotting just the ticker
mpf.plot(
    day_data,
    type='candle',        # candlestick
    style='yahoo',        # a nice preset style
    title=f"MNQ - {day_str}",
    ylabel='Price',
    volume=True,          # add volume panel
    ylabel_lower='Volume'
)

#Level two adding IBL/IBH
initial_balance.day = pd.to_datetime(initial_balance.day)

IBH = initial_balance[initial_balance.day == date].IBH.item()
start_time = pd.to_datetime("09:30").time()
line = pd.Series(np.nan, index=day_data.index)
line[day_data.index.time >= start_time] = IBH

IBL = initial_balance[initial_balance.day == date].IBL.item()
line2 = pd.Series(np.nan, index=day_data.index)
line2[day_data.index.time >= start_time] = IBL

apds = [mpf.make_addplot(line, color='green', width=2),
        mpf.make_addplot(line2, color='red', width=2)]
mpf.plot(day_data, type='candle', style='yahoo', volume=True, addplot=apds)

#Level three zooming in on only pre-market to power hour
session_data = day_data.between_time("08:30", "16:00")

# define your key times (clock times)
session_open_time = pd.to_datetime("09:30").time()
power_hour_time   = pd.to_datetime("15:00").time()

# get unique days in your session_data
days = session_data.index.normalize().unique()

# build a list of full timestamps for vlines
vlines = []
for d in days:
    vlines.append(d + pd.Timedelta(hours=session_open_time.hour, 
                                   minutes=session_open_time.minute))
    vlines.append(d + pd.Timedelta(hours=power_hour_time.hour, 
                                   minutes=power_hour_time.minute))

# mplfinance vlines config
vline_kwargs = dict(
    vlines=vlines,
    linewidths=0.5,
    colors="grey",
    linestyle="--",
)

line = pd.Series(np.nan, index=session_data.index)
line[session_data.index.time >= start_time] = IBH
line2 = pd.Series(np.nan, index=session_data.index)
line2[session_data.index.time >= start_time] = IBL

apds = [mpf.make_addplot(line, color='green', width=2),
        mpf.make_addplot(line2, color='red', width=2)]

mpf.plot(
    session_data,
    type="candle",
    style="yahoo",
    volume=True,
    vlines=vline_kwargs,
    title="Pre-market → Power Hour with session markers",
    addplot=apds
)