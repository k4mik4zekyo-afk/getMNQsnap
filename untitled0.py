#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 12:49:02 2026

VERSION: v0.1.0

CHANGELOG (v0.1.0)
- Timeframe-selectable execution (1min / 5min / 15min) via df_map OR resampling from base TF
- ATR(14) added
- Risk architecture added:
    risk_mode = 'atr' with stop_points = atr_mult * ATR
    risk_mode = 'fixed' with stop_points = fixed_risk_points
- Take profit added via RR multiple: tp_points = rr * stop_points
- Parameters designed for later optimization via bt.optimize()

NOTES / ASSUMPTIONS
- DataFrames must have DatetimeIndex and columns: Open, High, Low, Close (Volume optional but recommended).
- VWAP is anchored daily at session_start (default 06:30 local).
- "Opening HV candle" is chosen within first open_window_minutes after session_start
  using max Volume (fallback: max range if Volume missing).
- Fib definition: 0.0=HV Low, 1.0=HV High, 0.5=midpoint.
- Stage 2 entries only (acceptance-based). Stage 1 probe omitted (needs machine-readable “pre-defined levels”).
@author: kylesuico
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# ---------------------------
# Data utilities
# ---------------------------

TF_TO_PANDAS = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
}

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to rule (e.g., '5min', '15min')."""
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"

    out = df.resample(rule).agg(agg).dropna()
    return out

def select_timeframe_df(
    df_map: dict[str, pd.DataFrame] | None,
    timeframe: str,
    base_df: pd.DataFrame | None = None,
    base_timeframe: str | None = None,
) -> pd.DataFrame:
    """
    Choose df for timeframe.
    - If df_map contains timeframe, return it.
    - Else resample base_df (must provide base_timeframe).
    """
    if timeframe not in TF_TO_PANDAS:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {list(TF_TO_PANDAS.keys())}")

    if df_map is not None and timeframe in df_map:
        df = df_map[timeframe].copy()
        return df.sort_index()

    if base_df is None or base_timeframe is None:
        raise ValueError("If df_map doesn't have the timeframe, provide base_df and base_timeframe for resampling.")

    base_rule = TF_TO_PANDAS[base_timeframe]
    target_rule = TF_TO_PANDAS[timeframe]

    # If already at desired tf
    if base_rule == target_rule:
        return base_df.copy().sort_index()

    df = base_df.copy().sort_index()
    out = resample_ohlcv(df, target_rule)
    return out


# ---------------------------
# Indicators / Features
# ---------------------------

def add_atr(df: pd.DataFrame, length: int = 14, wilder: bool = True) -> pd.DataFrame:
    """
    Adds ATR column.
    - Wilder ATR uses EMA(alpha=1/length) on TR.
    - Simple ATR uses rolling mean of TR.
    """
    out = df.copy()
    h, l, c = out["High"], out["Low"], out["Close"]
    prev_c = c.shift(1)

    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    if wilder:
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
    else:
        atr = tr.rolling(length).mean()

    out["ATR"] = atr
    return out


def add_anchored_vwap_bands(
    df: pd.DataFrame,
    session_start: str = "06:30",
    vol_col: str = "Volume",
) -> pd.DataFrame:
    """
    Adds:
      vwap, vwap_sigma, vwap_u1, vwap_l1
    VWAP anchored daily at session_start.
    sigma computed as running weighted sqrt(E[p^2] - (E[p])^2) using typical price.
    """
    out = df.copy()

    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["_typical"] = typical

    if vol_col not in out.columns:
        out[vol_col] = 1.0

    start_t = pd.to_datetime(session_start).time()

    vwap = np.full(len(out), np.nan, dtype=float)
    sigma = np.full(len(out), np.nan, dtype=float)

    # group by calendar date
    dates = pd.Series(out.index.date, index=out.index)
    for d, idx in dates.groupby(dates):
        pos = out.index.get_indexer(idx.index)
        day_idx = out.index[pos]
        mask = day_idx.time >= start_t
        pos2 = pos[mask]
        if len(pos2) == 0:
            continue

        p = out.iloc[pos2]["_typical"].astype(float).values
        w = out.iloc[pos2][vol_col].astype(float).values

        cw = np.cumsum(w)
        cwp = np.cumsum(w * p)
        mean = cwp / np.where(cw == 0, np.nan, cw)

        cwp2 = np.cumsum(w * p * p)
        mean2 = cwp2 / np.where(cw == 0, np.nan, cw)

        var = np.maximum(mean2 - mean * mean, 0.0)
        sd = np.sqrt(var)

        vwap[pos2] = mean
        sigma[pos2] = sd

    out["vwap"] = vwap
    out["vwap_sigma"] = sigma
    out["vwap_u1"] = out["vwap"] + out["vwap_sigma"]
    out["vwap_l1"] = out["vwap"] - out["vwap_sigma"]
    out.drop(columns=["_typical"], inplace=True)
    return out


def add_opening_hv_fib(
    df: pd.DataFrame,
    session_start: str = "06:30",
    open_window_minutes: int = 30,
    vol_col: str = "Volume",
) -> pd.DataFrame:
    """
    Finds opening HV candle in first open_window_minutes after session_start.
    Writes fib_0_0 (low), fib_0_5 (mid), fib_1_0 (high) AFTER window ends (prevents lookahead).
    """
    out = df.copy()
    start_t = pd.to_datetime(session_start).time()
    out["_d"] = out.index.date

    fib00 = np.full(len(out), np.nan)
    fib05 = np.full(len(out), np.nan)
    fib10 = np.full(len(out), np.nan)

    for d, g in out.groupby("_d"):
        day = pd.Timestamp(d)
        window_start = pd.Timestamp.combine(day, start_t)
        window_end = window_start + pd.Timedelta(minutes=open_window_minutes)

        w = g[(g.index >= window_start) & (g.index < window_end)]
        if len(w) == 0:
            continue

        if vol_col in w.columns:
            hv_row = w.loc[w[vol_col].astype(float).idxmax()]
        else:
            rng = (w["High"] - w["Low"]).astype(float)
            hv_row = w.loc[rng.idxmax()]

        lo = float(hv_row["Low"])
        hi = float(hv_row["High"])
        mid = lo + 0.5 * (hi - lo)

        apply_mask = (g.index >= window_end)
        locs = out.index.get_indexer(g.index[apply_mask])
        fib00[locs] = lo
        fib05[locs] = mid
        fib10[locs] = hi

    out["fib_0_0"] = fib00
    out["fib_0_5"] = fib05
    out["fib_1_0"] = fib10
    out.drop(columns=["_d"], inplace=True)
    return out


def add_vwap_chop_filter(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Adds vwap_crosses_N = rolling count of VWAP crossovers."""
    out = df.copy()
    above = out["Close"] > out["vwap"]
    cross = (above != above.shift(1)).astype(int)
    out[f"vwap_crosses_{lookback}"] = cross.rolling(lookback).sum()
    return out


def build_features(
    df: pd.DataFrame,
    session_start: str = "06:30",
    open_window_minutes: int = 30,
    chop_lookback: int = 10,
    atr_len: int = 14,
    atr_wilder: bool = True,
) -> pd.DataFrame:
    df2 = df.copy().sort_index()
    df2 = add_atr(df2, length=atr_len, wilder=atr_wilder)
    df2 = add_anchored_vwap_bands(df2, session_start=session_start)
    df2 = add_opening_hv_fib(df2, session_start=session_start, open_window_minutes=open_window_minutes)
    df2 = add_vwap_chop_filter(df2, lookback=chop_lookback)
    return df2


# ---------------------------
# Strategy (Stage 2 only) + Risk/Reward via SL/TP
# ---------------------------

class OpeningFibVWAP_RR_ATR(Strategy):
    # Session
    session_end = "13:00"        # PT for equities; adjust for your product
    session_start = "06:30"

    # Chop filter
    chop_lookback = 10
    max_vwap_crosses = 3

    # Acceptance / entry
    entry_mode = "pullback"      # "pullback" or "continuation"
    max_bars_after_accept = 24   # how long acceptance is “valid” for entry

    # VWAP grading / alignment
    require_vwap_alignment = True
    grade_min = "B"              # "A" or "B"
    size_A = 2
    size_B = 1

    # ATR / Risk / Reward
    atr_len = 14                 # computed in data, but keep param for optimization bookkeeping
    risk_mode = "atr"            # "atr" or "fixed"
    atr_mult = 2.0               # if risk_mode='atr'
    fixed_risk_points = 50.0     # if risk_mode='fixed'
    rr = 2.0                     # TP = rr * stop_points

    def init(self):
        self._cur_date = None
        self.accept_dir = 0     # +1 long accepted, -1 short accepted, 0 none
        self.accept_i = None

    def _grade_long(self, c, v, u1):
        if c < v: return "S"
        return "A" if c > u1 else "B"

    def _grade_short(self, c, v, l1):
        if c > v: return "S"
        return "A" if c < l1 else "B"

    def _grade_ok(self, g):
        return (g == "A") if self.grade_min == "A" else (g in ("A", "B"))

    def _stop_points(self) -> float | None:
        """Compute stop distance in POINTS (price units)."""
        atr = self.data.ATR[-1]
        if np.isnan(atr):
            return None
        if self.risk_mode == "atr":
            return float(self.atr_mult) * float(atr)
        if self.risk_mode == "fixed":
            return float(self.fixed_risk_points)
        raise ValueError("risk_mode must be 'atr' or 'fixed'")

    def next(self):
        ts = self.data.index[-1]
        d = ts.date()

        # Reset daily state
        if self._cur_date != d:
            self._cur_date = d
            self.accept_dir = 0
            self.accept_i = None

        # End-of-session flat
        if ts.time() >= pd.to_datetime(self.session_end).time():
            if self.position:
                self.position.close()
            return

        # Require fib + vwap
        fib05 = self.data.fib_0_5[-1]
        fib10 = self.data.fib_1_0[-1]
        vwap  = self.data.vwap[-1]
        u1    = self.data.vwap_u1[-1]
        l1    = self.data.vwap_l1[-1]

        if np.isnan(fib05) or np.isnan(fib10) or np.isnan(vwap):
            return

        # Chop filter
        crosses = getattr(self.data, f"vwap_crosses_{self.chop_lookback}")[-1]
        if crosses > self.max_vwap_crosses:
            return

        # Invalidation exit: close back inside 0.5–1.0 region
        c = float(self.data.Close[-1])
        if self.position.is_long:
            if (fib05 <= c <= fib10) or (c < fib05):
                self.position.close()
                return
        if self.position.is_short:
            if (fib05 <= c <= fib10) or (c > fib10):
                self.position.close()
                return

        # One position at a time
        if self.position:
            return

        # Need prev bar for acceptance logic
        if len(self.data.Close) < 2:
            return
        c1 = float(self.data.Close[-2])
        lo = float(self.data.Low[-1])
        hi = float(self.data.High[-1])

        # Acceptance detection
        if self.accept_dir == 0:
            long_accept = ((c1 > fib10 and c > fib10) or
                           (c1 > fib10 and lo <= fib10 and c > fib10))

            short_accept = ((c1 < fib05 and c < fib05) or
                            (c1 < fib05 and hi >= fib05 and c < fib05))

            if long_accept:
                self.accept_dir = +1
                self.accept_i = len(self.data.Close) - 1
            elif short_accept:
                self.accept_dir = -1
                self.accept_i = len(self.data.Close) - 1

        # Acceptance expiry
        if self.accept_dir != 0 and self.accept_i is not None:
            if (len(self.data.Close) - 1) - self.accept_i > self.max_bars_after_accept:
                self.accept_dir = 0
                self.accept_i = None
                return

        # Risk / Reward setup
        stop_points = self._stop_points()
        if stop_points is None or stop_points <= 0:
            return

        tp_points = float(self.rr) * stop_points

        # Entry rules
        if self.accept_dir == +1:
            if self.require_vwap_alignment and c < vwap:
                return
            g = self._grade_long(c, vwap, u1)
            if not self._grade_ok(g):
                return
            size = self.size_A if g == "A" else self.size_B

            if self.entry_mode == "continuation":
                entry_ok = (c > fib10)
            else:
                entry_ok = (lo <= fib10 and c > fib10)

            if entry_ok:
                sl = c - stop_points
                tp = c + tp_points
                self.buy(size=size, sl=sl, tp=tp)

        elif self.accept_dir == -1:
            if self.require_vwap_alignment and c > vwap:
                return
            g = self._grade_short(c, vwap, l1)
            if not self._grade_ok(g):
                return
            size = self.size_A if g == "A" else self.size_B

            if self.entry_mode == "continuation":
                entry_ok = (c < fib05)
            else:
                entry_ok = (hi >= fib05 and c < fib05)

            if entry_ok:
                sl = c + stop_points
                tp = c - tp_points
                self.sell(size=size, sl=sl, tp=tp)


# ---------------------------
# Runner
# ---------------------------

def run_backtest(
    df_map: dict[str, pd.DataFrame] | None = None,
    timeframe: str = "5min",
    base_df: pd.DataFrame | None = None,
    base_timeframe: str | None = None,
    cash: float = 10_000,
    commission: float = 0.0,
    spread: float = 0.0,
    trade_on_close: bool = True,
    # Feature params
    session_start: str = "06:30",
    open_window_minutes: int = 30,
    chop_lookback: int = 10,
    atr_len: int = 14,
    atr_wilder: bool = True,
    # Strategy params (optimization-ready)
    strat_kwargs: dict | None = None,
):
    """
    strat_kwargs example:
      dict(risk_mode="atr", atr_mult=2.0, rr=2.0, fixed_risk_points=50,
           grade_min="B", max_vwap_crosses=3, entry_mode="pullback")
    """
    df = select_timeframe_df(df_map, timeframe, base_df=base_df, base_timeframe=base_timeframe)

    # Build features on chosen timeframe
    df_feat = build_features(
        df,
        session_start=session_start,
        open_window_minutes=open_window_minutes,
        chop_lookback=chop_lookback,
        atr_len=atr_len,
        atr_wilder=atr_wilder,
    )

    bt = Backtest(
        df_feat,
        OpeningFibVWAP_RR_ATR,
        cash=cash,
        commission=commission,
        spread=spread,
        trade_on_close=trade_on_close,
        exclusive_orders=True,
    )

    strat_kwargs = strat_kwargs or {}
    # keep Strategy aligned with session_start/end if you pass them
    strat_kwargs.setdefault("session_start", session_start)
    stats = bt.run(**strat_kwargs)
    return bt, stats


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Example: you already have these dataframes
    # df1, df5, df15 = ...
    # df_map = {"1min": df1, "5min": df5, "15min": df15}

    # Run default (ATR*2 stop, RR=2) on 5min
    # bt, stats = run_backtest(
    #     df_map=df_map,
    #     timeframe="5min",
    #     strat_kwargs=dict(
    #         risk_mode="atr",
    #         atr_mult=2.0,
    #         rr=2.0,
    #         grade_min="B",
    #     )
    # )
    # print(stats)
    # bt.plot()

    pass
