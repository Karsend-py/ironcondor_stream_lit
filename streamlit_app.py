
# ============================================
# Iron Condor Backtester (Streamlit)
# EDITED for Daily ↔ Minute toggle, scalability, and performance
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple
import json

# =========================================================
# Page config & Global CSS (dark app look)
# =========================================================
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")
st.markdown(""" """, unsafe_allow_html=True)

# Header (no tabs, single page)
st.markdown("""
<div style="font-size:20px; font-weight:600;">Iron Condor Backtester</div>
<div style="opacity:0.8;">Options analytics & strategy management</div>
<div style="opacity:0.6;">v1.3</div>
""", unsafe_allow_html=True)  # CHANGED: version bump for tracking

# =========================================================
# HELPERS
# =========================================================

def parse_blackout_txt(file) -> List[pd.Timestamp]:
    """Parse blackout dates .txt with one date per line, allow comments (#)."""
    if file is None:
        return []
    try:
        content = file.read()
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1")
        loaded: List[pd.Timestamp] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                dt = pd.to_datetime(s)
                loaded.append(pd.Timestamp(dt).normalize())
            except Exception:
                # Ignore unparseable lines
                pass
        return loaded
    except Exception:
        return []


def build_blackout_mask(index: pd.DatetimeIndex,
                        earnings: List[pd.Timestamp],
                        pre: int, post: int) -> np.ndarray:
    """Vectorized blackout mask marking blackout intervals around each earnings date."""
    if not earnings or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    idx_dates = index.normalize()
    mask = np.zeros(len(index), dtype=bool)
    # CHANGED: accumulate blackout ranges across *all* earnings (bug fix)
    for e in earnings:
        e_n = pd.Timestamp(e).normalize()
        start = e_n - pd.Timedelta(days=int(pre))
        end = e_n + pd.Timedelta(days=int(post))
        mask |= (idx_dates >= start) & (idx_dates <= end)
    return mask


def infer_bars_per_day(index: pd.DatetimeIndex) -> int:
    """Infer bars per day by grouping timestamps by date and taking the median count."""
    if len(index) == 0:
        return 1
    s = pd.Series(1, index=index).groupby(index.normalize()).sum()
    bpd = int(float(s.median())) if len(s) > 0 else 1
    return max(bpd, 1)


def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling for plotting large time series."""
    n = len(x)
    if threshold >= n or threshold <= 0:
        return x, y
    sampled_x = [x[0]]
    sampled_y = [y[0]]
    every = (n - 2) / (threshold - 2)
    a = 0
    for i in range(0, threshold - 2):
        avg_range_start = int(np.floor((i + 1) * every)) + 1
        avg_range_end = int(np.floor((i + 2) * every)) + 1
        avg_range_end = min(avg_range_end, n)
        avg_x = np.mean(x[avg_range_start:avg_range_end])
        avg_y = np.mean(y[avg_range_start:avg_range_end])
        range_offs = int(np.floor(i * every)) + 1
        range_to = int(np.floor((i + 1) * every)) + 1
        range_to = min(range_to, n - 1)

        max_area = -1.0
        next_a = None
        ax = x[a]; ay = y[a]
        for j in range(range_offs, range_to + 1):
            area = abs((ax - avg_x) * (y[j] - ay) - (ay - avg_y) * (x[j] - ax))
            if area > max_area:
                max_area = area
                next_a = j
        if next_a is None:
            next_a = range_offs
        sampled_x.append(x[next_a])
        sampled_y.append(y[next_a])
        a = next_a
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])
    return np.array(sampled_x), np.array(sampled_y)


# =========================================================
# INDICATORS (Daily unchanged + Intraday mode-aware)
# =========================================================

def compute_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Original daily indicator logic unchanged."""
    df = df.copy()

    # Bollinger bands on VWAP (20, 2σ)
    ma = df["vwap"].rolling(20).mean()
    std = df["vwap"].rolling(20).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    # RSI (14) on close
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

    # ADX (14)
    up_move = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().bfill().ffill()

    # SMA20 on VWAP
    df["vwap_sma20"] = df["vwap"].rolling(20).mean()

    # DI lines for "ADX + DI" trend method
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Bollinger width & tightening flag
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(20, min_periods=5).median())
    ).fillna(False)

    # Historical Volatility (21-d rolling, annualized)
    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(21).std(ddof=0) * np.sqrt(252) * 100).bfill().ffill()

    return df


def compute_indicators_hourly(df: pd.DataFrame, bars_per_day: int, days_per_year: int) -> pd.DataFrame:
    """Intraday indicator logic: scale windows by bars_per_day and annualization by bars_per_day*days_per_year."""
    df = df.copy()
    bpd = max(int(bars_per_day), 1)
    bb_n = 20 * bpd
    rsi_n = 14 * bpd
    adx_n = 14 * bpd
    hv_n = 21 * bpd
    annual_factor = np.sqrt(bpd * days_per_year)

    # Bollinger bands on VWAP
    minp = max(5, int(0.2*bb_n))
    ma = df["vwap"].rolling(bb_n, min_periods=minp).mean()
    std = df["vwap"].rolling(bb_n, min_periods=minp).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    # RSI on close (use EWM alpha scaled)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha_rsi = 1 / max(rsi_n, 1)
    avg_gain = gain.ewm(alpha=alpha_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

    # ADX (use EWM alpha scaled)
    up_move = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    alpha_adx = 1 / max(adx_n, 1)
    atr = tr.ewm(alpha=alpha_adx, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha_adx, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha_adx, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(alpha=alpha_adx, adjust=False).mean().bfill().ffill()

    # VWAP SMA (scaled)
    df["vwap_sma20"] = df["vwap"].rolling(bb_n, min_periods=minp).mean()

    # DI lines for "ADX + DI" trend method
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Bollinger width & tightening flag (scaled window)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(bb_n, min_periods=minp).median())
    ).fillna(False)

    # Historical Volatility (rolling hv_n, annualized by sqrt(bpd * days_per_year))
    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(hv_n).std(ddof=0) * annual_factor * 100).bfill().ffill()

    return df


def compute_trend_flags(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == "VWAP Slope":
        df["vwap_delta"] = df["vwap"].diff()
        df["trend_up"] = df["vwap_delta"] > 0
        df["trend_down"] = df["vwap_delta"] < 0
    elif method == "VWAP vs SMA20":
        df["trend_up"] = df["vwap"] > df["vwap_sma20"]
        df["trend_down"] = df["vwap"] < df["vwap_sma20"]
    elif method == "ADX + DI":
        df["trend_up"] = (df["plus_di"] > df["minus_di"]) & (df["adx"] > 20)
        df["trend_down"] = (df["minus_di"] > df["plus_di"]) & (df["adx"] > 20)
    else:
        df["trend_up"] = False
        df["trend_down"] = False
    return df


def in_blackout(day, earnings: List[pd.Timestamp], pre: int, post: int):
    day_n = pd.Timestamp(day).normalize()
    for e in earnings:
        e_n = pd.Timestamp(e).normalize()
        if (e_n - timedelta(days=pre)) <= day_n <= e_n:
            return True
        if e_n <= day_n <= (e_n + timedelta(days=post)):
            return True
    return False


def next_friday_within(idx: pd.DatetimeIndex, start_loc: int, max_dte=5):
    end_loc = min(start_loc + max_dte, len(idx) - 1)
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            return idx[j]
    return idx[end_loc]


def next_friday_close_within(idx: pd.DatetimeIndex, start_loc: int, max_dte=5) -> pd.Timestamp:
    """For hourly data: pick the last timestamp on the first Friday within window."""
    end_loc = min(start_loc + max_dte * 24, len(idx) - 1)
    fridays = []
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            fridays.append(idx[j])
    if not fridays:
        return idx[end_loc]
    first_friday_date = pd.Timestamp(fridays[0]).normalize()
    same_day = [t for t in fridays if pd.Timestamp(t).normalize() == first_friday_date]
    return max(same_day) if same_day else fridays[0]


# ADDED: Generic intraday expiry helper (minute-safe)
def next_friday_close_within_generic(
    idx: pd.DatetimeIndex, start_loc: int, max_dte: int, bars_per_day: int
) -> pd.Timestamp:
    """
    Generic intraday expiry picker:
    - Scan forward up to max_dte * bars_per_day bars.
    - Return the *last* timestamp on the first Friday encountered.
    - If no Friday found, return the last timestamp in the window.
    """
    if len(idx) == 0:
        raise ValueError("Empty index provided to expiry helper.")
    forward = max(1, int(max_dte) * max(1, int(bars_per_day)))
    end_loc = min(start_loc + forward, len(idx) - 1)
    fridays = []
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            fridays.append(idx[j])
    if not fridays:
        return idx[end_loc]
    first_friday_date = pd.Timestamp(fridays[0]).normalize()
    same_day = [t for t in fridays if pd.Timestamp(t).normalize() == first_friday_date]
    return max(same_day) if same_day else fridays[0]


# =========================================================
# DAILY BACKTEST (optimized loop to match intraday speed)
# =========================================================
def run_backtest(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int
):
    # Preprocess (same as before)
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators & trend flags
    df = compute_indicators_daily(df)
    df = compute_trend_flags(df, trend_method)

    # Entry filters incl. HV bounds
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # CHANGED: use vectorized blackout mask (performance & correctness)
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # CHANGED: array-based loop for speed (mirrors hourly approach)
    idx = df.index
    n = len(idx)

    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)

    # Optional progress bar hook (non-blocking)
    pbar = st.session_state.get("_backtest_pbar")  # ADDED
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            v_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else v_today
            delta_today = v_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - v_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > v_today) or
                             (delta_today < 0 and current_close < v_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions (if eligible)
        if eligible_arr[i]:
            # Bias shifts strikes if enabled
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            # adaptive wing widths on trend transitions + BB tightening
            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)  # same as GUI
            expiry = next_friday_within(df.index, i, max_dte=5)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # log equity
        eq_dates.append(d)
        eq_cash.append(cash)

        # ADDED: progress updates for long runs (every ~1% or 1000 rows)
        if pbar and n > 0 and (i - last_update_i >= max(1000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    # Summary
    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# HOURLY BACKTEST (optimized, same entry/exit logic)
# =========================================================
def run_backtest_hourly(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int,
    bars_per_day: int, days_per_year: int
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators tuned for hourly
    df = compute_indicators_hourly(df, bars_per_day=bars_per_day, days_per_year=days_per_year)
    df = compute_trend_flags(df, trend_method)

    # Filters
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # Vectorized blackout mask
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # Use numpy arrays for speed
    idx = df.index
    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)
    n = len(idx)

    pbar = st.session_state.get("_backtest_pbar")  # ADDED
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            vwap_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else vwap_today
            delta_today = vwap_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - vwap_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > vwap_today) or
                             (delta_today < 0 and current_close < vwap_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement at the selected Friday close
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions (if eligible row)
        if eligible_arr[i]:
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)
            expiry = next_friday_close_within(idx, i, max_dte=5)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # log equity
        eq_dates.append(d)
        eq_cash.append(cash)

        # ADDED: progress updates
        if pbar and (i - last_update_i >= max(2000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# ADDED: MINUTE BACKTEST (intraday using bars_per_day scaling)
# =========================================================
def run_backtest_minute(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int,
    bars_per_day: int, days_per_year: int
):
    """
    Minute backtest: identical logic to hourly, but allows much larger bars_per_day
    (e.g., 390 for US equities, 1440 for 24/7 assets).
    Reuses compute_indicators_hourly() with scaled windows by bars_per_day.
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators tuned for intraday using scaled windows
    df = compute_indicators_hourly(df, bars_per_day=bars_per_day, days_per_year=days_per_year)
    df = compute_trend_flags(df, trend_method)

    # Filters
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # Vectorized blackout mask
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # Arrays for speed (non-blocking)
    idx = df.index
    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)
    n = len(idx)

    pbar = st.session_state.get("_backtest_pbar")
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            vwap_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else vwap_today
            delta_today = vwap_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - vwap_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > vwap_today) or
                             (delta_today < 0 and current_close < vwap_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions
        if eligible_arr[i]:
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)

            # ADDED: minute/intraday generic expiry
            expiry = next_friday_close_within_generic(idx, i, max_dte=5, bars_per_day=bars_per_day)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # Equity log
        eq_dates.append(d)
        eq_cash.append(cash)

        # Progress (keeps UI responsive on large minute runs)
        if pbar and (i - last_update_i >= max(5000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# ADDED: Optimized CSV loader (chunked assembly + dtypes)
# =========================================================
def _read_csv_optimized(file, expect_full_cols=True):
    """
    Efficiently read CSV with dtype down-casting, parse_dates, and optional chunk assembly.
    - Keeps memory usage reasonable for 200k+ rows.
    - Validates required columns and sorts by timestamp.
    """
    # Ensure file pointer at start (Streamlit uploader objects)
    try:
        file.seek(0)
    except Exception:
        pass

    # Expected columns
    usecols = ["timestamp", "close", "high", "low", "vwap"]
    dtype = {
        "close": "float32", "high": "float32",
        "low": "float32", "vwap": "float32",
    }

    # Assemble in chunks (avoid single huge read peak)
    chunks = []
    prog = 0  # ADDED: local progress counter
    try:
        for chunk in pd.read_csv(
            file,
            usecols=usecols,  # read only what we need
            dtype=dtype,
            parse_dates=["timestamp"],
            infer_datetime_format=True,
            chunksize=100_000,  # CHANGED: chunked streaming read
            low_memory=True,
            memory_map=True
        ):
            chunk = chunk.dropna(subset=["timestamp"])
            chunks.append(chunk)
            # ADDED: lightweight progress hint (optional)
            p = st.session_state.get("_csv_pbar")
            if p:
                prog = min(100, prog + 10)
                p.progress(prog)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
        # Ensure progress bar completes
        p = st.session_state.get("_csv_pbar")
        if p:
            p.progress(100)
    except ValueError:
        # If file doesn't have expected cols, read what it has
        try:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Sort by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Defensive bounds check
    if expect_full_cols:
        missing_full = set(usecols) - set(df.columns)
        if missing_full:
            # Let caller decide fallback
            return df, missing_full
        else:
            return df, set()
    return df, set()


# =========================================================
# TOP: Uploads + Settings + Run button (ONE CARD)
# =========================================================
st.subheader("Setup")
with st.container():
    st.markdown('<hr style="opacity:0.15;">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div style="margin-bottom:0.5rem;">
            <b>Upload CSV</b><br>
            Drag/drop or browse • Limit 200MB • CSV • Expected columns: timestamp, close, high, low, vwap
        </div>
        """, unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv is not None:
            st.session_state["uploaded_csv"] = uploaded_csv

        st.markdown("""
        <div style="margin-top:0.75rem; margin-bottom:0.5rem;">
            <b>Upload Blackout Dates (.txt)</b><br>
            One date per line (YYYY-MM-DD). Lines starting with # are ignored.
        </div>
        """, unsafe_allow_html=True)
        uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type=["txt"], label_visibility="collapsed")
        if uploaded_txt is not None:
            st.session_state["uploaded_txt"] = uploaded_txt

        # ---- Timeframe toggle (Daily vs Hourly vs Minute) ----
        # CHANGED ➜ add "Minute"
        timeframe_mode = st.radio("Timeframe", options=["Daily", "Hourly", "Minute"], horizontal=True)

        # ADDED: single source of truth for downstream branching/logging
        timeframe_key = (
            "daily" if timeframe_mode == "Daily"
            else "hourly" if timeframe_mode == "Hourly"
            else "minute"
        )

    with col_right:
        hv_min = st.number_input("HV Min (%)", value=35.0)
        hv_max = st.number_input("HV Max (%)", value=75.0)
        adx_exit = st.number_input("ADX Exit ≥", value=25)
        vwap_accept_k = st.number_input("VWAP Accept k×(BBU−BBM)", value=0.5)
        use_trend_bias = st.checkbox("Use Trend Bias")
        trend_bias_strength = st.number_input("Bias Strength", value=2.0, disabled=not use_trend_bias)
        trend_method = st.selectbox("Trend Method", ["VWAP Slope", "VWAP vs SMA20", "ADX + DI"])
        wing_ext_pct = st.number_input("Wing Extension %", value=25.0)
        days_before = st.number_input("Days before blackout", value=5)
        days_after = st.number_input("Days after blackout", value=5)

        # Hourly- / Minute-specific settings
        # CHANGED: broaden condition to intraday modes
        if timeframe_mode in ("Hourly", "Minute"):
            st.markdown("**Intraday Settings**")  # CHANGED: label

            # ADDED: sensible defaults per mode
            default_bpd = 24 if timeframe_mode == "Hourly" else 1440  # 1-min bars default for 24/7 assets
            bars_per_day_override = st.number_input(
                "Bars per day (override)",
                value=default_bpd,
                help=(
                    "Use ~6–7 for HOURLY equities, 24 for hourly 24/7 assets; "
                    "Use ~390 for MINUTE equities (US market hours) or 1440 for minute 24/7 assets. "
                    "If 0, the app infers bars/day from the data."
                ),
                min_value=0, max_value=2000  # CHANGED: allow minute values like 390 or 1440
            )

            days_per_year = st.number_input(
                "Days per year (annualization)",
                value=252,  # equities default; user can set 365 for crypto
                help="252 for equities; 365 for 24/7 assets.",
                min_value=200, max_value=366
            )

            max_chart_points = st.slider(
                "Max chart points (downsample)",
                min_value=2000, max_value=30000, value=12000, step=1000
            )
        else:
            bars_per_day_override = 0
            days_per_year = 252
            max_chart_points = st.slider(
                "Max chart points (downsample)",
                min_value=2000, max_value=30000, value=12000, step=1000
            )

        # ---- Preset save/load (no f-strings with braces) ----
        preset = {
            "hv_min": hv_min, "hv_max": hv_max, "adx_exit": adx_exit,
            "vwap_accept_k": vwap_accept_k,
            "use_trend_bias": use_trend_bias, "trend_bias_strength": trend_bias_strength,
            "trend_method": trend_method, "wing_ext_pct": wing_ext_pct,
            "days_before": days_before, "days_after": days_after,
            "timeframe_mode": timeframe_mode,
            "bars_per_day_override": bars_per_day_override,
            "days_per_year": days_per_year, "max_chart_points": max_chart_points
        }
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            if st.button("Save Preset"):
                st.session_state["icb_preset"] = preset
                st.success("Preset saved in session. Use Download to save locally.")
        with pc2:
            if "icb_preset" in st.session_state:
                st.download_button(
                    "⬇️ Download Preset JSON",
                    data=json.dumps(st.session_state["icb_preset"], indent=2),
                    file_name="icb_preset.json",
                    mime="application/json"
                )

        # Optional: Load Preset from uploaded JSON
        preset_file = st.file_uploader("Load Preset (.json) [optional]", type=["json"])
        if preset_file is not None:
            try:
                loaded_preset = json.load(preset_file)
                st.session_state["icb_preset_loaded"] = loaded_preset
                st.info("Preset loaded below. Apply values manually if desired.")
                st.json(loaded_preset, expanded=False)
            except Exception as e:
                st.error(f"Failed to parse preset JSON: {e}")

    # Run button (top, within same card)
    run_clicked = st.button("Run Backtest")
    st.markdown('<hr style="opacity:0.15;">', unsafe_allow_html=True)

# =========================================================
# OUTPUTS (compact box ABOVE charts)
# =========================================================
if not run_clicked:
    st.info("Upload files, set your parameters above, choose **Daily**, **Hourly**, or **Minute**, then press **Run Backtest**.")
else:
    csv_file = st.session_state.get("uploaded_csv")
    txt_file = st.session_state.get("uploaded_txt")
    if not csv_file or not txt_file:
        st.error("Please upload both CSV and blackout dates (.txt) at the top.")
        st.stop()

    # ADDED: progress bars for non-blocking UX
    st.subheader("Reading & Preparing Data")
    st.session_state["_csv_pbar"] = st.progress(0)        # CSV read progress
    st.session_state["_backtest_pbar"] = st.progress(0)   # Backtest progress

    # Read CSV defensively (optimized)
    df_raw, missing_full = _read_csv_optimized(csv_file, expect_full_cols=True)  # CHANGED: optimized loader
    blackout_dates = parse_blackout_txt(txt_file)

    # Fallback if missing columns
    if missing_full:
        st.warning(f"CSV missing columns for full backtest: {sorted(missing_full)}")
        st.info("Showing basic Bollinger chart on CLOSE. For full logic, include: timestamp, close, high, low, vwap.")

        if "timestamp" not in df_raw.columns or "close" not in df_raw.columns:
            st.error("CSV must include at least 'close' and 'timestamp' to render the fallback chart.")
            st.stop()

        df = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        # ---- Outputs box (placeholders) ----
        st.subheader("Outputs")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.markdown('<div><b>Trades</b><br>—</div>', unsafe_allow_html=True)
        with o2:
            st.markdown('<div><b>Win Rate</b><br>—</div>', unsafe_allow_html=True)
        with o3:
            st.markdown('<div><b>Total P&L</b><br>$—</div>', unsafe_allow_html=True)
        with o4:
            st.markdown('<div><b>Max Drawdown</b><br>$—<br>—%</div>', unsafe_allow_html=True)

        # ---- Chart (fallback) ----
        ma = df["close"].rolling(20).mean()
        ub = ma + 2 * df["close"].rolling(20).std()
        lb = ma - 2 * df["close"].rolling(20).std()

        x = df.index.values
        y = df["close"].values
        x_ds, y_ds = lttb_downsample(x.astype("datetime64[ns]").astype(np.int64),
                                     y.astype(float), threshold=int(st.session_state.get("max_chart_points", 12000)))

        fig_px = go.Figure()
        ts_ds = pd.to_datetime(x_ds)
        fig_px.add_trace(go.Scatter(x=ts_ds, y=y_ds, name="Close Price", mode="lines",
                                    line=dict(color="#60A5FA", width=2.5)))
        fig_px.add_trace(go.Scatter(x=df.index, y=ma, name="BB Mid (20)", mode="lines",
                                    line=dict(color="gray", width=1.2)))
        fig_px.add_trace(go.Scatter(x=df.index, y=ub, name="BB Upper (20,2σ)", mode="lines",
                                    line=dict(color="orange", width=1.2)))
        fig_px.add_trace(go.Scatter(x=df.index, y=lb, name="BB Lower (20,2σ)", mode="lines",
                                    line=dict(color="orange", width=1.2)))

        if blackout_dates and len(df) > 0:
            for e in blackout_dates:
                start = e - timedelta(days=int(days_before))
                end = e + timedelta(days=int(days_after))
                fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

        fig_px.update_layout(
            template="plotly_dark",
            title="Price with Bollinger Bands (fallback)",
            margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_px, use_container_width=True)
        st.stop()

    # ---- Full run ----
    st.subheader("Running Backtest")
    df_sorted = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Backtest with spinner & progress
    with st.spinner("Running backtest…"):
        if timeframe_mode == "Daily":
            df_out, trades_df, equity_df, summary = run_backtest(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after
            )
            bpd_inferred = 1

        elif timeframe_mode == "Hourly":
            idx_sorted = pd.DatetimeIndex(df_sorted["timestamp"].values)
            bpd_inferred = infer_bars_per_day(idx_sorted)
            bars_per_day = bars_per_day_override if int(bars_per_day_override) > 0 else bpd_inferred
            df_out, trades_df, equity_df, summary = run_backtest_hourly(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after,
                bars_per_day=bars_per_day, days_per_year=days_per_year
            )

        else:  # ADDED: Minute
            idx_sorted = pd.DatetimeIndex(df_sorted["timestamp"].values)
            bpd_inferred = infer_bars_per_day(idx_sorted)  # respects session gaps
            # If override is set, trust it; otherwise use inference
            bars_per_day = bars_per_day_override if int(bars_per_day_override) > 0 else max(1, bpd_inferred)

            # Optional UX feedback
            if 300 <= bars_per_day <= 500:
                st.info("Detected minute data with market sessions (~390 bars/day). "
                        "Ensure blackout/expiry expectations align with trading hours.")
            elif bars_per_day >= 1000:
                st.info("Detected minute data for 24/7 assets (~1440 bars/day). HV annualization uses your 'Days per year' setting.")

            df_out, trades_df, equity_df, summary = run_backtest_minute(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after,
                bars_per_day=bars_per_day, days_per_year=days_per_year
            )

    # =====================================================
    # OUTPUTS BOX (compact, above charts)
    # =====================================================
    st.subheader("Outputs")
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    r1c1.markdown(f'<div><b>Trades</b><br>{summary["trades"]}</div>', unsafe_allow_html=True)
    r1c2.markdown(f'<div><b>Win Rate</b><br>{summary["win_rate"]:.2f}%</div>', unsafe_allow_html=True)
    r1c3.markdown(f'<div><b>Total P&L</b><br>${summary["total_pnl"]:.2f}</div>', unsafe_allow_html=True)
    r1c4.markdown(
        f'<div><b>Max Drawdown</b><br>${summary["max_drawdown"]:.2f}<br>{summary["max_drawdown_pct"]:.2f}%</div>',
        unsafe_allow_html=True
    )
    r1c5.markdown(
        f'<div><b>Exits (Bre/ADX/VWAP/Broke)</b><br>{summary["breaches"]}/{summary["adx_exits"]}/{summary["vwap_exits"]}/{summary["brokes"]}</div>',
        unsafe_allow_html=True
    )

    dd_current_val = 0.0
    dd_current_pct = 0.0
    peak_equity = 0.0
    if not equity_df.empty and len(equity_df) > 1:
        equity_df = equity_df.sort_index()
        running_peak = equity_df["cash"].cummax()
        dd_curve = equity_df["cash"] - running_peak
        dd_pct_curve = (dd_curve / running_peak.replace(0, np.nan)).fillna(0.0) * 100.0
        dd_current_val = float(dd_curve.iloc[-1])
        dd_current_pct = float(dd_pct_curve.iloc[-1])
        peak_equity = float(running_peak.max())

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(f'<div><b>Current Drawdown</b><br>${abs(dd_current_val):,.2f}<br>{abs(dd_current_pct):.2f}%</div>', unsafe_allow_html=True)
    r2c2.markdown(f'<div><b>Max Drawdown</b><br>${summary["max_drawdown"]:,.2f}<br>{summary["max_drawdown_pct"]:.2f}%</div>', unsafe_allow_html=True)
    r2c3.markdown(f'<div><b>Peak Equity</b><br>${peak_equity:,.2f}</div>', unsafe_allow_html=True)

    df_losses = trades_df[trades_df["pnl"] < 0].copy()
    total_loss = float(-df_losses["pnl"].sum()) if not df_losses.empty else 0.0
    num_losses = int(len(df_losses))
    avg_loss = float((-df_losses["pnl"].mean()) if num_losses else 0.0)
    worst_loss = float((-df_losses["pnl"].min()) if num_losses else 0.0)

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.markdown(f'<div><b>Total Loss</b><br>${total_loss:,.2f}</div>', unsafe_allow_html=True)
    r3c2.markdown(f'<div><b># Losing Trades</b><br>{num_losses}</div>', unsafe_allow_html=True)
    r3c3.markdown(f'<div><b>Avg Loss</b><br>${avg_loss:,.2f}</div>', unsafe_allow_html=True)
    r3c4.markdown(f'<div><b>Worst Loss</b><br>${worst_loss:,.2f}</div>', unsafe_allow_html=True)

    # =====================================================
    # CHARTS (immediately below outputs)
    # =====================================================
    st.subheader("Charts")

    c_eq, c_hist = st.columns([2, 1])
    with c_eq:
        if equity_df.empty:
            st.info("No equity curve to display.")
        else:
            x_e = equity_df.index.values.astype("datetime64[ns]").astype(np.int64)
            y_e = equity_df["cash"].values.astype(float)
            x_e_ds, y_e_ds = lttb_downsample(x_e, y_e, threshold=int(max_chart_points))
            ts_e_ds = pd.to_datetime(x_e_ds)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=ts_e_ds, y=y_e_ds, name="Equity", mode="lines",
                                        line=dict(color="#22D3EE", width=3)))
            fig_eq.update_layout(template="plotly_dark",
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="", yaxis_title="Cash ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

    with c_hist:
        if df_losses.empty:
            st.info("No losses to chart.")
        else:
            hist_counts, hist_bins = np.histogram(-df_losses["pnl"].values,
                                                  bins=min(20, max(5, len(df_losses))))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=[f"{round(hist_bins[i],2)}–{round(hist_bins[i+1],2)}" for i in range(len(hist_bins)-1)],
                y=hist_counts, marker=dict(color="#F59E0B"), name="Loss size (USD)"
            ))
            fig_hist.update_layout(
                template="plotly_dark", title="Loss distribution",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Loss bucket ($)", yaxis_title="Count", showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### VWAP & Bollinger Bands with Trade Markers")
    fig_px = go.Figure()
    x_full = df_out.index.values.astype("datetime64[ns]").astype(np.int64)
    vwap_full = df_out["vwap"].values.astype(float)
    x_ds, vwap_ds = lttb_downsample(x_full, vwap_full, threshold=int(max_chart_points))
    ts_ds = pd.to_datetime(x_ds)
    fig_px.add_trace(go.Scatter(x=ts_ds, y=vwap_ds, name="VWAP", mode="lines",
                                line=dict(color="steelblue", width=2)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_upper"], name="BB Upper (20,2σ)", mode="lines",
                                line=dict(color="orange", width=1.5)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_mid"], name="BB Mid (20)", mode="lines",
                                line=dict(color="gray", width=1.0)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_lower"], name="BB Lower (20,2σ)", mode="lines",
                                line=dict(color="orange", width=1.5)))

    if blackout_dates:
        for e in blackout_dates:
            start = e - timedelta(days=int(days_before))
            end = e + timedelta(days=int(days_after))
            fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

    if use_trend_bias:
        up_idx = df_out.index[df_out["trend_up"]]
        down_idx = df_out.index[df_out["trend_down"]]
        fig_px.add_trace(go.Scatter(x=up_idx, y=df_out.loc[up_idx, "vwap"], name="Uptrend", mode="markers",
                                    marker=dict(color="green", size=5, opacity=0.5), hoverinfo="skip"))
        fig_px.add_trace(go.Scatter(x=down_idx, y=df_out.loc[down_idx, "vwap"], name="Downtrend", mode="markers",
                                    marker=dict(color="red", size=5, opacity=0.5), hoverinfo="skip"))

    if not trades_df.empty:
        wins_m = (trades_df["outcome"] == "win")
        losses_m = (trades_df["outcome"] == "loss")
        breachBelow is your **fully patched** `streamlit_app.py` with a clean **Daily ↔ Hourly ↔ Minute** timeframe toggle and a generic intraday expiry helper. I preserved all existing daily behavior, reused your intraday indicator scaling for minute bars, and kept the optimized ingestion & progress UX. All edits are clearly annotated with `# ADDED` / `# CHANGED`.

> Drop this file in place of your current `streamlit_app.py`.

```python
# ============================================
# Iron Condor Backtester (Streamlit)
# EDITED for 200k+ scalability, stability, performance
# + Minute timeframe toggle (non-destructive)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple
import json

# =========================================================
# Page config & Global CSS (dark app look)
# =========================================================
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")
st.markdown(""" """, unsafe_allow_html=True)

# Header (no tabs, single page)
st.markdown("""
<div style="font-size:20px; font-weight:600;">Iron Condor Backtester</div>
<div style="opacity:0.8;">Options analytics & strategy management</div>
<div style="opacity:0.6;">v1.2</div>
""", unsafe_allow_html=True)  # CHANGED: version bump for tracking

# =========================================================
# HELPERS
# =========================================================

def parse_blackout_txt(file) -> List[pd.Timestamp]:
    """Parse blackout dates .txt with one date per line, allow comments (#)."""
    if file is None:
        return []
    try:
        content = file.read()
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1")
        loaded: List[pd.Timestamp] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                dt = pd.to_datetime(s)
                loaded.append(pd.Timestamp(dt).normalize())
            except Exception:
                # Ignore unparseable lines
                pass
        return loaded
    except Exception:
        return []


def build_blackout_mask(index: pd.DatetimeIndex,
                        earnings: List[pd.Timestamp],
                        pre: int, post: int) -> np.ndarray:
    """Vectorized blackout mask marking blackout intervals around each earnings date."""
    if not earnings or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    idx_dates = index.normalize()
    mask = np.zeros(len(index), dtype=bool)
    # CHANGED: accumulate blackout ranges across *all* earnings (bug fix)
    for e in earnings:
        e_n = pd.Timestamp(e).normalize()
        start = e_n - pd.Timedelta(days=int(pre))
        end = e_n + pd.Timedelta(days=int(post))
        mask |= (idx_dates >= start) & (idx_dates <= end)
    return mask


def infer_bars_per_day(index: pd.DatetimeIndex) -> int:
    """Infer bars per day by grouping timestamps by date and taking the median count."""
    if len(index) == 0:
        return 1
    s = pd.Series(1, index=index).groupby(index.normalize()).sum()
    bpd = int(float(s.median())) if len(s) > 0 else 1
    return max(bpd, 1)


def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling for plotting large time series."""
    n = len(x)
    if threshold >= n or threshold <= 0:
        return x, y
    sampled_x = [x[0]]
    sampled_y = [y[0]]
    every = (n - 2) / (threshold - 2)
    a = 0
    for i in range(0, threshold - 2):
        avg_range_start = int(np.floor((i + 1) * every)) + 1
        avg_range_end = int(np.floor((i + 2) * every)) + 1
        avg_range_end = min(avg_range_end, n)
        avg_x = np.mean(x[avg_range_start:avg_range_end])
        avg_y = np.mean(y[avg_range_start:avg_range_end])
        range_offs = int(np.floor(i * every)) + 1
        range_to = int(np.floor((i + 1) * every)) + 1
        range_to = min(range_to, n - 1)

        max_area = -1.0
        next_a = None
        ax = x[a]; ay = y[a]
        for j in range(range_offs, range_to + 1):
            area = abs((ax - avg_x) * (y[j] - ay) - (ay - avg_y) * (x[j] - ax))
            if area > max_area:
                max_area = area
                next_a = j
        if next_a is None:
            next_a = range_offs
        sampled_x.append(x[next_a])
        sampled_y.append(y[next_a])
        a = next_a
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])
    return np.array(sampled_x), np.array(sampled_y)


# =========================================================
# INDICATORS (Daily unchanged + Hourly/Minute mode-aware)
# =========================================================

def compute_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Original daily indicator logic unchanged."""
    df = df.copy()

    # Bollinger bands on VWAP (20, 2σ)
    ma = df["vwap"].rolling(20).mean()
    std = df["vwap"].rolling(20).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    # RSI (14) on close
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

    # ADX (14)
    up_move = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().bfill().ffill()

    # SMA20 on VWAP
    df["vwap_sma20"] = df["vwap"].rolling(20).mean()

    # DI lines for "ADX + DI" trend method
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Bollinger width & tightening flag
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(20, min_periods=5).median())
    ).fillna(False)

    # Historical Volatility (21-d rolling, annualized)
    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(21).std(ddof=0) * np.sqrt(252) * 100).bfill().ffill()

    return df


def compute_indicators_hourly(df: pd.DataFrame, bars_per_day: int, days_per_year: int) -> pd.DataFrame:
    """Intraday indicator logic: scale windows by bars_per_day and annualization by bars_per_day*days_per_year."""
    df = df.copy()
    bpd = max(int(bars_per_day), 1)
    bb_n = 20 * bpd
    rsi_n = 14 * bpd
    adx_n = 14 * bpd
    hv_n = 21 * bpd
    annual_factor = np.sqrt(bpd * days_per_year)

    # Bollinger bands on VWAP
    minp = max(5, int(0.2*bb_n))
    ma = df["vwap"].rolling(bb_n, min_periods=minp).mean()
    std = df["vwap"].rolling(bb_n, min_periods=minp).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    # RSI on close (use EWM alpha scaled)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha_rsi = 1 / max(rsi_n, 1)
    avg_gain = gain.ewm(alpha=alpha_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

    # ADX (use EWM alpha scaled)
    up_move = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    alpha_adx = 1 / max(adx_n, 1)
    atr = tr.ewm(alpha=alpha_adx, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha_adx, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha_adx, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(alpha=alpha_adx, adjust=False).mean().bfill().ffill()

    # VWAP SMA (scaled)
    df["vwap_sma20"] = df["vwap"].rolling(bb_n, min_periods=minp).mean()

    # DI lines for "ADX + DI" trend method
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Bollinger width & tightening flag (scaled window)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(bb_n, min_periods=minp).median())
    ).fillna(False)

    # Historical Volatility (rolling hv_n, annualized by sqrt(bpd * days_per_year))
    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(hv_n).std(ddof=0) * annual_factor * 100).bfill().ffill()

    return df


def compute_trend_flags(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == "VWAP Slope":
        df["vwap_delta"] = df["vwap"].diff()
        df["trend_up"] = df["vwap_delta"] > 0
        df["trend_down"] = df["vwap_delta"] < 0
    elif method == "VWAP vs SMA20":
        df["trend_up"] = df["vwap"] > df["vwap_sma20"]
        df["trend_down"] = df["vwap"] < df["vwap_sma20"]
    elif method == "ADX + DI":
        df["trend_up"] = (df["plus_di"] > df["minus_di"]) & (df["adx"] > 20)
        df["trend_down"] = (df["minus_di"] > df["plus_di"]) & (df["adx"] > 20)
    else:
        df["trend_up"] = False
        df["trend_down"] = False
    return df


def in_blackout(day, earnings: List[pd.Timestamp], pre: int, post: int):
    day_n = pd.Timestamp(day).normalize()
    for e in earnings:
        e_n = pd.Timestamp(e).normalize()
        if (e_n - timedelta(days=pre)) <= day_n <= e_n:
            return True
        if e_n <= day_n <= (e_n + timedelta(days=post)):
            return True
    return False


def next_friday_within(idx: pd.DatetimeIndex, start_loc: int, max_dte=5):
    end_loc = min(start_loc + max_dte, len(idx) - 1)
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            return idx[j]
    return idx[end_loc]


def next_friday_close_within(idx: pd.DatetimeIndex, start_loc: int, max_dte=5) -> pd.Timestamp:
    """For hourly data: pick the last timestamp on the first Friday within window."""
    end_loc = min(start_loc + max_dte * 24, len(idx) - 1)
    fridays = []
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            fridays.append(idx[j])
    if not fridays:
        return idx[end_loc]
    first_friday_date = pd.Timestamp(fridays[0]).normalize()
    same_day = [t for t in fridays if pd.Timestamp(t).normalize() == first_friday_date]
    return max(same_day) if same_day else fridays[0]


# ADDED: Generic intraday expiry helper (minute-safe)
def next_friday_close_within_generic(
    idx: pd.DatetimeIndex, start_loc: int, max_dte: int, bars_per_day: int
) -> pd.Timestamp:
    """
    Generic intraday expiry picker:
    - Scan forward up to max_dte * bars_per_day bars.
    - Return the *last* timestamp on the first Friday encountered.
    - If no Friday found, return the last timestamp in the window.
    """
    if len(idx) == 0:
        raise ValueError("Empty index provided to expiry helper.")
    forward = max(1, int(max_dte) * max(1, int(bars_per_day)))
    end_loc = min(start_loc + forward, len(idx) - 1)
    fridays = []
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            fridays.append(idx[j])
    if not fridays:
        return idx[end_loc]
    first_friday_date = pd.Timestamp(fridays[0]).normalize()
    same_day = [t for t in fridays if pd.Timestamp(t).normalize() == first_friday_date]
    return max(same_day) if same_day else fridays[0]


# =========================================================
# DAILY BACKTEST (optimized loop to match hourly speed)
# =========================================================
def run_backtest(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int
):
    # Preprocess (same as before)
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators & trend flags
    df = compute_indicators_daily(df)
    df = compute_trend_flags(df, trend_method)

    # Entry filters incl. HV bounds
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # CHANGED: use vectorized blackout mask (performance & correctness)
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # CHANGED: array-based loop for speed (mirrors hourly approach)
    idx = df.index
    n = len(idx)

    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)

    # Optional progress bar hook (non-blocking)
    pbar = st.session_state.get("_backtest_pbar")  # ADDED
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            v_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else v_today
            delta_today = v_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - v_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > v_today) or
                             (delta_today < 0 and current_close < v_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions (if eligible)
        if eligible_arr[i]:
            # Bias shifts strikes if enabled
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            # adaptive wing widths on trend transitions + BB tightening
            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)  # same as GUI
            expiry = next_friday_within(df.index, i, max_dte=5)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # log equity
        eq_dates.append(d)
        eq_cash.append(cash)

        # ADDED: progress updates for long runs (every ~1% or ≥1000 rows)
        if pbar and n > 0 and (i - last_update_i >= max(1000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    # Summary
    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# HOURLY BACKTEST (optimized, same entry/exit logic)
# =========================================================
def run_backtest_hourly(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int,
    bars_per_day: int, days_per_year: int
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators tuned for hourly
    df = compute_indicators_hourly(df, bars_per_day=bars_per_day, days_per_year=days_per_year)
    df = compute_trend_flags(df, trend_method)

    # Filters
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # Vectorized blackout mask
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # Use numpy arrays for speed
    idx = df.index
    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)
    n = len(idx)

    pbar = st.session_state.get("_backtest_pbar")  # ADDED
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            vwap_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else vwap_today
            delta_today = vwap_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - vwap_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > vwap_today) or
                             (delta_today < 0 and current_close < vwap_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement at the selected Friday close
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions (if eligible row)
        if eligible_arr[i]:
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)
            expiry = next_friday_close_within(idx, i, max_dte=5)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # log equity
        eq_dates.append(d)
        eq_cash.append(cash)

        # ADDED: progress updates
        if pbar and (i - last_update_i >= max(2000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# ADDED: MINUTE BACKTEST (intraday using bars_per_day scaling)
# =========================================================
def run_backtest_minute(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int, days_after: int,
    bars_per_day: int, days_per_year: int
):
    """
    Minute backtest: identical logic to hourly, but allows much larger bars_per_day
    (e.g., ~390 for US equities, 1440 for 24/7 assets). Reuses compute_indicators_hourly().
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators tuned for intraday using scaled windows
    df = compute_indicators_hourly(df, bars_per_day=bars_per_day, days_per_year=days_per_year)
    df = compute_trend_flags(df, trend_method)

    # Filters
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # Vectorized blackout mask
    mask_blackout = build_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = combined & (~mask_blackout)

    # Fees/model params
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, net_credit):
        put_width = sp - lp
        call_width = lc - sc
        if sp <= exp_close <= sc:
            gross = net_credit * mult
            fees = 4 * per_leg_fee
            return gross - fees, 'win'
        loss_w = call_width if exp_close > sc else put_width
        gross_loss = (loss_w - net_credit) * mult
        fees = 4 * per_leg_fee
        return -gross_loss - fees, 'loss'

    # Numpy arrays
    idx = df.index
    close = df["close"].to_numpy()
    vwap = df["vwap"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy()
    bb_mid = df["bb_mid"].to_numpy()
    bb_lower = df["bb_lower"].to_numpy()
    adx_arr = df["adx"].to_numpy()
    trend_up = df["trend_up"].astype(bool).to_numpy()
    trend_down = df["trend_down"].astype(bool).to_numpy()
    tightening = df["bb_tightening"].astype(bool).to_numpy()
    eligible_arr = eligible.to_numpy()

    open_positions = []
    trades = []
    cash = 0.0
    eq_dates = []
    eq_cash = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)
    n = len(idx)

    pbar = st.session_state.get("_backtest_pbar")
    last_update_i = 0

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            vwap_today = vwap[i]
            v_prev = vwap[i - 1] if i > 0 else vwap_today
            delta_today = vwap_today - v_prev
            delta_prev = (vwap[i - 1] - vwap[i - 2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - vwap_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > vwap_today) or
                             (delta_today < 0 and current_close < vwap_today))
            vwap_exit = slope_flip and away_enough and on_slope_side

            for pos in open_positions:
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])

                exited = False
                outcome_flag = None
                if (d < pos["expiry"]) and broke:
                    exited = True; outcome_flag = "broke"
                elif (d < pos["expiry"]) and breach:
                    exited = True; outcome_flag = "breach"
                elif (d < pos["expiry"]) and adx_exit_now:
                    exited = True; outcome_flag = "adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited = True; outcome_flag = "vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": current_close, "pnl": pnl_today,
                        "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"],
                        "expiry_close": exp_close, "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions
        if eligible_arr[i]:
            if use_bias:
                bias = float(bias_strength)
                if trend_up[i]:
                    sp = round_to(float(bb_lower[i]) + 0.5 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) + 1.0 * bias, 1.0)
                elif trend_down[i]:
                    sp = round_to(float(bb_lower[i]) - 1.0 * bias, 1.0)
                    sc = round_to(float(bb_upper[i]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(bb_lower[i]), 1.0)
                    sc = round_to(float(bb_upper[i]), 1.0)
            else:
                sp = round_to(float(bb_lower[i]), 1.0)
                sc = round_to(float(bb_upper[i]), 1.0)

            prev_up = bool(trend_up[i - 1]) if i > 0 else False
            prev_down = bool(trend_down[i - 1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)

            # ADDED: minute/intraday generic expiry
            expiry = next_friday_close_within_generic(idx, i, max_dte=5, bars_per_day=bars_per_day)

            open_positions.append({
                "entry": d, "expiry": expiry, "sp": sp, "lp": lp,
                "sc": sc, "lc": lc, "credit": credit
            })

        # Equity log
        eq_dates.append(d)
        eq_cash.append(cash)

        # Progress (keeps UI responsive on large minute runs)
        if pbar and (i - last_update_i >= max(5000, n // 100)):
            pbar.progress(int((i + 1) / n * 100))
            last_update_i = i

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame({"date": eq_dates, "cash": eq_cash}).set_index("date") if len(eq_dates) else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown_mag = running_max - equity_df["cash"]
        max_dd_val = float(drawdown_mag.max()) if not drawdown_mag.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown_mag.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    wins = int((trades_df["outcome"] == "win").sum()) if not trades_df.empty else 0
    losses = int((trades_df["outcome"] == "loss").sum()) if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits = int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits = int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses, "breaches": breaches,
        "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# ADDED: Optimized CSV loader (chunked assembly + dtypes)
# =========================================================
def _read_csv_optimized(file, expect_full_cols=True):
    """
    Efficiently read CSV with dtype down-casting, parse_dates, and optional chunk assembly.
    - Keeps memory usage reasonable for 200k+ rows.
    - Validates required columns and sorts by timestamp.
    """
    # Ensure file pointer at start (Streamlit uploader objects)
    try:
        file.seek(0)
    except Exception:
        pass

    # Expected columns
    usecols = ["timestamp", "close", "high", "low", "vwap"]
    dtype = {
        "close": "float32", "high": "float32",
        "low": "float32", "vwap": "float32",
    }

    chunks = []
    # ADDED: track CSV progress in session (safe default)
    csv_progress = st.session_state.get("_csv_progress", 0)
    try:
        for chunk in pd.read_csv(
            file,
            usecols=usecols,  # read only what we need
            dtype=dtype,
            parse_dates=["timestamp"],
            infer_datetime_format=True,
            chunksize=100_000,  # CHANGED: chunked streaming read
            low_memory=True,
            memory_map=True
        ):
            chunk = chunk.dropna(subset=["timestamp"])
            chunks.append(chunk)
            # ADDED: lightweight progress hint (increments without relying on internal attributes)
            p = st.session_state.get("_csv_pbar")
            if p:
                csv_progress = min(100, csv_progress + 5)
                p.progress(csv_progress)
                st.session_state["_csv_progress"] = csv_progress

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    except ValueError:
        # If file doesn't have expected cols, read what it has
        try:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Sort by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Defensive bounds check
    if expect_full_cols:
        missing_full = set(usecols) - set(df.columns)
        if missing_full:
            return df, missing_full
        else:
            return df, set()
    return df, set()


# =========================================================
# TOP: Uploads + Settings + Run button (ONE CARD)
# =========================================================
st.subheader("Setup")
with st.container():
    st.markdown('<hr style="opacity:0.15;">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div style="margin-bottom:0.5rem;">
            <b>Upload CSV</b><br>
            Drag/drop or browse • Limit 200MB • CSV • Expected columns: timestamp, close, high, low, vwap
        </div>
        """, unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv is not None:
            st.session_state["uploaded_csv"] = uploaded_csv

        st.markdown("""
        <div style="margin-top:0.75rem; margin-bottom:0.5rem;">
            <b>Upload Blackout Dates (.txt)</b><br>
            One date per line (YYYY-MM-DD). Lines starting with # are ignored.
        </div>
        """, unsafe_allow_html=True)
        uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type=["txt"], label_visibility="collapsed")
        if uploaded_txt is not None:
            st.session_state["uploaded_txt"] = uploaded_txt

        # ---- Timeframe toggle (Daily vs Hourly, add Minute) ----
        # CHANGED: include "Minute"
        timeframe_mode = st.radio("Timeframe", options=["Daily", "Hourly", "Minute"], horizontal=True)

        # ADDED: single source of truth for downstream branching/logging
        timeframe_key = (
            "daily" if timeframe_mode == "Daily"
            else "hourly" if timeframe_mode == "Hourly"
            else "minute"
        )

    with col_right:
        hv_min = st.number_input("HV Min (%)", value=35.0)
        hv_max = st.number_input("HV Max (%)", value=75.0)
        adx_exit = st.number_input("ADX Exit ≥", value=25)
        vwap_accept_k = st.number_input("VWAP Accept k×(BBU−BBM)", value=0.5)
        use_trend_bias = st.checkbox("Use Trend Bias")
        trend_bias_strength = st.number_input("Bias Strength", value=2.0, disabled=not use_trend_bias)
        trend_method = st.selectbox("Trend Method", ["VWAP Slope", "VWAP vs SMA20", "ADX + DI"])
        wing_ext_pct = st.number_input("Wing Extension %", value=25.0)
        days_before = st.number_input("Days before blackout", value=5)
        days_after = st.number_input("Days after blackout", value=5)

        # Hourly- / Minute-specific settings
        # CHANGED: broaden condition to intraday modes
        if timeframe_mode in ("Hourly", "Minute"):
            st.markdown("**Intraday Settings**")  # CHANGED: label

            # ADDED: sensible defaults per mode
            default_bpd = 24 if timeframe_mode == "Hourly" else 1440
            bars_per_day_override = st.number_input(
                "Bars per day (override)",
                value=default_bpd,
                help=(
                    "Use ~6–7 for HOURLY equities, 24 for hourly 24/7 assets; "
                    "Use ~390 for MINUTE equities (US market hours) or 1440 for minute 24/7 assets. "
                    "If 0, the app infers bars/day from the data."
                ),
                min_value=0, max_value=2000  # allow minute values like 390 or 1440
            )

            days_per_year = st.number_input(
                "Days per year (annualization)",
                value=252,  # equities default; user can set 365 for crypto
                help="252 for equities; 365 for 24/7 assets.",
                min_value=200, max_value=366
            )

            max_chart_points = st.slider(
                "Max chart points (downsample)",
                min_value=2000, max_value=30000, value=12000, step=1000
            )
        else:
            bars_per_day_override = 0
            days_per_year = 252
            max_chart_points = st.slider(
                "Max chart points (downsample)",
                min_value=2000, max_value=30000, value=12000, step=1000
            )

        # ---- Preset save/load (no f-strings with braces) ----
        preset = {
            "hv_min": hv_min, "hv_max": hv_max, "adx_exit": adx_exit,
            "vwap_accept_k": vwap_accept_k,
            "use_trend_bias": use_trend_bias, "trend_bias_strength": trend_bias_strength,
            "trend_method": trend_method, "wing_ext_pct": wing_ext_pct,
            "days_before": days_before, "days_after": days_after,
            "timeframe_mode": timeframe_mode,
            "bars_per_day_override": bars_per_day_override,
            "days_per_year": days_per_year, "max_chart_points": max_chart_points
        }
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            if st.button("Save Preset"):
                st.session_state["icb_preset"] = preset
                st.success("Preset saved in session. Use Download to save locally.")
        with pc2:
            if "icb_preset" in st.session_state:
                st.download_button(
                    "⬇️ Download Preset JSON",
                    data=json.dumps(st.session_state["icb_preset"], indent=2),
                    file_name="icb_preset.json",
                    mime="application/json"
                )

        # Optional: Load Preset from uploaded JSON
        preset_file = st.file_uploader("Load Preset (.json) [optional]", type=["json"])
        if preset_file is not None:
            try:
                loaded_preset = json.load(preset_file)
                st.session_state["icb_preset_loaded"] = loaded_preset
                st.info("Preset loaded below. Apply values manually if desired.")
                st.json(loaded_preset, expanded=False)
            except Exception as e:
                st.error(f"Failed to parse preset JSON: {e}")

    # Run button (top, within same card)
    run_clicked = st.button("Run Backtest")
    st.markdown('<hr style="opacity:0.15;">', unsafe_allow_html=True)

# =========================================================
# OUTPUTS (compact box ABOVE charts)
# =========================================================
if not run_clicked:
    st.info("Upload files, set your parameters above, choose **Daily**, **Hourly**, or **Minute**, then press **Run Backtest**.")
else:
    csv_file = st.session_state.get("uploaded_csv")
    txt_file = st.session_state.get("uploaded_txt")
    if not csv_file or not txt_file:
        st.error("Please upload both CSV and blackout dates (.txt) at the top.")
        st.stop()

    # ADDED: progress bars for non-blocking UX
    st.subheader("Reading & Preparing Data")
    st.session_state["_csv_pbar"] = st.progress(0)        # CSV read progress
    st.session_state["_csv_progress"] = 0                 # track value safely
    st.session_state["_backtest_pbar"] = st.progress(0)   # Backtest progress

    # Read CSV defensively (optimized)
    df_raw, missing_full = _read_csv_optimized(csv_file, expect_full_cols=True)  # CHANGED: optimized loader
    blackout_dates = parse_blackout_txt(txt_file)

    # Fallback if missing columns
    if missing_full:
        st.warning(f"CSV missing columns for full backtest: {sorted(missing_full)}")
        st.info("Showing basic Bollinger chart on CLOSE. For full logic, include: timestamp, close, high, low, vwap.")

        if "timestamp" not in df_raw.columns or "close" not in df_raw.columns:
            st.error("CSV must include at least 'close' and 'timestamp' to render the fallback chart.")
            st.stop()

        df = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        # ---- Outputs box (placeholders) ----
        st.subheader("Outputs")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.markdown('<div><b>Trades</b><br>—</div>', unsafe_allow_html=True)
        with o2:
            st.markdown('<div><b>Win Rate</b><br>—</div>', unsafe_allow_html=True)
        with o3:
            st.markdown('<div><b>Total P&L</b><br>$—</div>', unsafe_allow_html=True)
        with o4:
            st.markdown('<div><b>Max Drawdown</b><br>$—<br>—%</div>', unsafe_allow_html=True)

        # ---- Chart (fallback) ----
        ma = df["close"].rolling(20).mean()
        ub = ma + 2 * df["close"].rolling(20).std()
        lb = ma - 2 * df["close"].rolling(20).std()

        x = df.index.values
        y = df["close"].values
        x_ds, y_ds = lttb_downsample(
            x.astype("datetime64[ns]").astype(np.int64),
            y.astype(float),
            threshold=12000
        )

        fig_px = go.Figure()
        ts_ds = pd.to_datetime(x_ds)
        fig_px.add_trace(go.Scatter(x=ts_ds, y=y_ds, name="Close Price", mode="lines",
                                    line=dict(color="#60A5FA", width=2.5)))
        fig_px.add_trace(go.Scatter(x=df.index, y=ma, name="BB Mid (20)", mode="lines",
                                    line=dict(color="gray", width=1.2)))
        fig_px.add_trace(go.Scatter(x=df.index, y=ub, name="BB Upper (20,2σ)", mode="lines",
                                    line=dict(color="orange", width=1.2)))
        fig_px.add_trace(go.Scatter(x=df.index, y=lb, name="BB Lower (20,2σ)", mode="lines",
                                    line=dict(color="orange", width=1.2)))

        if blackout_dates and len(df) > 0:
            for e in blackout_dates:
                start = e - timedelta(days=int(days_before))
                end = e + timedelta(days=int(days_after))
                fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

        fig_px.update_layout(
            template="plotly_dark",
            title="Price with Bollinger Bands (fallback)",
            margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_px, use_container_width=True)
        st.stop()

    # ---- Full run ----
    st.subheader("Running Backtest")
    df_sorted = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Backtest with spinner & progress
    with st.spinner("Running backtest…"):
        if timeframe_mode == "Daily":
            df_out, trades_df, equity_df, summary = run_backtest(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after
            )
            bpd_inferred = 1

        elif timeframe_mode == "Hourly":
            idx_sorted = pd.DatetimeIndex(df_sorted["timestamp"].values)
            bpd_inferred = infer_bars_per_day(idx_sorted)
            bars_per_day = bars_per_day_override if int(bars_per_day_override) > 0 else bpd_inferred
            df_out, trades_df, equity_df, summary = run_backtest_hourly(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after,
                bars_per_day=bars_per_day, days_per_year=days_per_year
            )

        else:  # ADDED: Minute
            idx_sorted = pd.DatetimeIndex(df_sorted["timestamp"].values)
            bpd_inferred = infer_bars_per_day(idx_sorted)
            # If override is set, trust it; otherwise use inference
            bars_per_day = bars_per_day_override if int(bars_per_day_override) > 0 else max(1, bpd_inferred)

            # Optional UX hint
            if 300 <= bars_per_day <= 500:
                st.info("Detected minute data with market sessions (~390 bars/day). Adjust Days/Year as needed.")
            elif bars_per_day >= 1000:
                st.info("Detected minute data for 24/7 assets (~1440 bars/day). HV annualization uses your Days/Year.")

            df_out, trades_df, equity_df, summary = run_backtest_minute(
                df_raw=df_sorted,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method,
                wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after,
                bars_per_day=bars_per_day, days_per_year=days_per_year
            )

    # =====================================================
    # OUTPUTS BOX (compact, above charts)
    # =====================================================
    st.subheader("Outputs")
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    r1c1.markdown(f'<div><b>Trades</b><br>{summary["trades"]}</div>', unsafe_allow_html=True)
    r1c2.markdown(f'<div><b>Win Rate</b><br>{summary["win_rate"]:.2f}%</div>', unsafe_allow_html=True)
    r1c3.markdown(f'<div><b>Total P&L</b><br>${summary["total_pnl"]:.2f}</div>', unsafe_allow_html=True)
    r1c4.markdown(
        f'<div><b>Max Drawdown</b><br>${summary["max_drawdown"]:.2f}<br>{summary["max_drawdown_pct"]:.2f}%</div>',
        unsafe_allow_html=True
    )
    r1c5.markdown(
        f'<div><b>Exits (Bre/ADX/VWAP/Broke)</b><br>{summary["breaches"]}/{summary["adx_exits"]}/{summary["vwap_exits"]}/{summary["brokes"]}</div>',
        unsafe_allow_html=True
    )

    dd_current_val = 0.0
    dd_current_pct = 0.0
    peak_equity = 0.0
    if not equity_df.empty and len(equity_df) > 1:
        equity_df = equity_df.sort_index()
        running_peak = equity_df["cash"].cummax()
        dd_curve = equity_df["cash"] - running_peak
        dd_pct_curve = (dd_curve / running_peak.replace(0, np.nan)).fillna(0.0) * 100.0
        dd_current_val = float(dd_curve.iloc[-1])
        dd_current_pct = float(dd_pct_curve.iloc[-1])
        peak_equity = float(running_peak.max())

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(f'<div><b>Current Drawdown</b><br>${abs(dd_current_val):,.2f}<br>{abs(dd_current_pct):.2f}%</div>', unsafe_allow_html=True)
    r2c2.markdown(f'<div><b>Max Drawdown</b><br>${summary["max_drawdown"]:,.2f}<br>{summary["max_drawdown_pct"]:.2f}%</div>', unsafe_allow_html=True)
    r2c3.markdown(f'<div><b>Peak Equity</b><br>${peak_equity:,.2f}</div>', unsafe_allow_html=True)

    df_losses = trades_df[trades_df["pnl"] < 0].copy()
    total_loss = float(-df_losses["pnl"].sum()) if not df_losses.empty else 0.0
    num_losses = int(len(df_losses))
    avg_loss = float((-df_losses["pnl"].mean()) if num_losses else 0.0)
    worst_loss = float((-df_losses["pnl"].min()) if num_losses else 0.0)

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.markdown(f'<div><b>Total Loss</b><br>${total_loss:,.2f}</div>', unsafe_allow_html=True)
    r3c2.markdown(f'<div><b># Losing Trades</b><br>{num_losses}</div>', unsafe_allow_html=True)
    r3c3.markdown(f'<div><b>Avg Loss</b><br>${avg_loss:,.2f}</div>', unsafe_allow_html=True)
    r3c4.markdown(f'<div><b>Worst Loss</b><br>${worst_loss:,.2f}</div>', unsafe_allow_html=True)

    # =====================================================
    # CHARTS (immediately below outputs)
    # =====================================================
    st.subheader("Charts")

    c_eq, c_hist = st.columns([2, 1])
    with c_eq:
        if equity_df.empty:
            st.info("No equity curve to display.")
        else:
            x_e = equity_df.index.values.astype("datetime64[ns]").astype(np.int64)
            y_e = equity_df["cash"].values.astype(float)
            x_e_ds, y_e_ds = lttb_downsample(x_e, y_e, threshold=int(max_chart_points))
            ts_e_ds = pd.to_datetime(x_e_ds)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=ts_e_ds, y=y_e_ds, name="Equity", mode="lines",
                                        line=dict(color="#22D3EE", width=3)))
            fig_eq.update_layout(template="plotly_dark",
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="", yaxis_title="Cash ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

    with c_hist:
        if df_losses.empty:
            st.info("No losses to chart.")
        else:
            hist_counts, hist_bins = np.histogram(-df_losses["pnl"].values,
                                                  bins=min(20, max(5, len(df_losses))))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=[f"{round(hist_bins[i],2)}–{round(hist_bins[i+1],2)}" for i in range(len(hist_bins)-1)],
                y=hist_counts, marker=dict(color="#F59E0B"), name="Loss size (USD)"
            ))
            fig_hist.update_layout(
                template="plotly_dark", title="Loss distribution",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Loss bucket ($)", yaxis_title="Count", showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### VWAP & Bollinger Bands with Trade Markers")
    fig_px = go.Figure()
    x_full = df_out.index.values.astype("datetime64[ns]").astype(np.int64)
    vwap_full = df_out["vwap"].values.astype(float)
    x_ds, vwap_ds = lttb_downsample(x_full, vwap_full, threshold=int(max_chart_points))
    ts_ds = pd.to_datetime(x_ds)
    fig_px.add_trace(go.Scatter(x=ts_ds, y=vwap_ds, name="VWAP", mode="lines",
                                line=dict(color="steelblue", width=2)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_upper"], name="BB Upper (20,2σ)", mode="lines",
                                line=dict(color="orange", width=1.5)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_mid"], name="BB Mid (20)", mode="lines",
                                line=dict(color="gray", width=1.0)))
    fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_lower"], name="BB Lower (20,2σ)", mode="lines",
                                line=dict(color="orange", width=1.5)))

    if blackout_dates:
        for e in blackout_dates:
            start = e - timedelta(days=int(days_before))
            end = e + timedelta(days=int(days_after))
            fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

    if use_trend_bias:
        up_idx = df_out.index[df_out["trend_up"]]
        down_idx = df_out.index[df_out["trend_down"]]
        fig_px.add_trace(go.Scatter(x=up_idx, y=df_out.loc[up_idx, "vwap"], name="Uptrend", mode="markers",
                                    marker=dict(color="green", size=5, opacity=0.5), hoverinfo="skip"))
        fig_px.add_trace(go.Scatter(x=down_idx, y=df_out.loc[down_idx, "vwap"], name="Downtrend", mode="markers",
                                    marker=dict(color="red", size=5, opacity=0.5), hoverinfo="skip"))

    if not trades_df.empty:
        wins_m = (trades_df["outcome"] == "win")
        losses_m = (trades_df["outcome"] == "loss")
        breach_m = (trades_df["outcome"] == "breach")
        adx_m = (trades_df["outcome"] == "adx_exit")
        vwap_m = (trades_df["outcome"] == "vwap_exit")
        broke_m = (trades_df["outcome"] == "broke")

        fig_px.add_trace(go.Scatter(
            x=trades_df.loc[wins_m, "entry_date"],
            y=df_out.loc[trades_df.loc[wins_m, "entry_date"], "vwap"],
            name="Entry (win)", mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=9)
        ))
        fig_px.add_trace(go.Scatter(
            x=trades_df.loc[losses_m, "entry_date"],
            y=df_out.loc[trades_df.loc[losses_m, "entry_date"], "vwap"],
            name="Entry (loss)", mode="markers",
            marker=dict(symbol="triangle-up", color="red", size=9)
        ))

        def add_exit(mask, name, color, symbol="x"):
            fig_px.add_trace(go.Scatter(
                x=trades_df.loc[mask, "expiry_date"],
                y=df_out.loc[trades_df.loc[mask, "expiry_date"], "close"],
                name=name, mode="markers",
                marker=dict(symbol=symbol, color=color, size=9)
            ))

        add_exit(wins_m, "Exit (win)", "green", "x")
        add_exit(losses_m, "Exit (loss)", "red", "x")
        add_exit(breach_m, "Exit (breach)", "red", "triangle-down")
        add_exit(adx_m, "Exit (ADX)", "purple", "square")
        add_exit(vwap_m, "Exit (VWAP)", "orange", "diamond")
        add_exit(broke_m, "Exit (broke)", "black", "star")

    fig_px.update_layout(
        template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="", yaxis_title="Price ($)",
