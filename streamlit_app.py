
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple
import json

# =========================================================
# Page config (lightweight at startup)
# =========================================================
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")

# === UPDATED: Basic layout scaffolding & placeholders only at top ===
# Keep header minimal; avoid doing any heavy work until the user clicks Run.
st.markdown(
    """
    <h2 style="margin-bottom:0">Iron Condor Backtester</h2>
    <div style="opacity:.8">Options analytics & strategy management &mdash; v1.2</div>
    <hr style="margin-top:0"/>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPERS (definitions only; no execution)
# ---------------------------------------------------------
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
                pass
        return loaded
    except Exception:
        return []


def build_blackout_mask(index: pd.DatetimeIndex, earnings: List[pd.Timestamp], pre: int, post: int) -> np.ndarray:
    """Vectorized blackout mask marking blackout intervals around each earnings date."""
    if not earnings or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    idx_dates = index.normalize()
    mask = np.zeros(len(index), dtype=bool)
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
        sampled_x.append(x[next_a]); sampled_y.append(y[next_a])
        a = next_a
    sampled_x.append(x[-1]); sampled_y.append(y[-1])
    return np.array(sampled_x), np.array(sampled_y)


# =========================================================
# INDICATORS (Daily unchanged + Hourly mode-aware)
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
    """
    === FIXED: Hourly Bollinger Bands time-normalization ===
    - Use a *day-based* lookback converted to bars: bb_n = 20 days × bars_per_day (e.g., equities ~6.5–7 → ~130–140 bars).
    - Keep SMA/SD logic identical to daily; fully vectorized rolling.
    - RSI/ADX/HV windows similarly scaled by bars/day; annualization by sqrt(bars_per_day * days_per_year).
    """
    df = df.copy()
    bpd = max(int(bars_per_day), 1)

    # Day-based lookbacks converted to bars
    bb_n = 20 * bpd
    rsi_n = 14 * bpd
    adx_n = 14 * bpd
    hv_n = 21 * bpd

    annual_factor = np.sqrt(bpd * days_per_year)

    # Bollinger bands on VWAP
    minp = max(5, int(0.2 * bb_n))
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


# =========================================================
# DAILY BACKTEST (unchanged)
# =========================================================
def run_backtest(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float,
    hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool,
    bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int,
    days_after: int
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    df = compute_indicators_daily(df)
    df = compute_trend_flags(df, trend_method)

    # Entry filters incl. HV bounds
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    # Earnings blackout
    mask_blackout = df.index.to_series().apply(lambda d: in_blackout(d, blackout_dates, days_before, days_after))
    eligible = combined & (~mask_blackout.values)

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

    # Sim loop
    idx = df.index
    open_positions = []
    trades = []
    cash = 0.0
    eq = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)

    for i in range(len(idx)):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            for pos in open_positions:
                current_close = df.loc[d, "close"]
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])

                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke = (current_close < pos["lp"]) or (current_close > pos["lc"])
                adx_exit_now = (df.loc[d, "adx"] >= ADX_EXIT)

                # VWAP exit
                vwap_today = df.loc[d, "vwap"]
                vwap_prev = df.iloc[i-1]["vwap"] if i > 0 else vwap_today
                delta_today = vwap_today - vwap_prev
                delta_prev = (df.iloc[i-1]["vwap"] - df.iloc[i-2]["vwap"]) if i > 1 else 0.0
                sign_today = np.sign(delta_today)
                sign_prev = np.sign(delta_prev)
                slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
                bb_halfwidth = df.loc[d, "bb_upper"] - df.loc[d, "bb_mid"]
                accept_dist = VWAP_ACCEPT_K * bb_halfwidth
                away_enough = abs(current_close - vwap_today) >= accept_dist
                on_slope_side = ((delta_today > 0 and current_close > vwap_today) or (delta_today < 0 and current_close < vwap_today))
                vwap_exit = slope_flip and away_enough and on_slope_side

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
                        "net_credit": pos["credit"], "expiry_close": current_close,
                        "pnl": pnl_today, "outcome": outcome_flag
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Expiry settlement
        if open_positions:
            still_open = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = df.loc[d, "close"]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"], "expiry_close": exp_close,
                        "pnl": pnl, "outcome": out
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Open new positions (if eligible)
        if eligible.iloc[i]:
            row = df.iloc[i]
            # bias shifts strikes if enabled
            if use_bias:
                bias = float(bias_strength)
                if df["trend_up"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) + 0.5 * bias, 1.0)
                    sc = round_to(float(row["bb_upper"]) + 1.0 * bias, 1.0)
                elif df["trend_down"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) - 1.0 * bias, 1.0)
                    sc = round_to(float(row["bb_upper"]) - 0.5 * bias, 1.0)
                else:
                    sp = round_to(float(row["bb_lower"]), 1.0)
                    sc = round_to(float(row["bb_upper"]), 1.0)
            else:
                sp = round_to(float(row["bb_lower"]), 1.0)
                sc = round_to(float(row["bb_upper"]), 1.0)

            # adaptive wing widths on trend transitions + BB tightening
            prev_up = bool(df["trend_up"].iloc[i-1]) if i > 0 else False
            prev_down = bool(df["trend_down"].iloc[i-1]) if i > 0 else False
            tightening_now = bool(df["bb_tightening"].iloc[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct) / 100.0)

            put_w = 5.0 * ext_factor if (df["trend_up"].iloc[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (df["trend_down"].iloc[i] and prev_up and tightening_now) else 5.0

            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)  # same as GUI

            expiry = next_friday_within(df.index, i, max_dte=5)
            open_positions.append({"entry": d, "expiry": expiry, "sp": sp, "lp": lp, "sc": sc, "lc": lc, "credit": credit})

        # log equity
        eq.append({"date": d, "cash": cash})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    equity_df = pd.DataFrame(eq).set_index("date") if eq else pd.DataFrame(columns=["cash"])

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
    adx_exits= int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits=int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins,
        "losses": losses,
        "breaches": breaches,
        "adx_exits": adx_exits,
        "vwap_exits": vwap_exits,
        "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val,
        "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# HOURLY BACKTEST (optimized, same entry/exit logic)
# =========================================================
def run_backtest_hourly(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float,
    hv_max: float,
    adx_exit_thr: int,
    vwap_k: float,
    use_bias: bool,
    bias_strength: float,
    trend_method: str,
    wing_ext_pct: float,
    days_before: int,
    days_after: int,
    bars_per_day: int,
    days_per_year: int
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # === FIXED: Hourly Bollinger Bands time-normalization ===
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

    for i in range(n):
        d = idx[i]

        # Early exits
        if open_positions:
            still_open = []
            current_close = close[i]
            adx_exit_now = (adx_arr[i] >= ADX_EXIT)

            # VWAP exit components
            vwap_today = vwap[i]
            v_prev = vwap[i-1] if i > 0 else vwap_today
            delta_today = vwap_today - v_prev
            delta_prev = (vwap[i-1] - vwap[i-2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            bb_halfwidth = bb_upper[i] - bb_mid[i]
            accept_dist = VWAP_ACCEPT_K * bb_halfwidth
            away_enough = abs(current_close - vwap_today) >= accept_dist
            on_slope_side = ((delta_today > 0 and current_close > vwap_today) or (delta_today < 0 and current_close < vwap_today))
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
                        "net_credit": pos["credit"], "expiry_close": current_close,
                        "pnl": pnl_today, "outcome": outcome_flag
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
                        "net_credit": pos["credit"], "expiry_close": exp_close,
                        "pnl": pnl, "outcome": out
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

            prev_up = bool(trend_up[i-1]) if i > 0 else False
            prev_down = bool(trend_down[i-1]) if i > 0 else False
            tightening_now = bool(tightening[i])
            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0

            put_w = 5.0 * ext_factor if (trend_up[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (trend_down[i] and prev_up and tightening_now) else 5.0

            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)
            expiry = next_friday_close_within(idx, i, max_dte=5)

            open_positions.append({"entry": d, "expiry": expiry, "sp": sp, "lp": lp, "sc": sc, "lc": lc, "credit": credit})

        # log equity
        eq_dates.append(d)
        eq_cash.append(cash)

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
    adx_exits= int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits=int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes = int((trades_df["outcome"] == "broke").sum()) if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins,
        "losses": losses,
        "breaches": breaches,
        "adx_exits": adx_exits,
        "vwap_exits": vwap_exits,
        "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val,
        "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# === NEW: Deferred backtest execution after file upload/button click ===
# =========================================================
st.subheader("Setup")

with st.container():
    # Logical grouping of controls
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            """
            **Upload CSV**  
            Drag/drop or browse • Limit 200MB • CSV  
            Expected columns: `timestamp, close, high, low, vwap`
            """,
            unsafe_allow_html=True,
        )
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv is not None:
            st.session_state["uploaded_csv"] = uploaded_csv

        st.markdown(
            """
            **Upload Blackout Dates (.txt)**  
            One date per line (YYYY-MM-DD). Lines starting with `#` are ignored.
            """,
            unsafe_allow_html=True,
        )
        uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type=["txt"], label_visibility="collapsed")
        if uploaded_txt is not None:
            st.session_state["uploaded_txt"] = uploaded_txt

        # Timeframe toggle (Daily vs Hourly)
        timeframe_mode = st.radio("Timeframe", options=["Daily", "Hourly"], horizontal=True)

    with col_right:
        # Parameter controls
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

        # Hourly-specific settings
        if timeframe_mode == "Hourly":
            st.markdown("**Hourly Settings**")
            # === UPDATED: sensible default (0 = infer from data) ===
            bars_per_day_override = st.number_input(
                "Bars per day (override)",
                value=0,  # default to inference to avoid incorrect fixed 24 for equities
                help="Use 24 for crypto; ~6–7 for equities. If 0, app infers from data.",
                min_value=0, max_value=96
            )
            days_per_year = st.number_input(
                "Days per year (annualization)",
                value=252,
                help="252 for equities, 365 for crypto.",
                min_value=200, max_value=366
            )
            max_chart_points = st.slider("Max chart points (downsample)", min_value=2000, max_value=30000, value=12000, step=1000)
        else:
            bars_per_day_override = 0
            days_per_year = 252
            max_chart_points = st.slider("Max chart points (downsample)", min_value=2000, max_value=30000, value=12000, step=1000)

    # Preset save/load
    preset = {
        "hv_min": hv_min, "hv_max": hv_max, "adx_exit": adx_exit, "vwap_accept_k": vwap_accept_k,
        "use_trend_bias": use_trend_bias, "trend_bias_strength": trend_bias_strength,
        "trend_method": trend_method, "wing_ext_pct": wing_ext_pct,
        "days_before": days_before, "days_after": days_after,
        "timeframe_mode": timeframe_mode, "bars_per_day_override": bars_per_day_override,
        "days_per_year": days_per_year, "max_chart_points": max_chart_points
    }
    pc1, pc2 = st.columns([1,1])
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

# === NEW: Placeholders for metrics and charts so UI is responsive at startup ===
outputs_box = st.empty()
charts_box = st.container()
trades_table_box = st.container()

# Run button (deferred execution)
run_clicked = st.button("Run Backtest")

# =========================================================
# === NEW: Deferred execution & status/progress indicators ===
# =========================================================
if not run_clicked:
    st.info("Upload files, set parameters, choose **Daily** or **Hourly**, then press **Run Backtest**.")
else:
    csv_file = st.session_state.get("uploaded_csv")
    txt_file = st.session_state.get("uploaded_txt")

    if not csv_file or not txt_file:
        st.error("Please upload both CSV and blackout dates (.txt) at the top.")
        st.stop()

    with st.status("Running backtest…", expanded=True) as status:
        st.write("Reading data...")
        # Read CSV defensively
        df_raw = pd.read_csv(csv_file)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        required_full = {"timestamp", "close", "high", "low", "vwap"}
        missing_full = required_full - set(df_raw.columns)
        blackout_dates = parse_blackout_txt(txt_file)

        # Fallback if missing columns
        if missing_full:
            st.warning(f"CSV missing columns for full backtest: {sorted(missing_full)}")
            st.info("Showing basic Bollinger chart on CLOSE. For full logic, include: timestamp, close, high, low, vwap.")
            df_raw["timestamp"] = pd.to_datetime(df_raw.get("timestamp"), errors="coerce")
            df = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            if "close" not in df.columns:
                st.error("CSV must include at least 'close' and 'timestamp' to render the fallback chart.")
                st.stop()

            # === UPDATED: Metrics box above charts (fallback shows zeros) ===
            with outputs_box.container():
                st.subheader("Outputs")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Trades Taken", 0)
                m2.metric("Exits (Loss/Breach/Broke/Win)", "0/0/0/0")
                m3.metric("P&L", "$0.00")
                m4.metric("Drawdown", "$0.00")
                m5.metric("Drawdown %", "0.00%")

            # Fallback chart with downsampled Close & simple BB on Close
            st.write("Rendering fallback chart...")
            ma = df["close"].rolling(20).mean()
            ub = ma + 2 * df["close"].rolling(20).std(ddof=0)
            lb = ma - 2 * df["close"].rolling(20).std(ddof=0)

            x = df.index.values
            y = df["close"].values
            x_ds, y_ds = lttb_downsample(x.astype("datetime64[ns]").astype(np.int64), y.astype(float), threshold=int(max_chart_points))
            fig_px = go.Figure()
            ts_ds = pd.to_datetime(x_ds)
            fig_px.add_trace(go.Scatter(x=ts_ds, y=y_ds, name="Close Price", mode="lines", line=dict(color="#60A5FA", width=2.5)))
            fig_px.add_trace(go.Scatter(x=df.index, y=ma, name="BB Mid (20)", mode="lines", line=dict(color="gray", width=1.2)))
            fig_px.add_trace(go.Scatter(x=df.index, y=ub, name="BB Upper (20,2σ)", mode="lines", line=dict(color="orange", width=1.2)))
            fig_px.add_trace(go.Scatter(x=df.index, y=lb, name="BB Lower (20,2σ)", mode="lines", line=dict(color="orange", width=1.2)))

            if blackout_dates and len(df) > 0:
                for e in blackout_dates:
                    start = e - timedelta(days=int(days_before))
                    end = e + timedelta(days=int(days_after))
                    fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

            fig_px.update_layout(
                template="plotly_dark", title="Price with Bollinger Bands (fallback)",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="", yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            with charts_box:
                st.subheader("Charts")
                st.plotly_chart(fig_px, use_container_width=True)

            status.update(label="Completed (fallback)", state="complete")
            st.stop()

        # Full run
        st.write("Preparing and sorting data...")
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
        df_sorted = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp")

        st.write("Running strategy and computing indicators...")
        if timeframe_mode == "Daily":
            df_out, trades_df, equity_df, summary = run_backtest(
                df_raw=df_sorted, blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max, adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k, use_bias=use_trend_bias,
                bias_strength=trend_bias_strength, trend_method=trend_method,
                wing_ext_pct=wing_ext_pct, days_before=days_before, days_after=days_after
            )
            bpd_inferred = 1
        else:
            idx_sorted = pd.DatetimeIndex(df_sorted["timestamp"].values)
            bpd_inferred = infer_bars_per_day(idx_sorted)
            bars_per_day = bars_per_day_override if int(bars_per_day_override) > 0 else bpd_inferred

            df_out, trades_df, equity_df, summary = run_backtest_hourly(
                df_raw=df_sorted, blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max, adx_exit_thr=adx_exit,
                vwap_k=vwap_accept_k, use_bias=use_trend_bias,
                bias_strength=trend_bias_strength, trend_method=trend_method,
                wing_ext_pct=wing_ext_pct, days_before=days_before, days_after=days_after,
                bars_per_day=bars_per_day, days_per_year=days_per_year
            )

        st.write("Rendering outputs...")

        # =====================================================
        # === UPDATED: Metrics box above charts (clean, professional) ===
        # =====================================================
        with outputs_box.container():
            st.subheader("Outputs")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trades Taken", summary["trades"])
            m2.metric("Exits (Loss/Breach/Broke/Win)", f"{summary['losses']}/{summary['breaches']}/{summary['brokes']}/{summary['wins']}")
            m3.metric("P&L", f"${summary['total_pnl']:.2f}")
            m4.metric("Drawdown", f"${summary['max_drawdown']:.2f}")
            m5.metric("Drawdown %", f"{summary['max_drawdown_pct']:.2f}%")

        # =====================================================
        # === UPDATED: Charts (loss chart removed) ===
        # =====================================================
        with charts_box:
            st.subheader("Charts")

            # Equity curve
            if equity_df.empty:
                st.info("No equity curve to display.")
            else:
                x_e = equity_f_df = equity_df.index.values.astype("datetime64[ns]").astype(np.int64)
                y_e = equity_df["cash"].values.astype(float)
                x_e_ds, y_e_ds = lttb_downsample(x_e, y_e, threshold=int(max_chart_points))
                ts_e_ds = pd.to_datetime(x_e_ds)
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=ts_e_ds, y=y_e_ds, name="Equity", mode="lines", line=dict(color="#22D3EE", width=3)))
                fig_eq.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="Cash ($)")
                st.plotly_chart(fig_eq, use_container_width=True)

            # VWAP & Bollinger Bands with Trade Markers
            st.markdown("#### VWAP & Bollinger Bands with Trade Markers")
            fig_px = go.Figure()
            x_full = df_out.index.values.astype("datetime64[ns]").astype(np.int64)
            vwap_full = df_out["vwap"].values.astype(float)
            x_ds, vwap_ds = lttb_downsample(x_full, vwap_full, threshold=int(max_chart_points))
            ts_ds = pd.to_datetime(x_ds)
            fig_px.add_trace(go.Scatter(x=ts_ds, y=vwap_ds, name="VWAP", mode="lines", line=dict(color="steelblue", width=2)))
            fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_upper"], name="BB Upper (20,2σ)", mode="lines", line=dict(color="orange", width=1.5)))
            fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_mid"],   name="BB Mid (20)",       mode="lines", line=dict(color="gray",   width=1.0)))
            fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_lower"], name="BB Lower (20,2σ)", mode="lines", line=dict(color="orange", width=1.5)))

            if blackout_dates:
                for e in blackout_dates:
                    start = e - timedelta(days=int(days_before))
                    end = e + timedelta(days=int(days_after))
                    fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

            if use_trend_bias:
                up_idx = df_out.index[df_out["trend_up"]]
                down_idx = df_out.index[df_out["trend_down"]]
                fig_px.add_trace(go.Scatter(x=up_idx,   y=df_out.loc[up_idx, "vwap"],   name="Uptrend",   mode="markers",
                                            marker=dict(color="green", size=5, opacity=0.5), hoverinfo="skip"))
                fig_px.add_trace(go.Scatter(x=down_idx, y=df_out.loc[down_idx, "vwap"], name="Downtrend", mode="markers",
                                            marker=dict(color="red", size=5, opacity=0.5), hoverinfo="skip"))

            if not trades_df.empty:
                wins_m   = (trades_df["outcome"] == "win")
                losses_m = (trades_df["outcome"] == "loss")
                breach_m = (trades_df["outcome"] == "breach")
                adx_m    = (trades_df["outcome"] == "adx_exit")
                vwap_m   = (trades_df["outcome"] == "vwap_exit")
                broke_m  = (trades_df["outcome"] == "broke")

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

                add_exit(wins_m,   "Exit (win)",   "green",  "x")
                add_exit(losses_m, "Exit (loss)",  "red",    "x")
                add_exit(breach_m, "Exit (breach)","red",    "triangle-down")
                add_exit(adx_m,    "Exit (ADX)",   "purple", "square")
                add_exit(vwap_m,   "Exit (VWAP)",  "orange", "diamond")
                add_exit(broke_m,  "Exit (broke)", "black",  "star")

            fig_px.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="", yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_px, use_container_width=True)

        # =====================================================
        # === UPDATED: Trades table (below charts) ===
        # =====================================================
        with trades_table_box:
            st.subheader("Trades")
            if trades_df.empty:
                st.info("No trades generated under current settings.")
            else:
                show_df = trades_df.copy()
                show_df["entry_date"] = pd.to_datetime(show_df["entry_date"])
                show_df["expiry_date"] = pd.to_datetime(show_df["expiry_date"])
                st.dataframe(show_df, use_container_width=True, hide_index=True)

        status.update(label="Completed", state="complete")

# =========================================================
# Overview (only when not running and no files)
# =========================================================
if not run_clicked:
    st.markdown(
        """
        **Overview**  
        Keep your **Uploads** and **Settings** at the top. Click **Run Backtest** to populate the **Outputs** box and charts below.
        Toggle **Daily / Hourly** without changing your core logic.
        """,
        unsafe_allow_html=True,
    )
