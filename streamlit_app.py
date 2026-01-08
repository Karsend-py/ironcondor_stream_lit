
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit.components.v1 import html
from datetime import timedelta
from typing import List

# =========================================================
# Page config & Global CSS (dark app look)
# =========================================================
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#0f172a;          /* page background */
  --panel:#111827;       /* card/sidebar bg */
  --panel2:#0b1221;      /* control surface */
  --text:#e5e7eb;        /* base text */
  --muted:#9ca3af;       /* muted text */
  --primary:#2563EB;     /* brand primary */
  --primary2:#60A5FA;    /* gradient end */
  --border:#263143;
}

/* Base dark polish */
body { background-color: var(--bg); color: var(--text); }
h1,h2,h3 { color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--panel); }

/* Header strip */
.header {
  padding: 14px 18px;
  border-radius: 14px;
  background: linear-gradient(90deg,var(--primary),var(--primary2));
  color: white;
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  box-shadow: 0 8px 30px rgba(0,0,0,.30);
}
.header .title { font-size: 22px; font-weight: 700; }
.header .sub { opacity: .9; }

/* Card container */
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 8px 30px rgba(0,0,0,.30);
}

/* Upload zones */
.upload-box {
  border: 2px dashed #35507a; border-radius: 12px;
  padding: 18px; text-align: left; color: #cde1ff; background-color: var(--panel2);
  margin-bottom: 10px;
}
.upload-box:hover { border-color: var(--primary2); }
.upload-head { font-weight: 700; margin-bottom: 6px; }
.upload-help { font-size: 12px; color: var(--muted); }

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,var(--primary),var(--primary2));
  color: #0b1221; font-weight: 800; border-radius: 10px; padding: 10px 16px;
  border: none;
}

/* Compact KPI cards */
.kpi {
  background:#0b1221;
  border:1px solid #2b3e59;
  border-radius:10px;
  padding:10px 12px;
}
.kpi .label { font-size:12px; color:#9ca3af; margin-bottom:4px; }
.kpi .value { font-size:18px; font-weight:700; color:#cde1ff; }
.kpi .delta { font-size:12px; color:#9ca3af; }

/* Section anchors spacing */
.section { scroll-margin-top: 90px; } /* avoid header overlap */
</style>
""", unsafe_allow_html=True)

# Header (no tabs, single page)
st.markdown("""
<div class="header">
  <div>
    <div class="title">Iron Condor Backtester</div>
    <div class="sub">Options analytics & strategy management</div>
  </div>
  <div style="display:flex;gap:8px">
    <span>v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Frequency selector
freq = st.radio("Select data frequency", ["Daily", "Hourly"])

# Upload blackout dates (shared)
uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type=["txt"], label_visibility="collapsed")
blackout_dates = parse_blackout_txt(uploaded_txt)

# Upload CSV depending on frequency
if freq == "Daily":
    uploaded_csv = st.file_uploader("Upload Daily CSV", type=["csv"], label_visibility="collapsed")
else:
    uploaded_csv = st.file_uploader("Upload Hourly CSV", type=["csv"], label_visibility="collapsed")

# --- Your existing daily indicator function ---
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ma = df["vwap"].rolling(20).mean()
    std = df["vwap"].rolling(20).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

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

    df["vwap_sma20"] = df["vwap"].rolling(20).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(20, min_periods=5).median())
    ).fillna(False)

    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(21).std(ddof=0) * np.sqrt(252) * 100).bfill().ffill()

    return df

# --- Hourly indicator function ---
def compute_indicators_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ma = df["vwap"].rolling(20).mean()
    std = df["vwap"].rolling(20).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_upper"] = ma + 2.0 * std
    df["bb_lower"] = ma - 2.0 * std

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = (100 - (100 / (1 + rs))).bfill().ffill()

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

    df["vwap_sma20"] = df["vwap"].rolling(20).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_tightening"] = (
        (df["bb_width"] < df["bb_width"].shift(1)) &
        (df["bb_width"] < df["bb_width"].rolling(20, min_periods=5).median())
    ).fillna(False)

    log_ret = np.log(df["close"]).diff()
    df["hv"] = (log_ret.rolling(21*6, min_periods=10).std(ddof=0) * np.sqrt(252*6.5) * 100).bfill().ffill()

    return df

# --- Trend flags (shared) ---
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

# --- Blackout helper (shared) ---
def in_blackout(day, earnings: List[pd.Timestamp], pre: int, post: int):
    day_n = pd.Timestamp(day).normalize()
    for e in earnings:
        e_n = pd.Timestamp(e).normalize()
        if (e_n - timedelta(days=pre)) <= day_n <= e_n:
            return True
        if e_n <= day_n <= (e_n + timedelta(days=post)):
            return True
    return False

# --- Next Friday finder (shared) ---
def next_friday_within(idx, start_loc, max_dte=5):
    end_loc = min(start_loc + max_dte, len(idx) - 1)
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            return idx[j]
    return idx[end_loc]

# --- Backtest for daily data (your existing function) ---
def run_backtest(
    df_raw: pd.DataFrame,
    blackout_dates: List[pd.Timestamp],
    hv_min: float, hv_max: float,
    adx_exit_thr: int, vwap_k: float,
    use_bias: bool, bias_strength: float,
    trend_method: str, wing_ext_pct: float,
    days_before: int, days_after: int
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    df = compute_indicators(df)
    df = compute_trend_flags(df, trend_method)

    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv  = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    mask_blackout = df.index.to_series().apply(lambda d: in_blackout(d, blackout_dates, days_before, days_after))
    eligible = combined & (~mask_blackout.values)

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

    idx = df.index
    open_positions = []
    trades = []
    cash = 0.0
    eq = []

    ADX_EXIT = int(adx_exit_thr)
    VWAP_ACCEPT_K = float(vwap_k)

    for i in range(len(idx)):
        d = idx[i]

        if open_positions:
            still_open = []
            for pos in open_positions:
                current_close = df.loc[d, "close"]
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke  = (current_close < pos["lp"]) or (current_close > pos["lc"])
                adx_exit_now = (df.loc[d, "adx"] >= ADX_EXIT)

                vwap_today = df.loc[d, "vwap"]
                vwap_prev  = df.iloc[i-1]["vwap"] if i > 0 else vwap_today
                delta_today = vwap_today - vwap_prev
                delta_prev  = (df.iloc[i-1]["vwap"] - df.iloc[i-2]["vwap"]) if i > 1 else 0.0
                sign_today = np.sign(delta_today)
                sign_prev  = np.sign(delta_prev)
                slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
                bb_halfwidth = df.loc[d, "bb_upper"] - df.loc[d, "bb_mid"]
                accept_dist = VWAP_ACCEPT_K * bb_halfwidth
                away_enough = abs(current_close - vwap_today) >= accept_dist
                on_slope_side = ((delta_today > 0 and current_close > vwap_today) or
                                 (delta_today < 0 and current_close < vwap_today))
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

        if eligible.iloc[i]:
            row = df.iloc[i]
            if use_bias:
                bias = float(bias_strength)
                if df["trend_up"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) + 0.5 * bias, 1.0)
                    sc = round_to(float(row["bb_upper"]) + 1.0 * bias, 1.0)
                elif df["trend_down"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) - 1.0 * bias, 1.0)
                    sc = round_to(float(row["bb
