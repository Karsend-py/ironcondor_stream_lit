
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import List

# ------------------------------------------------------------------------------------------
# FAST BOOT
# ------------------------------------------------------------------------------------------
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")

# Minimal header
st.markdown("## 📈 Iron Condor Backtester")

# ------------------------------------------------------------------------------------------
# === Single-page layout: File Upload first ===
# ------------------------------------------------------------------------------------------
with st.container():
    st.subheader("Data Files")
    st.write("Upload the input CSV and blackout dates file to begin.")
    uploaded_csv = st.file_uploader("CSV File (OHLCV + VWAP)", type=["csv"])
    uploaded_txt = st.file_uploader("Blackout Dates (TXT)", type=["txt"])
    st.caption("TXT: one date per line (YYYY-MM-DD). Lines beginning with '#' are ignored.")

has_csv = uploaded_csv is not None
has_txt = uploaded_txt is not None

# === NEW: Timeframe toggle under file upload ===
if has_csv:
    timeframe_choice = st.radio(
        "Timeframe",
        ["Daily", "Hourly", "30-minute", "15-minute", "5-minute", "1-minute"],
        index=0,
        horizontal=True,
        help="Resample the uploaded data to this frequency before indicators/backtest."
    )
else:
    timeframe_choice = "Daily"  # default until CSV is uploaded

# === NEW: Timeframe-aware Bollinger Band window ===
def bb_window_from_timeframe(choice: str, base_days: int = 20) -> int:
    """
    Return number of bars that represent ~base_days trading days for the selected timeframe.
    1 trading day ≈ 390 minutes; 1 hour ≈ 60 minutes.
    """
    mapping = {
        "Daily":      base_days,               # 20 bars
        "Hourly":     int(base_days * 6.5),    # 20 * 6.5 = 130
        "30-minute":  int(base_days * 13),     # 20 * 13  = 260
        "15-minute":  int(base_days * 26),     # 20 * 26  = 520
        "5-minute":   int(base_days * 78),     # 20 * 78  = 1,560
        "1-minute":   int(base_days * 390),    # 20 * 390 = 7,800
    }
    return mapping.get(choice, base_days)

bb_window = bb_window_from_timeframe(timeframe_choice)

# ------------------------------------------------------------------------------------------
# Helpers (cached) — unchanged, except BB uses dynamic window inside compute_indicators
# ------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_blackout_txt(file) -> List[pd.Timestamp]:
    if not file:
        return []
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1")
    out: List[pd.Timestamp] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(pd.Timestamp(pd.to_datetime(s)).normalize())
        except Exception:
            pass
    return out

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    return df

# Timeframe resampling helper
def _freq_from_choice(choice: str) -> str:
    return {
        "Daily": "D",
        "Hourly": "H",
        "30-minute": "30min",
        "15-minute": "15min",
        "5-minute": "5min",
        "1-minute": "T",
    }.get(choice, "D")

def resample_to_timeframe(df_raw: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Resample required columns to the selected timeframe, preserving lower-case column names."""
    if df_raw.empty or "timestamp" not in df_raw.columns:
        return df_raw
    freq = _freq_from_choice(choice)
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    agg = {}
    if "open" in df.columns:   agg["open"]  = "first"
    if "high" in df.columns:   agg["high"]  = "max"
    if "low" in df.columns:    agg["low"]   = "min"
    if "close" in df.columns:  agg["close"] = "last"
    if "vwap" in df.columns:   agg["vwap"]  = "mean"  # simple mean if no volume
    df_res = df.resample(freq).agg(agg)
    df_res = df_res.dropna(how="any")
    return df_res.reset_index()  # keep "timestamp" for downstream code

@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame, bb_window: int) -> pd.DataFrame:
    """
    Compute indicators; Bollinger Bands use a dynamic window tied to the selected timeframe.
    Other indicators remain unchanged to preserve backtest behavior.
    """
    df = df.copy()

    # --- Bollinger Bands on VWAP with timeframe-aware window ---
    ma_vwap  = df["vwap"].rolling(bb_window, min_periods=bb_window).mean()
    std_vwap = df["vwap"].rolling(bb_window, min_periods=bb_window).std(ddof=0)
    df["bb_mid"]   = ma_vwap
    df["bb_upper"] = ma_vwap + 2.0 * std_vwap
    df["bb_lower"] = ma_vwap - 2.0 * std_vwap

    # --- Existing indicators (unchanged) ---
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

    # Keep existing SMA20 for "VWAP vs SMA20" trend method (non-goal to modify other indicators)
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
        df["trend_up"] = False; df["trend_down"] = False
    return df

# -------------------------------- FULL BACKTEST (cached) -----------------------------------
@st.cache_data(show_spinner=True)
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
    days_after: int,
    bb_window: int,   # pass timeframe-aware BB window
):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Indicators with timeframe-aware Bollinger window
    df = compute_indicators(df, bb_window=bb_window)
    df = compute_trend_flags(df, trend_method)

    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
    combined = cond_adx & cond_rsi & cond_hv

    def in_blackout(day):
        day_n = pd.Timestamp(day).normalize()
        for e in blackout_dates:
            e_n = pd.Timestamp(e).normalize()
            if (e_n - timedelta(days=days_before)) <= day_n <= e_n:
                return True
            if e_n <= day_n <= (e_n + timedelta(days=days_after)):
                return True
        return False

    mask_blackout = df.index.to_series().apply(in_blackout)
    eligible = combined & (~mask_blackout.values)

    per_leg_fee = 0.65; mult = 100
    def round_to(x, step=1.0):
        return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, credit):
        put_w = sp - lp; call_w = lc - sc
        if sp <= exp_close <= sc:
            return credit * mult - 4 * per_leg_fee, "win"
        loss_w = call_w if exp_close > sc else put_w
        return -(loss_w - credit) * mult - 4 * per_leg_fee, "loss"

    idx = df.index
    open_positions = []; trades = []; cash = 0.0; eq = []

    for i in range(len(idx)):
        d = idx[i]

        if open_positions:
            keep = []
            for pos in open_positions:
                cur = df.loc[d, "close"]
                pnl_today, _ = eval_condor(cur, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (cur < pos["sp"]) or (cur > pos["sc"])
                broke  = (cur < pos["lp"]) or (cur > pos["lc"])

                adx_out = df.loc[d,"adx"] >= int(adx_exit_thr)
                vwap_today = df.loc[d,"vwap"]; vwap_prev = df.iloc[i-1]["vwap"] if i>0 else vwap_today
                delta_today = vwap_today - vwap_prev
                delta_prev = (df.iloc[i-1]["vwap"] - df.iloc[i-2]["vwap"]) if i>1 else 0.0
                slope_flip = (np.sign(delta_today) != 0) and (np.sign(delta_prev) != 0) and (np.sign(delta_today) != np.sign(delta_prev))
                bb_half = df.loc[d,"bb_upper"] - df.loc[d,"bb_mid"]
                accept_dist = float(vwap_k) * bb_half
                away_enough = abs(cur - vwap_today) >= accept_dist
                on_slope = ((delta_today > 0 and cur > vwap_today) or (delta_today < 0 and cur < vwap_today))
                vwap_exit = slope_flip and away_enough and on_slope

                exited = False; flag=None
                if (d < pos["expiry"]) and broke:
                    exited=True; flag="broke"
                elif (d < pos["expiry"]) and breach:
                    exited=True; flag="breach"
                elif (d < pos["expiry"]) and adx_out:
                    exited=True; flag="adx_exit"
                elif (d < pos["expiry"]) and vwap_exit:
                    exited=True; flag="vwap_exit"

                if exited:
                    cash += pnl_today
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"], "expiry_close": cur,
                        "pnl": pnl_today, "outcome": flag
                    })
                else:
                    keep.append(pos)
            open_positions = keep

        if open_positions:
            keep = []
            for pos in open_positions:
                if d == pos["expiry"]:
                    exp_close = df.loc[d,"close"]
                    pnl,out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": pos["entry"], "expiry_date": d,
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"], "expiry_close": exp_close,
                        "pnl": pnl, "outcome": out
                    })
                else:
                    keep.append(pos)
            open_positions = keep

        if eligible.iloc[i]:
            row = df.iloc[i]
            if use_bias:
                bias = float(bias_strength)
                if df["trend_up"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) + 0.5*bias, 1.0)
                    sc = round_to(float(row["bb_upper"]) + 1.0*bias, 1.0)
                elif df["trend_down"].iloc[i]:
                    sp = round_to(float(row["bb_lower"]) - 1.0*bias, 1.0)
                    sc = round_to(float(row["bb_upper"]) - 0.5*bias, 1.0)
                else:
                    sp = round_to(float(row["bb_lower"]), 1.0)
                    sc = round_to(float(row["bb_upper"]), 1.0)
            else:
                sp = round_to(float(row["bb_lower"]), 1.0)
                sc = round_to(float(row["bb_upper"]), 1.0)

            prev_up = bool(df["trend_up"].iloc[i-1]) if i>0 else False
            prev_dn = bool(df["trend_down"].iloc[i-1]) if i>0 else False
            tightening = bool(df["bb_tightening"].iloc[i])

            ext = 1.0 + max(0.0, float(wing_ext_pct))/100.0
            put_w  = 5.0 * ext if (df["trend_up"].iloc[i] and prev_dn and tightening) else 5.0
            call_w = 5.0 * ext if (df["trend_down"].iloc[i] and prev_up and tightening) else 5.0

            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)

            # find next Friday (≤ 5 DTE)
            end = min(i+5, len(idx)-1)
            expiry = None
            for j in range(i, end+1):
                if idx[j].weekday() == 4:
                    expiry = idx[j]; break
            if expiry is None:
                expiry = idx[end]

            open_positions.append({
                "entry": d, "expiry": expiry,
                "sp": sp, "lp": lp, "sc": sc, "lc": lc, "credit": credit
            })

        eq.append({"date": d, "cash": cash})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame(eq).set_index("date") if eq else pd.DataFrame(columns=["cash"])

    # summary (unchanged)
    wins     = (trades_df["outcome"]=="win").sum()     if not trades_df.empty else 0
    losses   = (trades_df["outcome"]=="loss").sum()    if not trades_df.empty else 0
    breaches = (trades_df["outcome"]=="breach").sum()  if not trades_df.empty else 0
    adx_exits= (trades_df["outcome"]=="adx_exit").sum()if not trades_df.empty else 0
    vwap_exits=(trades_df["outcome"]=="vwap_exit").sum()if not trades_df.empty else 0
    brokes   = (trades_df["outcome"]=="broke").sum()   if not trades_df.empty else 0

    win_rate = (100*wins/len(trades_df)) if not trades_df.empty else 0.0
    total_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0

    # max drawdown (unchanged)
    if not equity_df.empty and not equity_df["cash"].empty:
        run_max = equity_df["cash"].cummax()
        dd = run_max - equity_df["cash"]
        max_dd_val = float(dd.max()) if not dd.empty else 0.0
        if max_dd_val > 0:
            dd_idx = dd.idxmax()
            max_dd_pct = (max_dd_val/run_max.loc[dd_idx])*100 if run_max.loc[dd_idx]!=0 else 0.0
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0; max_dd_pct = 0.0

    summary = {
        "trades": int(len(trades_df)),
        "wins": int(wins), "losses": int(losses),
        "breaches": int(breaches), "adx_exits": int(adx_exits),
        "vwap_exits": int(vwap_exits), "brokes": int(brokes),
        "win_rate": float(win_rate), "total_pnl": total_pnl,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }
    return df, trades_df, equity_df, summary

# ------------------------------------------------------------------------------------------
# 2) Configuration (only shown after CSV upload)
# ------------------------------------------------------------------------------------------
if has_csv:
    with st.container():
        st.subheader("Configuration")
        left, right = st.columns(2)

        with left:
            st.markdown("**Backtest Parameters**")
            days_before = st.number_input("Days Before Earnings", value=7)
            days_after  = st.number_input("Days After Earnings",  value=1)
            adx_exit    = st.number_input("ADX Exit Threshold",   value=30)
            vwap_accept_k = st.number_input("VWAP Exit Distance (k)", value=1.0)
            hv_min      = st.number_input("Min Historical Vol (%)", value=15.0)
            hv_max      = st.number_input("Max Historical Vol (%)", value=40.0)

        with right:
            st.markdown("**Trend Bias Settings**")
            use_trend_bias = st.checkbox("Enable Trend Bias")
            trend_method = st.selectbox("Trend Detection Method", ["VWAP Slope","VWAP vs SMA20","ADX + DI"])
            trend_bias_strength = st.number_input("Bias Strength ($)", value=2.0, disabled=not use_trend_bias)
            wing_ext_pct = st.number_input("Wing Extension (%)", value=20.0)

    run_disabled = not (has_csv and has_txt)
    run_clicked = st.button("Run Backtest", disabled=run_disabled)
    if run_disabled:
        st.caption("Upload both files (CSV and blackout .txt) to enable the backtest.")
else:
    st.info("Upload a CSV to reveal configuration and the backtest button.")

# ------------------------------------------------------------------------------------------
# 3) Summary + Charts (after run)
# ------------------------------------------------------------------------------------------
if has_csv and 'run_clicked' in locals() and run_clicked:
    if not uploaded_txt:
        st.error("Upload blackout .txt to proceed.")
        st.stop()

    # Load + resample to selected timeframe BEFORE running the backtest
    df_raw_original = load_csv(uploaded_csv)
    df_raw = resample_to_timeframe(df_raw_original, timeframe_choice)

    required = {"timestamp","close","high","low","vwap"}
    missing = required - set(df_raw.columns)
    blackout_dates = parse_blackout_txt(uploaded_txt)

    # Fallback chart if CSV lacks required columns for full logic
    if missing:
        st.markdown("### Summary")
        st.info("No results yet. CSV is missing required columns. Showing basic price chart.")
        df_raw["timestamp"] = pd.to_datetime(df_raw.get("timestamp"), errors="coerce")
        df = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        # Fallback BB also uses timeframe-aware window for visual consistency
        ma = df["close"].rolling(bb_window, min_periods=bb_window).mean()
        ub = ma + 2 * df["close"].rolling(bb_window, min_periods=bb_window).std(ddof=0)
        lb = ma - 2 * df["close"].rolling(bb_window, min_periods=bb_window).std(ddof=0)

        st.markdown("### Equity Curve")
        st.info("No equity curve.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index,y=df["close"],name="Close",mode="lines",line=dict(color="#60A5FA",width=2.5)))
        fig.add_trace(go.Scatter(x=df.index,y=ma,name="BB Mid (timeframe-aware)",mode="lines",line=dict(color="gray",width=1.2)))
        fig.add_trace(go.Scatter(x=df.index,y=ub,name="BB Upper",mode="lines",line=dict(color="orange",width=1.2)))
        fig.add_trace(go.Scatter(x=df.index,y=lb,name="BB Lower",mode="lines",line=dict(color="orange",width=1.2)))

        if blackout_dates and len(df)>0:
            for e in blackout_dates:
                start = e - timedelta(days=int(days_before))
                end   = e + timedelta(days=int(days_after))
                fig.add_vrect(x0=start,x1=end,fillcolor="red",opacity=0.08,line_width=0)

        fig.update_layout(template="plotly_dark",margin=dict(l=10,r=10,t=40,b=10),
                          xaxis_title="",yaxis_title="Price ($)",
                          legend=dict(orientation="h",y=1.02))
        st.markdown("### Price Chart with Indicators")
        st.plotly_chart(fig, use_container_width=True)
        st.stop()

    # Full backtest on resampled data with timeframe-aware Bollinger window
    with st.spinner("Running backtest…"):
        df_out, trades_df, equity_df, summary = run_backtest(
            df_raw=df_raw,
            blackout_dates=blackout_dates,
            hv_min=hv_min, hv_max=hv_max,
            adx_exit_thr=adx_exit,
            vwap_k=vwap_accept_k,
            use_bias=use_trend_bias,
            bias_strength=trend_bias_strength,
            trend_method=trend_method,
            wing_ext_pct=wing_ext_pct,
            days_before=days_before, days_after=days_after,
            bb_window=bb_window,
        )

    # Metrics box (unchanged)
    st.markdown("### Summary")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades Taken", f"{summary['trades']}")
        c2.metric("P&L",          f"${summary['total_pnl']:.2f}")
        c3.metric("Wins",         f"{summary['wins']}")
        c4.metric("Losses",       f"{summary['losses']}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Breach",       f"{summary['breaches']}")
        c6.metric("Broke",        f"{summary['brokes']}")
        c7.metric("Drawdown",     f"${summary['max_drawdown']:.2f}")
        c8.metric("Drawdown %",   f"{summary['max_drawdown_pct']:.2f}%")

    # === Equity Curve (with Plotly update_layout fix) ===
    st.markdown("### Equity Curve")
    if equity_df.empty:
        st.info("No equity curve.")
    else:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df["cash"],
                name="Equity",
                mode="lines",
                line=dict(color="#22D3EE", width=3),
            )
        )
        # === FIX: correct method name (no backslash)
        fig_eq.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="",
            yaxis_title="Cash ($)",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # Price Chart with Indicators (unchanged)
    st.markdown("### Price Chart with Indicators")
    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=df_out.index,y=df_out["vwap"],name="VWAP", mode="lines",line=dict(color="steelblue",width=2)))
    fig_px.add_trace(go.Scatter(x=df_out.index,y=df_out["bb_upper"],name="BB Upper", mode="lines",line=dict(color="orange",width=1.5)))
    fig_px.add_trace(go.Scatter(x=df_out.index,y=df_out["bb_mid"],  name="BB Mid",   mode="lines",line=dict(color="gray",width=1.0)))
    fig_px.add_trace(go.Scatter(x=df_out.index,y=df_out["bb_lower"],name="BB Lower", mode="lines",line=dict(color="orange",width=1.5)))

    if blackout_dates:
        for e in blackout_dates:
            start = e - timedelta(days=int(days_before))
            end   = e + timedelta(days=int(days_after))
            fig_px.add_vrect(x0=start,x1=end,fillcolor="red",opacity=0.08,line_width=0)

    if use_trend_bias:
        up_idx = df_out.index[df_out["trend_up"]]
        dn_idx = df_out.index[df_out["trend_down"]]
        fig_px.add_trace(go.Scatter(x=up_idx,y=df_out.loc[up_idx,"vwap"],name="Uptrend",
                                    mode="markers",marker=dict(color="green",size=5,opacity=0.5)))
        fig_px.add_trace(go.Scatter(x=dn_idx,y=df_out.loc[dn_idx,"vwap"],name="Downtrend",
                                    mode="markers",marker=dict(color="red",size=5,opacity=0.5)))

    if not trades_df.empty:
        def add_pts(mask, name, color, symbol):
            fig_px.add_trace(go.Scatter(
                x=trades_df.loc[mask,"entry_date"],
                y=df_out.loc[trades_df.loc[mask,"entry_date"],"vwap"],
                name=name, mode="markers",
                marker=dict(symbol=symbol,color=color,size=9)
            ))
        wins_m   = (trades_df["outcome"]=="win")
        losses_m = (trades_df["outcome"]=="loss")
        add_pts(wins_m,   "Entry (win)","green","triangle-up")
        add_pts(losses_m, "Entry (loss)","red","triangle-up")

        def add_exit(mask, name, color, symbol="x"):
            fig_px.add_trace(go.Scatter(
                x=trades_df.loc[mask,"expiry_date"],
                y=df_out.loc[trades_df.loc[mask,"expiry_date"],"close"],
                name=name, mode="markers",
                marker=dict(symbol=symbol,color=color,size=9)
            ))
        add_exit((trades_df["outcome"]=="win"),      "Exit (win)","green","x")
        add_exit((trades_df["outcome"]=="loss"),     "Exit (loss)","red","x")
        add_exit((trades_df["outcome"]=="breach"),   "Exit (breach)","red","triangle-down")
        add_exit((trades_df["outcome"]=="adx_exit"), "Exit (ADX)","purple","square")
        add_exit((trades_df["outcome"]=="vwap_exit"),"Exit (VWAP)","orange","diamond")
        add_exit((trades_df["outcome"]=="broke"),    "Exit (broke)","black","star")

    fig_px.update_layout(template="plotly_dark",margin=dict(l=10,r=10,t=40,b=10),
                         xaxis_title="",yaxis_title="Price ($)",
                         legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0))
    st.plotly_chart(fig_px, use_container_width=True)

    # === RESTORED: Trades box under price chart ===
    st.markdown("### Trades")
    if trades_df.empty:
        st.info("No trades to display for the current file and timeframe.")
    else:
        trades_display = trades_df.copy()
        # Format numeric columns if present
        for col in ["short_put","long_put","short_call","long_call","net_credit","expiry_close","pnl"]:
            if col in trades_display.columns:
                trades_display[col] = trades_display[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        # Format dates
        for dcol in ["entry_date","expiry_date"]:
            if dcol in trades_display.columns:
                trades_display[dcol] = pd.to_datetime(trades_display[dcol]).dt.strftime("%Y-%m-%d %H:%M")
        # Scrollable table under the chart
        st.dataframe(trades_display, use_container_width=True, height=320)
