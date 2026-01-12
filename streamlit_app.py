
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import List

st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")
st.markdown("## 📈 Iron Condor Backtester")

# ========================= Uploads =========================
with st.container():
    st.subheader("Data Files")
    st.write("Upload the input CSV and blackout dates file to begin.")
    uploaded_csv = st.file_uploader("CSV File (OHLCV + VWAP)", type=["csv"])
    uploaded_txt = st.file_uploader("Blackout Dates (TXT)", type=["txt"])
    st.caption("TXT: one date per line (YYYY-MM-DD). Lines beginning with '#' are ignored.")

has_csv = uploaded_csv is not None
has_txt = uploaded_txt is not None

# Initialize execution control state early
if "bt_running" not in st.session_state:
    st.session_state.bt_running = False
if "bt_results" not in st.session_state:
    st.session_state.bt_results = None
if "bt_params_key" not in st.session_state:
    st.session_state.bt_params_key = None
if "show_rejected_panel" not in st.session_state:
    st.session_state.show_rejected_panel = False

# Timeframe toggle
if has_csv:
    timeframe_choice = st.radio(
        "Timeframe",
        ["Daily", "Hourly", "30-minute", "15-minute", "5-minute", "1-minute"],
        index=0,
        horizontal=True,
        help="Resample the uploaded data to this frequency before indicators/backtest."
    )
else:
    timeframe_choice = "Daily"


# BB window from timeframe (constant ~20 trading days)
def bb_window_from_timeframe(choice: str, base_days: int = 20) -> int:
    mapping = {
        "Daily": base_days,
        "Hourly": int(base_days * 6.5),
        "30-minute": int(base_days * 13),
        "15-minute": int(base_days * 26),
        "5-minute": int(base_days * 78),
        "1-minute": int(base_days * 390),
    }
    return mapping.get(choice, base_days)


bb_window = bb_window_from_timeframe(timeframe_choice)

# ========================= Helpers (cached) =========================
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
    if df_raw.empty or "timestamp" not in df_raw.columns:
        return df_raw
    freq = _freq_from_choice(choice)
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    agg = {}
    if "open" in df.columns: agg["open"] = "first"
    if "high" in df.columns: agg["high"] = "max"
    if "low" in df.columns: agg["low"] = "min"
    if "close" in df.columns: agg["close"] = "last"
    if "vwap" in df.columns: agg["vwap"] = "mean"
    df_res = df.resample(freq).agg(agg)
    df_res = df_res.dropna(how="any")
    return df_res.reset_index()


@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame, bb_window: int) -> pd.DataFrame:
    df = df.copy()
    # Timeframe-aware Bollinger Bands on VWAP
    ma_vwap = df["vwap"].rolling(bb_window, min_periods=bb_window).mean()
    std_vwap = df["vwap"].rolling(bb_window, min_periods=bb_window).std(ddof=0)
    df["bb_mid"] = ma_vwap
    df["bb_upper"] = ma_vwap + 2.0 * std_vwap
    df["bb_lower"] = ma_vwap - 2.0 * std_vwap

    # Other indicators (unchanged)
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
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
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


def _vectorized_blackout_mask(index_ts: pd.Index, blackout_dates: List[pd.Timestamp], days_before: int, days_after: int) -> np.ndarray:
    """Vectorized blackout mask over the datetime index (normalized to date)."""
    if not blackout_dates or len(index_ts) == 0:
        return np.zeros(len(index_ts), dtype=bool)
    idx_norm = pd.to_datetime(index_ts).normalize()
    mask = np.zeros(len(idx_norm), dtype=bool)
    for e in blackout_dates:
        e = pd.Timestamp(e).normalize()
        before_start = e - timedelta(days=days_before)
        after_end = e + timedelta(days=days_after)
        # Union of [e - days_before, e] and [e, e + days_after] equals [e - days_before, e + days_after]
        rng = (idx_norm >= before_start) & (idx_norm <= after_end)
        # Keep as union to match original inclusive logic
        mask |= rng
    return mask


# ========================= Backtest (performance-refactored) =========================
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
    bb_window: int,
    progress=None,  # st.progress instance or None
):
    # --- Precompute indicators once ---
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = compute_indicators(df, bb_window=bb_window)
    df = compute_trend_flags(df, trend_method)

    # --- Prepare arrays for loop (integer indexing only) ---
    idx_ts = df.index.to_numpy()

    # Core series (numpy arrays)
    close = df["close"].to_numpy(dtype=float)
    vwap = df["vwap"].to_numpy(dtype=float)
    bb_mid = df["bb_mid"].to_numpy(dtype=float)
    bb_upper = df["bb_upper"].to_numpy(dtype=float)
    bb_lower = df["bb_lower"].to_numpy(dtype=float)
    adx = df["adx"].to_numpy(dtype=float)
    rsi = df["rsi"].to_numpy(dtype=float)
    hv = df["hv"].to_numpy(dtype=float)

    # Trend / features
    trend_up = df["trend_up"].to_numpy(dtype=bool)
    trend_down = df["trend_down"].to_numpy(dtype=bool)
    bb_tightening = df["bb_tightening"].to_numpy(dtype=bool)
    bb_half = (bb_upper - bb_mid).astype(float)

    # Entry eligibility conditions (vectorized)
    cond_adx = adx < 20
    cond_rsi = (rsi >= 40) & (rsi <= 60)
    cond_hv = (hv >= hv_min) & (hv <= hv_max)

    mask_blackout = _vectorized_blackout_mask(df.index, blackout_dates, days_before, days_after)
    eligible = cond_adx & cond_rsi & cond_hv & (~mask_blackout)

    # Collect diagnostics for rejected (evaluated but not taken)
    rejected = []
    # Build reasons list only for non-eligible points; integer indexing with arrays
    not_eligible_idx = np.where(~eligible)[0]
    for ii in not_eligible_idx:
        reasons = []
        if not cond_hv[ii]: reasons.append("HV outside range")
        if not cond_adx[ii]: reasons.append("ADX too high")
        if not cond_rsi[ii]: reasons.append("RSI out of range")
        if mask_blackout[ii]: reasons.append("Blackout window")
        rejected.append({
            "date": idx_ts[ii],
            "reasons": ", ".join(reasons) if reasons else "Not eligible",
            "adx": float(adx[ii]),
            "rsi": float(rsi[ii]),
            "hv": float(hv[ii]),
            "blackout": bool(mask_blackout[ii]),
        })

    # Fees/constants
    per_leg_fee = 0.65
    mult = 100

    def round_to(x, step=1.0): return float(np.round(x / step) * step)

    def eval_condor(exp_close, sp, lp, sc, lc, credit):
        put_w = sp - lp
        call_w = lc - sc
        if sp <= exp_close <= sc:
            return credit * mult - 4 * per_leg_fee, "win"
        loss_w = call_w if exp_close > sc else put_w
        return -(loss_w - credit) * mult - 4 * per_leg_fee, "loss"

    n = len(idx_ts)
    open_positions = []  # store dicts with integer indices/values
    trades = []
    cash = 0.0
    eq = []

    # Progress control
    if progress is not None:
        progress.progress(0)
    update_every = max(1, n // 100)  # ~1% steps

    # --- Lightweight loop: integer indexing + boolean checks/state updates ---
    for i in range(n):
        # Exit processing for existing positions
        if open_positions:
            keep = []
            cur = close[i]
            adx_out = adx[i] >= int(adx_exit_thr)

            # VWAP exit precomputations (integer-only)
            # delta_today and delta_prev from vwap series
            delta_today = vwap[i] - (vwap[i-1] if i > 0 else vwap[i])
            delta_prev = (vwap[i-1] - vwap[i-2]) if i > 1 else 0.0
            sign_today = np.sign(delta_today)
            sign_prev = np.sign(delta_prev)
            slope_flip = (sign_today != 0) and (sign_prev != 0) and (sign_today != sign_prev)
            accept_dist = float(vwap_k) * bb_half[i]
            away_enough = abs(cur - vwap[i]) >= accept_dist
            on_slope = ((delta_today > 0 and cur > vwap[i]) or (delta_today < 0 and cur < vwap[i]))
            vwap_exit = slope_flip and away_enough and on_slope

            for pos in open_positions:
                breach = (cur < pos["sp"]) or (cur > pos["sc"])
                broke = (cur < pos["lp"]) or (cur > pos["lc"])
                exited = False
                flag = None

                # Only exit before expiry via rules
                if (i < pos["expiry_idx"]) and broke:
                    exited = True; flag = "broke"
                elif (i < pos["expiry_idx"]) and breach:
                    exited = True; flag = "breach"
                elif (i < pos["expiry_idx"]) and adx_out:
                    exited = True; flag = "adx_exit"
                elif (i < pos["expiry_idx"]) and vwap_exit:
                    exited = True; flag = "vwap_exit"

                if exited:
                    pnl_today, _ = eval_condor(cur, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl_today
                    trades.append({
                        "entry_date": idx_ts[pos["entry_idx"]], "expiry_date": idx_ts[i],
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"], "expiry_close": cur,
                        "pnl": pnl_today, "outcome": flag
                    })
                else:
                    keep.append(pos)
            open_positions = keep

        # Expiration handling (positions that reach expiry today)
        if open_positions:
            keep = []
            for pos in open_positions:
                if i == pos["expiry_idx"]:
                    exp_close = close[i]
                    pnl, out = eval_condor(exp_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                    cash += pnl
                    trades.append({
                        "entry_date": idx_ts[pos["entry_idx"]], "expiry_date": idx_ts[i],
                        "short_put": pos["sp"], "long_put": pos["lp"],
                        "short_call": pos["sc"], "long_call": pos["lc"],
                        "net_credit": pos["credit"], "expiry_close": exp_close,
                        "pnl": pnl, "outcome": out
                    })
                else:
                    keep.append(pos)
            open_positions = keep

        # Entry evaluation
        if eligible[i]:
            # Bias-adjusted strikes derived from VWAP-based Bollinger Bands
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
            prev_dn = bool(trend_down[i-1]) if i > 0 else False
            tightening = bool(bb_tightening[i])
            ext = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext if (trend_up[i] and prev_dn and tightening) else 5.0
            call_w = 5.0 * ext if (trend_down[i] and prev_up and tightening) else 5.0
            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)

            # Expiry selection: next Friday within 5 bars, else last of window
            end = min(i + 5, n - 1)
            expiry_idx = None
            for j in range(i, end + 1):
                if pd.Timestamp(idx_ts[j]).weekday() == 4:
                    expiry_idx = j; break
            if expiry_idx is None:
                expiry_idx = end

            open_positions.append({
                "entry_idx": i, "expiry_idx": expiry_idx,
                "sp": sp, "lp": lp, "sc": sc, "lc": lc, "credit": credit
            })

        # Equity curve point each step (no slicing)
        eq.append({"date": idx_ts[i], "cash": cash})

        # Update progress (non-blocking UI, integer increments)
        if progress is not None and (i % update_every == 0 or i == n - 1):
            pct = int((i + 1) * 100 // n)
            progress.progress(pct)

    # Build outputs
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame(eq).set_index("date") if eq else pd.DataFrame(columns=["cash"])

    # Summary metrics (unchanged logic)
    wins = (trades_df["outcome"] == "win").sum() if not trades_df.empty else 0
    losses = (trades_df["outcome"] == "loss").sum() if not trades_df.empty else 0
    breaches = (trades_df["outcome"] == "breach").sum() if not trades_df.empty else 0
    adx_exits = (trades_df["outcome"] == "adx_exit").sum() if not trades_df.empty else 0
    vwap_exits = (trades_df["outcome"] == "vwap_exit").sum() if not trades_df.empty else 0
    brokes = (trades_df["outcome"] == "broke").sum() if not trades_df.empty else 0
    win_rate = (100 * wins / len(trades_df)) if not trades_df.empty else 0.0
    total_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0

    # Max drawdown ($ and %) from equity curve
    if not equity_df.empty and not equity_df["cash"].empty:
        run_max = equity_df["cash"].cummax()
        dd = run_max - equity_df["cash"]
        max_dd_val = float(dd.max()) if not dd.empty else 0.0
        if max_dd_val > 0:
            dd_idx = dd.idxmax()
            max_dd_pct = (max_dd_val / run_max.loc[dd_idx]) * 100 if run_max.loc[dd_idx] != 0 else 0.0
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
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct,  # keep % for UI
    }
    rejected_df = pd.DataFrame(rejected)
    return df, trades_df, equity_df, summary, rejected_df


# ========================= Configuration =========================
if has_csv:
    with st.container():
        st.subheader("Configuration")
        left, right = st.columns(2)
        with left:
            st.markdown("**Backtest Parameters**")
            days_before = st.number_input("Days Before Earnings", value=7)
            days_after = st.number_input("Days After Earnings", value=1)
            adx_exit = st.number_input("ADX Exit Threshold", value=30)
            vwap_accept_k = st.number_input("VWAP Exit Distance (k)", value=1.0)
            hv_min = st.number_input("Min Historical Vol (%)", value=15.0)
            hv_max = st.number_input("Max Historical Vol (%)", value=40.0)
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

# ========================= Results =========================
if has_csv:
    # Prepare data and params key (for execution control)
    if uploaded_txt:
        blackout_dates = parse_blackout_txt(uploaded_txt)
    else:
        blackout_dates = []

    # Load and resample only once per run (cached)
    df_raw_original = load_csv(uploaded_csv) if uploaded_csv else pd.DataFrame()
    df_raw = resample_to_timeframe(df_raw_original, timeframe_choice)

    required = {"timestamp", "close", "high", "low", "vwap"}
    missing = required - set(df_raw.columns)

    # If the button is clicked, start a controlled run only once
    if 'run_clicked' in locals() and run_clicked:
        # Build a params key to avoid restarting the same run on subsequent reruns
        # Use dataset bounds + essential params
        if not df_raw.empty and "timestamp" in df_raw.columns:
            ts_sorted = pd.to_datetime(df_raw["timestamp"]).sort_values()
            params_key = (
                timeframe_choice, bb_window,
                days_before, days_after, adx_exit, vwap_accept_k, hv_min, hv_max,
                use_trend_bias, trend_method, trend_bias_strength, wing_ext_pct,
                int(len(df_raw)), ts_sorted.iloc[0], ts_sorted.iloc[-1]
            )
        else:
            params_key = ("empty",)

        # Only trigger if not currently running and (new params or no previous results)
        start_new_run = (not st.session_state.bt_running) and (st.session_state.bt_params_key != params_key)
        if start_new_run:
            st.session_state.bt_running = True
            st.session_state.bt_params_key = params_key
            st.session_state.bt_results = None  # clear previous

            # Early checks for missing columns fallback display
            if missing:
                st.markdown("### Summary")
                st.info("No results yet. CSV is missing required columns. Showing basic price chart.")
                df_raw["timestamp"] = pd.to_datetime(df_raw.get("timestamp"), errors="coerce")
                df_fallback = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
                # fallback BB uses timeframe-aware window
                ma = df_fallback["close"].rolling(bb_window, min_periods=bb_window).mean()
                ub = ma + 2 * df_fallback["close"].rolling(bb_window, min_periods=bb_window).std(ddof=0)
                lb = ma - 2 * df_fallback["close"].rolling(bb_window, min_periods=bb_window).std(ddof=0)

                st.markdown("### Equity Curve")
                st.info("No equity curve.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_fallback.index, y=df_fallback["close"], name="Close", mode="lines",
                                         line=dict(color="#60A5FA", width=2.5)))
                fig.add_trace(go.Scatter(x=df_fallback.index, y=ma, name="BB Mid (timeframe-aware)", mode="lines",
                                         line=dict(color="gray", width=1.2)))
                fig.add_trace(go.Scatter(x=df_fallback.index, y=ub, name="BB Upper", mode="lines",
                                         line=dict(color="orange", width=1.2)))
                fig.add_trace(go.Scatter(x=df_fallback.index, y=lb, name="BB Lower", mode="lines",
                                         line=dict(color="orange", width=1.2)))
                if blackout_dates and len(df_fallback) > 0:
                    for e in blackout_dates:
                        start = e - timedelta(days=int(days_before))
                        end = e + timedelta(days=int(days_after))
                        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                                  xaxis_title="", yaxis_title="Price ($)",
                                  legend=dict(orientation="h", y=1.02))
                st.markdown("### Price Chart with Indicators")
                st.plotly_chart(fig, use_container_width=True)

                # End the run attempt (no heavy backtest)
                st.session_state.bt_running = False
                st.stop()

            # Run with visible progress bar (deferred UI rendering until completion)
            progress_ph = st.empty()
            progress_bar = progress_ph.progress(0)
            with st.spinner("Running backtest…"):
                df_out, trades_df, equity_df, summary, rejected_df = run_backtest(
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
                    progress=progress_bar,
                )
            progress_ph.empty()  # remove progress bar when done
            st.session_state.bt_results = (df_out, trades_df, equity_df, summary, rejected_df)
            st.session_state.bt_running = False

    # If we have results (either just finished or cached in session_state), render the UI
    if st.session_state.bt_results is not None and not st.session_state.bt_running:
        df_out, trades_df, equity_df, summary, rejected_df = st.session_state.bt_results

        # ===== Top Stats (with Drawdown %) =====
        PER_LEG_FEE = 0.65
        MULT = 100.0

        if not trades_df.empty:
            put_w = trades_df["short_put"] - trades_df["long_put"]
            call_w = trades_df["long_call"] - trades_df["short_call"]
            width = np.minimum(put_w, call_w)
            max_risk_vec = (width - trades_df["net_credit"]) * MULT + 4 * PER_LEG_FEE
        else:
            max_risk_vec = pd.Series([], dtype=float)

        pnl_series = trades_df["pnl"] if not trades_df.empty else pd.Series([], dtype=float)
        total_pl = float(pnl_series.sum()) if len(pnl_series) > 0 else 0.0
        high_pl = float(pnl_series.max()) if len(pnl_series) > 0 else 0.0
        low_pl = float(pnl_series.min()) if len(pnl_series) > 0 else 0.0
        max_risk = float(max_risk_vec.max()) if len(max_risk_vec) > 0 else 0.0

        # $ and % drawdown
        max_dd_val = summary.get("max_drawdown", 0.0)
        max_dd_pct = summary.get("max_drawdown_pct", 0.0)

        wins_mask = trades_df["outcome"].eq("win") if not trades_df.empty else pd.Series([], dtype=bool)
        losses_mask = trades_df["outcome"].eq("loss") if not trades_df.empty else pd.Series([], dtype=bool)

        win_sum = float(pnl_series[wins_mask].clip(lower=0).sum()) if len(pnl_series) > 0 else 0.0
        loss_sum = float(pnl_series[losses_mask].clip(upper=0).sum()) if len(pnl_series) > 0 else 0.0
        profit_factor = float(win_sum / abs(loss_sum)) if loss_sum < 0 else np.nan

        count = int(len(trades_df))
        wins_n = int(wins_mask.sum()) if count > 0 else 0
        losses_n = int(losses_mask.sum()) if count > 0 else 0
        win_rate = (100.0 * wins_n / count) if count > 0 else 0.0
        best_win = float(pnl_series[wins_mask].max()) if wins_n > 0 else 0.0
        worst_loss = float(pnl_series[losses_mask].min()) if losses_n > 0 else 0.0

        def longest_streak(mask_bool):
            longest = cur = 0
            for v in mask_bool.tolist():
                if v: cur += 1
                else: cur = 0
                longest = max(longest, cur)
            return longest

        win_streak = longest_streak(wins_mask) if count > 0 else 0
        loss_streak = longest_streak(losses_mask) if count > 0 else 0
        avg_pl = float(pnl_series.mean()) if len(pnl_series) > 0 else 0.0
        avg_risk = float(max_risk_vec.mean()) if len(max_risk_vec) > 0 else 0.0
        ret_on_risk = (total_pl / float(max_risk_vec.sum()) * 100.0) if len(max_risk_vec) > 0 and max_risk_vec.sum() > 0 else 0.0
        avg_win = float(pnl_series[wins_mask].mean()) if wins_n > 0 else np.nan
        avg_loss = abs(float(pnl_series[losses_mask].mean())) if losses_n > 0 else np.nan
        reward_risk = float(avg_win / avg_loss) if (wins_n > 0 and losses_n > 0 and avg_loss > 0) else np.nan

        expired_n = wins_n + losses_n
        breach_n = int(trades_df["outcome"].eq("breach").sum()) if count > 0 else 0
        broke_n = int(trades_df["outcome"].eq("broke").sum()) if count > 0 else 0

        # ===== Summary metrics grid (no charts here) =====
        st.markdown("### Summary")
        st.markdown("#### Stats")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Total P/L", f"${total_pl:.2f}", delta=f"{total_pl:+.2f}")
        s2.metric("High P/L", f"${high_pl:.2f}", delta=f"{high_pl:+.2f}")
        s3.metric("Low P/L", f"${low_pl:.2f}", delta=f"{low_pl:+.2f}")
        s4.metric("Max Risk", f"${max_risk:.2f}")
        # Drawdown shows $ value and % delta
        s5.metric("Max Drawdown", f"${max_dd_val:.2f}", delta=f"{-abs(max_dd_pct):.2f}%")
        s6.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "—")

        st.markdown("#### Positions")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Count", f"{count}")
        p2.metric("Win Rate", f"{win_rate:.2f}%")
        p3.metric("Win / Loss", f"{wins_n}/{losses_n}")
        p4.metric("Best Win", f"${best_win:.2f}", delta=f"{best_win:+.2f}")
        p5.metric("Worst Loss", f"${worst_loss:.2f}", delta=f"{worst_loss:+.2f}")
        p6.metric("Win Streak", f"{win_streak}")

        st.markdown("#### Averages")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Avg P/L", f"${avg_pl:.2f}", delta=f"{avg_pl:+.2f}")
        a2.metric("Avg Risk", f"${avg_risk:.2f}")
        a3.metric("Return on Risk", f"{ret_on_risk:.2f}%")
        a4.metric("Reward / Risk", f"{reward_risk:.2f}" if not np.isnan(reward_risk) else "—")
        a5.metric("Loss Streak", f"{loss_streak}")

        st.markdown("#### Exits")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Wins", f"{wins_n}")
        e2.metric("Losses", f"{losses_n}")
        e3.metric("Expired", f"{expired_n}")
        e4.metric("Breach", f"{breach_n}")
        e5.metric("Broke", f"{broke_n}")

        # ===== Rejected (Filtered) Diagnostics Panel =====
        filtered_count = int(len(rejected_df)) if not rejected_df.empty else 0
        st.markdown("#### Filtered (Rejected) Diagnostics")
        st.checkbox(
            f"Show rejected diagnostics ({filtered_count})",
            key="show_rejected_panel",
            help="Toggle to open/close the diagnostic panel for rejected evaluations."
        )
        with st.expander("Rejected Diagnostics Panel", expanded=st.session_state.show_rejected_panel):
            st.caption(f"Total filtered (rejected) evaluations: {filtered_count}")
            if rejected_df.empty:
                st.info("No rejected evaluations to display.")
            else:
                # Reason chips with counts (non-interactive)
                reasons_series = rejected_df["reasons"].str.split(", ").explode()
                counts = reasons_series.value_counts().reset_index()
                counts.columns = ["Reason", "Count"]
                chips = st.columns(min(6, len(counts)) or 1)
                for i, (_, row) in enumerate(counts.iterrows()):
                    chips[i % len(chips)].button(f"{row['Reason']} • {row['Count']}", disabled=True)

                # Limit large table rows for performance (latest 1000 rows)
                display_cols = ["date", "reasons", "adx", "rsi", "hv", "blackout"]
                tbl = rejected_df.copy()
                tbl["date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d %H:%M")
                # Show the most recent evaluations first, capped
                max_rows = 1000
                tbl_sorted = tbl.sort_values("date")
                if len(tbl_sorted) > max_rows:
                    tbl_sorted = tbl_sorted.tail(max_rows)
                st.dataframe(tbl_sorted[display_cols], use_container_width=True, height=360)

        # ===== Monthly P/L (calculated once from finalized trades) =====
        with st.expander("Monthly P/L", expanded=False):
            if trades_df.empty:
                st.info("No trades for monthly summary.")
            else:
                mt = trades_df.copy()
                mt["month"] = pd.to_datetime(mt["expiry_date"]).dt.to_period("M").astype(str)
                monthly_tbl = mt.groupby("month", as_index=False)["pnl"].sum().rename(columns={"pnl": "monthly_pnl"})
                st.dataframe(monthly_tbl.sort_values("month"), use_container_width=True, height=260)

        # ===== Equity Curve =====
        st.markdown("### Equity Curve")
        if equity_df.empty:
            st.info("No equity curve.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity_df.index, y=equity_df["cash"], name="Equity",
                                        mode="lines", line=dict(color="#22D3EE", width=3)))
            fig_eq.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="", yaxis_title="Cash ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

        # ===== Price Chart with Indicators =====
        st.markdown("### Price Chart with Indicators")
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["vwap"], name="VWAP",
                                    mode="lines", line=dict(color="steelblue", width=2)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_upper"], name="BB Upper",
                                    mode="lines", line=dict(color="orange", width=1.5)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_mid"], name="BB Mid",
                                    mode="lines", line=dict(color="gray", width=1.0)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_lower"], name="BB Lower",
                                    mode="lines", line=dict(color="orange", width=1.5)))
        if blackout_dates:
            for e in blackout_dates:
                start = e - timedelta(days=int(days_before))
                end = e + timedelta(days=int(days_after))
                fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)
        if use_trend_biaS:=use_trend_bias:  # keep same behavior; small alias
            up_idx = df_out.index[df_out["trend_up"]]
            dn_idx = df_out.index[df_out["trend_down"]]
            fig_px.add_trace(go.Scatter(x=up_idx, y=df_out.loc[up_idx, "vwap"], name="Uptrend",
                                        mode="markers", marker=dict(color="green", size=5, opacity=0.5)))
            fig_px.add_trace(go.Scatter(x=dn_idx, y=df_out.loc[dn_idx, "vwap"], name="Downtrend",
                                        mode="markers", marker=dict(color="red", size=5, opacity=0.5)))

        if not trades_df.empty:
            def add_pts(mask, name, color, symbol):
                fig_px.add_trace(go.Scatter(
                    x=trades_df.loc[mask, "entry_date"],
                    y=df_out.loc[trades_df.loc[mask, "entry_date"], "vwap"],
                    name=name, mode="markers",
                    marker=dict(symbol=symbol, color=color, size=9)
                ))

            wins_m = (trades_df["outcome"] == "win")
            losses_m = (trades_df["outcome"] == "loss")
            add_pts(wins_m, "Entry (win)", "green", "triangle-up")
            add_pts(losses_m, "Entry (loss)", "red", "triangle-up")

            def add_exit(mask, name, color, symbol="x"):
                fig_px.add_trace(go.Scatter(
                    x=trades_df.loc[mask, "expiry_date"],
                    y=df_out.loc[trades_df.loc[mask, "expiry_date"], "close"],
                    name=name, mode="markers",
                    marker=dict(symbol=symbol, color=color, size=9)
                ))

            add_exit((trades_df["outcome"] == "win"), "Exit (win)", "green", "x")
            add_exit((trades_df["outcome"] == "loss"), "Exit (loss)", "red", "x")
            add_exit((trades_df["outcome"] == "breach"), "Exit (breach)", "red", "triangle-down")
            add_exit((trades_df["outcome"] == "adx_exit"), "Exit (ADX)", "purple", "square")
            add_exit((trades_df["outcome"] == "vwap_exit"), "Exit (VWAP)", "orange", "diamond")
            add_exit((trades_df["outcome"] == "broke"), "Exit (broke)", "black", "star")

        fig_px.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                             xaxis_title="", yaxis_title="Price ($)",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_px, use_container_width=True)

        # ===== Trades box (limit very large tables) =====
        st.markdown("### Trades")
        if trades_df.empty:
            st.info("No trades to display for the current file and timeframe.")
        else:
            trades_display = trades_df.copy()
            for col in ["short_put", "long_put", "short_call", "long_call", "net_credit", "expiry_close", "pnl"]:
                if col in trades_display.columns:
                    trades_display[col] = trades_display[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            for dcol in ["entry_date", "expiry_date"]:
                if dcol in trades_display.columns:
                    trades_display[dcol] = pd.to_datetime(trades_display[dcol]).dt.strftime("%Y-%m-%d %H:%M")
            # Show last 1000 rows for performance
            max_trade_rows = 1000
            if len(trades_display) > max_trade_rows:
                trades_display = trades_display.tail(max_trade_rows)
            st.dataframe(trades_display, use_container_width=True, height=360)
