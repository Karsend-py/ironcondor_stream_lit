import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

/* Tables */
[data-testid="stTable"] { background: var(--panel2); }

/* Sticky bottom action bar (visual CTA) */
.sticky-bar{
  position:sticky; bottom:0; background:rgba(12,18,32,.85);
  backdrop-filter:blur(6px); border-top:1px solid var(--border);
  padding:12px 0; z-index:999;
}
.sticky-inner{
  max-width:1100px; margin:0 auto; display:flex;
  justify-content:space-between; align-items:center; gap:12px; color:#cde1ff;
}
.pill { background:#14223a; border:1px solid #2b3e59; border-radius:999px; padding:6px 10px; font-size:13px }
.cta {
  background:linear-gradient(90deg,var(--primary),var(--primary2));
  border:none; border-radius:10px; color:#091426; font-weight:800; padding:10px 16px; cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

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

# =========================================================
# Tabs
# =========================================================
tab_overview, tab_uploads, tab_settings, tab_results = st.tabs(["Overview", "Uploads", "Settings", "Results"])

with tab_overview:
    st.markdown("""
    <div class="card">
      <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Welcome</div>
      <div style="color:#c9d4e3;">
        Use the <b>Uploads</b> and <b>Settings</b> tabs to configure, then press <b>Run Backtest</b>.
        Results will appear under <b>Results</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# Uploads (styled)
# =========================================================
with tab_uploads:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-box"><div class="upload-head">Upload CSV</div><div class="upload-help">Drag/drop or browse • Limit 200MB • CSV • Expected columns: timestamp, close, high, low, vwap</div></div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown('<div class="upload-box"><div class="upload-head">Upload Blackout Dates (.txt)</div><div class="upload-help">One date per line (YYYY-MM-DD). Lines starting with # are ignored.</div></div>', unsafe_allow_html=True)
    uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type=["txt"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Settings (keep your original controls)
# =========================================================
with tab_settings:
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

    # Presets (client-side localStorage only; no Firebase yet)
    from streamlit.components.v1 import html
    html(f"""
    <div style="display:flex;gap:10px;margin:8px 0;">
      <!-- IIFE and object braces must be doubled in a Python f-string -->
      <button onclick="(function(){{ 
        const preset = {{
          hv_min:{hv_min}, hv_max:{hv_max}, adx_exit:{adx_exit}, vwap_accept_k:{vwap_accept_k},
          use_trend_bias:{str(use_trend_bias).lower()}, trend_bias_strength:{trend_bias_strength},
          trend_method:'{trend_method}', wing_ext_pct:{wing_ext_pct},
          days_before:{days_before}, days_after:{days_after}
        }};
        localStorage.setItem('icb_preset', JSON.stringify(preset));
        alert('Preset saved');
      }})()" 
      style="background:#152846;border:1px solid #2f4d74;color:#cde1ff;border-radius:8px;padding:8px 12px;">
        Save Preset
      </button>

      <button onclick="(function(){{ 
        const raw = localStorage.getItem('icb_preset');
        if(!raw) return alert('No preset found');
        const p = JSON.parse(raw);
        alert('Preset loaded (values shown in alert). Apply them in Settings if needed.\\n' + JSON.stringify(p, null, 2));
      }})()" 
      style="background:#0b1221;border:1px solid #35507a;color:#cde1ff;border-radius:8px;padding:8px 12px;">
        Load Preset
      </button>
    </div>
    """, height=60)

# =========================================================
# Sticky bottom bar (visual summary & CTA) — braces FIXED
# =========================================================
st.markdown(f"""
<div class="sticky-bar">
  <div class="sticky-inner">
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <span class="pill">HV: {hv_min}–{hv_max}%</span>
      <span class="pill">ADX Exit ≥ {adx_exit}</span>
      <span class="pill">VWAP κ {vwap_accept_k}</span>
      <span class="pill">Trend: {trend_method}</span>
      <span class="pill">Bias: {'On' if use_trend_bias else 'Off'} ({trend_bias_strength if use_trend_bias else '—'})</span>
      <span class="pill">Wing Ext: {wing_ext_pct}%</span>
      <span class="pill">Blackout: {days_before}d before / {days_after}d after</span>
    </div>
    <div>
      <!-- Double braces to escape JS object literal -->
      <button class="cta" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">Run Backtest ↑</button>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Helpers (ported logic)
# =========================================================
def parse_blackout_txt(file) -> List[pd.Timestamp]:
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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Bollinger bands on VWAP (20, 2σ)
    ma = df["vwap"].rolling(20).mean()
    std = df["vwap"].rolling(20).std(ddof=0)
    df["bb_mid"]   = ma
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
    up_move   = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm  = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
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

def next_friday_within(idx, start_loc, max_dte=5):
    end_loc = min(start_loc + max_dte, len(idx) - 1)
    for j in range(start_loc, end_loc + 1):
        if idx[j].weekday() == 4:
            return idx[j]
    return idx[end_loc]

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

    # Entry filters incl. HV bounds
    cond_adx = df["adx"] < 20
    cond_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 60)
    cond_hv  = (df["hv"] >= hv_min) & (df["hv"] <= hv_max)
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

        # Early exits on open positions
        if open_positions:
            still_open = []
            for pos in open_positions:
                current_close = df.loc[d, "close"]
                pnl_today, _ = eval_condor(current_close, pos["sp"], pos["lp"], pos["sc"], pos["lc"], pos["credit"])
                breach = (current_close < pos["sp"]) or (current_close > pos["sc"])
                broke  = (current_close < pos["lp"]) or (current_close > pos["lc"])
                adx_exit_now = (df.loc[d, "adx"] >= ADX_EXIT)

                # vwap exit (slope flip + far enough + on slope side)
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

            ext_factor = 1.0 + max(0.0, float(wing_ext_pct)) / 100.0
            put_w = 5.0 * ext_factor if (df["trend_up"].iloc[i] and prev_down and tightening_now) else 5.0
            call_w = 5.0 * ext_factor if (df["trend_down"].iloc[i] and prev_up and tightening_now) else 5.0

            lp = round_to(sp - put_w, 1.0)
            lc = round_to(sc + call_w, 1.0)
            credit = 0.30 * min(call_w, put_w)  # same as GUI

            expiry = next_friday_within(df.index, i, max_dte=5)
            open_positions.append({
                "entry": d, "expiry": expiry,
                "sp": sp, "lp": lp, "sc": sc, "lc": lc,
                "credit": credit
            })

        # log equity
        eq.append({"date": d, "cash": cash})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    equity_df = pd.DataFrame(eq).set_index("date") if eq else pd.DataFrame(columns=["cash"])

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["cash"].cummax()
        drawdown = running_max - equity_df["cash"]
        max_dd_val = float(drawdown.max()) if not drawdown.empty else 0.0
        if max_dd_val > 0 and not running_max.empty:
            dd_idx = drawdown.idxmax()
            max_dd_pct = (max_dd_val / running_max.loc[dd_idx]) * 100
        else:
            max_dd_pct = 0.0
    else:
        max_dd_val = 0.0
        max_dd_pct = 0.0

    # Summary
    wins     = int((trades_df["outcome"] == "win").sum())    if not trades_df.empty else 0
    losses   = int((trades_df["outcome"] == "loss").sum())   if not trades_df.empty else 0
    breaches = int((trades_df["outcome"] == "breach").sum()) if not trades_df.empty else 0
    adx_exits= int((trades_df["outcome"] == "adx_exit").sum()) if not trades_df.empty else 0
    vwap_exits=int((trades_df["outcome"] == "vwap_exit").sum()) if not trades_df.empty else 0
    brokes   = int((trades_df["outcome"] == "broke").sum())  if not trades_df.empty else 0

    summary = {
        "trades": int(len(trades_df)),
        "wins": wins, "losses": losses,
        "breaches": breaches, "adx_exits": adx_exits, "vwap_exits": vwap_exits, "brokes": brokes,
        "win_rate": float(100 * wins / len(trades_df)) if not trades_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
        "max_drawdown": max_dd_val, "max_drawdown_pct": max_dd_pct
    }

    return df, trades_df, equity_df, summary


# =========================================================
# Results tab (Run & display everything)
# =========================================================
with tab_results:
    # Main Run button
    run_clicked = st.button("Run Backtest")

    if run_clicked:
        if not uploaded_csv or not uploaded_txt:
            st.error("Please upload both CSV and blackout dates (.txt) in the Uploads tab.")
            st.stop()

        # Read CSV defensively
        df_raw = pd.read_csv(uploaded_csv)
        df_raw.columns = [c.lower() for c in df_raw.columns]

        # Full backtest requires these:
        required_full = {"timestamp", "close", "high", "low", "vwap"}
        missing_full = required_full - set(df_raw.columns)

        # Parse blackout dates
        blackout_dates = parse_blackout_txt(uploaded_txt)

        # If missing columns, show basic Bollinger on CLOSE (fallback), no trades
        if missing_full:
            st.warning(f"CSV missing columns for full backtest: {sorted(missing_full)}")
            st.info("Showing basic Bollinger chart on CLOSE. For full logic, include: timestamp, close, high, low, vwap.")

            df_raw["timestamp"] = pd.to_datetime(df_raw.get("timestamp"), errors="coerce")
            df = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            if "close" not in df.columns:
                st.error("CSV must include at least 'close' and 'timestamp' to render the fallback chart.")
                st.stop()

            # Bollinger on CLOSE
            ma = df["close"].rolling(20).mean()
            ub = ma + 2 * df["close"].rolling(20).std()
            lb = ma - 2 * df["close"].rolling(20).std()

            fig_px = go.Figure()
            fig_px.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close Price",
                                        mode="lines", line=dict(color="#60A5FA", width=2.5)))
            fig_px.add_trace(go.Scatter(x=df.index, y=ma, name="BB Mid (20)",
                                        mode="lines", line=dict(color="gray", width=1.2)))
            fig_px.add_trace(go.Scatter(x=df.index, y=ub, name="BB Upper (20,2σ)",
                                        mode="lines", line=dict(color="orange", width=1.2)))
            fig_px.add_trace(go.Scatter(x=df.index, y=lb, name="BB Lower (20,2σ)",
                                        mode="lines", line=dict(color="orange", width=1.2)))
            # blackout shading
            if blackout_dates and len(df) > 0:
                for e in blackout_dates:
                    start = e - timedelta(days=int(days_before))
                    end = e + timedelta(days=int(days_after))
                    fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

            fig_px.update_layout(
                template="plotly_dark",
                title="Price with Bollinger Bands (fallback)",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="", yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_px, use_container_width=True)

            # Placeholder metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", "—")
            m2.metric("Win Rate", "—")
            m3.metric("Total P&L", "$—")
            m4.metric("Max Drawdown", "$—", "—%")

            st.write("✅ Backtest (fallback) complete. Add required columns for full logic.")
            st.stop()

        # Full run
        with st.spinner("Running backtest…"):
            df_out, trades_df, equity_df, summary = run_backtest(
                df_raw=df_raw,
                blackout_dates=blackout_dates,
                hv_min=hv_min, hv_max=hv_max,
                adx_exit_thr=adx_exit, vwap_k=vwap_accept_k,
                use_bias=use_trend_bias, bias_strength=trend_bias_strength,
                trend_method=trend_method, wing_ext_pct=wing_ext_pct,
                days_before=days_before, days_after=days_after
            )

        # Summary metric cards
        st.subheader("Summary")
        m1, m2, m3, m4, m5 = st.columns(5, gap="small")
        m1.metric("Trades", f"{summary['trades']}")
        m2.metric("Win Rate", f"{summary['win_rate']:.2f}%")
        m3.metric("Total P&L", f"${summary['total_pnl']:.2f}")
        m4.metric("Max Drawdown", f"${summary['max_drawdown']:.2f}", f"{summary['max_drawdown_pct']:.2f}%")
        m5.metric("Exits (Bre/ADX/VWAP/Broke)", f"{summary['breaches']}/{summary['adx_exits']}/{summary['vwap_exits']}/{summary['brokes']}")

        # Trades table
        st.subheader("Trades")
        if trades_df.empty:
            st.info("No trades generated under current settings.")
        else:
            show_df = trades_df.copy()
            show_df["entry_date"] = pd.to_datetime(show_df["entry_date"])
            show_df["expiry_date"] = pd.to_datetime(show_df["expiry_date"])
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Equity curve chart
        st.subheader("Equity Curve")
        if equity_df.empty:
            st.info("No equity curve to display.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity_df.index, y=equity_df["cash"],
                                        name="Equity", mode="lines", line=dict(color="#22D3EE", width=3)))
            fig_eq.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                                 xaxis_title="", yaxis_title="Cash ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

        # VWAP + Bollinger Bands + markers
        st.subheader("VWAP & Bollinger Bands with Trade Markers")
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["vwap"], name="VWAP",
                                    mode="lines", line=dict(color="steelblue", width=2)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_upper"], name="BB Upper (20,2σ)",
                                    mode="lines", line=dict(color="orange", width=1.5)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_mid"], name="BB Mid (20)",
                                    mode="lines", line=dict(color="gray", width=1.0)))
        fig_px.add_trace(go.Scatter(x=df_out.index, y=df_out["bb_lower"], name="BB Lower (20,2σ)",
                                    mode="lines", line=dict(color="orange", width=1.5)))

        # Blackout shading
        if blackout_dates:
            for e in blackout_dates:
                start = e - timedelta(days=int(days_before))
                end   = e + timedelta(days=int(days_after))
                fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

        # Trend overlay (if bias used—like GUI)
        if use_trend_bias:
            up_idx = df_out.index[df_out["trend_up"]]
            down_idx = df_out.index[df_out["trend_down"]]
            fig_px.add_trace(go.Scatter(x=up_idx, y=df_out.loc[up_idx, "vwap"],
                                        name="Uptrend", mode="markers",
                                        marker=dict(color="green", size=5, opacity=0.5), hoverinfo="skip"))
            fig_px.add_trace(go.Scatter(x=down_idx, y=df_out.loc[down_idx, "vwap"],
                                        name="Downtrend", mode="markers",
                                        marker=dict(color="red", size=5, opacity=0.5), hoverinfo="skip"))

        # Entry/Exit markers
        if not trades_df.empty:
            wins_m   = (trades_df["outcome"] == "win")
            losses_m = (trades_df["outcome"] == "loss")
            breach_m = (trades_df["outcome"] == "breach")
            adx_m    = (trades_df["outcome"] == "adx_exit")
            vwap_m   = (trades_df["outcome"] == "vwap_exit")
            broke_m  = (trades_df["outcome"] == "broke")

            # Entry markers
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
            # Exit markers
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
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="", yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_px, use_container_width=True)

    else:
        st.info("Press **Run Backtest** to generate results (after uploads & settings).")
