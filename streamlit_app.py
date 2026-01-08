
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------------
# Page config & CSS (dark app look)
# ----------------------------
st.set_page_config(page_title="Iron Condor Backtester", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* Base dark polish */
body { background-color: #0f172a; color: #e5e7eb; }
h1,h2,h3 { color: #e5e7eb; }
section[data-testid="stSidebar"] { background-color: #111827; }

/* Header strip */
.header {
  padding: 14px 18px;
  border-radius: 14px;
  background: linear-gradient(90deg,#2563EB,#60A5FA);
  color: white;
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  box-shadow: 0 8px 30px rgba(0,0,0,.30);
}
.header .title { font-size: 22px; font-weight: 700; }
.header .sub { opacity: .9; }

/* Card container */
.card {
  background: #111827; border: 1px solid #263143; border-radius: 12px;
  padding: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.30);
}

/* Upload zones */
.upload-box {
  border: 2px dashed #35507a; border-radius: 12px;
  padding: 18px; text-align: left; color: #cde1ff; background-color: #0b1221;
  margin-bottom: 10px;
}
.upload-box:hover { border-color: #60a5fa; }
.upload-head { font-weight: 700; margin-bottom: 6px; }
.upload-help { font-size: 12px; color: #9ca3af; }

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#2563EB,#60A5FA);
  color: #0b1221; font-weight: 800; border-radius: 10px; padding: 10px 16px;
  border: none;
}

/* Tables */
[data-testid="stTable"] { background: #0b1221; }
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

st.title("Iron Condor Backtester")

# ----------------------------
# Uploads (styled cards)
# ----------------------------
st.subheader("Uploads")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="upload-box"><div class="upload-head">Upload CSV</div><div class="upload-help">Drag/drop or browse • Limit 200MB • CSV</div></div>', unsafe_allow_html=True)
uploaded_csv = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

st.markdown('<div class="upload-box"><div class="upload-head">Upload Blackout Dates (.txt)</div><div class="upload-help">One date per line (YYYY-MM-DD). Lines starting with # are ignored.</div></div>', unsafe_allow_html=True)
uploaded_txt = st.file_uploader("Upload Blackout Dates (.txt)", type="txt", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Parameter controls (unchanged names/logic)
# ----------------------------
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

# ----------------------------
# Helper: parse blackout dates (safe)
# ----------------------------
def parse_blackout_txt(file) -> list[pd.Timestamp]:
    if file is None:
        return []
    try:
        content = file.read()
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1")
        loaded = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                dt = pd.to_datetime(s)
                loaded.append(pd.Timestamp(dt).normalize())
            except Exception:
                # skip invalid
                pass
        return loaded
    except Exception:
        return []

# ----------------------------
# When both files are present
# ----------------------------
if uploaded_csv and uploaded_txt:
    # Read CSV defensively (won't crash if cols missing)
    df = pd.read_csv(uploaded_csv)
    # Normalize column names to lowercase for robustness
    df.columns = [c.lower() for c in df.columns]

    required_cols = {"timestamp", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {sorted(missing)}")
        st.info("Expected at minimum: 'timestamp', 'close'. For advanced charts, also include 'high', 'low', 'vwap'.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        blackout_dates = parse_blackout_txt(uploaded_txt)
        st.success(f"Loaded {len(df)} rows and {len(blackout_dates)} blackout dates")

        # ----------------------------
        # Run Backtest (button)
        # ----------------------------
        if st.button("Run Backtest"):
            # ---- Chart A: Modern Plotly (preferred) ----
            try:
                # Bollinger Bands on CLOSE (as in your example)
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

                # Blackout shading if timestamps align
                if blackout_dates and len(df) > 0:
                    for e in blackout_dates:
                        start = e - timedelta(days=int(days_before))
                        end = e + timedelta(days=int(days_after))
                        fig_px.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0)

                fig_px.update_layout(
                    template="plotly_dark",
                    title="Price with Bollinger Bands",
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="", yaxis_title="Price ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig_px, use_container_width=True)

            except Exception as e:
                # ---- Fallback: your original Matplotlib plot ----
                st.warning(f"Plotly chart failed ({e}). Falling back to Matplotlib.")
                ma = df['close'].rolling(20).mean()
                ub = ma + 2 * df['close'].rolling(20).std()
                lb = ma - 2 * df['close'].rolling(20).std()

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index, df['close'], label='Close Price', color='skyblue')
                ax.plot(df.index, ma, label='BB Mid', color='gray')
                ax.plot(df.index, ub, label='BB Upper', color='orange')
                ax.plot(df.index, lb, label='BB Lower', color='orange')
                ax.set_title("Price with Bollinger Bands")
                ax.legend()
                st.pyplot(fig)

            # ---- Summary placeholder (kept from your code) ----
            st.write("✅ Backtest complete! (Add full logic here)")

else:
    st.info("Upload your CSV and blackout dates to enable the backtest.")
