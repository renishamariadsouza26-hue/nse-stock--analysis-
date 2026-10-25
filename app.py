import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core

st.set_page_config(page_title="NSE Desk", page_icon="📈", layout="wide")
UP, DOWN, ACCENT = "#0B8F5A", "#CB3A31", "#2F4BD8"


@st.cache_data(ttl=600, show_spinner=False)
def get(symbol, period):
    return core.fetch(symbol, period)


@st.cache_data(ttl=900, show_spinner=False)
def get_universe(period="1y"):
    return core.fetch_many(core.NIFTY50, period)


# ------------------------------------------------------------------ sidebar
if "watch" not in st.session_state:
    st.session_state.watch = ["RELIANCE", "TCS", "HDFCBANK"]

with st.sidebar:
    st.title("NSE Desk")
    symbol = st.text_input("Stock symbol", "RELIANCE").strip().upper().replace(".NS", "")
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2)
    st.caption("Quick picks from your watchlist")
    pick = st.radio("Watchlist", st.session_state.watch, index=None, label_visibility="collapsed")
    if pick:
        symbol = pick
    c1, c2 = st.columns(2)
    if c1.button("Add", width="stretch") and symbol not in st.session_state.watch:
        st.session_state.watch.append(symbol)
        st.rerun()
    if c2.button("Remove", width="stretch") and symbol in st.session_state.watch:
        st.session_state.watch.remove(symbol)
        st.rerun()
    st.caption("Data from Yahoo Finance, delayed. For learning only, not investment advice.")

try:
    df = get(symbol, "5y" if period == "5y" else "2y" if period in ("2y", "1y", "6mo") else period)
except Exception as e:
    st.error(str(e))
    st.stop()

last, prev = df.Close.iloc[-1], df.Close.iloc[-2]
sig_table, score, verdict = core.signals(df)
hi52, lo52 = df.High.tail(252).max(), df.Low.tail(252).min()

st.header(f"{symbol}  ₹{last:,.2f}")
m = st.columns(5)
m[0].metric("Day change", f"{last - prev:+.2f}", f"{(last / prev - 1) * 100:+.2f}%")
m[1].metric("RSI (14)", f"{core.rsi(df.Close).iloc[-1]:.1f}")
m[2].metric("Signal", verdict, f"score {score}")
m[3].metric("52-week range", f"{lo52:,.0f} – {hi52:,.0f}")
m[4].metric("From 52w high", f"{(last / hi52 - 1) * 100:.1f}%")
st.progress(float((last - lo52) / (hi52 - lo52)), text="Where the price sits in its 52-week range")

tabs = st.tabs(["Chart", "Signals", "Market map", "Screener", "Compare", "Backtest", "Future paths", "Seasonality", "Risk calculator"])

# ------------------------------------------------------------------ chart
with tabs[0]:
    o = st.columns([3, 2])
    overlays = o[0].multiselect("Overlays", ["SMA 20", "SMA 50", "SMA 200", "Bollinger", "Supertrend"], ["SMA 20", "SMA 50"])
    lower = o[1].selectbox("Lower panel", ["RSI", "MACD", "None"])
    bars = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}[period]
    full = df
    view = full.tail(bars)
    rows = 2 if lower == "None" else 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.62, 0.15, 0.23] if rows == 3 else [0.8, 0.2])
    fig.add_trace(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close, name="Price",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    for name, n, col in (("SMA 20", 20, "#E0A100"), ("SMA 50", 50, "#8E44AD"), ("SMA 200", 200, ACCENT)):
        if name in overlays:
            fig.add_trace(go.Scatter(x=view.index, y=core.sma(full.Close, n).reindex(view.index), name=name, line=dict(width=1.3, color=col)), row=1, col=1)
    if "Bollinger" in overlays:
        u, _, l = core.bollinger(full.Close)
        for s, n in ((u, "BB upper"), (l, "BB lower")):
            fig.add_trace(go.Scatter(x=view.index, y=s.reindex(view.index), name=n, line=dict(width=1, color="#7F8AA3", dash="dot")), row=1, col=1)
    if "Supertrend" in overlays:
        stl, _ = core.supertrend(full)
        fig.add_trace(go.Scatter(x=view.index, y=stl.reindex(view.index), name="Supertrend", line=dict(width=1.5, color="#E8590C")), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view.Volume, name="Volume", marker_color=np.where(view.Close >= view.Open, UP, DOWN), opacity=0.5), row=2, col=1)
    if lower == "RSI":
        fig.add_trace(go.Scatter(x=view.index, y=core.rsi(full.Close).reindex(view.index), name="RSI", line=dict(color="#7B61FF")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=DOWN, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=UP, row=3, col=1)
    elif lower == "MACD":
        line, sg, h = core.macd(full.Close)
        fig.add_trace(go.Bar(x=view.index, y=h.reindex(view.index), name="Histogram", marker_color=np.where(h.reindex(view.index) >= 0, UP, DOWN), opacity=0.6), row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=line.reindex(view.index), name="MACD", line=dict(color=ACCENT)), row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=sg.reindex(view.index), name="Signal", line=dict(color="#E8590C")), row=3, col=1)
    sup, res = core.levels(full)
    for v in sup:
        fig.add_hline(y=v, line_dash="dot", line_color=UP, opacity=0.5, row=1, col=1)
    for v in res:
        fig.add_hline(y=v, line_dash="dot", line_color=DOWN, opacity=0.5, row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.03), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Dotted green lines are support levels {sup}, dotted red lines are resistance levels {res}.")

# ------------------------------------------------------------------ signals
with tabs[1]:
    a, b = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": verdict},
                                   gauge={"axis": {"range": [-100, 100]}, "bar": {"color": ACCENT},
                                          "steps": [{"range": [-100, -20], "color": "#F2C9C5"}, {"range": [-20, 20], "color": "#E3E7EE"}, {"range": [20, 100], "color": "#BFE5D4"}]}))
    gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    a.plotly_chart(gauge, width="stretch")
    b.dataframe(sig_table.drop(columns="Weight"), hide_index=True, width="stretch")
    st.subheader("Candlestick patterns in the last 10 sessions")
    pat = core.patterns(df)
    st.dataframe(pat, hide_index=True, width="stretch") if len(pat) else st.info("No classic patterns in the last 10 sessions.")
    st.caption("The score blends five signals by weight. It describes the chart today, it does not predict tomorrow.")

# ------------------------------------------------------------------ market map
with tabs[2]:
    if st.button("Load the Nifty 50 map", key="map") or st.session_state.get("map_on"):
        st.session_state.map_on = True
        with st.spinner("Downloading 50 stocks..."):
            mt = core.market_table(get_universe())
        fig = px.treemap(mt, path=["Symbol"], values="Turnover (Cr)", color="Change %", color_continuous_scale=["#CB3A31", "#E9ECF2", "#0B8F5A"],
                         color_continuous_midpoint=0, hover_data={"Price": ":.2f", "Change %": ":.2f"})
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{color:.2f}%")
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        g, l = st.columns(2)
        g.subheader("Top gainers"); g.dataframe(mt.nlargest(5, "Change %"), hide_index=True, width="stretch")
        l.subheader("Top losers"); l.dataframe(mt.nsmallest(5, "Change %"), hide_index=True, width="stretch")
        st.caption("Box size is today's traded value, colour is today's price change.")
    else:
        st.info("Click the button to see the whole market at a glance. It downloads 50 stocks, so it takes a few seconds.")

# ------------------------------------------------------------------ screener
with tabs[3]:
    f = st.columns(5)
    r_lo, r_hi = f[0].slider("RSI between", 0, 100, (0, 100))
    a50, a200 = f[1].checkbox("Above 50-day average"), f[1].checkbox("Above 200-day average")
    nh = f[2].number_input("Within % of 52-week high (0 = off)", 0.0, 50.0, 0.0)
    vr = f[3].number_input("Volume at least x times average (0 = off)", 0.0, 10.0, 0.0)
    if f[4].button("Run screener", width="stretch"):
        with st.spinner("Scanning the Nifty 50..."):
            res_df = core.screen(get_universe(), r_lo, r_hi, a50, a200, nh or None, vr or None)
        st.write(f"{len(res_df)} stocks match.")
        st.dataframe(res_df.sort_values("Change %", ascending=False) if len(res_df) else res_df, hide_index=True, width="stretch")

# ------------------------------------------------------------------ compare
with tabs[4]:
    syms = st.text_input("Symbols to compare, separated by commas", f"{symbol},TCS,INFY").upper().replace(".NS", "")
    names = [s.strip() for s in syms.split(",") if s.strip()][:6]
    if len(names) >= 2:
        try:
            closes = pd.concat({s: get(s, "1y").Close for s in names}, axis=1).dropna()
            fig = go.Figure([go.Scatter(x=closes.index, y=closes[s] / closes[s].iloc[0] * 100, name=s) for s in names])
            fig.update_layout(height=420, yaxis_title="Rebased to 100", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            st.subheader("How closely do they move together?")
            corr = closes.pct_change().corr().round(2)
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale=["#CB3A31", "#FFFFFF", "#0B8F5A"], zmin=-1, zmax=1, height=380), width="stretch")
            st.caption("Near 1 means they rise and fall together, so holding both gives little diversification.")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Enter at least two symbols.")

# ------------------------------------------------------------------ backtest
with tabs[5]:
    b1 = st.columns(4)
    strat = b1[0].selectbox("Strategy", ["Moving-average crossover", "RSI mean reversion"])
    if strat.startswith("Moving"):
        fa, sl = b1[1].number_input("Fast average", 5, 100, 20), b1[2].number_input("Slow average", 10, 250, 50)
        curve, stats = core.backtest(get(symbol, "5y"), "ma_cross", fa, sl)
    else:
        lo_, hi_ = b1[1].number_input("Buy when RSI below", 5, 50, 30), b1[2].number_input("Sell when RSI above", 50, 95, 70)
        curve, stats = core.backtest(get(symbol, "5y"), "rsi", lo=lo_, hi=hi_)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, f"{v:.1f}" if "Trades" not in k else int(v))
    fig = go.Figure([go.Scatter(x=curve.index, y=curve[c], name=c, line=dict(color=ACCENT if c == "Strategy" else "#8B96A8")) for c in curve])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Growth of ₹1", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Long-only, trades the day after a signal, 0.05% cost per side. Past results do not predict future returns.")

# ------------------------------------------------------------------ future paths
with tabs[6]:
    days = st.slider("Days ahead", 20, 250, 60)
    bands, end = core.monte_carlo(get(symbol, "5y").Close, days)
    x = list(range(1, days + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands[95], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=bands[5], fill="tonexty", fillcolor="rgba(47,75,216,0.12)", line=dict(width=0), name="90% of paths"))
    fig.add_trace(go.Scatter(x=x, y=bands[75], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=bands[25], fill="tonexty", fillcolor="rgba(47,75,216,0.25)", line=dict(width=0), name="50% of paths"))
    fig.add_trace(go.Scatter(x=x, y=bands[50], line=dict(color=ACCENT, width=2), name="Median"))
    fig.add_hline(y=last, line_dash="dot", line_color="#8B96A8")
    fig.update_layout(height=420, xaxis_title="Trading days from today", yaxis_title="Price (₹)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    p = st.columns(3)
    p[0].metric("Chance of ending higher", f"{(end > last).mean() * 100:.0f}%")
    p[1].metric("Bad case (5th percentile)", f"₹{np.percentile(end, 5):,.0f}", f"{(np.percentile(end, 5) / last - 1) * 100:.1f}%")
    p[2].metric("Good case (95th percentile)", f"₹{np.percentile(end, 95):,.0f}", f"{(np.percentile(end, 95) / last - 1) * 100:+.1f}%")
    st.caption("500 simulated futures built by reshuffling this stock's own past daily moves. It shows the range of risk, it is not a forecast.")

# ------------------------------------------------------------------ seasonality
with tabs[7]:
    sea = core.seasonality(get(symbol, "5y").Close)
    sea.columns = [pd.Timestamp(2000, int(mo), 1).strftime("%b") for mo in sea.columns]
    st.plotly_chart(px.imshow(sea.round(1), text_auto=True, aspect="auto", color_continuous_scale=["#CB3A31", "#FFFFFF", "#0B8F5A"], color_continuous_midpoint=0, height=360), width="stretch")
    avg = sea.mean().round(1)
    st.write(f"Best month on average: **{avg.idxmax()}** ({avg.max():+.1f}%). Weakest: **{avg.idxmin()}** ({avg.min():+.1f}%).")
    st.caption("Monthly returns in %, last five years. Five data points per month is a small sample, so treat it as a curiosity.")

# ------------------------------------------------------------------ risk calculator
with tabs[8]:
    st.write("How many shares can you buy while risking only what you can afford to lose?")
    k = st.columns(4)
    cap = k[0].number_input("Trading capital (₹)", 1000.0, 1e9, 100000.0, step=5000.0)
    rk = k[1].number_input("Risk per trade (%)", 0.1, 10.0, 1.0, step=0.1)
    en = k[2].number_input("Entry price (₹)", 0.01, 1e6, float(round(last, 2)))
    sp = k[3].number_input("Stop-loss price (₹)", 0.01, 1e6, float(round(last * 0.97, 2)))
    rr = st.slider("Reward-to-risk target", 1.0, 5.0, 2.0, 0.5)
    try:
        ps = core.position_size(cap, rk, en, sp, rr)
        q = st.columns(5)
        for col, (name, v) in zip(q, ps.items()):
            col.metric(name, f"{v:,.0f}" if name == "Quantity" else f"₹{v:,.2f}" if "%" not in name else f"{v:.1f}%")
        if ps["Capital used %"] > 100:
            st.warning("This position is bigger than your capital. Widen the risk or tighten the stop.")
    except ValueError as e:
        st.error(str(e))
