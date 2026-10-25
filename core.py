"""All the market maths. No Streamlit in here, so it is easy to test and reuse."""
import numpy as np
import pandas as pd
import yfinance as yf

NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]  # check niftyindices.com: the index changes twice a year


def norm(sym):
    s = sym.strip().upper()
    return s if s.startswith("^") or s.endswith((".NS", ".BO")) else s + ".NS"


def fetch(symbol, period="2y"):
    df = yf.Ticker(norm(symbol)).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol.upper()}. Check the NSE symbol.")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_many(symbols, period="1y"):
    syms = [norm(s) for s in symbols]
    raw = yf.download(syms, period=period, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    out = {}
    for s, n in zip(symbols, syms):
        try:
            d = raw[n][["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(d) > 30:
                out[s] = d
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- indicators
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(dn != 0, 100.0)


def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def bollinger(s, n=20, k=2.0):
    mid, sd = sma(s, n), s.rolling(n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=10, mult=3.0):
    a, hl2, c = atr(df, n).values, ((df.High + df.Low) / 2).values, df.Close.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb, d, line = ub.copy(), lb.copy(), np.ones(len(df)), np.full(len(df), np.nan)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        d[i] = -1 if (d[i - 1] == 1 and c[i] < flb[i]) else 1 if (d[i - 1] == -1 and c[i] > fub[i]) else d[i - 1]
        line[i] = flb[i] if d[i] == 1 else fub[i]
    return pd.Series(line, index=df.index), pd.Series(d, index=df.index)


def levels(df, win=5, lookback=120):
    d = df.tail(lookback)
    ph = d.High[d.High == d.High.rolling(2 * win + 1, center=True).max()]
    pl = d.Low[d.Low == d.Low.rolling(2 * win + 1, center=True).min()]
    p = d.Close.iloc[-1]
    return (sorted({round(float(v), 2) for v in pl if v < p}, reverse=True)[:3],
            sorted({round(float(v), 2) for v in ph if v > p})[:3])


# ---------------------------------------------------------------- signals
def signals(df):
    c = df.Close
    last, r = c.iloc[-1], rsi(c).iloc[-1]
    m50, m200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
    h = macd(c)[2]
    st, sd = supertrend(df)
    u, _, lo = bollinger(c)
    pb = (last - lo.iloc[-1]) / (u.iloc[-1] - lo.iloc[-1])
    rows = [
        ("RSI (14)", "bullish" if r < 30 else "bearish" if r > 70 else "neutral", f"{r:.1f}", 1),
        ("Trend: price vs 50/200-day", "bullish" if last > m50 > m200 else "bearish" if last < m50 < m200 else "neutral",
         "above both" if last > m50 > m200 else "below both" if last < m50 < m200 else "mixed", 2),
        ("MACD histogram", "bullish" if h.iloc[-1] > 0 and h.iloc[-1] > h.iloc[-2] else "bearish" if h.iloc[-1] < 0 and h.iloc[-1] < h.iloc[-2] else "neutral", f"{h.iloc[-1]:.2f}", 1.5),
        ("Supertrend", "bullish" if sd.iloc[-1] == 1 else "bearish", "uptrend" if sd.iloc[-1] == 1 else "downtrend", 2),
        ("Bollinger %B", "bullish" if pb < 0.05 else "bearish" if pb > 0.95 else "neutral", f"{pb:.2f}", 1),
    ]
    t = pd.DataFrame(rows, columns=["Signal", "Reading", "Detail", "Weight"])
    sign = t.Reading.map({"bullish": 1, "bearish": -1, "neutral": 0})
    score = round(float((sign * t.Weight).sum() / t.Weight.sum() * 100))
    verdict = ("Strong bullish" if score >= 50 else "Bullish" if score >= 20 else
               "Strong bearish" if score <= -50 else "Bearish" if score <= -20 else "Neutral")
    return t, score, verdict


def patterns(df, last_n=10):
    """Simple candlestick patterns over the last few bars."""
    o, h, l, c = df.Open, df.High, df.Low, df.Close
    body, rng = (c - o).abs(), (h - l).replace(0, np.nan)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    found = []
    for i in range(len(df) - last_n, len(df)):
        d = df.index[i].strftime("%d %b")
        if body.iloc[i] / rng.iloc[i] < 0.1:
            found.append((d, "Doji", "Indecision: buyers and sellers are balanced."))
        if lower.iloc[i] > 2 * body.iloc[i] and upper.iloc[i] < body.iloc[i] and body.iloc[i] > 0:
            found.append((d, "Hammer", "Long lower shadow: sellers were rejected. Bullish after a fall."))
        if upper.iloc[i] > 2 * body.iloc[i] and lower.iloc[i] < body.iloc[i] and body.iloc[i] > 0:
            found.append((d, "Shooting star", "Long upper shadow: buyers were rejected. Bearish after a rise."))
        if i > 0:
            pb = c.iloc[i - 1] - o.iloc[i - 1]
            cb = c.iloc[i] - o.iloc[i]
            if pb < 0 < cb and o.iloc[i] <= c.iloc[i - 1] and c.iloc[i] >= o.iloc[i - 1]:
                found.append((d, "Bullish engulfing", "A green candle swallowed the previous red one."))
            if pb > 0 > cb and o.iloc[i] >= c.iloc[i - 1] and c.iloc[i] <= o.iloc[i - 1]:
                found.append((d, "Bearish engulfing", "A red candle swallowed the previous green one."))
    return pd.DataFrame(found, columns=["Date", "Pattern", "Meaning"])


# ---------------------------------------------------------------- market tools
def day_change(df):
    return float((df.Close.iloc[-1] / df.Close.iloc[-2] - 1) * 100) if len(df) > 1 else 0.0


def market_table(data):
    rows = [{"Symbol": s, "Price": d.Close.iloc[-1], "Change %": day_change(d),
             "Turnover (Cr)": float(d.Close.iloc[-1] * d.Volume.iloc[-1]) / 1e7} for s, d in data.items()]
    return pd.DataFrame(rows)


def screen(data, rsi_min=0, rsi_max=100, above50=False, above200=False, near_high=None, vol_ratio=None):
    rows = []
    for s, d in data.items():
        c = d.Close
        last, m50, m200 = c.iloc[-1], sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
        r = rsi(c).iloc[-1]
        away = (last / d.High.max() - 1) * 100
        vr = d.Volume.iloc[-1] / d.Volume.tail(20).mean()
        ok = rsi_min <= r <= rsi_max
        ok &= (not above50) or bool(last > m50)
        ok &= (not above200) or bool(last > m200)
        ok &= near_high is None or away >= -near_high
        ok &= vol_ratio is None or vr >= vol_ratio
        if ok:
            rows.append({"Symbol": s, "Price": last, "Change %": day_change(d), "RSI": r,
                         "From 52w high %": away, "Volume vs avg (x)": vr})
    return pd.DataFrame(rows)


def backtest(df, strategy="ma_cross", fast=20, slow=50, lo=30, hi=70, cost_bps=5):
    c = df.Close
    if strategy == "ma_cross":
        pos = (sma(c, fast) > sma(c, slow)).astype(float)
    else:
        r = rsi(c)
        pos = pd.Series(np.nan, index=c.index)
        pos[r < lo], pos[r > hi] = 1.0, 0.0
        pos = pos.ffill().fillna(0.0)
    pos = pos.shift(1).fillna(0.0)  # act on the next day: no peeking at the future
    ret = c.pct_change().fillna(0.0)
    strat = pos * ret - pos.diff().abs().fillna(0) * cost_bps / 10000
    eq, bh = (1 + strat).cumprod(), (1 + ret).cumprod()
    n = len(c)
    stats = {
        "Strategy return %": (eq.iloc[-1] - 1) * 100, "Buy and hold %": (bh.iloc[-1] - 1) * 100,
        "Max drawdown %": ((eq / eq.cummax()) - 1).min() * 100,
        "Trades": int((pos.diff() == 1).sum()), "Time in market %": pos.mean() * 100,
    }
    return pd.DataFrame({"Strategy": eq, "Buy and hold": bh}), stats


def monte_carlo(close, days=60, paths=500, seed=7):
    """Bootstrap past daily returns to draw possible futures. An illustration of risk, not a forecast."""
    r = np.log(close).diff().dropna().values
    rng = np.random.default_rng(seed)
    sims = close.iloc[-1] * np.exp(np.cumsum(rng.choice(r, size=(days, paths)), axis=0))
    return {p: np.percentile(sims, p, axis=1) for p in (5, 25, 50, 75, 95)}, sims[-1]


def seasonality(close):
    m = close.resample("ME").last().pct_change().dropna() * 100
    t = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Ret": m.values})
    return t.pivot_table(index="Year", columns="Month", values="Ret")


def position_size(capital, risk_pct, entry, stop, rr=2.0):
    per_share = abs(entry - stop)
    if per_share <= 0:
        raise ValueError("Entry and stop-loss must be different.")
    qty = int((capital * risk_pct / 100) // per_share)
    long = entry > stop
    target = entry + rr * per_share if long else entry - rr * per_share
    return {"Quantity": qty, "Money at risk": qty * per_share, "Position value": qty * entry,
            "Target (1:%g)" % rr: target, "Capital used %": qty * entry / capital * 100}
