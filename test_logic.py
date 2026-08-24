"""Offline logic test using synthetic OHLCV data (no network) to validate
compute_indicators / compute_rs_ratings / the signal-detection logic in
minervini_screen.py before it ever touches real Yahoo Finance data."""

import numpy as np
import pandas as pd
from minervini_screen import compute_indicators, compute_rs_ratings, TRANSITION_WINDOW, RS_MIN_FOR_BUY

np.random.seed(0)
dates = pd.bdate_range("2024-01-01", periods=500)


def make_series(kind):
    n = len(dates)
    if kind == "strong_uptrend_breakout":
        # steady uptrend for 400 days, tight consolidation for last 20, breakout on last day
        base = np.linspace(50, 150, 481)
        base = np.concatenate([base, np.linspace(150, 152, 15), [152, 152.5, 153, 165]])
        close = base[:n] + np.random.normal(0, 0.3, n)
        vol = np.random.randint(1_000_000, 1_500_000, n).astype(float)
        vol[-1] = 4_000_000  # breakout volume spike
    elif kind == "downtrend_broken":
        base = np.linspace(150, 60, n)
        close = base + np.random.normal(0, 0.5, n)
        vol = np.random.randint(800_000, 1_200_000, n).astype(float)
    elif kind == "recent_breakdown":
        # was in a Trend-Template-qualifying uptrend, breaks below 50sma in the last few days
        base = np.linspace(50, 140, n - 5)
        base = np.concatenate([base, [138, 134, 128, 120, 110]])
        close = base + np.random.normal(0, 0.2, n)
        vol = np.random.randint(1_000_000, 1_500_000, n).astype(float)
    else:  # flat/choppy, never qualifies
        close = 80 + 10 * np.sin(np.linspace(0, 40, n)) + np.random.normal(0, 1, n)
        vol = np.random.randint(500_000, 900_000, n).astype(float)

    high = close * (1 + np.random.uniform(0.001, 0.02, n))
    low = close * (1 - np.random.uniform(0.001, 0.02, n))
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)
    return df


tickers = {
    "BUY1": make_series("strong_uptrend_breakout"),
    "SELL1": make_series("recent_breakdown"),
    "DOWN1": make_series("downtrend_broken"),
    "FLAT1": make_series("flat"),
}

ind = {t: compute_indicators(df) for t, df in tickers.items()}
closes = {t: df["Close"].dropna() for t, df in tickers.items()}

common_dates = sorted(set.intersection(*[set(v.index) for v in ind.values()]))
eval_dates = common_dates[-TRANSITION_WINDOW:]
today = eval_dates[-1]

rs_by_date = compute_rs_ratings(closes, eval_dates)
print("RS ratings today:", rs_by_date[today])

for t, df in ind.items():
    row = df.loc[today]
    crit_cols = [c for c in df.columns if c.startswith("c")]
    crits = {c: bool(row[c]) for c in crit_cols}
    print(f"\n{t}: price={row['close']:.2f} vcp={row['vcp_score']:.1f} breakout={bool(row['breakout'])}")
    print("  criteria:", crits)
    print("  all_true:", all(crits.values()), "rs:", rs_by_date[today].get(t))

print("\nOK - no exceptions, sanity check the printed values above make directional sense.")
