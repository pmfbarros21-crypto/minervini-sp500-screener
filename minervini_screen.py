#!/usr/bin/env python3
"""
Minervini-style daily screener for the S&P 500.

Implements (as objective, computable rules — see README for the reasoning):
  - Trend Template (8 criteria) from Mark Minervini's SEPA methodology
  - An approximate IBD-style Relative Strength (RS) Rating vs. the S&P 500 universe
  - A heuristic VCP (Volatility Contraction Pattern) score + breakout-with-volume detector
  - Transition detection over the last N trading days to flag BUY / SELL-WARNING signals
    without needing any state persisted between runs (each run recomputes history fresh).

Output: signals.json (machine-readable) and report.md (human-readable) written to the
repo root, committed by the GitHub Actions workflow that calls this script daily.

Data source: Yahoo Finance via the `yfinance` package (free, unofficial API).
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOOKBACK_PERIOD = "2y"        # history window to download (need >252 trading days)
TRANSITION_WINDOW = 10        # trading days used to detect "just qualified" / "just broke down"
BREAKOUT_LOOKBACK = 20        # days used to define a new pivot high for breakout detection
RS_MIN_FOR_BUY = 70           # Minervini's minimum RS Rating for consideration
RS_STRONG = 80                # "strong" RS Rating threshold
BATCH_SIZE = 60               # tickers per yfinance batch download (keeps requests reliable)
INDEX_TICKER = "^GSPC"        # S&P 500 index, used as the market benchmark


def get_sp500_tickers():
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    names = dict(zip(tickers, df["Security"].astype(str)))
    sectors = dict(zip(tickers, df["GICS Sector"].astype(str))) if "GICS Sector" in df.columns else {}
    return tickers, names, sectors


def download_batches(tickers, period=LOOKBACK_PERIOD, batch_size=BATCH_SIZE):
    """Download OHLCV history in batches; tolerate individual batch failures."""
    frames = {}
    failed = []
    all_tickers = tickers + [INDEX_TICKER]
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        for attempt in range(3):
            try:
                data = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Batch failed permanently: {batch[:3]}... ({e})", file=sys.stderr)
                    data = None
                else:
                    time.sleep(5)
        if data is None:
            failed.extend(batch)
            continue
        for t in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[t]
                df = df.dropna(how="all")
                if df.empty or len(df) < 60:
                    failed.append(t)
                    continue
                frames[t] = df
            except Exception:
                failed.append(t)
        time.sleep(1)  # be polite between batches
    return frames, failed


def compute_indicators(df):
    """Compute moving averages, 52w range, and daily True/False Trend Template
    sub-criteria (all but RS Rating, which needs the cross-sectional universe)."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    out["close"] = close
    out["volume"] = vol
    out["sma50"] = close.rolling(50).mean()
    out["sma150"] = close.rolling(150).mean()
    out["sma200"] = close.rolling(200).mean()
    out["sma200_1m_ago"] = out["sma200"].shift(21)
    out["high_52w"] = high.rolling(252, min_periods=100).max()
    out["low_52w"] = low.rolling(252, min_periods=100).min()

    out["c1_above_150_200"] = (close > out["sma150"]) & (close > out["sma200"])
    out["c2_150_above_200"] = out["sma150"] > out["sma200"]
    out["c3_200_trending_up"] = out["sma200"] > out["sma200_1m_ago"]
    out["c4_50_above_150_200"] = (out["sma50"] > out["sma150"]) & (out["sma50"] > out["sma200"])
    out["c5_close_above_50"] = close > out["sma50"]
    out["c6_30pct_above_low"] = close >= 1.30 * out["low_52w"]
    out["c7_within_25pct_high"] = close >= 0.75 * out["high_52w"]

    # VCP heuristic
    daily_range = high - low
    range_recent = daily_range.rolling(10).mean()
    range_base = daily_range.rolling(50).mean()
    range_ratio = (range_recent / range_base).clip(0, 2)
    contraction_score = ((1 - range_ratio) * 200).clip(0, 100)

    vol_avg50 = vol.rolling(50).mean()
    vol_recent10 = vol.rolling(10).mean()
    vol_ratio = (vol_recent10 / vol_avg50).clip(0, 2)
    dryup_score = ((1 - vol_ratio) * 150).clip(0, 100)

    out["vcp_score"] = (0.6 * contraction_score + 0.4 * dryup_score).clip(0, 100)

    pivot_high = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    out["breakout"] = (close > pivot_high) & (vol > 1.5 * vol_avg50)

    return out


def compute_rs_ratings(closes_by_ticker, dates):
    """Cross-sectional RS Rating (1-99 percentile) for each date in `dates`,
    using an IBD-style weighted-return formula: 40% * 3mo + 20% each of 6/9/12mo."""
    rs_by_date = {}
    for d in dates:
        raw = {}
        for t, s in closes_by_ticker.items():
            s = s[s.index <= d]
            if len(s) < 253:
                continue
            try:
                p0 = s.iloc[-1]
                r3 = p0 / s.iloc[-63] - 1
                r6 = p0 / s.iloc[-126] - 1
                r9 = p0 / s.iloc[-189] - 1
                r12 = p0 / s.iloc[-252] - 1
                raw[t] = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12
            except Exception:
                continue
        if not raw:
            rs_by_date[d] = {}
            continue
        s = pd.Series(raw)
        pct = s.rank(pct=True)
        rating = (pct * 98 + 1).round().astype(int)
        rs_by_date[d] = rating.to_dict()
    return rs_by_date


def main():
    print("Fetching S&P 500 constituent list...")
    tickers, names, sectors = get_sp500_tickers()
    print(f"{len(tickers)} tickers found.")

    print("Downloading price history (this can take several minutes)...")
    frames, failed = download_batches(tickers)
    if INDEX_TICKER in frames:
        frames.pop(INDEX_TICKER, None)
    print(f"Downloaded {len(frames)} tickers, {len(failed)} failed.")

    print("Computing indicators...")
    ind = {t: compute_indicators(df) for t, df in frames.items()}
    closes = {t: df["Close"].dropna() for t, df in frames.items()}

    # dates to evaluate: last TRANSITION_WINDOW trading days present across the universe
    common_dates = sorted(set.intersection(*[set(v.index) for v in ind.values()])) if ind else []
    eval_dates = common_dates[-TRANSITION_WINDOW:] if len(common_dates) >= TRANSITION_WINDOW else common_dates
    if not eval_dates:
        print("No overlapping dates found — aborting.", file=sys.stderr)
        sys.exit(1)
    today = eval_dates[-1]

    print("Computing RS Ratings (cross-sectional, per day)...")
    rs_by_date = compute_rs_ratings(closes, eval_dates)

    buy_signals = []
    sell_signals = []
    watch_list = []

    for t, df in ind.items():
        if today not in df.index:
            continue
        sub = df.loc[[d for d in eval_dates if d in df.index]]
        if sub.empty:
            continue

        crit_cols = [
            "c1_above_150_200", "c2_150_above_200", "c3_200_trending_up",
            "c4_50_above_150_200", "c5_close_above_50", "c6_30pct_above_low",
            "c7_within_25pct_high",
        ]

        def qualifies(day):
            if day not in sub.index:
                return False
            row = sub.loc[day]
            if not all(bool(row[c]) for c in crit_cols):
                return False
            rs = rs_by_date.get(day, {}).get(t)
            return rs is not None and rs >= RS_MIN_FOR_BUY

        qualifies_today = qualifies(today)
        qualified_any_recent = any(qualifies(d) for d in eval_dates[:-1])

        row_today = sub.loc[today]
        rs_today = rs_by_date.get(today, {}).get(t)
        price = round(float(row_today["close"]), 2)

        record_base = {
            "ticker": t,
            "name": names.get(t, t),
            "sector": sectors.get(t, ""),
            "price": price,
            "rs_rating": rs_today,
            "vcp_score": round(float(row_today["vcp_score"]), 1) if not pd.isna(row_today["vcp_score"]) else None,
            "pct_below_52w_high": round((1 - price / float(row_today["high_52w"])) * 100, 1)
                if row_today["high_52w"] and not pd.isna(row_today["high_52w"]) else None,
        }

        if qualifies_today:
            breakout_today = bool(row_today.get("breakout", False))
            just_qualified = qualifies_today and not qualified_any_recent
            if breakout_today or just_qualified:
                signal_type = "BUY - breakout with volume" if breakout_today else "BUY - new Trend Template qualifier"
                buy_signals.append({**record_base, "signal": signal_type, "strong_rs": (rs_today or 0) >= RS_STRONG})
            elif (row_today["vcp_score"] or 0) >= 60:
                watch_list.append({**record_base, "signal": "WATCH - VCP forming, in base"})
        else:
            if qualified_any_recent:
                reasons = []
                if not bool(row_today["c5_close_above_50"]):
                    reasons.append("fechou abaixo da SMA50")
                if not bool(row_today["c4_50_above_150_200"]):
                    reasons.append("SMA50 cruzou abaixo da SMA150/200")
                if not bool(row_today["c1_above_150_200"]):
                    reasons.append("preço abaixo da SMA150/200")
                if rs_today is not None and rs_today < RS_MIN_FOR_BUY:
                    reasons.append(f"RS Rating caiu para {rs_today}")
                if reasons:
                    sell_signals.append({
                        **record_base,
                        "signal": "SELL WARNING - saiu do Trend Template",
                        "reasons": reasons,
                    })

    buy_signals.sort(key=lambda r: (r["rs_rating"] or 0), reverse=True)
    sell_signals.sort(key=lambda r: (r["rs_rating"] or 0), reverse=True)
    watch_list.sort(key=lambda r: (r["vcp_score"] or 0), reverse=True)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_trading_date": str(today.date()) if hasattr(today, "date") else str(today),
        "universe_size": len(tickers),
        "downloaded_ok": len(frames),
        "download_failed": failed,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "watch_list": watch_list[:30],
    }

    with open("signals.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    write_report_md(result)
    print(f"Done. {len(buy_signals)} buy signals, {len(sell_signals)} sell warnings, {len(watch_list)} on watch.")


def write_report_md(result):
    lines = []
    lines.append(f"# Minervini S&P 500 Screen — {result['as_of_trading_date']}\n")
    lines.append(f"_Gerado em {result['generated_at_utc']} · {result['downloaded_ok']}/{result['universe_size']} tickers processados"
                  f"{', ' + str(len(result['download_failed'])) + ' falharam' if result['download_failed'] else ''}._\n")

    lines.append("\n## 🟢 BUY signals\n")
    if result["buy_signals"]:
        lines.append("| Ticker | Nome | Preço | RS Rating | VCP Score | % abaixo da máx. 52w | Sinal |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in result["buy_signals"]:
            lines.append(f"| **{r['ticker']}** | {r['name']} | ${r['price']} | {r['rs_rating']} | {r['vcp_score']} | "
                          f"{r['pct_below_52w_high']}% | {r['signal']} |")
    else:
        lines.append("_Nenhum sinal de compra hoje._")

    lines.append("\n## 🔴 SELL warnings\n")
    if result["sell_signals"]:
        lines.append("| Ticker | Nome | Preço | RS Rating | Motivo |")
        lines.append("|---|---|---|---|---|")
        for r in result["sell_signals"]:
            lines.append(f"| **{r['ticker']}** | {r['name']} | ${r['price']} | {r['rs_rating']} | {', '.join(r['reasons'])} |")
    else:
        lines.append("_Nenhum aviso de venda hoje._")

    lines.append("\n## 👀 Watchlist (a formar base, VCP em contração)\n")
    if result["watch_list"]:
        lines.append("| Ticker | Nome | Preço | RS Rating | VCP Score |")
        lines.append("|---|---|---|---|---|")
        for r in result["watch_list"][:15]:
            lines.append(f"| {r['ticker']} | {r['name']} | ${r['price']} | {r['rs_rating']} | {r['vcp_score']} |")
    else:
        lines.append("_Vazio._")

    lines.append("\n---\n_Isto é uma ferramenta de screening baseada numa metodologia pública (SEPA / Trend Template "
                  "de Mark Minervini). Não é aconselhamento financeiro. Dados: Yahoo Finance via yfinance._")

    with open("report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
