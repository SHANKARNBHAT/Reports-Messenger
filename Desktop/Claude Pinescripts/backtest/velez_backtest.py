"""
Oliver Velez's best-known setups as exact rules, on the futures stocks, on 2H, 4H, Daily, Weekly.

His full rules are not published in one place, so these are one fair reading of them:
  Trend     long: 20 SMA above 200 SMA (for the bar setups also close above the 20 SMA); short mirrored
  Pullback  3 bars of lower highs and lower lows, the first two red, with the 20 SMA rising
  Elephant  green bar, body 70%+ of its range, range 1.5x the average of the last 10 bars
  180       green bar engulfing the previous red bar's body
  Tail      bar dips to the 20 SMA and closes above it, lower tail 2x+ its body and half its range
Entry: next bar only, on a break of the signal bar's high (at the open if it gaps past).
Stop: the setup's low. Target: 2x the risk. If a bar touches both, the stop counts first.
`edge` is a trade's net % less the stock's own average move over the same days.
"""
import math

import numpy as np
import pandas as pd

import backtest as bt
import test_long_only as lo

REWARD = 2.0
SPLIT = lo.RECENT
SETUPS = ("Pullback", "Elephant", "180", "Tail")


def signals(bars: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """setup -> (signal, stop) for longs, on arrays where a short is the same bars turned upside down."""
    o, h, l, c = (bars[column].to_numpy() for column in ("Open", "High", "Low", "Close"))
    fast, slow = pd.Series(c).rolling(20).mean(), pd.Series(c).rolling(200).mean()
    rising = (fast > fast.shift(5)).to_numpy()
    fast, slow = fast.to_numpy(), slow.to_numpy()
    aligned = fast > slow
    trend = (c > fast) & aligned

    def back(values, n):
        out = np.roll(values, n)
        out[:n] = np.nan
        return out

    red, green = c < o, c > o
    body, span = np.abs(c - o), h - l
    pullback = (aligned & rising & (h < back(h, 1)) & (back(h, 1) < back(h, 2)) & (l < back(l, 1)) & (back(l, 1) < back(l, 2))
                & (back(red.astype(float), 1) == 1) & (back(red.astype(float), 2) == 1))
    average = pd.Series(span).shift(1).rolling(10).mean().to_numpy()
    elephant = trend & green & (body >= 0.7 * span) & (span >= 1.5 * average)
    engulf = trend & green & (back(red.astype(float), 1) == 1) & (o <= back(c, 1)) & (c >= back(o, 1))
    tail_low = np.minimum(o, c) - l
    tail = trend & aligned & (l <= fast) & (tail_low >= 2 * body) & (tail_low >= 0.5 * span)
    two_low = np.minimum(l, back(l, 1))
    return {"Pullback": (pullback, l), "Elephant": (elephant, l), "180": (engulf, two_low), "Tail": (tail, l)}


def simulate(o, h, l, signal, stops) -> list[tuple]:
    """Long trades on these bars: (entry bar, exit bar, entry, exit, reason)."""
    trades, i, n = [], 1, len(o)
    while i < n:
        if not signal[i - 1]:
            i += 1
            continue
        trigger, stop = h[i - 1], stops[i - 1]
        if not h[i] >= trigger:
            i += 1
            continue
        entry = max(o[i], trigger)
        risk = entry - stop
        if not risk > 0:
            i += 1
            continue
        target, start = entry + REWARD * risk, i
        while i < n:
            if l[i] <= stop:
                trades.append((start, i, entry, min(o[i], stop) if i > start else stop, "stop"))
                break
            if h[i] >= target:
                trades.append((start, i, entry, max(o[i], target) if i > start else target, "target"))
                break
            i += 1
        i += 1
    return trades


def run() -> pd.DataFrame:
    lot_sizes, rows = bt.lots(), []
    for path in sorted((bt.DATA / "1d").glob("*.csv")):
        symbol = path.stem
        daily = bt.clean(bt.read_prices(path))
        daily.index = daily.index.normalize()
        daily = daily[~daily.index.duplicated(keep="last")]
        if len(daily) < 250:
            continue
        charts = {"Daily": daily, "Weekly": bt.weekly_bars(daily)}
        hourly_path = bt.DATA / "60m" / path.name
        if hourly_path.exists():
            hourly = bt.clean(bt.read_prices(hourly_path))
            charts["2H"], charts["4H"] = bt.session_bars(hourly, 120), bt.session_bars(hourly, 240)
        for timeframe, bars in charts.items():
            if len(bars) < 220:
                continue
            flipped = pd.DataFrame({"Open": -bars["Open"], "High": -bars["Low"], "Low": -bars["High"], "Close": -bars["Close"]})
            for side, frame in ((1, bars), (-1, flipped)):
                o, h, l = frame["Open"].to_numpy(), frame["High"].to_numpy(), frame["Low"].to_numpy()
                for setup, (signal, stops) in signals(frame).items():
                    for start, end, entry, exit_, reason in simulate(o, h, l, signal, stops):
                        entry, exit_ = entry * side, exit_ * side
                        gross = (exit_ / entry - 1) * side * 100
                        cost = bt.charges_pct(entry, lot_sizes.get(symbol, 1)) + 2 * bt.SLIPPAGE_PCT
                        rows.append({"symbol": symbol, "timeframe": timeframe, "setup": setup, "side": side,
                                     "variant": setup, "entry_time": bars.index[start], "exit_time": bars.index[end],
                                     "entry": entry, "exit": exit_, "reason": reason, "bars": end - start,
                                     "gross": gross, "net": gross - cost})
    return pd.DataFrame(rows)


def table(trades: pd.DataFrame, returns: dict[str, pd.Series], start, end) -> pd.DataFrame:
    frame = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)].copy()
    drift = {s: r[(r.index >= start) & (r.index < end)].mean() for s, r in returns.items()}
    frame["edge"] = frame["net"] - frame["symbol"].map(drift) * frame["days"] * frame["side"]

    def summary(group: pd.DataFrame) -> pd.Series:
        by_stock = group.groupby("symbol")["edge"].mean()
        return pd.Series({"trades": len(group), "win_pct": (group["net"] > 0).mean() * 100, "avg_net": group["net"].mean(),
                          "median_net": group["net"].median(), "avg_edge": group["edge"].mean(), "avg_days": group["days"].mean(),
                          "stocks_positive_pct": (by_stock > 0).mean() * 100,
                          "t_stat": by_stock.mean() / (by_stock.std() / math.sqrt(len(by_stock))) if len(by_stock) > 1 else math.nan})

    return frame.groupby(["timeframe", "setup", "side"]).apply(summary, include_groups=False)


def main() -> None:
    bt.RESULTS.mkdir(exist_ok=True)
    trades = bt.drop_breaks(run())
    trades["entry_time"], trades["exit_time"] = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    trades["days"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400 * lo.BARS_A_YEAR / 365
    trades.to_csv(bt.RESULTS / "velez_trades.csv.gz", index=False)
    returns = lo.daily_returns()
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 120)
    tables = []
    for label, (start, end) in lo.PERIODS.items():
        result = table(trades, returns, start, end)
        tables.append(result.assign(period=label).set_index("period", append=True))
        print("\n==", label)
        print(result.round(2).to_string())
    pd.concat(tables).to_csv(bt.RESULTS / "velez_summary.csv")


if __name__ == "__main__":
    main()
