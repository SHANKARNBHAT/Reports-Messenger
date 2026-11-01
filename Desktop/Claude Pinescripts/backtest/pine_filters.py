"""
The Pine strategy (backtest.py's copy of All_in_one_Strategy_fixed.pine) with the filters that
helped the Auto trader: close past the previous day's high/low, close past SMA 50 and 60, and
longs only / shorts only. 2H and 4H, same period as the intraday data. Charges are for
₹10 lakh of position value (the lot-size file lives outside this folder), plus SLIPPAGE_PCT each side.

    python pine_filters.py K N   (chunk K of N, saved in results/pine_parts)
    python pine_filters.py       (summary of all chunks)
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import backtest as bt

FILTERS = ("none", "pdh", "sma50_60", "pdh+sma50_60")
SIDES = ("both", "long", "short")
MODES = ("Pure RSI", "RSI+HA colour entry")


def simulate(p, mode, longs_extra, shorts_extra, sides):
    opens, highs, lows, closes = p["open"], p["high"], p["low"], p["close"]
    if mode == "Pure RSI":
        long_ok = short_ok = np.ones(len(closes), bool)
    else:
        long_ok, short_ok = p["green"], p["red"]
    longs = (p["filter"] > bt.LONG_SMA) & p["cross_long"] & long_ok & longs_extra & (sides != "short")
    shorts = (p["filter"] < bt.SHORT_SMA) & p["cross_short"] & short_ok & shorts_extra & (sides != "long")
    level_l, level_s = p["ha_low_prev"], p["ha_high_prev"]
    exit_long, exit_short = bt.cross_below(p["rsi"], bt.LONG_EXIT), bt.cross_above(p["rsi"], bt.SHORT_EXIT)
    ha = mode != "Pure RSI"
    trades, side, entry, stop, opened = [], 0, 0.0, math.nan, 0
    for i in range(len(closes)):
        if side == 1 and lows[i] <= stop:
            trades.append((opened, i, side, entry, min(opens[i], stop), "stop")); side = 0
        elif side == -1 and highs[i] >= stop:
            trades.append((opened, i, side, entry, max(opens[i], stop), "stop")); side = 0
        if side == 1 and ((closes[i] < level_l[i]) if ha else exit_long[i]):
            trades.append((opened, i, side, entry, closes[i], "exit")); side = 0
        elif side == -1 and ((closes[i] > level_s[i]) if ha else exit_short[i]):
            trades.append((opened, i, side, entry, closes[i], "exit")); side = 0
        wanted = 1 if longs[i] else -1 if shorts[i] else 0
        if wanted and side != wanted:
            if side:
                trades.append((opened, i, side, entry, closes[i], "reverse"))
            side, entry, opened = wanted, closes[i], i
            stop = closes[i] - wanted * p["atr"][i] * bt.ATR_MULTIPLIER
    return trades


def one_stock(path):
    rows = []
    if True:
        symbol = path.stem
        daily = bt.clean(bt.read_prices(path))
        daily.index = daily.index.normalize()
        daily = daily[~daily.index.duplicated(keep="last")]
        hourly_path = bt.DATA / "60m" / path.name
        if len(daily) < 60 or not hourly_path.exists():
            return None
        daily_sma = bt.rsi(daily["Close"]).rolling(bt.RSI_SMA_LENGTH).mean()
        hourly = bt.clean(bt.read_prices(hourly_path))
        charts = {"2H": bt.session_bars(hourly, 120), "4H": bt.session_bars(hourly, 240)}
        prev_high, prev_low = daily["High"].shift(1), daily["Low"].shift(1)
        for timeframe, bars in charts.items():
            p = bt.prepare(bars, daily_sma)
            close = bars["Close"]
            days = bars.index.normalize()
            pdh, pdl = prev_high.reindex(days).to_numpy(), prev_low.reindex(days).to_numpy()
            s50, s60 = close.rolling(50).mean().to_numpy(), close.rolling(60).mean().to_numpy()
            c = close.to_numpy()
            masks = {"none": (np.ones(len(c), bool),) * 2,
                     "pdh": (c > pdh, c < pdl),
                     "sma50_60": ((c > s50) & (c > s60), (c < s50) & (c < s60))}
            masks["pdh+sma50_60"] = (masks["pdh"][0] & masks["sma50_60"][0], masks["pdh"][1] & masks["sma50_60"][1])
            for mode in MODES:
                for flt in FILTERS:
                    for sides in SIDES:
                        found = simulate(p, mode, *masks[flt], np.array(sides))
                        if not found:
                            continue
                        o, i, side, entry, exit_price, _ = (np.array(x) for x in zip(*found))
                        entry, exit_price = entry.astype(float), exit_price.astype(float)
                        cost = np.array([bt.charges_pct(e, max(1, round(1_000_000 / e))) for e in entry]) + 2 * bt.SLIPPAGE_PCT
                        rows.append(pd.DataFrame({"symbol": symbol, "timeframe": timeframe, "mode": mode, "filter": flt, "sides": sides,
                                                  "entry_time": bars.index[o.astype(int)], "exit_time": bars.index[i.astype(int)],
                                                  "side": side.astype(int), "net": (exit_price / entry - 1) * side * 100 - cost}))
    return pd.concat(rows, ignore_index=True) if rows else None


PARTS = bt.RESULTS / "pine_parts"


def run_chunk(k: int, n: int):
    PARTS.mkdir(exist_ok=True)
    paths = sorted((bt.DATA / "1d").glob("*.csv"))[k::n]
    with Pool(2) as pool:
        frames = [f for f in pool.map(one_stock, paths) if f is not None]
    pd.concat(frames, ignore_index=True).to_pickle(PARTS / f"part_{k}.pkl")


def main():
    if len(sys.argv) == 3:
        return run_chunk(int(sys.argv[1]), int(sys.argv[2]))
    trades = bt.drop_breaks(pd.concat([pd.read_pickle(p) for p in sorted(PARTS.glob("part_*.pkl"))], ignore_index=True))
    start = trades.loc[trades["timeframe"] == "2H", "entry_time"].min()
    trades = trades[trades["entry_time"] >= start]
    split = start + (trades["exit_time"].max() - start) / 2
    keys = ["timeframe", "mode", "sides", "filter"]
    g = trades.groupby(keys)["net"]
    first = trades["entry_time"] < split
    per_stock = trades.groupby(keys + ["symbol"])["net"].sum()
    out = pd.DataFrame({"trades": g.size(), "win_pct": g.apply(lambda n: (n > 0).mean() * 100), "avg_net_pct": g.mean(),
                        "sum_pct": g.sum(), "first_half_avg": trades[first].groupby(keys)["net"].mean(),
                        "second_half_avg": trades[~first].groupby(keys)["net"].mean(),
                        "profit_factor": g.apply(lambda n: n[n > 0].sum() / -n[n <= 0].sum()),
                        "stocks_up_pct": per_stock.groupby(level=keys).apply(lambda n: (n > 0).mean() * 100),
                        "t_stat": g.apply(lambda n: n.mean() / n.std() * math.sqrt(len(n)))}).reset_index()
    out.round(3).to_csv(bt.RESULTS / "pine_filters_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print("From", start, "split", split)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
