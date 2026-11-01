"""
The best entries from level_filters.py with longs only and shorts only, each side on its own
(one position per stock per side), on the Auto trader's RSI system, 2H and 4H, RSI and trailing exits.

    python level_sides.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import auto_trader_backtest as sim
import backtest as bt
import eod_backtest as eod
import level_filters as lf


def entries(chart, charts, timeframe, side, price):
    past, _, _ = sim.compare(side)
    app = sim.rsi_signal(chart, timeframe, side)
    novol = lf.base_no_volume(chart, timeframe, side)
    pdh = past(price, lf.prev_level(charts, "Daily", side))
    s5060 = lf.past_all(chart, side, (50, 60), price)
    sall = lf.past_all(chart, side, lf.PAIRS["sma_all"], price)
    return {"base": app, "no_volume": novol, "pdh": app & pdh, "no_volume+pdh": novol & pdh,
            "sma50_60": app & s5060, "no_volume+sma50_60": novol & s5060, "sma_all": app & sall,
            "no_volume+sma_all": novol & sall}


def stock_trades(path):
    symbol = path.stem
    hourly, daily = sim.load(path)
    if len(hourly) < 100:
        return []
    charts = sim.build_charts(hourly, daily)
    replay = sim.Replay(hourly)
    slots = eod.minutes(hourly.index)
    days = hourly.index.normalize()
    first_bar = np.r_[True, days[1:] != days[:-1]]
    day_open = hourly["Open"].groupby(days).transform("first").to_numpy()
    last_close = daily["Close"].shift(1).reindex(days).to_numpy()
    day_gap = (day_open / last_close - 1) * 100
    price = hourly["Close"].to_numpy()
    exits = sim.rsi_exits(charts, hourly)
    allowed = (slots <= sim.LAST_SLOT) & (hourly.index >= sim.START)
    rows = []
    for timeframe in lf.TIMEFRAMES:
        chart = charts[timeframe]
        for side in (1, -1):
            hit = sim.opened_past_trigger(chart, timeframe, "RSI", side, day_open) & first_bar
            blocked = pd.Series(hit, index=hourly.index).groupby(days).transform("any").to_numpy() | (day_gap * side >= sim.POINT_FIVE_GAP)
            for name, sig in entries(chart, charts, timeframe, side, price).items():
                ok = sig & ~blocked & allowed
                for style in lf.STYLES:
                    free_from, entered = 0, set()
                    for i in np.flatnonzero(ok):
                        if i < free_from or days[i] in entered:
                            continue
                        end, exit_price, reason = replay.run(i, side, style, exits[(timeframe, side)], None)
                        free_from, entry = end, sim.tick(replay.c[i])
                        entered.add(days[i])
                        gross, charges, slippage = sim.net_rupees(entry, float(exit_price), side)
                        rows.append({"variant": name, "side": "long" if side == 1 else "short", "style": style, "timeframe": timeframe,
                                     "symbol": symbol, "entry_time": hourly.index[i], "exit_time": hourly.index[end], "net": gross - charges - slippage})
    return rows


def main():
    paths = [p for p in sorted((bt.DATA / "60m").glob("*.csv")) if (bt.DATA / "1d" / p.name).exists()]
    with Pool() as pool:
        rows = [r for part in pool.map(stock_trades, paths) for r in part]
    trades = bt.drop_breaks(pd.DataFrame(rows))
    keys = ["side", "timeframe", "style", "variant"]
    first = trades["entry_time"] < sim.SPLIT
    g = trades.groupby(keys)["net"]
    per_stock = trades.groupby(keys + ["symbol"])["net"].sum()
    out = pd.DataFrame({"trades": g.size(), "win_pct": g.apply(lambda n: (n > 0).mean() * 100), "avg_net_rs": g.mean(),
                        "total_lakh": g.sum() / 1e5, "first_half_lakh": trades[first].groupby(keys)["net"].sum() / 1e5,
                        "second_half_lakh": trades[~first].groupby(keys)["net"].sum() / 1e5,
                        "stocks_in_profit_pct": per_stock.groupby(level=keys).apply(lambda n: (n > 0).mean() * 100),
                        "t_stat": g.apply(lambda n: n.mean() / n.std() * np.sqrt(len(n)))}).reset_index()
    out.round(2).to_csv(bt.RESULTS / "level_sides_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
