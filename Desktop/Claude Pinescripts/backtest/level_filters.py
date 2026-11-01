"""
Level and moving-average tests on the Auto trader's best setups (auto_trader_backtest.py):
RSI system, RSI exit and trailing exit, on 2H and 4H, same costs, gap rules and dates.

Filters: the app's RSI entry, taken only when price is also past a level at that moment.
  base        the app's RSI entry as it is
  no_volume   the app's RSI entry without its volume condition
  sma5_8, sma14_20, sma50_60, sma_all   price past both SMAs (all six for sma_all) of the trade's timeframe
  pdh, pwh    price past the previous day's high (long) or low (short), or the previous week's
Triggers: the entry is the cross itself, confirmed by SMA RSI (45 long / 50 short), RSI past 60 / 40
on the trade's timeframe and a candle in the trade's direction. No volume condition.
  x_sma5_8, x_sma14_20, x_sma50_60   price crosses above both SMAs of the pair (below for shorts)
  x_pdh, x_pwh   price crosses the previous day's or week's high (low); skipped on days that open past it

    python level_filters.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import auto_trader_backtest as sim
import backtest as bt
import eod_backtest as eod

TIMEFRAMES = ("2H", "4H")
STYLES = ("rsi", "trail")
PAIRS = {"sma5_8": (5, 8), "sma14_20": (14, 20), "sma50_60": (50, 60), "sma_all": (5, 8, 14, 20, 50, 60)}


def past_all(chart, side, lengths, price):
    past, _, _ = sim.compare(side)
    ok = np.ones(len(price), bool)
    for length in lengths:
        ok &= past(price, chart.sma(length, price))
    return ok


def was_past_all(chart, side, lengths):
    past, _, _ = sim.compare(side)
    last = chart.ago(chart.finished_close)
    ok = np.ones(len(last), bool)
    for length in lengths:
        ok &= past(last, chart.ago(chart.finished_sma(length)))
    return ok


def prev_level(charts, name, side):
    chart = charts[name]
    return chart.ago(chart.candles["High" if side == 1 else "Low"].to_numpy())


def base_no_volume(chart, timeframe, side):
    past, reached, _ = sim.compare(side)
    price = chart.close
    fast, slow = sim.TREND[(timeframe, side)]
    triggered = sim.rsi_triggers(chart, timeframe, side, price)
    for length in (16, 20, 60):
        triggered |= chart.crossed_at_last_candle(side, length)
    sma_rsi = chart.sma_rsi()
    trend = reached(chart.sma(fast, price), chart.sma(slow, price)) & reached(price, chart.sma(200, price))
    strength = sma_rsi >= bt.LONG_SMA if side == 1 else sma_rsi <= bt.SHORT_SMA
    return past(price, chart.open) & trend & strength & triggered


def confirm(chart, side, price):
    past, _, _ = sim.compare(side)
    sma_rsi = chart.sma_rsi()
    strength = sma_rsi >= bt.LONG_SMA if side == 1 else sma_rsi <= bt.SHORT_SMA
    rsi = chart.rsi(price)
    momentum = rsi > bt.LONG_CROSS if side == 1 else rsi < bt.SHORT_CROSS
    return strength & momentum & past(price, chart.open)


def variants(chart, charts, timeframe, side, price, prev_close, opened_past_block):
    """name -> (signal, extra block). Blocks are on top of the 0.5% gap rule."""
    past, _, before = sim.compare(side)
    app = sim.rsi_signal(chart, timeframe, side)
    app_block = opened_past_block("app")
    out = {"base": (app, app_block), "no_volume": (base_no_volume(chart, timeframe, side), app_block)}
    for name, lengths in PAIRS.items():
        out[name] = (app & past_all(chart, side, lengths, price), app_block)
    for name, frame in (("pdh", "Daily"), ("pwh", "Weekly")):
        out[name] = (app & past(price, prev_level(charts, frame, side)), app_block)
    ok = confirm(chart, side, price)
    for name, lengths in PAIRS.items():
        if name != "sma_all":
            out["x_" + name] = (ok & past_all(chart, side, lengths, price) & ~was_past_all(chart, side, lengths), None)
    for name, frame in (("x_pdh", "Daily"), ("x_pwh", "Weekly")):
        level = prev_level(charts, frame, side)
        out[name] = (ok & past(price, level) & before(prev_close, level), opened_past_block(level))
    return out


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
    prev_close = np.r_[np.nan, price[:-1]]
    exits = sim.rsi_exits(charts, hourly)
    allowed = (slots <= sim.LAST_SLOT) & (hourly.index >= sim.START)
    rows = []
    for timeframe in TIMEFRAMES:
        chart = charts[timeframe]
        per_side = {}
        for side in (1, -1):
            past, _, _ = sim.compare(side)

            def opened_past_block(level, side=side, past=past):
                if isinstance(level, str):
                    hit = sim.opened_past_trigger(chart, timeframe, "RSI", side, day_open) & first_bar
                else:
                    hit = past(day_open, level) & first_bar
                return pd.Series(hit, index=hourly.index).groupby(days).transform("any").to_numpy()

            gap = day_gap * side >= sim.POINT_FIVE_GAP
            per_side[side] = {name: (sig, gap | (blk if blk is not None else False))
                              for name, (sig, blk) in variants(chart, charts, timeframe, side, price, prev_close, opened_past_block).items()}
        for name in per_side[1]:
            sig_l, blk_l = per_side[1][name]
            sig_s, blk_s = per_side[-1][name]
            longs, shorts = sig_l & ~blk_l & allowed, sig_s & ~blk_s & allowed
            for style in STYLES:
                free_from, entered = 0, set()
                for i in np.flatnonzero(sig_l & allowed | sig_s & allowed):
                    if sig_l[i] and sig_s[i]:
                        continue
                    wanted = 1 if longs[i] else -1 if shorts[i] else 0
                    if not wanted or i < free_from or (days[i], wanted) in entered:
                        continue
                    end, exit_price, reason = replay.run(i, wanted, style, exits[(timeframe, wanted)], None)
                    free_from, entry = end, sim.tick(replay.c[i])
                    entered.add((days[i], wanted))
                    gross, charges, slippage = sim.net_rupees(entry, float(exit_price), wanted)
                    rows.append({"variant": name, "style": style, "timeframe": timeframe, "symbol": symbol, "side": wanted,
                                 "entry_time": hourly.index[i], "exit_time": hourly.index[end], "reason": reason,
                                 "net": gross - charges - slippage})
    return rows


def main():
    paths = [p for p in sorted((bt.DATA / "60m").glob("*.csv")) if (bt.DATA / "1d" / p.name).exists()]
    with Pool() as pool:
        rows = [r for part in pool.map(stock_trades, paths) for r in part]
    trades = bt.drop_breaks(pd.DataFrame(rows))
    trades.to_csv(bt.RESULTS / "level_filters_trades.csv.gz", index=False)
    keys = ["variant", "timeframe", "style"]
    first = trades["entry_time"] < sim.SPLIT
    per_stock = trades.groupby(keys + ["symbol"])["net"].sum()
    g = trades.groupby(keys)["net"]
    out = pd.DataFrame({
        "trades": g.size(), "win_pct": g.apply(lambda n: (n > 0).mean() * 100), "avg_net_rs": g.mean(),
        "total_lakh": g.sum() / 1e5,
        "first_half_lakh": trades[first].groupby(keys)["net"].sum() / 1e5,
        "second_half_lakh": trades[~first].groupby(keys)["net"].sum() / 1e5,
        "long_lakh": trades[trades.side == 1].groupby(keys)["net"].sum() / 1e5,
        "short_lakh": trades[trades.side == -1].groupby(keys)["net"].sum() / 1e5,
        "stocks_in_profit_pct": per_stock.groupby(level=keys).apply(lambda n: (n > 0).mean() * 100),
        "t_stat": g.apply(lambda n: n.mean() / n.std() * np.sqrt(len(n)) if len(n) > 1 else np.nan),
    })
    order = ["base", "no_volume", *PAIRS, "pdh", "pwh", "x_sma5_8", "x_sma14_20", "x_sma50_60", "x_pdh", "x_pwh"]
    out = out.reset_index()
    out["variant"] = pd.Categorical(out["variant"], order, ordered=True)
    out = out.sort_values(["timeframe", "style", "variant"])
    out.round(2).to_csv(bt.RESULTS / "level_filters_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
