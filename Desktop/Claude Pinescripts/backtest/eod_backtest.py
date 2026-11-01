"""
The RSI SMA strategy held only until the end of the day, on the futures stocks.

2H and 4H enter at the close of a signal candle. Daily and Weekly enter on the hour the live
RSI crosses the level, with the day's or week's candle still forming, as Chartink signals do.
As in the app's Auto trader: entries up to 3:00 PM, square-off at 3:15 PM, one trade per stock,
side and day. Stops are checked on hourly bars, and if a bar touches stop and target the stop
counts first. `base` is the stock's usual move from that hour to 3:15 PM, on all its days, and
`edge` the trade's net % less that move, so a normal day's drift does not count as the strategy.
"""
import math

import numpy as np
import pandas as pd

import backtest as bt
import test_long_only as lo

LAST_ENTRY_MINUTE = 13 * 60 + 15  # start of the last hourly bar to close before 3:00 PM: it ends at 2:15 PM
SQUARE_OFF_MINUTE = 15 * 60 + 15
SPLIT = pd.Timestamp("2025-05-12")
STOPS = ("EOD only", "EOD + ATR stop", "EOD + 1.5% SL, 3% target")


def minutes(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour * 60 + index.minute


def entries_on_candles(bars: pd.DataFrame, length: int, daily_sma: pd.Series, hourly_at: dict) -> list[dict]:
    """Signals of 2H or 4H candles, at the hourly bar each candle ends with."""
    p = bt.prepare(bars, daily_sma)
    rows = []
    last_hour = bars.index + pd.to_timedelta(length - 60, unit="min")
    for side, signal, passes in ((1, (p["filter"] > bt.LONG_SMA) & p["cross_long"], p["close"] > p["ha_high_prev"]),
                                 (-1, (p["filter"] < bt.SHORT_SMA) & p["cross_short"], p["close"] < p["ha_low_prev"])):
        for i in np.flatnonzero(signal):
            position = hourly_at.get(last_hour[i])
            if position is not None and not math.isnan(p["atr"][i]):
                rows.append({"position": position, "side": side, "ha": bool(passes[i]), "atr": p["atr"][i]})
    return rows


def entries_live(hourly: pd.DataFrame, higher: pd.DataFrame, daily_sma: pd.Series) -> list[dict]:
    """Signals of a forming Daily or Weekly candle, from its finished candles and the live price."""
    close = higher["Close"]
    change = close.diff()
    up, down = bt.rma(change.clip(lower=0), bt.RSI_LENGTH).to_numpy(), bt.rma((-change).clip(lower=0), bt.RSI_LENGTH).to_numpy()
    rsi = bt.rsi(close).to_numpy()
    atr_now = bt.atr(higher).to_numpy()
    candles = bt.heikin_ashi(higher)
    ends = higher.index.to_numpy()
    days = hourly.index.normalize()
    before = np.searchsorted(ends, days.to_numpy(), side="left") - 1
    sma_before = daily_sma.reindex(daily_sma.index.union(days.unique())).ffill().shift(1).reindex(days).to_numpy()
    price = hourly["Close"].to_numpy()

    valid = before >= 0
    previous = before.clip(min=0)
    gain = price - close.to_numpy()[previous]
    live_up = (up[previous] * (bt.RSI_LENGTH - 1) + np.clip(gain, 0, None)) / bt.RSI_LENGTH
    live_down = (down[previous] * (bt.RSI_LENGTH - 1) + np.clip(-gain, 0, None)) / bt.RSI_LENGTH
    with np.errstate(divide="ignore", invalid="ignore"):
        live = 100 - 100 / (1 + live_up / live_down)
    earlier = rsi[previous]
    rows = []
    allowed = valid & (minutes(hourly.index) <= LAST_ENTRY_MINUTE)
    high_before, low_before = candles["high"].to_numpy()[previous], candles["low"].to_numpy()[previous]
    for side, signal, passes in ((1, (sma_before > bt.LONG_SMA) & (earlier <= bt.LONG_CROSS) & (live > bt.LONG_CROSS), price > high_before),
                                 (-1, (sma_before < bt.SHORT_SMA) & (earlier >= bt.SHORT_CROSS) & (live < bt.SHORT_CROSS), price < low_before)):
        for position in np.flatnonzero(signal & allowed):
            if not math.isnan(atr_now[previous[position]]):
                rows.append({"position": position, "side": side, "ha": bool(passes[position]), "atr": atr_now[previous[position]]})
    return rows


def first_each_day(rows: list[dict], days: np.ndarray, ha: bool) -> list[dict]:
    """One trade per side and day: the day's first signal that passes the mode's filter."""
    seen, kept = set(), []
    for row in sorted(rows, key=lambda r: r["position"]):
        if ha and not row["ha"]:
            continue
        key = (days[row["position"]], row["side"])
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def trade(h: dict, position: int, side: int, stop: float, target: float) -> tuple[int, float, str]:
    """From the close of `position` to the stop, the target or 3:15 PM, whichever comes first."""
    for j in range(position + 1, len(h["open"])):
        if h["day"][j] != h["day"][position]:
            return j - 1, h["close"][j - 1], "eod"
        if h["minute"][j] >= SQUARE_OFF_MINUTE:
            return j, h["open"][j], "eod"
        low, high = h["low"][j], h["high"][j]
        if side == 1 and low <= stop:
            return j, min(h["open"][j], stop), "stop"
        if side == -1 and high >= stop:
            return j, max(h["open"][j], stop), "stop"
        if side == 1 and high >= target:
            return j, max(h["open"][j], target), "target"
        if side == -1 and low <= target:
            return j, min(h["open"][j], target), "target"
    return position, h["close"][position], "none"


def usual_moves(hourly: pd.DataFrame) -> pd.Series:
    """The average % move from each hour's close to 3:15 PM, on all the stock's days."""
    days = hourly.index.normalize()
    square_off = hourly["Open"].where(minutes(hourly.index) == SQUARE_OFF_MINUTE).groupby(days).transform("last")
    move = (square_off / hourly["Close"] - 1) * 100
    return move.groupby(minutes(hourly.index)).mean()


def run() -> pd.DataFrame:
    lot_sizes, rows = bt.lots(), []
    for path in sorted((bt.DATA / "60m").glob("*.csv")):
        symbol = path.stem
        hourly = bt.clean(bt.read_prices(path))
        hourly = hourly[~hourly.index.duplicated(keep="last")]
        daily = bt.clean(bt.read_prices(bt.DATA / "1d" / path.name))
        daily.index = daily.index.normalize()
        daily = daily[~daily.index.duplicated(keep="last")]
        daily_sma = bt.rsi(daily["Close"]).rolling(bt.RSI_SMA_LENGTH).mean()
        hourly_at = {time: i for i, time in enumerate(hourly.index)}
        h = {"open": hourly["Open"].to_numpy(), "high": hourly["High"].to_numpy(), "low": hourly["Low"].to_numpy(),
             "close": hourly["Close"].to_numpy(), "day": hourly.index.normalize().to_numpy(), "minute": minutes(hourly.index)}
        base = usual_moves(hourly)
        allowed = minutes(hourly.index) <= LAST_ENTRY_MINUTE
        signals = {"2H": entries_on_candles(bt.session_bars(hourly, 120), 120, daily_sma, hourly_at),
                   "4H": entries_on_candles(bt.session_bars(hourly, 240), 240, daily_sma, hourly_at),
                   "Daily": entries_live(hourly, daily, daily_sma),
                   "Weekly": entries_live(hourly, bt.weekly_bars(daily), daily_sma)}
        lot = lot_sizes.get(symbol, 1)
        for timeframe, found in signals.items():
            found = [row for row in found if allowed[row["position"]]]
            for mode, ha in (("Pure RSI", False), ("RSI + HA", True)):
                for row in first_each_day(found, h["day"], ha):
                    position, side = row["position"], row["side"]
                    entry = h["close"][position]
                    exits = {"EOD only": (-side * math.inf, side * math.inf),
                             "EOD + ATR stop": (entry - side * row["atr"] * bt.ATR_MULTIPLIER, side * math.inf),
                             "EOD + 1.5% SL, 3% target": (entry * (1 - side * 0.015), entry * (1 + side * 0.03))}
                    for name, (stop, target) in exits.items():
                        end, price, reason = trade(h, position, side, stop, target)
                        if reason == "none":
                            continue
                        gross = (price / entry - 1) * side * 100
                        net = gross - bt.charges_pct(entry, lot) - 2 * bt.SLIPPAGE_PCT
                        rows.append({"symbol": symbol, "timeframe": timeframe, "mode": mode, "stop": name, "side": side,
                                     "variant": name, "entry_time": hourly.index[position], "exit_time": hourly.index[end],
                                     "entry": entry, "exit": price, "reason": reason, "bars": end - position, "gross": gross,
                                     "net": net, "base": base.get(minutes(hourly.index[[position]])[0], 0.0) * side})
    return pd.DataFrame(rows)


def summary(group: pd.DataFrame) -> pd.Series:
    by_stock = group.groupby("symbol")["edge"].mean()
    first = group[group["entry_time"] < SPLIT]["edge"]
    second = group[group["entry_time"] >= SPLIT]["edge"]
    return pd.Series({"trades": len(group), "win_pct": (group["net"] > 0).mean() * 100, "avg_gross": group["gross"].mean(),
                      "avg_net": group["net"].mean(), "median_net": group["net"].median(), "avg_base": group["base"].mean(),
                      "edge": group["edge"].mean(), "edge_first_half": first.mean(), "edge_second_half": second.mean(),
                      "stopped_pct": (group["reason"] == "stop").mean() * 100, "target_pct": (group["reason"] == "target").mean() * 100,
                      "stocks_positive_pct": (by_stock > 0).mean() * 100,
                      "t_stat": by_stock.mean() / (by_stock.std() / math.sqrt(len(by_stock))) if len(by_stock) > 1 else math.nan})


def main() -> None:
    trades = bt.drop_breaks(run())
    trades["entry_time"], trades["exit_time"] = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    trades["edge"] = trades["net"] - trades["base"]
    trades.to_csv(bt.RESULTS / "eod_trades.csv.gz", index=False)
    order = {"2H": 0, "4H": 1, "Daily": 2, "Weekly": 3}
    result = trades.groupby(["timeframe", "mode", "stop", "side"]).apply(summary, include_groups=False)
    result = result.sort_index(level=["timeframe"], key=lambda level: level.map(order), sort_remaining=True)
    result.to_csv(bt.RESULTS / "eod_summary.csv")
    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_rows", 120)
    print("From", trades["entry_time"].min(), "to", trades["exit_time"].max())
    print(result.round(2).to_string())
    years = trades.assign(year=trades["exit_time"].dt.year).groupby(["timeframe", "mode", "stop", "side", "year"])["net"].mean()
    print("\n== Average net % a trade, by year")
    print(years.unstack("year").round(2).to_string())
    print("\nFirst entry times:", trades.groupby("timeframe")["entry_time"].apply(lambda t: sorted(set(t.dt.strftime("%H:%M")))).to_dict())


if __name__ == "__main__":
    main()
