"""
The two main strategies, EOD exit, entered the moment a signal triggers in the first two hours.

No waiting for a candle to close. On the candle forming at 9:15, every trigger is a price known
before the open, so it works like a stop order: filled at that price when the hourly bars of 9:15
or 10:15 touch it. A stock that opens past its trigger, up or down, is skipped on that side for the
day, as no gap entries are wanted. Each trigger is worked out from finished candles:
  Close crosses SMA n   the average of the last n-1 closes, where price equals its own live SMA
  RSI crosses a level   the price that puts Wilder's RSI on that level with this one new close
  HA break              the previous HA candle's high (long) or low (short)
SMA RSI + HA: daily SMA(RSI 14, 20) above 45 or below 50, RSI through 60 or 40, and the HA break.
SMA RSI + Crossover: SMA(RSI 14, 14) on the timeframe above 45 or below 50, and any one of close
through SMA 14, close through SMA 60, or RSI through 60, whichever price is reached first.
One trade per stock, strategy, timeframe, side and day, squared off at 3:15 PM. The wait-for-close
version is kept beside it in first_two_hours_close.csv.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

import backtest as bt
import eod_backtest as eod

LAST_SLOT = 10 * 60 + 15  # the bar starting 10:15 ends at 11:15
SIDES = {1: "Long", -1: "Short"}
NAMES = ("SMA RSI + HA", "SMA RSI + Crossover / RSI cross 60")
RS_60, RS_40 = 1.5, 2 / 3  # up/down average at RSI 60 and 40
SKIPPED = Counter()


def level_for_rs(close: np.ndarray, up: np.ndarray, down: np.ndarray, rs: float, rising: bool) -> np.ndarray:
    """The next close that puts RSI on the level whose up/down ratio is `rs`."""
    keep = bt.RSI_LENGTH - 1
    return close + keep * (rs * down - up) if rising else close - keep * (up / rs - down)


def prepare(bars: pd.DataFrame) -> dict:
    close = bars["Close"]
    change = close.diff()
    up = bt.rma(change.clip(lower=0), bt.RSI_LENGTH).to_numpy()
    down = bt.rma((-change).clip(lower=0), bt.RSI_LENGTH).to_numpy()
    rsi = bt.rsi(close)
    values = close.to_numpy()
    candles = bt.heikin_ashi(bars)
    return {"index": bars.index.to_numpy(), "close": values, "rsi": rsi.to_numpy(), "sma_rsi": rsi.rolling(14).mean().to_numpy(),
            "sma_14": close.rolling(14).mean().to_numpy(), "sma_60": close.rolling(60).mean().to_numpy(),
            "cross_14": close.rolling(13).mean().to_numpy(), "cross_60": close.rolling(59).mean().to_numpy(),
            "rsi_up_60": level_for_rs(values, up, down, RS_60, True), "rsi_down_60": level_for_rs(values, up, down, RS_60, False),
            "rsi_down_40": level_for_rs(values, up, down, RS_40, False),
            "ha_high": candles["high"].to_numpy(), "ha_low": candles["low"].to_numpy()}


def levels(p: dict, i: int, daily_sma: float) -> dict:
    """(strategy, side) -> the day's trigger price, for the finished candle `i`."""
    close, rsi, out = p["close"][i], p["rsi"][i], {}
    if math.isnan(rsi):
        return out
    if daily_sma > bt.LONG_SMA and rsi <= 60:
        out[(NAMES[0], 1)] = max(p["rsi_up_60"][i], p["ha_high"][i])
    if daily_sma < bt.SHORT_SMA and rsi >= 40:
        out[(NAMES[0], -1)] = min(p["rsi_down_40"][i], p["ha_low"][i])
    sma_rsi = p["sma_rsi"][i]
    if sma_rsi > 45:
        found = [p["rsi_up_60"][i]] if rsi <= 60 else []
        found += [p["cross_14"][i]] if close <= p["sma_14"][i] else []
        found += [p["cross_60"][i]] if close <= p["sma_60"][i] else []
        if found:
            out[(NAMES[1], 1)] = min(found)
    if sma_rsi < 50:
        found = [p["rsi_down_60"][i]] if rsi >= 60 else []
        found += [p["cross_14"][i]] if close >= p["sma_14"][i] else []
        found += [p["cross_60"][i]] if close >= p["sma_60"][i] else []
        if found:
            out[(NAMES[1], -1)] = max(found)
    return out


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
        charts = {"2H": bt.session_bars(hourly, 120), "4H": bt.session_bars(hourly, 240), "Daily": daily, "Weekly": bt.weekly_bars(daily)}
        prepared = {timeframe: prepare(bars) for timeframe, bars in charts.items()}
        base = eod.usual_moves(hourly)
        lot = lot_sizes.get(symbol, 1)
        slots = eod.minutes(hourly.index)
        o, h, l, c = (hourly[column].to_numpy() for column in ("Open", "High", "Low", "Close"))
        for day, group in pd.Series(np.arange(len(hourly)), index=hourly.index).groupby(hourly.index.normalize()):
            where = group.to_numpy()
            window = [j for j in where if slots[j] <= LAST_SLOT]
            final = [j for j in where if slots[j] == eod.SQUARE_OFF_MINUTE]
            if not window or not final:
                continue
            exit_price = o[final[0]]
            before = daily_sma[daily_sma.index < day]
            sma_value = before.iloc[-1] if len(before) else math.nan
            closes = daily["Close"][daily.index < day]
            last_close = closes.iloc[-1] if len(closes) else math.nan
            for timeframe, p in prepared.items():
                i = int(np.searchsorted(p["index"], np.datetime64(day), side="left")) - 1
                if i < 0:
                    continue
                for (strategy, side), level in levels(p, i, sma_value).items():
                    if math.isnan(level):
                        continue
                    for j in window:
                        if j == where[0] and ((o[j] > level) if side == 1 else (o[j] < level)):
                            SKIPPED[(strategy, timeframe, SIDES[side])] += 1  # gapped past the trigger: no trade today
                            break
                        reached = h[j] >= level if side == 1 else l[j] <= level
                        if not reached:
                            continue
                        entry = max(o[j], level) if side == 1 else min(o[j], level)
                        gross = (exit_price / entry - 1) * side * 100
                        net = gross - bt.charges_pct(entry, lot) - 2 * bt.SLIPPAGE_PCT
                        rows.append({"strategy": strategy, "symbol": symbol, "timeframe": timeframe, "side": SIDES[side],
                                     "entry_time": hourly.index[j], "exit_time": hourly.index[final[0]], "entry": entry, "exit": exit_price,
                                     "day_gap": (o[where[0]] / last_close - 1) * 100, "reason": "eod", "gross": gross, "net": net,
                                     "base": base.get(slots[j], 0.0) * side})
                        break
    return pd.DataFrame(rows)


def at_close() -> pd.DataFrame:
    """The same two strategies entered at the close of the signal candle, from the EOD tests."""
    dates = ["entry_time", "exit_time"]
    rsi = pd.read_csv(bt.RESULTS / "eod_trades.csv.gz", parse_dates=dates)
    rsi = rsi[(rsi["mode"] == "RSI + HA") & (rsi["stop"] == "EOD only")].assign(strategy=NAMES[0])
    cross = pd.read_csv(bt.RESULTS / "sma_rsi_eod_trades.csv.gz", parse_dates=dates)
    cross = cross[cross["setup"].isin(("Long", "Short, RSI 60"))].assign(strategy=NAMES[1])
    columns = ["strategy", "symbol", "timeframe", "side", "entry_time", "exit_time", "reason", "gross", "net", "base", "edge"]
    frame = pd.concat([rsi[columns], cross[columns]], ignore_index=True)
    slot = frame["entry_time"].dt.hour * 60 + frame["entry_time"].dt.minute
    frame = frame[slot <= LAST_SLOT]
    return frame.assign(side=frame["side"].map(SIDES))


def main() -> None:
    trades = bt.drop_breaks(run())
    trades["entry_time"], trades["exit_time"] = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    trades["edge"] = trades["net"] - trades["base"]
    trades.to_csv(bt.RESULTS / "first_two_hours_trades.csv.gz", index=False)
    keys = ["strategy", "timeframe", "side"]
    result = trades.groupby(keys).apply(eod.summary, include_groups=False)
    result["skipped_gaps"] = pd.Series(SKIPPED).rename_axis(keys)
    close_result = at_close().groupby(keys).apply(eod.summary, include_groups=False)
    result["avg_net_at_close"] = close_result["avg_net"]
    result["trades_at_close"] = close_result["trades"]
    result.to_csv(bt.RESULTS / "first_two_hours.csv")
    close_result.to_csv(bt.RESULTS / "first_two_hours_close.csv")
    stocks = trades.groupby([*keys, "symbol"]).agg(trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100),
                                                   total_net=("net", "sum"), avg_net=("net", "mean")).reset_index()
    stocks.round(3).to_csv(bt.RESULTS / "first_two_hours_stocks.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 80)
    columns = ["trades", "win_pct", "avg_gross", "avg_net", "median_net", "edge", "edge_first_half", "edge_second_half",
               "stocks_positive_pct", "t_stat", "skipped_gaps", "trades_at_close", "avg_net_at_close"]
    print(result[columns].round(2).to_string())
    years = trades.assign(year=trades["entry_time"].dt.year).groupby([*keys, "year"])["net"].mean()
    print("\nAverage net % a trade, by year")
    print(years.unstack("year").round(2).to_string())
    print("\nStricter gap rule, day's open within 0.5% of yesterday's close: trades and average net %")
    print(trades[trades["day_gap"].abs() <= 0.5].groupby(keys)["net"].agg(["count", "mean"]).round(2).unstack("side").to_string())
    for strategy in NAMES:
        for (timeframe, side), group in stocks[(stocks["strategy"] == strategy) & (stocks["trades"] >= 10)].groupby(["timeframe", "side"]):
            print(strategy, timeframe, side, "| stocks with 10+ trades:", len(group), "| in profit:", int((group["total_net"] > 0).sum()),
                  "| best:", ", ".join(f"{r.symbol} {r.total_net:+.0f}%" for r in group.nlargest(3, "total_net").itertuples()))

    sample = trades[(trades["symbol"] == "RELIANCE") & (trades["timeframe"] == "Daily") & (trades["strategy"] == NAMES[1]) & (trades["side"] == "Long")].iloc[-1]
    day = sample["entry_time"].normalize()
    daily = bt.clean(bt.read_prices(bt.DATA / "1d" / "RELIANCE.csv"))
    daily.index = daily.index.normalize()
    close = daily["Close"]
    print("\nRELIANCE Daily long:", sample["entry_time"], "entry", round(sample["entry"], 2), "exit", round(sample["exit"], 2))
    before = close[close.index < day]
    print("Yesterday close", round(before.iloc[-1], 2), "SMA14", round(before.tail(14).mean(), 2), "SMA60", round(before.tail(60).mean(), 2),
          "| cross-14 level", round(before.tail(13).mean(), 2), "cross-60 level", round(before.tail(59).mean(), 2), "| RSI", round(bt.rsi(before).iloc[-1], 2))
    print(bt.clean(bt.read_prices(bt.DATA / "60m" / "RELIANCE.csv")).loc[str(day.date())][["Open", "High", "Low", "Close"]].round(2).head(3).to_string())
    print("RSI if the day closes at the entry price:", round(bt.rsi(pd.concat([before, pd.Series([sample["entry"]], index=[day])])).iloc[-1], 2))


if __name__ == "__main__":
    main()
