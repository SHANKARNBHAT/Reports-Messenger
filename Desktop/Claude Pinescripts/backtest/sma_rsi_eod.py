"""
SMA RSI strategy, held to the end of the day, on the futures stocks.

Long: SMA(RSI 14, 14) above 45, and the close crosses above SMA 14 or SMA 60, or RSI crosses above 60.
Short: SMA(RSI 14, 14) below 50, and the close crosses below SMA 14 or SMA 60, or RSI crosses below
60 (as asked) or 40 (the earlier strategy's level). Any one trigger is enough. All on the trade's
timeframe. Entries, square-off, costs and `edge` as in eod_backtest.py, with no stop or target.
"""
import pandas as pd
import numpy as np

import backtest as bt
import eod_backtest as eod

FILTER_LENGTH = 14
LONG_FILTER, SHORT_FILTER = 45.0, 50.0
TRIGGERS = {("Long", 1): ("SMA 14", "SMA 60", "RSI 60"),
            ("Short, RSI 60", -1): ("SMA 14", "SMA 60", "RSI 60"),
            ("Short, RSI 40", -1): ("SMA 14", "SMA 60", "RSI 40")}


def candle_entries(bars: pd.DataFrame, length: int, hourly_at: dict) -> list[dict]:
    close = bars["Close"]
    rsi = bt.rsi(close)
    filter_ = rsi.rolling(FILTER_LENGTH).mean().to_numpy()
    above_14 = (close - close.rolling(14).mean()).to_numpy()
    above_60 = (close - close.rolling(60).mean()).to_numpy()
    values = rsi.to_numpy()
    fired = {(1, "SMA 14"): bt.cross_above(above_14, 0), (1, "SMA 60"): bt.cross_above(above_60, 0),
             (1, "RSI 60"): bt.cross_above(values, 60), (-1, "SMA 14"): bt.cross_below(above_14, 0),
             (-1, "SMA 60"): bt.cross_below(above_60, 0), (-1, "RSI 60"): bt.cross_below(values, 60),
             (-1, "RSI 40"): bt.cross_below(values, 40)}
    last_hour = bars.index + pd.to_timedelta(length - 60, unit="min")
    rows = []
    for (side, trigger), signal in fired.items():
        passed = filter_ > LONG_FILTER if side == 1 else filter_ < SHORT_FILTER
        for i in np.flatnonzero(signal & passed):
            position = hourly_at.get(last_hour[i])
            if position is not None:
                rows.append({"position": position, "side": side, "trigger": trigger})
    return rows


def live_entries(hourly: pd.DataFrame, higher: pd.DataFrame) -> list[dict]:
    """The same triggers on a forming Daily or Weekly candle, with the live price as its close."""
    close = higher["Close"]
    change = close.diff()
    up = bt.rma(change.clip(lower=0), bt.RSI_LENGTH).to_numpy()
    down = bt.rma((-change).clip(lower=0), bt.RSI_LENGTH).to_numpy()
    rsi = bt.rsi(close)
    days = hourly.index.normalize()
    before = np.searchsorted(higher.index.to_numpy(), days.to_numpy(), side="left") - 1
    valid, previous = before >= 0, before.clip(min=0)
    price, last = hourly["Close"].to_numpy(), close.to_numpy()[previous]
    gain = price - last
    live_up = (up[previous] * (bt.RSI_LENGTH - 1) + np.clip(gain, 0, None)) / bt.RSI_LENGTH
    live_down = (down[previous] * (bt.RSI_LENGTH - 1) + np.clip(-gain, 0, None)) / bt.RSI_LENGTH
    with np.errstate(divide="ignore", invalid="ignore"):
        live_rsi = 100 - 100 / (1 + live_up / live_down)
    live_filter = (rsi.rolling(FILTER_LENGTH - 1).sum().to_numpy()[previous] + live_rsi) / FILTER_LENGTH
    sma_14 = close.rolling(14).mean().to_numpy()[previous]
    sma_60 = close.rolling(60).mean().to_numpy()[previous]
    live_14 = (close.rolling(13).sum().to_numpy()[previous] + price) / 14
    live_60 = (close.rolling(59).sum().to_numpy()[previous] + price) / 60
    earlier = rsi.to_numpy()[previous]
    fired = {(1, "SMA 14"): (last <= sma_14) & (price > live_14), (1, "SMA 60"): (last <= sma_60) & (price > live_60),
             (1, "RSI 60"): (earlier <= 60) & (live_rsi > 60), (-1, "SMA 14"): (last >= sma_14) & (price < live_14),
             (-1, "SMA 60"): (last >= sma_60) & (price < live_60), (-1, "RSI 60"): (earlier >= 60) & (live_rsi < 60),
             (-1, "RSI 40"): (earlier >= 40) & (live_rsi < 40)}
    allowed = valid & (eod.minutes(hourly.index) <= eod.LAST_ENTRY_MINUTE)
    rows = []
    for (side, trigger), signal in fired.items():
        passed = live_filter > LONG_FILTER if side == 1 else live_filter < SHORT_FILTER
        for position in np.flatnonzero(signal & passed & allowed):
            rows.append({"position": position, "side": side, "trigger": trigger})
    return rows


def run() -> pd.DataFrame:
    lot_sizes, rows = bt.lots(), []
    for path in sorted((bt.DATA / "60m").glob("*.csv")):
        symbol = path.stem
        hourly = bt.clean(bt.read_prices(path))
        hourly = hourly[~hourly.index.duplicated(keep="last")]
        daily = bt.clean(bt.read_prices(bt.DATA / "1d" / path.name))
        daily.index = daily.index.normalize()
        daily = daily[~daily.index.duplicated(keep="last")]
        hourly_at = {time: i for i, time in enumerate(hourly.index)}
        h = {"open": hourly["Open"].to_numpy(), "high": hourly["High"].to_numpy(), "low": hourly["Low"].to_numpy(),
             "close": hourly["Close"].to_numpy(), "day": hourly.index.normalize().to_numpy(), "minute": eod.minutes(hourly.index)}
        base = eod.usual_moves(hourly)
        allowed = eod.minutes(hourly.index) <= eod.LAST_ENTRY_MINUTE
        signals = {"2H": candle_entries(bt.session_bars(hourly, 120), 120, hourly_at),
                   "4H": candle_entries(bt.session_bars(hourly, 240), 240, hourly_at),
                   "Daily": live_entries(hourly, daily), "Weekly": live_entries(hourly, bt.weekly_bars(daily))}
        lot = lot_sizes.get(symbol, 1)
        for timeframe, found in signals.items():
            for (name, side), triggers in TRIGGERS.items():
                mine = [r for r in found if r["side"] == side and r["trigger"] in triggers and allowed[r["position"]]]
                seen = set()
                for row in sorted(mine, key=lambda r: r["position"]):
                    position = row["position"]
                    if h["day"][position] in seen:
                        continue
                    seen.add(h["day"][position])
                    entry = h["close"][position]
                    end, price, reason = eod.trade(h, position, side, -side * float("inf"), side * float("inf"))
                    if reason == "none":
                        continue
                    gross = (price / entry - 1) * side * 100
                    net = gross - bt.charges_pct(entry, lot) - 2 * bt.SLIPPAGE_PCT
                    rows.append({"symbol": symbol, "timeframe": timeframe, "setup": name, "side": side, "trigger": row["trigger"],
                                 "entry_time": hourly.index[position], "exit_time": hourly.index[end], "entry": entry, "exit": price,
                                 "reason": reason, "bars": end - position, "gross": gross, "net": net,
                                 "base": base.get(eod.minutes(hourly.index[[position]])[0], 0.0) * side})
    return pd.DataFrame(rows)


def main() -> None:
    trades = bt.drop_breaks(run())
    trades["entry_time"], trades["exit_time"] = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    trades["edge"] = trades["net"] - trades["base"]
    trades.to_csv(bt.RESULTS / "sma_rsi_eod_trades.csv.gz", index=False)
    columns = ["trades", "win_pct", "avg_gross", "avg_net", "median_net", "edge", "edge_first_half", "edge_second_half", "stocks_positive_pct", "t_stat"]
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 120)
    result = trades.groupby(["timeframe", "setup"]).apply(eod.summary, include_groups=False)[columns]
    result.to_csv(bt.RESULTS / "sma_rsi_eod_summary.csv")
    print(result.round(2).to_string())
    print("\n== By the trigger that fired first")
    print(trades.groupby(["timeframe", "setup", "trigger"]).apply(eod.summary, include_groups=False)[columns].round(2).to_string())
    print("\n== Average net % a trade, by year")
    years = trades.assign(year=trades["exit_time"].dt.year).groupby(["timeframe", "setup", "year"])["net"].mean()
    print(years.unstack("year").round(2).to_string())
    first = trades[(trades["entry_time"].dt.hour == 9) & trades["timeframe"].isin(("Daily", "Weekly"))]
    print("\n== Daily and Weekly, first-hour signals")
    print(first.groupby(["timeframe", "setup"]).apply(eod.summary, include_groups=False)[columns].round(2).to_string())

    sample = trades[(trades["symbol"] == "RELIANCE") & (trades["timeframe"] == "2H") & (trades["setup"] == "Long")].iloc[-1]
    bars = bt.session_bars(bt.clean(bt.read_prices(bt.DATA / "60m" / "RELIANCE.csv")), 120)
    check = pd.DataFrame({"close": bars["Close"], "sma14": bars["Close"].rolling(14).mean(), "sma60": bars["Close"].rolling(60).mean(),
                          "rsi": bt.rsi(bars["Close"]), "sma_rsi": bt.rsi(bars["Close"]).rolling(14).mean()}).round(2)
    print("\nRELIANCE 2H long:", sample["trigger"], sample["entry_time"], sample["entry"], "exit", sample["exit_time"], sample["exit"])
    place = check.index.searchsorted(sample["entry_time"] - pd.Timedelta(minutes=60))
    print(check.iloc[place - 2:place + 1].to_string())


if __name__ == "__main__":
    main()
