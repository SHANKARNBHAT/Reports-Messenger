"""
The SMA RSI EOD strategy (sma_rsi_eod.py, shorts on RSI 60) with RVOL and ATR entry filters.

RVOL: the day's volume up to the signal bar, over its average at that time on the last 20 days.
ATR: yesterday's Daily ATR 14 as a % of the entry price. Both are known at entry. The day's first
signal that passes the filters is traded, so a later signal is not lost to an earlier failed one.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eod_backtest as eod
import sma_rsi_eod as sma

RVOL_LEVELS = (0.0, 1.0, 1.5, 2.0)
ATR_LEVELS = (0.0, 2.0, 3.0)
SIDES = {1: "Long", -1: "Short"}
FIRST_BAR = 9 * 60 + 15


def relative_volume(hourly: pd.DataFrame) -> pd.Series:
    """From 10:15 on: Yahoo's 9:15 bar has no volume on most days (and a burst on some), so it is
    left out of every count, and a first-hour signal has no RVOL."""
    days = hourly.index.normalize()
    slots = eod.minutes(hourly.index)
    volume = hourly["Volume"].where(slots != FIRST_BAR, 0.0)
    so_far = volume.groupby(days).cumsum()
    usual = so_far.groupby(slots).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    return (so_far / usual.replace(0, np.nan)).where(slots != FIRST_BAR)


def candidates() -> pd.DataFrame:
    """Every signal with its RVOL, ATR % and EOD result, before any filter."""
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
        rvol = relative_volume(hourly).to_numpy()
        atr_before = bt.atr(daily).shift(1).reindex(hourly.index.normalize()).to_numpy()
        signals = {"2H": sma.candle_entries(bt.session_bars(hourly, 120), 120, hourly_at),
                   "4H": sma.candle_entries(bt.session_bars(hourly, 240), 240, hourly_at),
                   "Daily": sma.live_entries(hourly, daily), "Weekly": sma.live_entries(hourly, bt.weekly_bars(daily))}
        lot = lot_sizes.get(symbol, 1)
        for timeframe, found in signals.items():
            positions = {(r["position"], r["side"]) for r in found if allowed[r["position"]] and r["trigger"] != "RSI 40"}
            for position, side in positions:
                entry = h["close"][position]
                end, price, reason = eod.trade(h, position, side, -side * float("inf"), side * float("inf"))
                if reason == "none":
                    continue
                gross = (price / entry - 1) * side * 100
                rows.append({"symbol": symbol, "timeframe": timeframe, "side": side, "entry_time": hourly.index[position],
                             "exit_time": hourly.index[end], "reason": reason, "gross": gross,
                             "net": gross - bt.charges_pct(entry, lot) - 2 * bt.SLIPPAGE_PCT,
                             "base": base.get(h["minute"][position], 0.0) * side,
                             "rvol": rvol[position], "atr_pct": atr_before[position] / entry * 100})
    return pd.DataFrame(rows)


def main() -> None:
    signals = bt.drop_breaks(candidates())
    signals["entry_time"], signals["exit_time"] = pd.to_datetime(signals["entry_time"]), pd.to_datetime(signals["exit_time"])
    signals["edge"] = signals["net"] - signals["base"]
    signals["day"] = signals["entry_time"].dt.normalize()
    signals = signals.sort_values("entry_time")
    rows = []
    for rvol_min in RVOL_LEVELS:
        for atr_min in ATR_LEVELS:
            mask = pd.Series(True, index=signals.index)
            if rvol_min:
                mask &= signals["rvol"] >= rvol_min
            if atr_min:
                mask &= signals["atr_pct"] >= atr_min
            traded = signals[mask].drop_duplicates(["symbol", "timeframe", "side", "day"])
            for (timeframe, side), group in traded.groupby(["timeframe", "side"]):
                rows.append({"timeframe": timeframe, "side": SIDES[side], "rvol": rvol_min, "atr": atr_min,
                             **eod.summary(group).to_dict()})
    result = pd.DataFrame(rows)
    result.to_csv(bt.RESULTS / "sma_rsi_filters.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 120)
    print("Average net % a trade (trades): columns RVOL, rows timeframe, side, ATR %")
    print(result.pivot_table(index=["timeframe", "side", "atr"], columns="rvol", values="avg_net").round(2).to_string())
    print(result.pivot_table(index=["timeframe", "side", "atr"], columns="rvol", values="trades").round(0).to_string())
    columns = ["timeframe", "side", "rvol", "atr", "trades", "win_pct", "avg_gross", "avg_net", "median_net", "edge",
               "edge_first_half", "edge_second_half", "stocks_positive_pct", "t_stat"]
    print("\nBest 12 by average net")
    print(result.sort_values("avg_net", ascending=False)[columns].head(12).round(2).to_string(index=False))
    print("\nSignal RVOL quantiles", signals["rvol"].quantile([.1, .5, .9]).round(2).to_dict(),
          "| ATR % quantiles", signals["atr_pct"].quantile([.1, .5, .9]).round(2).to_dict(),
          "| RVOL missing", round(float(signals["rvol"].isna().mean() * 100), 1), "%")


if __name__ == "__main__":
    main()
