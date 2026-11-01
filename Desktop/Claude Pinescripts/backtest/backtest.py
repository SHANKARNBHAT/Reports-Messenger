"""
Backtest of All_in_one_Strategy_fixed.pine on the futures stocks, on 2H, 4H, Daily and Weekly.

Each bar follows the order TradingView uses with process_orders_on_close: the ATR stop can
fill inside the bar, then the script runs at the close and exits or enters at the close.
Results are per trade, in % of the position's value (no leverage), after the charges of one
lot (the table in the app's futures_simulator.py) and SLIPPAGE_PCT on each side.
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FUTURES_CSV = Path(r"C:\Users\shank\Report_Messenger_claude\futures stocks.csv")

RSI_LENGTH = 14
RSI_SMA_LENGTH = 20
LONG_SMA, SHORT_SMA = 45.0, 50.0
LONG_CROSS, SHORT_CROSS = 60.0, 40.0
LONG_EXIT, SHORT_EXIT = 60.0, 40.0
ATR_LENGTH = 14
ATR_MULTIPLIER = 2.0
SLIPPAGE_PCT = 0.03
BREAK_PCT = 35.0

RATES = {"order": 20.0, "brokerage": 0.0003, "stt_sell": 0.0005, "exchange": 0.0000173,
         "sebi": 0.000001, "stamp_buy": 0.00002, "gst": 0.18}

TIMEFRAMES = ("2H", "4H", "Daily", "Weekly")


@dataclass(frozen=True)
class Variant:
    name: str
    ha: bool = False
    atr: bool = True
    entry: str = "break"  # break: close past previous HA high/low; colour: green/red HA candle
    exit: str = "previous"  # previous: previous HA candle; colour: last green/red HA candle
    shorts: bool = True
    long_exit: float = LONG_EXIT
    short_exit: float = SHORT_EXIT


VARIANTS = (
    Variant("Pure RSI + ATR"),
    Variant("Pure RSI, no stop", atr=False),
    Variant("RSI+HA + ATR (default)", ha=True),
    Variant("RSI+HA, no stop", ha=True, atr=False),
    Variant("RSI+HA + ATR, last-colour exit", ha=True, exit="colour"),
    Variant("RSI+HA + ATR, colour entry", ha=True, entry="colour"),
    Variant("RSI+HA + ATR, colour entry+exit", ha=True, entry="colour", exit="colour"),
)


def rma(series: pd.Series, length: int) -> pd.Series:
    """ta.rma: seeded with the SMA of the first `length` values, then Wilder smoothing."""
    values = series.to_numpy(float)
    out = np.full(len(values), np.nan)
    start = np.where(~np.isnan(values))[0]
    if len(start) == 0 or start[0] + length > len(values):
        return pd.Series(out, index=series.index)
    first = start[0] + length - 1
    out[first] = np.mean(values[start[0]:first + 1])
    for i in range(first + 1, len(values)):
        out[i] = (out[i - 1] * (length - 1) + values[i]) / length
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    change = close.diff()
    up = rma(change.clip(lower=0), length)
    down = rma((-change).clip(lower=0), length)
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(bars: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    previous = bars["Close"].shift(1)
    true_range = pd.concat([bars["High"] - bars["Low"], (bars["High"] - previous).abs(),
                            (bars["Low"] - previous).abs()], axis=1).max(axis=1)
    return rma(true_range, length)


def heikin_ashi(bars: pd.DataFrame) -> pd.DataFrame:
    ha_close = ((bars["Open"] + bars["High"] + bars["Low"] + bars["Close"]) / 4).to_numpy()
    ha_open = np.empty(len(bars))
    ha_open[0] = (bars["Open"].iloc[0] + bars["Close"].iloc[0]) / 2
    for i in range(1, len(bars)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum.reduce([bars["High"].to_numpy(), ha_open, ha_close])
    ha_low = np.minimum.reduce([bars["Low"].to_numpy(), ha_open, ha_close])
    return pd.DataFrame({"open": ha_open, "close": ha_close, "high": ha_high, "low": ha_low}, index=bars.index)


def read_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
    return frame[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def clean(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[(bars[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]


def session_bars(hourly: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """2H or 4H candles as TradingView builds them on NSE: from 9:15, the last one to the close."""
    since_open = (hourly.index.hour * 60 + hourly.index.minute) - (9 * 60 + 15)
    start = (since_open // minutes) * minutes + 9 * 60 + 15
    key = pd.to_datetime(hourly.index.normalize()) + pd.to_timedelta(start, unit="min")
    grouped = hourly.groupby(key)
    return pd.DataFrame({"Open": grouped["Open"].first(), "High": grouped["High"].max(),
                         "Low": grouped["Low"].min(), "Close": grouped["Close"].last(),
                         "Volume": grouped["Volume"].sum()})


def weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly candles, indexed by the week's last trading day."""
    grouped = daily.groupby(daily.index.to_period("W-SUN"))
    weekly = pd.DataFrame({"Open": grouped["Open"].first(), "High": grouped["High"].max(),
                           "Low": grouped["Low"].min(), "Close": grouped["Close"].last(),
                           "Volume": grouped["Volume"].sum()})
    weekly.index = grouped.apply(lambda week: week.index[-1], include_groups=False).to_numpy()
    return weekly


def filter_values(index: pd.DatetimeIndex, daily_sma: pd.Series) -> np.ndarray:
    """mtfRsiSma: the daily SMA RSI as request.security gives it with lookahead_off. A bar sees
    the last finished day, and the day's last bar sees that day's own value."""
    days = index.normalize()
    last_of_day = pd.Series(days, index=index).groupby(days).transform(lambda s: s.index == s.index[-1]).to_numpy(bool)
    finished = daily_sma.reindex(daily_sma.index.union(days.unique())).ffill()
    today = finished.reindex(days).to_numpy()
    before = finished.shift(1).reindex(days).to_numpy()
    return np.where(last_of_day, today, before)


def cross_above(values: np.ndarray, level: float) -> np.ndarray:
    previous = np.roll(values, 1)
    previous[0] = np.nan
    return (values > level) & (previous <= level)


def cross_below(values: np.ndarray, level: float) -> np.ndarray:
    previous = np.roll(values, 1)
    previous[0] = np.nan
    return (values < level) & (previous >= level)


def prepare(bars: pd.DataFrame, daily_sma: pd.Series) -> dict:
    """Everything the bar loop needs, worked out once for all the variants."""
    candles = heikin_ashi(bars)
    rsi_now = rsi(bars["Close"]).to_numpy()
    green, red = (candles["close"] > candles["open"]).to_numpy(), (candles["close"] < candles["open"]).to_numpy()
    last_green = candles["low"].where(green).ffill().shift(1).to_numpy()
    last_red = candles["high"].where(red).ffill().shift(1).to_numpy()
    return {
        "open": bars["Open"].to_numpy(), "high": bars["High"].to_numpy(), "low": bars["Low"].to_numpy(),
        "close": bars["Close"].to_numpy(), "time": bars.index, "rsi": rsi_now,
        "atr": atr(bars).shift(1).to_numpy(), "filter": filter_values(bars.index, daily_sma),
        "cross_long": cross_above(rsi_now, LONG_CROSS), "cross_short": cross_below(rsi_now, SHORT_CROSS),
        "green": green, "red": red, "ha_high_prev": candles["high"].shift(1).to_numpy(),
        "ha_low_prev": candles["low"].shift(1).to_numpy(),
        "green_prev": last_green, "red_prev": last_red,
    }


def simulate(p: dict, variant: Variant) -> list[dict]:
    opens, highs, lows, closes = p["open"], p["high"], p["low"], p["close"]
    if variant.ha:
        long_ok = p["green"] if variant.entry == "colour" else closes > p["ha_high_prev"]
        short_ok = p["red"] if variant.entry == "colour" else closes < p["ha_low_prev"]
        long_level = p["green_prev"] if variant.exit == "colour" else p["ha_low_prev"]
        short_level = p["red_prev"] if variant.exit == "colour" else p["ha_high_prev"]
    else:
        long_ok = short_ok = np.ones(len(closes), bool)
        long_level = short_level = None
    longs = (p["filter"] > LONG_SMA) & p["cross_long"] & long_ok
    shorts = (p["filter"] < SHORT_SMA) & p["cross_short"] & short_ok & variant.shorts
    exit_long, exit_short = cross_below(p["rsi"], variant.long_exit), cross_above(p["rsi"], variant.short_exit)

    trades, side, entry, stop, opened = [], 0, 0.0, math.nan, 0

    def close_trade(i: int, price: float, reason: str) -> None:
        nonlocal side
        trades.append({"entry_time": p["time"][opened], "exit_time": p["time"][i], "side": side, "entry": entry,
                       "exit": price, "reason": reason, "bars": i - opened})
        side = 0

    for i in range(len(closes)):
        if variant.atr and side == 1 and lows[i] <= stop:
            close_trade(i, min(opens[i], stop), "stop")
        elif variant.atr and side == -1 and highs[i] >= stop:
            close_trade(i, max(opens[i], stop), "stop")

        at_open = side  # strategy.position_size as the script sees it at this bar's close
        if at_open == 1:
            if (closes[i] < long_level[i]) if variant.ha else exit_long[i]:
                close_trade(i, closes[i], "exit")
        elif at_open == -1:
            if (closes[i] > short_level[i]) if variant.ha else exit_short[i]:
                close_trade(i, closes[i], "exit")

        wanted = 1 if longs[i] else -1 if shorts[i] else 0
        if wanted and at_open != wanted:
            if side:
                close_trade(i, closes[i], "reverse")
            side, entry, opened = wanted, closes[i], i
            stop = closes[i] - wanted * p["atr"][i] * ATR_MULTIPLIER
    return trades


def charges_pct(price: float, lot: int) -> float:
    """Round-trip charges of one lot, in % of its value (as charges() in auto_trader.py)."""
    value = price * lot
    brokerage = 2 * min(RATES["order"], value * RATES["brokerage"])
    exchange = 2 * value * RATES["exchange"]
    sebi = 2 * value * RATES["sebi"]
    total = (brokerage + exchange + sebi + value * RATES["stt_sell"] + value * RATES["stamp_buy"]
             + (brokerage + exchange + sebi) * RATES["gst"])
    return total / value * 100


def lots() -> dict[str, int]:
    with FUTURES_CSV.open(encoding="utf-8-sig", newline="") as file:
        return {row["Symbol"].strip(): int(row["Lot size"].replace(",", "")) for row in csv.DictReader(file)}


def run() -> pd.DataFrame:
    lot_sizes = lots()
    all_trades = []
    for path in sorted((DATA / "1d").glob("*.csv")):
        symbol = path.stem
        daily = clean(read_prices(path))
        daily.index = daily.index.normalize()
        daily = daily[~daily.index.duplicated(keep="last")]
        if len(daily) < 60:
            continue
        daily_sma = rsi(daily["Close"]).rolling(RSI_SMA_LENGTH).mean()
        charts = {"Daily": daily, "Weekly": weekly_bars(daily)}
        hourly_path = DATA / "60m" / path.name
        if hourly_path.exists():
            hourly = clean(read_prices(hourly_path))
            charts["2H"], charts["4H"] = session_bars(hourly, 120), session_bars(hourly, 240)
        for timeframe, bars in charts.items():
            prepared = prepare(bars, daily_sma)
            for variant in VARIANTS:
                for trade in simulate(prepared, variant):
                    gross = (trade["exit"] / trade["entry"] - 1) * trade["side"] * 100
                    cost = charges_pct(trade["entry"], lot_sizes.get(symbol, 1)) + 2 * SLIPPAGE_PCT
                    all_trades.append({**trade, "symbol": symbol, "timeframe": timeframe,
                                       "variant": variant.name, "gross": gross, "net": gross - cost})
    return pd.DataFrame(all_trades)


def data_breaks() -> dict[str, list[pd.Timestamp]]:
    """Days a stock's price jumped over BREAK_PCT from one bar to the next: unadjusted splits
    and demergers, not real moves (F&O stocks have no daily circuit, but none moves this far)."""
    breaks: dict[str, set] = {}
    for folder in ("60m", "1d"):
        for path in sorted((DATA / folder).glob("*.csv")):
            close = clean(read_prices(path))["Close"]
            jumps = close.index[close.pct_change().abs() > BREAK_PCT / 100]
            if len(jumps):
                breaks.setdefault(path.stem, set()).update(jumps.normalize())
    return {symbol: sorted(days) for symbol, days in breaks.items()}


def drop_breaks(trades: pd.DataFrame) -> pd.DataFrame:
    """Without the trades open across a data break."""
    breaks = data_breaks()
    print("Data breaks:", {symbol: [str(day.date()) for day in days] for symbol, days in breaks.items()})
    keep = np.ones(len(trades), bool)
    entries, exits = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    for symbol, days in breaks.items():
        mine = (trades["symbol"] == symbol).to_numpy()
        for day in days:
            keep &= ~(mine & (entries.dt.normalize() <= day).to_numpy() & (exits.dt.normalize() >= day).to_numpy())
    print("Trades dropped across data breaks:", int((~keep).sum()), "of", len(trades))
    return trades[keep]


def stats(group: pd.DataFrame) -> pd.Series:
    net = group["net"]
    wins, losses = net[net > 0], net[net <= 0]
    per_stock = group.groupby("symbol")["net"].sum()
    without_top = net.sort_values().iloc[:-max(1, len(net) // 100)]
    return pd.Series({
        "trades": len(net), "stocks": group["symbol"].nunique(),
        "win_pct": (net > 0).mean() * 100, "avg_net": net.mean(), "median_net": net.median(),
        "avg_win": wins.mean(), "avg_loss": losses.mean(),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() else math.nan,
        "stop_pct": (group["reason"] == "stop").mean() * 100, "avg_bars": group["bars"].mean(),
        "long_avg": net[group["side"] == 1].mean(), "short_avg": net[group["side"] == -1].mean(),
        "avg_no_top1pct": without_top.mean(), "stocks_up_pct": (per_stock > 0).mean() * 100,
        "t_stat": net.mean() / (net.std() / math.sqrt(len(net))) if len(net) > 1 else math.nan,
    })


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    trades = drop_breaks(run())
    trades.to_csv(RESULTS / "all_trades.csv.gz", index=False)
    first_intraday = trades.loc[trades["timeframe"] == "2H", "entry_time"].min()
    common = trades[trades["entry_time"] >= first_intraday]

    full = trades.groupby(["timeframe", "variant"]).apply(stats, include_groups=False)
    same_period = common.groupby(["timeframe", "variant"]).apply(stats, include_groups=False)
    by_year = (trades.assign(year=pd.to_datetime(trades["exit_time"]).dt.year)
               .groupby(["timeframe", "variant", "year"])["net"].agg(["count", "mean", "sum"]))
    full.to_csv(RESULTS / "summary_full_history.csv")
    same_period.to_csv(RESULTS / "summary_same_period.csv")
    by_year.to_csv(RESULTS / "by_year.csv")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    columns = ["trades", "win_pct", "avg_net", "median_net", "avg_no_top1pct", "profit_factor", "stop_pct", "avg_bars",
               "long_avg", "short_avg", "stocks_up_pct", "t_stat"]
    print("Same period from", first_intraday, "to", trades["exit_time"].max())
    print(same_period[columns].round(2).to_string())
    print("\nFull history")
    print(full.loc[["Daily", "Weekly"], columns].round(2).to_string())


if __name__ == "__main__":
    main()
