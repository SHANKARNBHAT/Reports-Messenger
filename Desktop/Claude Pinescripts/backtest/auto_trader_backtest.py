"""
The app's Auto trader (auto_trader.py, futures_simulator.replay) replayed on the futures stocks.

Entries are the widgets of the two Chartink dashboards, 501779 (RSI) and 502621 (Heikin Ashi), on
2H, 4H and Daily, condition by condition. Each timeframe is tested on its own. As Chartink does, a
widget reads the forming candle: at the end of every hourly bar up to 3:00 PM its candle is built
from the hours so far, with the finished candles behind it, and a stock that matches is entered at
that hourly close. The candles are Chartink's: 2H and 4H from 9:15, the day's last one to the close.

Exits are the Auto trader's, 1.5% SL on all of them, one position per stock, style and timeframe:
  eod       3% target, squared off at 3:15 PM
  trail     held; at the 3% target the stop trails 1.5% behind in 0.5% jumps
  rsi       held, no target; out when any timeframe's RSI crosses the level (the "RSI crossed"
            widgets) and the RSI of the trade's own timeframe is past it, 60 for a long, 40 for a short
  ha        held, no target; out when the price closes past the previous HA candle's low or high
  ha_green  the same with the last green (long) or red (short) HA candle
Held positions close at 3:15 PM on expiry day, the last Tuesday of the month as in the app.

No gap entries: on a day the stock opens past its trigger, or POINT_FIVE_GAP or more away from
yesterday's close in the trade's direction, that side is skipped for the day. Stops are replayed on
hourly bars, not the app's 5-minute ones: the SL counts first if a bar touches SL and target.
Each trade is INVESTMENT rupees of position value, no compounding, after the app's charges and
bt.SLIPPAGE_PCT on each side. Prices are Yahoo's (download_prices.py), the cash stocks, not futures.

    python auto_trader_backtest.py
"""
import calendar
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

import backtest as bt
import eod_backtest as eod

INVESTMENT = 1_000_000
TOP_PCT = 10.0
START = pd.Timestamp("2024-05-01")  # 200 4H candles of warm-up on the hourly data from Nov 2023
SPLIT = pd.Timestamp("2025-08-01")  # middle of the test, for first half against second half
POINT_FIVE_GAP = 0.5
SL_PCT, TP_PCT, TRAIL_PCT, STEP_PCT = 1.5, 3.0, 1.5, 0.5
LAST_SLOT = 13 * 60 + 15  # the last hourly bar to end by 3:00 PM starts here and ends at 2:15 PM
TIMEFRAMES = ("2H", "4H", "Daily")
STYLES = {"RSI": ("eod", "trail", "rsi"), "HA": ("eod", "trail", "rsi", "ha", "ha_green")}
EXIT_NAMES = {"eod": "EOD 3:15 PM", "trail": "Trailing SL", "rsi": "RSI exit", "ha": "HA exit", "ha_green": "HA green/red exit"}
# fast and slow SMA of each RSI widget, by timeframe and side; they differ from widget to widget.
TREND = {("2H", 1): (16, 60), ("2H", -1): (16, 50), ("4H", 1): (14, 50), ("4H", -1): (14, 50),
         ("Daily", 1): (16, 50), ("Daily", -1): (16, 50)}


def tick(price: float) -> float:
    return round(round(price * 20) / 20, 2)


def compare(side: int):
    """past: strictly past in the trade's direction; reached: past or on it; before: not past."""
    if side == 1:
        return (lambda a, b: a > b), (lambda a, b: a >= b), (lambda a, b: a <= b)
    return (lambda a, b: a < b), (lambda a, b: a <= b), (lambda a, b: a >= b)


def expiry_after(day: date) -> date:
    for ahead in (0, 1):
        year, month = divmod(day.year * 12 + day.month - 1 + ahead, 12)
        last = date(year, month + 1, calendar.monthrange(year, month + 1)[1])
        tuesday = last - timedelta(days=(last.weekday() - calendar.TUESDAY) % 7)
        if tuesday > day:
            return tuesday


def fill_open_volume(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Yahoo leaves the volume of most 9:15 bars at 0: the day's volume less its other hours."""
    days = hourly.index.normalize()
    slots = eod.minutes(hourly.index)
    rest = hourly["Volume"].where(slots != 9 * 60 + 15, 0.0).groupby(days).transform("sum")
    total = daily["Volume"].reindex(days).to_numpy()
    missing = (slots == 9 * 60 + 15) & (hourly["Volume"].to_numpy() == 0)
    volume = hourly["Volume"].to_numpy().copy()
    volume[missing] = np.nan_to_num(np.clip(total - rest.to_numpy(), 0, None))[missing]
    return hourly.assign(Volume=volume)


def chartink_candles(hourly: pd.DataFrame, minutes: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Candles from 9:15 with the day's last one running to the close, and each hour's candle."""
    since_open = (hourly.index.hour * 60 + hourly.index.minute) - (9 * 60 + 15)
    slot = np.minimum(since_open // minutes, {120: 2, 240: 1}[minutes])
    key = hourly.index.normalize() + pd.to_timedelta(slot * minutes, unit="min")
    grouped = hourly.groupby(key)
    bars = pd.DataFrame({"Open": grouped["Open"].first(), "High": grouped["High"].max(), "Low": grouped["Low"].min(),
                         "Close": grouped["Close"].last(), "Volume": grouped["Volume"].sum()})
    return bars, np.searchsorted(bars.index.to_numpy(), key.to_numpy())


class Chart:
    """One timeframe seen from each hourly bar: finished candles and the forming one."""

    def __init__(self, bars: pd.DataFrame, where: np.ndarray, hourly: pd.DataFrame):
        self.where, self.close = where, hourly["Close"].to_numpy()
        self.candles = bars
        self.closes = bars["Close"]
        self.finished_close = bars["Close"].to_numpy()
        groups = pd.Series(where, index=hourly.index)
        self.open = hourly["Open"].groupby(groups).transform("first").to_numpy()
        self.high = hourly["High"].groupby(groups).cummax().to_numpy()
        self.low = hourly["Low"].groupby(groups).cummin().to_numpy()
        self.volume = hourly["Volume"].groupby(groups).cumsum().to_numpy()
        change = self.closes.diff()
        self.up = bt.rma(change.clip(lower=0), bt.RSI_LENGTH).to_numpy()
        self.down = bt.rma((-change).clip(lower=0), bt.RSI_LENGTH).to_numpy()
        self.finished_rsi = bt.rsi(self.closes)
        self.smas = {}
        ha = bt.heikin_ashi(bars)
        self.ha_open, self.ha_close = ha["open"].to_numpy(), ha["close"].to_numpy()
        self.ha_high, self.ha_low = ha["high"].to_numpy(), ha["low"].to_numpy()
        green, red = self.ha_close > self.ha_open, self.ha_close < self.ha_open
        self.last_green_low = ha["low"].where(green).ffill().to_numpy()
        self.last_red_high = ha["high"].where(red).ffill().to_numpy()

    def ago(self, values, back: int = 1) -> np.ndarray:
        """A finished candle's value, `back` candles before the forming one."""
        values = np.asarray(values, float)
        index = self.where - back
        out = np.full(len(index), np.nan)
        out[index >= 0] = values[index[index >= 0]]
        return out

    def finished_sma(self, length: int) -> np.ndarray:
        if length not in self.smas:
            self.smas[length] = self.closes.rolling(length).mean().to_numpy()
        return self.smas[length]

    def sma(self, length: int, price: np.ndarray) -> np.ndarray:
        behind = self.closes.rolling(length - 1).sum().to_numpy()
        return (self.ago(behind) + price) / length

    def rsi(self, price: np.ndarray) -> np.ndarray:
        gain = price - self.ago(self.finished_close)
        keep = bt.RSI_LENGTH - 1
        up = (self.ago(self.up) * keep + np.clip(gain, 0, None)) / bt.RSI_LENGTH
        down = (self.ago(self.down) * keep + np.clip(-gain, 0, None)) / bt.RSI_LENGTH
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100 - 100 / (1 + up / down)

    def sma_rsi(self) -> np.ndarray:
        behind = self.finished_rsi.rolling(13).sum().to_numpy()
        return (self.ago(behind) + self.rsi(self.close)) / 14

    def volume_up(self) -> np.ndarray:
        behind = self.candles["Volume"].rolling(4).sum().to_numpy()
        return self.volume > (self.ago(behind) + self.volume) / 5

    def crossed_at_last_candle(self, side: int, length: int) -> np.ndarray:
        """Close went past SMA `length` on the last finished candle."""
        past, _, before = compare(side)
        sma = self.finished_sma(length)
        return past(self.ago(self.finished_close, 1), self.ago(sma, 1)) & before(self.ago(self.finished_close, 2), self.ago(sma, 2))

    def lagged_cross(self, side: int, length: int, price: np.ndarray) -> np.ndarray:
        """Price past the last candle's SMA, with the last close not past the one before."""
        past, _, before = compare(side)
        sma = self.finished_sma(length)
        return past(price, self.ago(sma, 1)) & before(self.ago(self.finished_close, 1), self.ago(sma, 2))

    def rsi_cross(self, side: int, price: np.ndarray) -> np.ndarray:
        """RSI through 60 (long) or 40 (short) on the forming candle."""
        past, _, before = compare(side)
        level = bt.LONG_CROSS if side == 1 else bt.SHORT_CROSS
        return past(self.rsi(price), level) & before(self.ago(self.finished_rsi), level)


def rsi_triggers(chart: Chart, timeframe: str, side: int, price: np.ndarray) -> np.ndarray:
    """The triggers of the RSI widgets that depend on the price now."""
    past, _, before = compare(side)
    near = 14 if timeframe == "Daily" else 16
    found = past(price, chart.sma(near, price)) & before(chart.ago(chart.finished_close), chart.ago(chart.finished_sma(near)))
    for length in (20, 60, 200) if timeframe == "Daily" else (60,):
        found |= chart.lagged_cross(side, length, price)
    return found | chart.rsi_cross(side, price)


def rsi_signal(chart: Chart, timeframe: str, side: int) -> np.ndarray:
    past, reached, _ = compare(side)
    price = chart.close
    fast, slow = TREND[(timeframe, side)]
    triggered = rsi_triggers(chart, timeframe, side, price)
    for length in (14, 20, 60, 200) if timeframe == "Daily" else (16, 20, 60):
        triggered |= chart.crossed_at_last_candle(side, length)
    sma_rsi = chart.sma_rsi()
    trend = reached(chart.sma(fast, price), chart.sma(slow, price)) & reached(price, chart.sma(200, price))
    strength = sma_rsi >= bt.LONG_SMA if side == 1 else sma_rsi <= bt.SHORT_SMA
    return chart.volume_up() & past(price, chart.open) & trend & strength & triggered


def ha_setup(chart: Chart, side: int) -> np.ndarray:
    """The last HA candle was against the trade and had not broken the level before it."""
    level = chart.ha_high if side == 1 else chart.ha_low
    _, _, before = compare(side)
    against = chart.ago(chart.ha_close) < chart.ago(chart.ha_open) if side == 1 else chart.ago(chart.ha_close) > chart.ago(chart.ha_open)
    return before(chart.ago(chart.ha_close), chart.ago(level, 2)) & against


def ha_signal(chart: Chart, side: int) -> np.ndarray:
    past, _, _ = compare(side)
    ha_close = (chart.open + chart.high + chart.low + chart.close) / 4
    level = chart.ago(chart.ha_high if side == 1 else chart.ha_low)
    sma_rsi = chart.sma_rsi()
    strength = sma_rsi < 50 if side == 1 else sma_rsi > 50
    return chart.volume_up() & past(chart.close, chart.open) & past(ha_close, level) & ha_setup(chart, side) & strength


def opened_past_trigger(chart: Chart, timeframe: str, system: str, side: int, day_open: np.ndarray) -> np.ndarray:
    past, _, _ = compare(side)
    if system == "RSI":
        return rsi_triggers(chart, timeframe, side, day_open)
    return past(day_open, chart.ago(chart.ha_high if side == 1 else chart.ha_low)) & ha_setup(chart, side)


def load(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = bt.clean(bt.read_prices(path))
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    daily = bt.clean(bt.read_prices(bt.DATA / "1d" / path.name))
    daily.index = daily.index.normalize()
    daily = daily[~daily.index.duplicated(keep="last")]
    hourly = hourly[hourly.index.normalize().isin(daily.index)]
    return fill_open_volume(hourly, daily), daily


def build_charts(hourly: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Chart]:
    charts = {}
    for name, minutes in (("2H", 120), ("4H", 240)):
        bars, where = chartink_candles(hourly, minutes)
        charts[name] = Chart(bars, where, hourly)
    days = hourly.index.normalize().to_numpy()
    charts["Daily"] = Chart(daily, np.searchsorted(daily.index.to_numpy(), days), hourly)
    weekly = bt.weekly_bars(daily)
    charts["Weekly"] = Chart(weekly, np.searchsorted(weekly.index.to_numpy(), days), hourly)
    return charts


class Replay:
    """The bars of one stock, and futures_simulator.replay on them."""

    def __init__(self, hourly: pd.DataFrame):
        self.o, self.h, self.l, self.c = (hourly[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
        self.times = hourly.index
        self.seconds = hourly.index.to_numpy().astype("datetime64[s]").astype(np.int64)
        self.days = hourly.index.normalize()

    def cutoff(self, i: int, held: bool) -> int:
        day = self.days[i].date()
        last = expiry_after(day) if held else day
        return int(pd.Timestamp(last).timestamp()) + 15 * 3600 + 15 * 60

    def run(self, i: int, side: int, style: str, rsi_out: np.ndarray, level: np.ndarray | None) -> tuple[int, float, str]:
        long = side == 1
        entry = tick(self.c[i])
        stop = tick(entry * (1 - SL_PCT / 100 * side))
        target = tick(entry * (1 + TP_PCT / 100 * side)) if style in ("eod", "trail") else None
        run_trail, run_step = (tick(entry * TRAIL_PCT / 100), tick(entry * STEP_PCT / 100)) if style == "trail" else (0.0, 0.0)
        trail, step, trailing, passed, trailed, anchor = 0.0, 0.0, False, False, False, entry
        cutoff = self.cutoff(i, style != "eod")
        j = i + 1
        last = len(self.c)
        while j < last and self.seconds[j] < cutoff:
            open_, high, low, close = self.o[j], self.h[j], self.l[j], self.c[j]
            gap = open_ <= stop if long else open_ >= stop
            if gap or (low <= stop if long else high >= stop):
                return j, open_ if gap else stop, "TSL after target" if passed else "Trailing SL" if trailed else "Stop loss"
            if target is not None:
                gap = open_ >= target if long else open_ <= target
                if gap or (high >= target if long else low <= target):
                    hit = open_ if gap else target
                    if run_trail <= 0 or run_step <= 0:
                        return j, hit, "Target"
                    anchor = max(anchor, hit) if long else min(anchor, hit)
                    trail, step, trailing, target, passed = run_trail, run_step, True, None, True
                    new = round(anchor - trail if long else anchor + trail, 2)
                    if new > stop if long else new < stop:
                        stop, trailed = new, True
            if trailing:
                move = high - anchor if long else anchor - low
                if move >= step:
                    anchor += math.floor(move / step) * step * side
                    new = round(anchor - trail if long else anchor + trail, 2)
                    if new > stop if long else new < stop:
                        stop, trailed = new, True
            if style == "rsi" and rsi_out[j]:
                return j, close, "RSI exit"
            if level is not None and not math.isnan(level[j]) and (close < level[j] if long else close > level[j]):
                return j, close, "HA exit"
            j += 1
        if j >= last:
            return last - 1, self.c[last - 1], "Open at end"
        if self.days[j] == self.days[j - 1] or self.times[j].date() == pd.Timestamp(cutoff, unit="s").date():
            return j, self.o[j], "EOD 3:15 PM" if style == "eod" else "Expiry 3:15 PM"
        return j - 1, self.c[j - 1], "EOD 3:15 PM" if style == "eod" else "Expiry 3:15 PM"


def net_rupees(entry: float, price: float, side: int) -> tuple[float, float, float]:
    """Gross, charges and slippage on INVESTMENT rupees of position value (auto_trader.charges)."""
    qty = INVESTMENT / entry
    buy, sell = (entry, price) if side == 1 else (price, entry)
    buy_value, sell_value = buy * qty, sell * qty
    rates = bt.RATES
    brokerage = sum(min(rates["order"], value * rates["brokerage"]) for value in (buy_value, sell_value))
    exchange = (buy_value + sell_value) * rates["exchange"]
    sebi = (buy_value + sell_value) * rates["sebi"]
    charges = brokerage + exchange + sebi + sell_value * rates["stt_sell"] + buy_value * rates["stamp_buy"] \
        + (brokerage + exchange + sebi) * rates["gst"]
    return (sell - buy) * qty, charges, (buy_value + sell_value) * bt.SLIPPAGE_PCT / 100


def rsi_exits(charts: dict[str, Chart], hourly: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """(timeframe, side) -> at which hourly bars the RSI exit is due for a position of it."""
    price = hourly["Close"].to_numpy()
    live = {name: chart.rsi(price) for name, chart in charts.items()}
    widget = {}
    for side, level in ((1, bt.LONG_EXIT), (-1, bt.SHORT_EXIT)):
        found = np.zeros(len(price), bool)
        for name, chart in charts.items():
            before = chart.ago(chart.finished_rsi)
            found |= (live[name] < level) & (before >= level) if side == 1 else (live[name] > level) & (before <= level)
        widget[side] = found
    return {(name, side): widget[side] & ((live[name] < level) if side == 1 else (live[name] > level))
            for name in TIMEFRAMES for side, level in ((1, bt.LONG_EXIT), (-1, bt.SHORT_EXIT))}


def stock_trades(path, gap_rule: str) -> list[dict]:
    symbol = path.stem
    hourly, daily = load(path)
    if len(hourly) < 100:
        return []
    charts = build_charts(hourly, daily)
    replay = Replay(hourly)
    slots = eod.minutes(hourly.index)
    days = hourly.index.normalize()
    first_bar = np.r_[True, days[1:] != days[:-1]]
    day_open = hourly["Open"].groupby(days).transform("first").to_numpy()
    last_close = daily["Close"].shift(1).reindex(days).to_numpy()
    day_gap = (day_open / last_close - 1) * 100
    exits = rsi_exits(charts, hourly)
    allowed = (slots <= LAST_SLOT) & (hourly.index >= START)
    rows = []
    for timeframe in TIMEFRAMES:
        chart = charts[timeframe]
        for system, styles in STYLES.items():
            signals, blocked = {}, {}
            for side in (1, -1):
                signals[side] = rsi_signal(chart, timeframe, side) if system == "RSI" else ha_signal(chart, side)
                past = pd.Series(opened_past_trigger(chart, timeframe, system, side, day_open) & first_bar, index=hourly.index)
                skip = past.groupby(days).transform("any").to_numpy()
                if gap_rule == "strict":
                    skip |= day_gap * side >= POINT_FIVE_GAP
                blocked[side] = skip
            longs = signals[1] & ~blocked[1] & allowed
            shorts = signals[-1] & ~blocked[-1] & allowed
            for style in styles:
                free_from, entered = 0, set()
                level = None
                for i in np.flatnonzero(signals[1] & allowed | signals[-1] & allowed):
                    wanted = (1 if longs[i] else 0) + (-1 if shorts[i] else 0)
                    if signals[1][i] and signals[-1][i]:
                        continue  # mixed: bullish and bearish
                    if not wanted or i < free_from or (days[i], wanted) in entered:
                        continue
                    if style in ("ha", "ha_green"):
                        source = {"ha": (chart.ha_low, chart.ha_high), "ha_green": (chart.last_green_low, chart.last_red_high)}[style]
                        level = chart.ago(source[0 if wanted == 1 else 1])
                    end, price, reason = replay.run(i, wanted, style, exits[(timeframe, wanted)], level)
                    free_from, entry = end, tick(replay.c[i])
                    entered.add((days[i], wanted))
                    gross, charges, slippage = net_rupees(entry, float(price), wanted)
                    rows.append({"system": system, "style": style, "timeframe": timeframe, "symbol": symbol, "side": wanted,
                                 "entry_time": hourly.index[i], "exit_time": hourly.index[end], "entry": entry, "exit": round(float(price), 2),
                                 "reason": reason, "day_gap": round(day_gap[i], 2), "gross": gross, "charges": charges,
                                 "slippage": slippage, "net": gross - charges - slippage})
    return rows


def run(gap_rule: str) -> pd.DataFrame:
    rows = []
    for path in sorted((bt.DATA / "60m").glob("*.csv")):
        if (bt.DATA / "1d" / path.name).exists():
            rows += stock_trades(path, gap_rule)
    trades = pd.DataFrame(rows)
    return bt.drop_breaks(trades)


def by_stock(trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["system", "style", "timeframe", "symbol"]
    trades = trades.assign(first=trades["net"].where(trades["entry_time"] < SPLIT, 0.0), second=trades["net"].where(trades["entry_time"] >= SPLIT, 0.0),
                           win=trades["net"] > 0, long=trades["net"].where(trades["side"] == 1, 0.0), short=trades["net"].where(trades["side"] == -1, 0.0))
    table = trades.groupby(keys).agg(trades=("net", "size"), win_pct=("win", "mean"), net=("net", "sum"), first_half=("first", "sum"),
                                     second_half=("second", "sum"), long_net=("long", "sum"), short_net=("short", "sum"),
                                     charges=("charges", "sum")).reset_index()
    table["win_pct"] = table["win_pct"] * 100
    table["profit_pct"] = table["net"] / INVESTMENT * 100
    table["both_halves"] = (table["first_half"] > 0) & (table["second_half"] > 0)
    return table.sort_values(["system", "style", "timeframe", "profit_pct"], ascending=[True, True, True, False])


def summary(trades: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    keys = ["system", "style", "timeframe"]
    result = trades.groupby(keys).agg(trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100), avg_net_rs=("net", "mean"), total_net_rs=("net", "sum"))
    group = stocks.groupby(keys)
    result["stocks"] = group.size()
    result["in_profit"] = group["net"].apply(lambda n: (n > 0).sum())
    result["above_10pct"] = group["profit_pct"].apply(lambda p: (p > TOP_PCT).sum())
    result["below_minus_10pct"] = group["profit_pct"].apply(lambda p: (p < -TOP_PCT).sum())
    result["above_10pct_both_halves"] = stocks[stocks["both_halves"] & (stocks["profit_pct"] > TOP_PCT)].groupby(keys).size()
    return result.fillna(0)


def common(stocks: pd.DataFrame) -> pd.DataFrame:
    """Stocks above TOP_PCT on all three timeframes, for one strategy and exit."""
    rows = []
    for (system, style), group in stocks.groupby(["system", "style"]):
        wide = group[group["profit_pct"] > TOP_PCT].pivot(index="symbol", columns="timeframe", values="profit_pct").reindex(columns=TIMEFRAMES)
        halves = group.pivot(index="symbol", columns="timeframe", values="both_halves").reindex(columns=TIMEFRAMES)
        counts = group.pivot(index="symbol", columns="timeframe", values="trades").reindex(columns=TIMEFRAMES)
        for symbol, row in wide.dropna().iterrows():
            rows.append({"system": system, "exit": EXIT_NAMES[style], "symbol": symbol, **{f"{tf}_pct": round(row[tf], 1) for tf in TIMEFRAMES},
                         **{f"{tf}_trades": int(counts.loc[symbol, tf]) for tf in TIMEFRAMES}, "total_pct": round(row.sum(), 1),
                         "both_halves_all": bool(halves.loc[symbol].eq(True).all())})
    columns = ["system", "exit", "symbol", *[f"{tf}_pct" for tf in TIMEFRAMES], *[f"{tf}_trades" for tf in TIMEFRAMES], "total_pct", "both_halves_all"]
    return pd.DataFrame(rows, columns=columns).sort_values(["system", "exit", "total_pct"], ascending=[True, True, False])


def main() -> None:
    trades = run("strict")
    trades.to_csv(bt.RESULTS / "auto_sim_trades.csv.gz", index=False)
    stocks = by_stock(trades)
    stocks.round(2).to_csv(bt.RESULTS / "auto_sim_stocks.csv", index=False)
    stocks[stocks["profit_pct"] > TOP_PCT].round(2).to_csv(bt.RESULTS / "auto_sim_top.csv", index=False)
    result = summary(trades, stocks)
    result.round(2).to_csv(bt.RESULTS / "auto_sim_summary.csv")
    shared = common(stocks)
    shared.to_csv(bt.RESULTS / "auto_sim_common.csv", index=False)

    loose = run("trigger")
    loose_shared = common(by_stock(loose))
    loose_shared.to_csv(bt.RESULTS / "auto_sim_common_trigger_gap_only.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print("Trades from", trades["entry_time"].min(), "to", trades["exit_time"].max(), "| stocks", trades["symbol"].nunique())
    print(result.round(1).to_string())
    print("\nExit reasons")
    print(trades.groupby(["system", "style", "reason"]).size().to_string())
    print("\nCommon to 2H, 4H and Daily")
    print(shared.to_string(index=False))
    print("\nCommon with the opened-past-trigger gap rule only")
    print(loose_shared.to_string(index=False))


if __name__ == "__main__":
    main()
