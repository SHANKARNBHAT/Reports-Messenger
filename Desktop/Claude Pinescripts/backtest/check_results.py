"""Checks on results/all_trades.csv.gz: outliers, a trade worked through, years, and market drift."""
import numpy as np
import pandas as pd

import backtest as bt

MAIN = ("Pure RSI + ATR", "RSI+HA + ATR (default)")
trades = pd.read_csv(bt.RESULTS / "all_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

print("== Largest moves per trade (bad data would show here)")
main = trades[trades["variant"] == MAIN[0]]
for timeframe, group in main.groupby("timeframe"):
    print(timeframe, "worst", group["gross"].round(1).nsmallest(3).tolist(), "best", group["gross"].round(1).nlargest(3).tolist())
print(main.nlargest(4, "gross")[["symbol", "timeframe", "side", "entry_time", "exit_time", "entry", "exit", "gross"]].round(2).to_string())

print("\n== Bar-to-bar moves over 35% in the hourly data (split errors would show here)")
jumps = []
for path in sorted((bt.DATA / "60m").glob("*.csv")):
    close = bt.read_prices(path)["Close"]
    moves = close.pct_change().abs()
    if (moves > 0.35).any():
        jumps.append((path.stem, round(float(moves.max()) * 100, 1), str(moves.idxmax())))
print(jumps)

print("\n== RELIANCE Daily, last 4 Pure RSI trades, worked through")
daily = bt.clean(bt.read_prices(bt.DATA / "1d" / "RELIANCE.csv"))
daily.index = daily.index.normalize()
rsi = bt.rsi(daily["Close"])
sma = rsi.rolling(bt.RSI_SMA_LENGTH).mean()
atr = bt.atr(daily).shift(1)
sample = trades[(trades["symbol"] == "RELIANCE") & (trades["timeframe"] == "Daily") & (trades["variant"] == MAIN[0])].tail(4)
for trade in sample.itertuples():
    day = trade.entry_time.normalize()
    before = rsi.shift(1)[day]
    stop = trade.entry - trade.side * atr[day] * bt.ATR_MULTIPLIER
    end = trade.exit_time.normalize()
    print(f"{day.date()} side {trade.side:+d} close {daily['Close'][day]:.2f} entry {trade.entry:.2f} "
          f"RSI {before:.1f}->{rsi[day]:.1f} SMA {sma[day]:.1f} stop {stop:.2f} | "
          f"{end.date()} {trade.reason} exit {trade.exit:.2f} RSI {rsi.shift(1)[end]:.1f}->{rsi[end]:.1f} low {daily['Low'][end]:.2f} high {daily['High'][end]:.2f}")

print("\n== By year (exit year): trades, average net %")
years = trades[trades["variant"].isin(MAIN) & trades["timeframe"].isin(("Daily", "Weekly"))]
table = years.assign(year=years["exit_time"].dt.year).groupby(["timeframe", "variant", "year"])["net"].agg(["count", "mean"])
print(table["mean"].unstack("year").round(2).to_string())
intraday = trades[trades["variant"].isin(MAIN)]
table = intraday.assign(year=intraday["exit_time"].dt.year).query("year >= 2024").groupby(["timeframe", "variant", "year"])["net"].mean()
print(table.unstack("year").round(2).to_string())

print("\n== Against simply holding: stock's own average drift over the same bars, per trade")
drifts = {}
for path in sorted((bt.DATA / "1d").glob("*.csv")):
    frame = bt.clean(bt.read_prices(path))
    drifts[path.stem] = float(np.log(frame["Close"]).diff().mean()) * 100  # % a day over its history
trades["days"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400 * 252 / 365
trades["drift"] = trades["symbol"].map(drifts) * trades["days"]
trades["edge"] = (trades["gross"] - trades["drift"] * trades["side"])
start = trades.loc[trades["timeframe"] == "2H", "entry_time"].min()
for label, group in (("full history", trades), ("same period", trades[trades["entry_time"] >= start])):
    frame = group[group["variant"].isin(MAIN)]
    table = frame.groupby(["timeframe", "variant"]).apply(lambda g: pd.Series({
        "net": g["net"].mean(), "drift_part": (g["drift"] * g["side"]).mean(), "edge_before_costs": g["edge"].mean(),
        "cost": (g["gross"] - g["net"]).mean()}), include_groups=False)
    print(label)
    print(table.round(3).to_string())
