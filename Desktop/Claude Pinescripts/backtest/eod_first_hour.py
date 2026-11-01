"""EOD-only trades of signals in the first hour (9:15 to 10:15), against those later in the day."""
import math

import pandas as pd

import backtest as bt
import eod_backtest as eod

trades = pd.read_csv(bt.RESULTS / "eod_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
trades = trades[(trades["stop"] == "EOD only") & trades["timeframe"].isin(("Daily", "Weekly"))]
trades["slot"] = (trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute).map(lambda m: "First hour" if m == 555 else "Later")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 80)


def short(group: pd.DataFrame) -> pd.Series:
    row = eod.summary(group)
    return row[["trades", "win_pct", "avg_gross", "avg_net", "median_net", "edge", "edge_first_half", "edge_second_half",
                "stocks_positive_pct", "t_stat"]]


table = trades.groupby(["timeframe", "mode", "side", "slot"]).apply(short, include_groups=False)
table.to_csv(bt.RESULTS / "eod_first_hour.csv")
print(table.round(2).to_string())

first = trades[trades["slot"] == "First hour"]
years = first.assign(year=first["entry_time"].dt.year).groupby(["timeframe", "mode", "side", "year"])["net"].mean()
print("\nFirst hour, average net % a trade, by year")
print(years.unstack("year").round(2).to_string())
print("\nDistinct stock-days in the first hour:", first[["symbol", "entry_time"]].drop_duplicates().shape[0], "| cost a trade:", round((first["gross"] - first["net"]).mean(), 3))
