"""Top stocks on the long-only Pure RSI setup of the last 3 years, with other timeframes and holding beside them."""
import pandas as pd

import backtest as bt
import test_long_only as lo

TOP = 20
trades = pd.read_csv(bt.RESULTS / "long_only_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
trades = trades[(trades["variant"] == "Long, exit 60") & (trades["entry_time"] >= lo.RECENT)]

two_hour = trades[trades["timeframe"] == "2H"]
table = two_hour.groupby("symbol").agg(trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100),
                                       total_2h=("net", "sum"), avg_2h=("net", "mean"))
for timeframe, name in (("4H", "total_4h"), ("Daily", "total_daily"), ("Weekly", "total_weekly")):
    table[name] = trades[trades["timeframe"] == timeframe].groupby("symbol")["net"].sum()

holding, worst = {}, {}
for symbol in table.index:
    close = bt.clean(bt.read_prices(bt.DATA / "1d" / f"{symbol}.csv"))["Close"]
    close = close[close.index >= lo.RECENT]
    holding[symbol] = (close.iloc[-1] / close.iloc[0] - 1) * 100
    worst[symbol] = (close / close.cummax() - 1).min() * 100
table["hold_pct"], table["hold_worst_fall"] = pd.Series(holding), pd.Series(worst)
table["timeframes_positive"] = (table[["total_2h", "total_4h", "total_daily", "total_weekly"]] > 0).sum(axis=1)

ranked = table.sort_values("total_2h", ascending=False)
ranked.round(2).to_csv(bt.RESULTS / "top_stocks.csv")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
print(ranked.head(TOP).round(1).to_string())
top = ranked.head(TOP)
print("\nAll", len(ranked), "| positive on 2H", int((ranked["total_2h"] > 0).sum()), "| median 2H total", round(float(ranked["total_2h"].median()), 1))
print("Top 20 beating their hold:", int((top["total_2h"] > top["hold_pct"]).sum()), "| positive on all 4:", int((top["timeframes_positive"] == 4).sum()),
      "| average hold of top 20:", round(float(top["hold_pct"].mean()), 1), "| of all:", round(float(ranked["hold_pct"].mean()), 1))
