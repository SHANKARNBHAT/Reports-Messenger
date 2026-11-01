"""DIXON against TradingView's report: buy and hold, this test's 2H trades, and where it ranks."""
import pandas as pd

import backtest as bt

daily = bt.clean(bt.read_prices(bt.DATA / "1d" / "DIXON.csv"))
window = daily[daily.index >= "2021-01-01"]["Close"]
print("Buy and hold from", window.index[0].date(), round(window.iloc[0], 1), "to", window.index[-1].date(), round(window.iloc[-1], 1),
      "=", round((window.iloc[-1] / window.iloc[0] - 1) * 100, 1), "%")
print("From Jan 2021, deepest fall from a high:", round(float((window / window.cummax() - 1).min() * 100), 1), "%")

trades = pd.read_csv(bt.RESULTS / "all_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
pure = trades[(trades["variant"] == "Pure RSI, no stop") & (trades["timeframe"] == "2H")]
dixon = pure[pure["symbol"] == "DIXON"]
sides = dixon.groupby("side")["net"].agg(["count", "mean", "sum"]).round(2)
print("\nDIXON 2H, Pure RSI, no stop, both sides, this test (from", dixon["entry_time"].min().date(), ")")
print(sides.to_string())
wins, losses = dixon.loc[dixon["net"] > 0, "gross"].sum(), -dixon.loc[dixon["gross"] <= 0, "gross"].sum()
print("trades", len(dixon), "win %", round((dixon["net"] > 0).mean() * 100, 1), "| avg net", round(dixon["net"].mean(), 2),
      "| compounded gross %", round(((1 + dixon["gross"] / 100).prod() - 1) * 100, 1), "| compounded net %", round(((1 + dixon["net"] / 100).prod() - 1) * 100, 1))

totals = pure.groupby("symbol")["net"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
place = totals.index.get_loc("DIXON") + 1
print("\nDIXON rank by total net, of", len(totals), ":", place)
print("Top 8:\n", totals.head(8).round(2).to_string())
print("\nBottom 5:\n", totals.tail(5).round(2).to_string())
print("\nStocks with positive total net:", int((totals["sum"] > 0).sum()), "of", len(totals), "| median total net", round(float(totals["sum"].median()), 1), "%")
