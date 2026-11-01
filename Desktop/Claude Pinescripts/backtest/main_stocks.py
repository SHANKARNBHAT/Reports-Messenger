"""Which stocks did well on the two main strategies: first two hours, no gaps, EOD exit."""
import pandas as pd

import backtest as bt

MIN_TRADES = 10
SPLIT = pd.Timestamp("2025-05-12")
TIMEFRAMES = ["2H", "4H", "Daily", "Weekly"]
trades = pd.read_csv(bt.RESULTS / "first_two_hours_trades.csv.gz", parse_dates=["entry_time"])
trades["half"] = (trades["entry_time"] >= SPLIT).map({False: "first", True: "second"})
keys = ["strategy", "side", "symbol", "timeframe"]
table = trades.groupby(keys).agg(trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100),
                                 total=("net", "sum"), avg=("net", "mean")).reset_index()
halves = trades.pivot_table(index=keys, columns="half", values="net", aggfunc="sum").reset_index()
table = table.merge(halves, on=keys)
table = table[table["trades"] >= MIN_TRADES]
table["both_halves"] = (table["first"] > 0) & (table["second"] > 0)
table.round(2).sort_values("total", ascending=False).to_csv(bt.RESULTS / "main_stocks.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
profit = table[table["total"] > 0]
print("Combinations", len(table), "| profitable", len(profit), "| profitable in both halves", int(table["both_halves"].sum()),
      "| stocks with one in profit", profit["symbol"].nunique(), "| stocks with both halves", table[table["both_halves"]]["symbol"].nunique())
print(table.groupby(["strategy", "side"]).apply(lambda g: pd.Series({"tested": len(g), "profitable": (g["total"] > 0).sum(), "both_halves": g["both_halves"].sum()}), include_groups=False).to_string())

for (strategy, side), group in table.groupby(["strategy", "side"]):
    wide = group.pivot(index="symbol", columns="timeframe", values="total").reindex(columns=TIMEFRAMES)
    wide["positive"] = (wide[TIMEFRAMES] > 0).sum(axis=1)
    wide["total"] = wide[TIMEFRAMES].sum(axis=1)
    held = group[group["both_halves"]].groupby("symbol").size().rename("both_halves")
    wide = wide.join(held).fillna({"both_halves": 0})
    print(f"\n== {strategy}, {side}: top 12 by timeframes positive, then total")
    print(wide.sort_values(["positive", "both_halves", "total"], ascending=False).head(12).round(1).to_string())

print("\n== Best single combinations, profitable in both halves, 20+ trades")
best = table[table["both_halves"] & (table["trades"] >= 20)].sort_values("total", ascending=False)
print(best.head(20).round(2).to_string(index=False))
