"""Stress test of the first-two-hours results: same-day clustering, months, fills, and big days."""
import math

import pandas as pd

import backtest as bt

trades = pd.read_csv(bt.RESULTS / "first_two_hours_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
trades["day"] = trades["entry_time"].dt.normalize()
trades["month"] = trades["entry_time"].dt.to_period("M")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
rows = []
for keys, group in trades.groupby(["strategy", "timeframe", "side"]):
    daily = group.groupby("day")["net"].mean()
    summed = group.groupby("day")["net"].sum().sort_values(ascending=False)
    monthly = group.groupby("month")["net"].mean()
    best_days = group[~group["day"].isin(summed.index[:10])]["net"].mean()
    row = dict(zip(("strategy", "timeframe", "side"), keys))
    row.update({"trades": len(group), "days": len(daily), "avg_net": group["net"].mean(),
                "day_avg": daily.mean(), "day_t": daily.mean() / (daily.std() / math.sqrt(len(daily))),
                "months_up_pct": (monthly > 0).mean() * 100, "months": len(monthly), "without_best_10_days": best_days})
    for extra in (0.03, 0.05, 0.10):
        row[f"extra_{extra:.2f}"] = group["net"].mean() - 2 * extra
    rows.append(row)
table = pd.DataFrame(rows).set_index(["strategy", "timeframe", "side"])
table.to_csv(bt.RESULTS / "first_two_hours_stress.csv")
print(table.round(2).to_string())
short = trades[(trades["strategy"] == "SMA RSI + HA") & (trades["side"] == "Short")]
print("\nSMA RSI + HA shorts by entry hour, average net %")
print(short.groupby(["timeframe", short["entry_time"].dt.strftime("%H:%M")])["net"].agg(["count", "mean"]).round(2).to_string())
