"""EOD results in short, and a few trades printed beside their hourly candles to check by hand."""
import pandas as pd

import backtest as bt

trades = pd.read_csv(bt.RESULTS / "eod_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
summary = pd.read_csv(bt.RESULTS / "eod_summary.csv").set_index(["timeframe", "mode", "stop", "side"])
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 80)
print(sorted(set(trades["entry_time"].dt.strftime("%H:%M"))))
print(trades.groupby("timeframe")["entry_time"].apply(lambda t: sorted(set(t.dt.strftime("%H:%M")))).to_dict())
columns = ["trades", "win_pct", "avg_gross", "avg_net", "median_net", "edge", "edge_first_half", "edge_second_half",
           "stopped_pct", "target_pct", "stocks_positive_pct", "t_stat"]
print(summary[columns].round(2).to_string())

hourly = bt.clean(bt.read_prices(bt.DATA / "60m" / "RELIANCE.csv"))
sample = trades[(trades["symbol"] == "RELIANCE") & (trades["mode"] == "Pure RSI") & (trades["side"] == 1)]
for timeframe, stop in (("2H", "EOD only"), ("Daily", "EOD + 1.5% SL, 3% target")):
    trade = sample[(sample["timeframe"] == timeframe) & (sample["stop"] == stop)].iloc[-1]
    print(f"\n{timeframe} {stop}: entry {trade['entry']:.2f} at close of bar {trade['entry_time']}, "
          f"{trade['reason']} {trade['exit']:.2f} at bar {trade['exit_time']}, net {trade['net']:.2f}")
    day = hourly[hourly.index.normalize() == trade["entry_time"].normalize()]
    print(day[["Open", "High", "Low", "Close"]].round(2).to_string())
