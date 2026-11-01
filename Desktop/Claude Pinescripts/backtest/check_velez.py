"""A few Velez trades printed beside the candles they came from, to check the rules by hand."""
import pandas as pd

import backtest as bt

trades = pd.read_csv(bt.RESULTS / "velez_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
daily = bt.clean(bt.read_prices(bt.DATA / "1d" / "RELIANCE.csv"))
daily.index = daily.index.normalize()
fast, slow = daily["Close"].rolling(20).mean(), daily["Close"].rolling(200).mean()
sample = trades[(trades["symbol"] == "RELIANCE") & (trades["timeframe"] == "Daily") & (trades["side"] == 1)]
for setup in ("Pullback", "Elephant", "180", "Tail"):
    trade = sample[sample["setup"] == setup].iloc[-1]
    place = daily.index.get_loc(trade["entry_time"])
    print(f"\n{setup}: entry {trade['entry']:.2f} on {trade['entry_time'].date()}, {trade['reason']} {trade['exit']:.2f} on {trade['exit_time'].date()}")
    window = daily.iloc[place - 3:place + 1][["Open", "High", "Low", "Close"]].round(2)
    window["sma20"], window["sma200"] = fast.reindex(window.index).round(1), slow.reindex(window.index).round(1)
    print(window.to_string())
    print("exit bar low/high", round(daily["Low"][trade["exit_time"]], 2), round(daily["High"][trade["exit_time"]], 2))
