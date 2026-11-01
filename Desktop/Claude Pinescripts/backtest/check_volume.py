"""Is Yahoo's hourly volume reliable? Hourly bars against the daily volume, and RVOL by hour and year."""
import numpy as np
import pandas as pd

import backtest as bt
import sma_rsi_filters as filters

rows = []
for path in sorted((bt.DATA / "60m").glob("*.csv")):
    hourly = bt.clean(bt.read_prices(path))
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    daily = bt.clean(bt.read_prices(bt.DATA / "1d" / path.name))
    daily.index = daily.index.normalize()
    days = hourly.index.normalize()
    total = hourly["Volume"].groupby(days).sum()
    ratio = total / daily["Volume"].reindex(total.index)
    frame = pd.DataFrame({"symbol": path.stem, "time": hourly.index, "volume": hourly["Volume"].to_numpy(),
                          "rvol": filters.relative_volume(hourly).to_numpy(), "ratio": ratio.reindex(days).to_numpy()})
    rows.append(frame)
data = pd.concat(rows)
data["year"], data["slot"] = data["time"].dt.year, data["time"].dt.strftime("%H:%M")
print("Zero-volume hourly bars, %:", round(float((data["volume"] == 0).mean() * 100), 2))
print("Hourly total / daily volume, median by year:")
print(data.groupby("year")["ratio"].median().round(2).to_string())
print("\nZero-volume bars %, by slot:")
print(data.groupby("slot")["volume"].apply(lambda v: round(float((v == 0).mean() * 100), 2)).to_string())
print("\nRVOL median by slot and year:")
print(data.pivot_table(index="slot", columns="year", values="rvol", aggfunc="median").round(2).to_string())
sample = data[(data["symbol"] == "RELIANCE") & (data["time"] >= "2026-10-26")][["time", "volume", "rvol", "ratio"]]
print(sample.round(2).to_string(index=False))
