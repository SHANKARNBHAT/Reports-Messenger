"""
Long-only trades with wider RSI exits, against simply holding the same stocks.

Per stock, per year: `net` is the sum of its trades' net %, `hold` buying and holding it, and
`excess` what the trades made beyond `hold` for the time they were in the market. Averaged
over the stocks. Two periods apart: the settings must hold up in both, not fit one of them.
"""
import math

import numpy as np
import pandas as pd

import backtest as bt

RECENT = pd.Timestamp("2023-11-22")
BARS_A_YEAR = 252
bt.VARIANTS = (
    bt.Variant("Both sides, exit 60/40 (now)"),
    bt.Variant("Both sides, exit 50/50", long_exit=50.0, short_exit=50.0),
    bt.Variant("Long, exit 60", shorts=False),
    bt.Variant("Long, exit 50", shorts=False, long_exit=50.0),
    bt.Variant("Long, exit 45", shorts=False, long_exit=45.0),
    bt.Variant("Long, exit 40", shorts=False, long_exit=40.0),
    bt.Variant("Long, exit 50, no stop", shorts=False, long_exit=50.0, atr=False),
    bt.Variant("Long, RSI+HA", shorts=False, ha=True),
)
PERIODS = {"Earlier (2015 to Nov 2023)": (pd.Timestamp("2015-01-01"), RECENT),
           "Recent (Nov 2023 to Oct 2026)": (RECENT, pd.Timestamp("2027-01-01"))}


def daily_returns() -> dict[str, pd.Series]:
    """Each stock's daily % returns, without its data-break days."""
    returns = {}
    for path in sorted((bt.DATA / "1d").glob("*.csv")):
        close = bt.clean(bt.read_prices(path))["Close"]
        close.index = close.index.normalize()
        change = close[~close.index.duplicated(keep="last")].pct_change() * 100
        returns[path.stem] = change[change.abs() <= bt.BREAK_PCT].dropna()
    return returns


def period_table(trades: pd.DataFrame, returns: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    held = {symbol: series[(series.index >= start) & (series.index < end)] for symbol, series in returns.items()}
    held = {symbol: series for symbol, series in held.items() if len(series) >= BARS_A_YEAR / 2}
    hold = pd.Series({symbol: series.mean() * BARS_A_YEAR for symbol, series in held.items()})
    years = pd.Series({symbol: len(series) / BARS_A_YEAR for symbol, series in held.items()})
    rows = []
    inside = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)]
    for (timeframe, variant), group in inside.groupby(["timeframe", "variant"], sort=False):
        stocks = group["symbol"].unique() if timeframe in ("2H", "4H") else hold.index
        stocks = hold.index.intersection(stocks)
        by_stock = group.groupby("symbol")
        net = (by_stock["net"].sum().reindex(stocks).fillna(0) / years[stocks])
        days = (by_stock["days"].sum().reindex(stocks).fillna(0) / (years[stocks] * BARS_A_YEAR)).clip(upper=1)
        excess = net - hold[stocks] * days
        rows.append({"timeframe": timeframe, "variant": variant, "trades": len(group),
                     "win_pct": (group["net"] > 0).mean() * 100, "avg_net": group["net"].mean(),
                     "median_net": group["net"].median(), "stop_pct": (group["reason"] == "stop").mean() * 100,
                     "avg_days": group["days"].mean(), "in_market_pct": days.mean() * 100,
                     "net_a_year": net.mean(), "hold_a_year": hold[stocks].mean(), "excess_a_year": excess.mean(),
                     "stocks_beating": (excess > 0).mean() * 100,
                     "t_stat": excess.mean() / (excess.std() / math.sqrt(len(excess)))})
    return pd.DataFrame(rows).set_index(["timeframe", "variant"])


def main() -> None:
    trades = bt.drop_breaks(bt.run())
    trades["entry_time"], trades["exit_time"] = pd.to_datetime(trades["entry_time"]), pd.to_datetime(trades["exit_time"])
    trades["days"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400 * BARS_A_YEAR / 365
    trades.to_csv(bt.RESULTS / "long_only_trades.csv.gz", index=False)
    returns = daily_returns()
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    tables = []
    for label, (start, end) in PERIODS.items():
        table = period_table(trades, returns, start, end)
        tables.append(table.assign(period=label).set_index("period", append=True))
        print("\n==", label)
        print(table.round(2).to_string())
    pd.concat(tables).to_csv(bt.RESULTS / "long_only_summary.csv")

    frame = trades[trades["exit_time"] >= pd.Timestamp("2024-01-01")]
    years = frame.assign(year=frame["exit_time"].dt.year).groupby(["timeframe", "variant", "year"], sort=False)["net"].mean()
    print("\n== Average net % a trade, by exit year")
    print(years.unstack("year").round(2).to_string())


if __name__ == "__main__":
    main()
