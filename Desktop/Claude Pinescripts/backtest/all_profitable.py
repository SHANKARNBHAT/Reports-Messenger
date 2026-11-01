"""
Every stock, strategy and timeframe of the tests, over the same 3 years, and which made a profit.
Then a fair check: combinations profitable in the first half, judged on the second half.
"""
import pandas as pd

import backtest as bt
import test_long_only as lo

START, SPLIT = lo.RECENT, pd.Timestamp("2025-05-12")
MIN_TRADES = 5
SIDES = {1: "Long", -1: "Short"}
COLUMNS = ["symbol", "timeframe", "family", "strategy", "entry_time", "net"]


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(bt.RESULTS / name, parse_dates=["entry_time"])
    return frame[frame["entry_time"] >= START].copy()


def collect() -> pd.DataFrame:
    frames = []
    swing = load("all_trades.csv.gz")
    swing = swing[swing["variant"].isin(("Pure RSI + ATR", "RSI+HA + ATR (default)"))]
    frames.append(swing.assign(family="RSI swing, both sides", strategy=swing["variant"]))
    long_only = load("long_only_trades.csv.gz")
    long_only = long_only[long_only["variant"].str.startswith("Long")]
    frames.append(long_only.assign(family="RSI swing, long only", strategy=long_only["variant"]))
    velez = load("velez_trades.csv.gz")
    frames.append(velez.assign(family="Velez", strategy="Velez " + velez["setup"] + " " + velez["side"].map(SIDES)))
    eod = load("eod_trades.csv.gz")
    frames.append(eod.assign(family="RSI EOD", strategy="RSI EOD " + eod["mode"] + ", " + eod["stop"] + ", " + eod["side"].map(SIDES)))
    sma = load("sma_rsi_eod_trades.csv.gz")
    frames.append(sma.assign(family="SMA RSI EOD", strategy="SMA RSI EOD " + sma["setup"]))
    return pd.concat([frame[COLUMNS] for frame in frames], ignore_index=True)


def totals(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.groupby(["symbol", "timeframe", "family", "strategy"]).agg(
        trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100), total_net=("net", "sum"), avg_net=("net", "mean"))
    return table[table["trades"] >= MIN_TRADES]


def main() -> None:
    trades = collect()
    table = totals(trades)
    table.round(2).reset_index().to_csv(bt.RESULTS / "all_combinations.csv", index=False)
    profit = table[table["total_net"] > 0].sort_values("total_net", ascending=False).reset_index()
    profit.round(2).to_csv(bt.RESULTS / "all_profitable.csv", index=False)

    tested = table.groupby("symbol").size()
    won = profit.groupby("symbol").size()
    best = profit.groupby("symbol").head(1).set_index("symbol")
    per_stock = pd.DataFrame({"profitable": won, "tested": tested}).fillna(0).astype(int)
    per_stock["share_pct"] = per_stock["profitable"] / per_stock["tested"] * 100
    per_stock["best_timeframe"], per_stock["best_strategy"] = best["timeframe"], best["strategy"]
    per_stock["best_total"], per_stock["best_trades"] = best["total_net"], best["trades"]
    per_stock = per_stock.sort_values("share_pct", ascending=False)
    per_stock.round(1).to_csv(bt.RESULTS / "profit_by_stock.csv")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60, "display.max_colwidth", 45)
    print("Strategies", table.index.get_level_values("strategy").nunique(), "| combinations", len(table), "| profitable", len(profit),
          f"({len(profit) / len(table) * 100:.1f}%)", "| stocks", len(tested), "| stocks with at least one", len(won))
    share = table.assign(win=table["total_net"] > 0).groupby(["family", "timeframe"])["win"].agg(["size", "sum", "mean"])
    print((share.assign(mean=share["mean"] * 100)).round(1).to_string())
    print("\nTop 25 stocks by share of combinations profitable")
    print(per_stock.head(25).round(1).to_string())
    print("\nBottom 10")
    print(per_stock.tail(10).round(1).to_string())
    print("\nStocks by timeframe with most profitable combos")
    print(profit.groupby("timeframe").size().to_string())

    first, second = totals(trades[trades["entry_time"] < SPLIT]), totals(trades[trades["entry_time"] >= SPLIT])
    both = first.join(second, lsuffix="_first", rsuffix="_second", how="inner")
    winners = both[both["total_net_first"] > 0]
    losers = both[both["total_net_first"] <= 0]
    print("\nFair check, combinations with 5+ trades in both halves:", len(both))
    print("Profitable in the first half:", len(winners), "| of those, profitable in the second:", int((winners["total_net_second"] > 0).sum()),
          f"({(winners['total_net_second'] > 0).mean() * 100:.1f}%)", "| avg second-half avg_net", round(float(winners["avg_net_second"].mean()), 3))
    print("Losing in the first half:", len(losers), "| of those, profitable in the second:", int((losers["total_net_second"] > 0).sum()),
          f"({(losers['total_net_second'] > 0).mean() * 100:.1f}%)", "| avg second-half avg_net", round(float(losers["avg_net_second"].mean()), 3))
    top = both.sort_values("avg_net_first", ascending=False).groupby("symbol").head(1)
    print("Each stock's best combo from the first half:", len(top), "| first-half avg", round(float(top["avg_net_first"].mean()), 2),
          "| second-half avg", round(float(top["avg_net_second"].mean()), 2), f"| still profitable {(top['total_net_second'] > 0).mean() * 100:.1f}%")
    keep = both[(both["total_net_first"] > 0) & (both["total_net_second"] > 0)].copy()
    keep["total"] = keep["total_net_first"] + keep["total_net_second"]
    keep.round(2).reset_index().sort_values("total", ascending=False).to_csv(bt.RESULTS / "profitable_both_halves.csv", index=False)
    print("Profitable in both halves:", len(keep), "| stocks", keep.index.get_level_values("symbol").nunique())
    print(keep.groupby(level="timeframe").size().to_string())
    print(keep.groupby(level="family").size().to_string())


if __name__ == "__main__":
    main()
