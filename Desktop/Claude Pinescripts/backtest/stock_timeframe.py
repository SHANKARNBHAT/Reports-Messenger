"""
Which stock works best on which timeframe, long only, and do those picks last?

Picks are made on the first half of the 3 years and judged on the second half, which they
never saw. `edge` is a trade's net % less the stock's own average rise over the same days in
that half, so a stock that simply went up does not count as the strategy working.
"""
import numpy as np
import pandas as pd

import backtest as bt
import test_long_only as lo

START, SPLIT, END = pd.Timestamp("2023-11-22"), pd.Timestamp("2025-05-12"), pd.Timestamp("2026-10-31")
VARIANTS = {"Long, exit 60": "Pure RSI", "Long, RSI+HA": "RSI + HA"}
MIN_TRADES = {"2H": 12, "4H": 8, "Daily": 6, "Weekly": 3}
TOP = 25
RNG = np.random.default_rng(7)


def with_edge(trades: pd.DataFrame, returns: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)].copy()
    drift = {symbol: series[(series.index >= start) & (series.index < end)].mean() for symbol, series in returns.items()}
    frame["edge"] = frame["net"] - frame["symbol"].map(drift) * frame["days"]
    return frame


def combos(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["symbol", "timeframe", "mode"], sort=False)
    table = grouped.agg(trades=("net", "size"), win_pct=("net", lambda n: (n > 0).mean() * 100), avg_net=("net", "mean"),
                        avg_edge=("edge", "mean"), total_edge=("edge", "sum"))
    return table[table["trades"] >= table.index.get_level_values("timeframe").map(MIN_TRADES)]


def main() -> None:
    trades = pd.read_csv(bt.RESULTS / "long_only_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
    trades = trades[trades["variant"].isin(VARIANTS)].assign(mode=lambda t: t["variant"].map(VARIANTS))
    returns = lo.daily_returns()
    first, second = with_edge(trades, returns, START, SPLIT), with_edge(trades, returns, SPLIT, END)
    train, test = combos(first), combos(second)
    both = train.join(test, lsuffix="_first", rsuffix="_second", how="inner")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    print("stock/timeframe/mode combos with enough trades in both halves:", len(both))

    print("\n== Rank match, first half to second, by timeframe and mode (0 = no link, 1 = same order)")
    match = both.groupby(["timeframe", "mode"]).apply(lambda g: g["avg_edge_first"].corr(g["avg_edge_second"], method="spearman"), include_groups=False)
    print(match.round(3).to_string())

    print(f"\n== Top {TOP} of each timeframe picked on the first half: their second half, against all the others")
    rows = []
    for (timeframe, mode), group in both.groupby(["timeframe", "mode"]):
        ranked = group.sort_values("avg_edge_first", ascending=False)
        picks, rest = ranked.head(TOP), ranked.iloc[TOP:]
        rows.append({"timeframe": timeframe, "mode": mode, "first_half_picks": picks["avg_edge_first"].mean(),
                     "second_half_picks": picks["avg_edge_second"].mean(), "second_half_rest": rest["avg_edge_second"].mean(),
                     "picks_still_positive_pct": (picks["avg_edge_second"] > 0).mean() * 100})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n== Each stock's single best timeframe from the first half, in the second half")
    best = both.sort_values("avg_edge_first", ascending=False).groupby("symbol").head(1)
    print("stocks", len(best), "| first half avg edge", round(best["avg_edge_first"].mean(), 2), "| second half", round(best["avg_edge_second"].mean(), 2),
          "| still positive", round((best["avg_edge_second"] > 0).mean() * 100, 1), "%")
    print("all combos, second half avg edge", round(both["avg_edge_second"].mean(), 2),
          "| share positive", round((both["avg_edge_second"] > 0).mean() * 100, 1), "%")

    positive = both[(both["avg_edge_first"] > 0) & (both["avg_edge_second"] > 0)]
    share_first, share_second = (both["avg_edge_first"] > 0).mean(), (both["avg_edge_second"] > 0).mean()
    print("\n== Positive in both halves:", len(positive), "of", len(both), "| expected from chance alone:", round(share_first * share_second * len(both)))
    shuffled = []
    for _ in range(2000):
        order = both.groupby(["timeframe", "mode"])["avg_edge_second"].transform(lambda s: RNG.permutation(s.to_numpy()))
        shuffled.append(((both["avg_edge_first"] > 0) & (order > 0)).sum())
    print("shuffled second halves: mean", round(float(np.mean(shuffled)), 1), "95th percentile", np.percentile(shuffled, 95))

    positive = positive.assign(worst_half=positive[["avg_edge_first", "avg_edge_second"]].min(axis=1))
    columns = ["trades_first", "trades_second", "win_pct_first", "win_pct_second", "avg_edge_first", "avg_edge_second", "avg_net_first", "avg_net_second"]
    listing = positive.sort_values("worst_half", ascending=False)
    listing[columns].round(2).to_csv(bt.RESULTS / "stock_timeframe_both_halves.csv")
    both.round(3).to_csv(bt.RESULTS / "stock_timeframe_all.csv")
    print("\n== Positive in both halves, strongest worst half first, per timeframe")
    for timeframe in ("2H", "4H", "Daily", "Weekly"):
        part = listing[listing.index.get_level_values("timeframe") == timeframe]
        print(timeframe, len(part))
        print(part[columns].head(10).round(2).to_string())
    print("\nBy timeframe share in both-halves list:", listing.groupby(level=["timeframe", "mode"]).size().to_dict())


if __name__ == "__main__":
    main()
