"""
Market regime filters on the two best setups.
  Pine strategy longs (pine_filters.py trades): take a long only when the market is in an uptrend.
  Auto trader shorts (level_sides.py logic): take a short only in a downtrend, or only when volatility is high.

The "market" is an equal-weight index of every stock in data/1d (Nifty and India VIX can't be
downloaded from here). Volatility = 20-day annualised std of that index's daily returns, a stand-in
for India VIX; "high" = above its own 70th percentile of the past year. Every regime uses the
PREVIOUS day's close, so there is no look-ahead.

Trades are filtered at entry. One position per stock means a skipped trade could have freed the stock
for a later one; that small effect is ignored here.

    python regime_filters.py pine        (uses results/pine_parts)
    python regime_filters.py auto K N    (re-runs level_sides, chunk K of N, into results/regime_auto_parts)
    python regime_filters.py auto        (joins the chunks into results/regime_auto_trades.csv.gz)
    python regime_filters.py summary
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import backtest as bt

AUTO_TRADES = bt.RESULTS / "regime_auto_trades.csv.gz"


def market() -> pd.DataFrame:
    closes = {}
    for path in sorted((bt.DATA / "1d").glob("*.csv")):
        c = bt.read_prices(path)["Close"]
        c.index = pd.to_datetime(c.index, utc=True).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
        closes[path.stem] = c[~c.index.duplicated(keep="last")]
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index >= "2020-01-01"]
    rets = px.pct_change(fill_method=None).clip(-0.25, 0.25)
    idx = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    sma50 = idx.rolling(50).mean()
    vol = rets.mean(axis=1).rolling(20).std() * math.sqrt(252) * 100
    breadth = (px > px.rolling(50).mean()).sum(axis=1) / px.notna().sum(axis=1) * 100
    m = pd.DataFrame({"index": idx, "sma50": sma50, "vol": vol, "breadth": breadth})
    m["up"] = m["index"] > m["sma50"]
    m["rising"] = m["sma50"] > m["sma50"].shift(10)
    m["high_vol"] = m["vol"] > m["vol"].rolling(252, min_periods=120).quantile(0.70)
    m["broad_up"] = m["breadth"] > 50
    m = m.shift(1)  # known before the day starts
    for col in ("up", "rising", "high_vol", "broad_up"):
        m[col] = m[col].astype(object).fillna(False).astype(bool)
    return m


def regimes(entry_times: pd.Series, m: pd.DataFrame) -> pd.DataFrame:
    days = pd.to_datetime(entry_times).dt.tz_localize(None).dt.normalize() if getattr(entry_times.dt, "tz", None) else pd.to_datetime(entry_times).dt.normalize()
    r = m.reindex(m.index.union(days.unique())).ffill().reindex(days).reset_index(drop=True)
    for col in ("up", "rising", "high_vol", "broad_up"):
        r[col] = r[col].astype(object).fillna(False).astype(bool)
    return r


def stats(net: pd.Series, first: pd.Series) -> dict:
    if len(net) < 2:
        return {"trades": len(net)}
    return {"trades": len(net), "win_pct": (net > 0).mean() * 100, "avg": net.mean(), "total": net.sum(),
            "first_half_avg": net[first].mean(), "second_half_avg": net[~first].mean(),
            "profit_factor": net[net > 0].sum() / -net[net <= 0].sum(), "t_stat": net.mean() / net.std() * math.sqrt(len(net))}


LONG_RULES = {
    "no filter": lambda r: np.ones(len(r), bool),
    "index > SMA50": lambda r: r["up"].to_numpy(bool),
    "index > SMA50 & SMA50 rising": lambda r: (r["up"] & r["rising"]).to_numpy(bool),
    "breadth > 50%": lambda r: r["broad_up"].to_numpy(bool),
    "uptrend & rising & not high vol": lambda r: (r["up"] & r["rising"] & ~r["high_vol"]).to_numpy(bool),
    "not high vol": lambda r: (~r["high_vol"]).to_numpy(bool),
    "(skipped) index < SMA50": lambda r: (~r["up"]).to_numpy(bool),
    "(skipped) high vol": lambda r: r["high_vol"].to_numpy(bool),
}
SHORT_RULES = {
    "no filter": lambda r: np.ones(len(r), bool),
    "index < SMA50": lambda r: (~r["up"]).to_numpy(bool),
    "index < SMA50 & SMA50 falling": lambda r: (~r["up"] & ~r["rising"]).to_numpy(bool),
    "breadth < 50%": lambda r: (~r["broad_up"]).to_numpy(bool),
    "high vol (VIX stand-in)": lambda r: r["high_vol"].to_numpy(bool),
    "not high vol": lambda r: (~r["high_vol"]).to_numpy(bool),
}


def pine(m: pd.DataFrame) -> pd.DataFrame:
    t = bt.drop_breaks(pd.concat([pd.read_pickle(p) for p in sorted((bt.RESULTS / "pine_parts").glob("part_*.pkl"))], ignore_index=True))
    start = t.loc[t["timeframe"] == "2H", "entry_time"].min()
    t = t[(t["entry_time"] >= start) & (t["sides"] == "long") & (t["filter"].isin(["none", "pdh"]))].reset_index(drop=True)
    split = start + (t["exit_time"].max() - start) / 2
    r = regimes(t["entry_time"], m)
    rows = []
    for keys, g in t.groupby(["timeframe", "mode", "filter"]):
        rg = r.loc[g.index]
        for name, rule in LONG_RULES.items():
            keep = rule(rg)
            s = g[keep]
            rows.append({"timeframe": keys[0], "mode": keys[1], "filter": keys[2], "regime": name,
                         **stats(s["net"], s["entry_time"] < split)})
    return pd.DataFrame(rows)


def auto_run(k=None, n=None):
    import level_filters as lf
    import level_sides as ls
    import auto_trader_backtest as sim
    keep = ("base", "no_volume", "no_volume+sma50_60", "no_volume+pdh")
    original = ls.entries
    ls.entries = lambda *a: {k: v for k, v in original(*a).items() if k in keep}
    paths = [p for p in sorted((bt.DATA / "60m").glob("*.csv")) if (bt.DATA / "1d" / p.name).exists()]
    parts = bt.RESULTS / "regime_auto_parts"
    parts.mkdir(exist_ok=True)
    if k is not None:
        with Pool() as pool:
            rows = [x for part in pool.map(ls.stock_trades, paths[k::n]) for x in part]
        pd.DataFrame(rows).to_pickle(parts / f"part_{k}.pkl")
        return print("chunk", k, "of", n, len(rows), "trades")
    trades = bt.drop_breaks(pd.concat([pd.read_pickle(p) for p in sorted(parts.glob("part_*.pkl"))], ignore_index=True))
    trades.to_csv(AUTO_TRADES, index=False)
    print(len(trades), "trades saved")


def auto(m: pd.DataFrame) -> pd.DataFrame:
    import auto_trader_backtest as sim
    t = pd.read_csv(AUTO_TRADES, parse_dates=["entry_time", "exit_time"])
    r = regimes(t["entry_time"], m)
    rows = []
    for (side, tf, style, var), g in t.groupby(["side", "timeframe", "style", "variant"]):
        rg = r.loc[g.index]
        rules = SHORT_RULES if side == "short" else LONG_RULES
        for name, rule in rules.items():
            s = g[rule(rg)]
            st = stats(s["net"], s["entry_time"] < sim.SPLIT)
            if "total" in st:
                st["total"] /= 1e5
            rows.append({"side": side, "timeframe": tf, "style": style, "variant": var, "regime": name, **st})
    return pd.DataFrame(rows)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if what == "auto":
        return auto_run(*(int(x) for x in sys.argv[2:4])) if len(sys.argv) == 4 else auto_run()
    m = market()
    pd.set_option("display.width", 250, "display.max_rows", 400)
    if what in ("pine", "summary"):
        p = pine(m)
        p.round(3).to_csv(bt.RESULTS / "regime_pine_summary.csv", index=False)
        print(p.round(2).to_string(index=False))
    if what in ("summary",) and AUTO_TRADES.exists():
        a = auto(m)
        a.round(3).to_csv(bt.RESULTS / "regime_auto_summary.csv", index=False)
        print(a.round(2).to_string(index=False))
    if what == "market":
        print(m.dropna().resample("QE").agg({"up": "mean", "rising": "mean", "high_vol": "mean", "vol": "mean"}).round(2).tail(14))


if __name__ == "__main__":
    main()
