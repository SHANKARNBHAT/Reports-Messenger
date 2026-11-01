"""
Same tests as regime_filters.py, with the real Nifty 50 and India VIX (Yahoo, fetched 1 Nov 2026).
data/index/nifty_vix_regime.txt holds only the days the regime changed, as YYMMDD + flags + VIX band:
  flags = up (Nifty > its 50-day SMA) + 2*rising (SMA50 above its value 10 days earlier)
          + 4*high VIX (VIX above its own 70th percentile of the past 252 days)
  band  = VIX <12, 12-14, 14-16, 16-18, 18-22, 22+  ->  0..5
Values are at that day's close and apply from the NEXT trading day (no look-ahead).

    python regime_real.py
"""
import contextlib
import glob
import io
import math

import numpy as np
import pandas as pd

import auto_trader_backtest as sim
import backtest as bt
import regime_filters as rf

BANDS = ["<12", "12-14", "14-16", "16-18", "18-22", "22+"]


def load():
    codes = open(bt.DATA / "index" / "nifty_vix_regime.txt").read().split()
    when = pd.to_datetime(["20" + c[:6] for c in codes])
    flags = np.array([int(c[6]) for c in codes])
    band = np.array([int(c[7]) for c in codes])
    return when, flags, band


def tag(entry_times, when, flags, band):
    days = pd.to_datetime(entry_times).dt.tz_localize(None).dt.normalize().to_numpy() if getattr(entry_times.dt, "tz", None) else pd.to_datetime(entry_times).dt.normalize().to_numpy()
    k = np.searchsorted(when.to_numpy(), days, side="left") - 1  # last change strictly before the trade's day
    f, b = flags[k], band[k]
    return pd.DataFrame({"up": f & 1 > 0, "rising": f & 2 > 0, "vix_high": f & 4 > 0, "band": b})


def row(net, first, **keys):
    if len(net) < 2:
        return {**keys, "trades": len(net)}
    return {**keys, "trades": len(net), "win_pct": (net > 0).mean() * 100, "avg": net.mean(), "total": net.sum(),
            "first_half_avg": net[first].mean(), "second_half_avg": net[~first].mean(),
            "profit_factor": net[net > 0].sum() / -net[net <= 0].sum(), "t_stat": net.mean() / net.std() * math.sqrt(len(net))}


LONG = {"no filter": lambda r: np.ones(len(r), bool), "Nifty > SMA50 & rising": lambda r: r.up & r.rising,
        "VIX high": lambda r: r.vix_high, "VIX not high": lambda r: ~r.vix_high, "VIX >= 14": lambda r: r.band >= 2,
        "VIX < 14": lambda r: r.band < 2, "VIX >= 14 & Nifty > SMA50": lambda r: (r.band >= 2) & r.up}
SHORT = {"no filter": lambda r: np.ones(len(r), bool), "Nifty < SMA50": lambda r: ~r.up,
         "Nifty < SMA50 & falling": lambda r: ~r.up & ~r.rising, "VIX high": lambda r: r.vix_high,
         "VIX not high": lambda r: ~r.vix_high, "VIX >= 16": lambda r: r.band >= 3}


def main():
    when, flags, band = load()
    with contextlib.redirect_stdout(io.StringIO()):
        t = bt.drop_breaks(pd.concat([pd.read_pickle(p) for p in sorted(glob.glob(str(bt.RESULTS / "pine_parts" / "part_*.pkl")))], ignore_index=True))
    start = t.loc[t["timeframe"] == "2H", "entry_time"].min()
    t = t[(t.entry_time >= start) & (t.sides == "long") & (t["filter"].isin(["none", "pdh"]))].reset_index(drop=True)
    split = start + (t.exit_time.max() - start) / 2
    r = tag(t.entry_time, when, flags, band)
    rows, bands = [], []
    for keys, g in t.groupby(["timeframe", "mode", "filter"]):
        rg = r.loc[g.index].reset_index(drop=True)
        g = g.reset_index(drop=True)
        for name, rule in LONG.items():
            s = g[np.asarray(rule(rg), bool)]
            rows.append(row(s.net, (s.entry_time < split).to_numpy(), timeframe=keys[0], mode=keys[1], filter=keys[2], regime=name))
        for b in range(6):
            s = g[(rg.band == b).to_numpy()]
            bands.append({"timeframe": keys[0], "mode": keys[1], "filter": keys[2], "vix": BANDS[b], "trades": len(s), "avg": s.net.mean()})
    pine = pd.DataFrame(rows)
    pine.round(3).to_csv(bt.RESULTS / "regime_real_pine.csv", index=False)
    pd.DataFrame(bands).round(3).to_csv(bt.RESULTS / "regime_real_pine_vix_bands.csv", index=False)

    a = pd.read_csv(rf.AUTO_TRADES, parse_dates=["entry_time", "exit_time"])
    r = tag(a.entry_time, when, flags, band)
    rows = []
    for keys, g in a.groupby(["side", "timeframe", "style", "variant"]):
        rg = r.loc[g.index].reset_index(drop=True)
        g = g.reset_index(drop=True)
        for name, rule in (SHORT if keys[0] == "short" else LONG).items():
            s = g[np.asarray(rule(rg), bool)]
            out = row(s.net, (s.entry_time < sim.SPLIT).to_numpy(), side=keys[0], timeframe=keys[1], style=keys[2], variant=keys[3], regime=name)
            if "total" in out:
                out["total"] /= 1e5
            rows.append(out)
    pd.DataFrame(rows).round(3).to_csv(bt.RESULTS / "regime_real_auto.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
