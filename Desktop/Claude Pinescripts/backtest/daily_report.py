"""
The daily run: new prices, every backtest again, then one HTML report with a section per strategy.

Writes reports/report_YYYY-MM-DD.html and reports/latest.html, keeps a snapshot in reports/history
to show what changed since the last run, and logs each step in logs/. Stops without a new report
if the prices did not update or a backtest failed, so an old report is never passed off as new.
"""
import datetime as dt
import html
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RESULTS, REPORTS, LOGS = HERE / "results", HERE / "reports", HERE / "logs"
HISTORY = REPORTS / "history"
DOWNLOAD = "download_prices.py"
BACKTESTS = ("backtest.py", "test_long_only.py", "velez_backtest.py", "eod_backtest.py", "sma_rsi_eod.py",
             "first_two_hours.py", "sma_rsi_filters.py", "top_stocks.py", "all_profitable.py")
MIN_UPDATED = 190  # stocks whose daily file must have been refreshed
TOP = 15
ORDER = ("2H", "4H", "Daily", "Weekly")
MAIN = {"SMA RSI + HA": ("Long, RSI+HA", "RSI+HA + ATR (default)", "RSI EOD RSI + HA, EOD only, Long", "RSI EOD RSI + HA, EOD only, Short"),
        "SMA RSI + Crossover / RSI cross 60": ("SMA RSI EOD Long", "SMA RSI EOD Short, RSI 60")}
STYLE = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--band:#f5f5f3;--up:#0a7d3e;--down:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#191a1c;--fg:#eee;--muted:#a5a5a5;--line:#3a3b3e;--band:#242528;--up:#5cd08c;--down:#f28b82}}
body{margin:0 auto;max-width:1050px;padding:20px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:16px 0 6px}
.muted{color:var(--muted)}.box{background:var(--band);border-radius:8px;padding:10px 14px;margin:12px 0}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.up{color:var(--up)}.down{color:var(--down)}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}summary{cursor:pointer;font-weight:600}
"""


def run(script: str, log) -> None:
    started = time.time()
    log.write(f"\n=== {script} {dt.datetime.now():%H:%M:%S}\n")
    log.flush()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    log.write(f"=== {script} exit {done.returncode} in {time.time() - started:.0f}s\n")
    log.flush()
    if done.returncode:
        raise RuntimeError(f"{script} failed, see the log")


def data_status(started: float) -> tuple[int, int, str]:
    files = list((HERE / "data" / "1d").glob("*.csv"))
    fresh = sum(path.stat().st_mtime >= started for path in files)
    last = pd.read_csv(HERE / "data" / "1d" / "RELIANCE.csv", index_col=0).index[-1]
    return fresh, len(files), str(pd.to_datetime(last, utc=True).tz_convert("Asia/Kolkata").date())


def signed(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f'<span class="{"up" if value > 0 else "down" if value < 0 else ""}">{value:+.{digits}f}%</span>'


def table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def summarise(combos: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy and timeframe, weighting each stock by its trades."""
    frame = combos.assign(wins=combos["trades"] * combos["win_pct"] / 100, profit=combos["total_net"] > 0)
    rows = frame.groupby(["family", "strategy", "timeframe"]).agg(
        trades=("trades", "sum"), wins=("wins", "sum"), total=("total_net", "sum"), stocks=("symbol", "size"), profitable=("profit", "sum")).reset_index()
    rows["avg_net"] = rows["total"] / rows["trades"]
    rows["win_pct"] = rows["wins"] / rows["trades"] * 100
    return rows


def previous_snapshot(today: str) -> tuple[pd.DataFrame | None, str | None]:
    earlier = sorted(path for path in HISTORY.glob("*.csv") if path.stem < today)
    return (pd.read_csv(earlier[-1]), earlier[-1].stem) if earlier else (None, None)


def strategy_section(name: str, rows: pd.DataFrame, combos: pd.DataFrame, old: pd.DataFrame | None, opened: bool = False) -> str:
    overall = rows["total"].sum() / rows["trades"].sum()
    lines = []
    for timeframe in ORDER:
        row = rows[rows["timeframe"] == timeframe]
        if row.empty:
            continue
        row = row.iloc[0]
        change = ""
        if old is not None:
            before = old[(old["strategy"] == name) & (old["timeframe"] == timeframe)]
            if not before.empty:
                change = signed(row["avg_net"] - before.iloc[0]["avg_net"], 3)
        lines.append([timeframe, f"{int(row['trades']):,}", f"{row['win_pct']:.1f}%", signed(row["avg_net"]), change,
                      f"{int(row['profitable'])} / {int(row['stocks'])}"])
    parts = [f'<details{" open" if opened else ""}><summary>{html.escape(name)} &nbsp;<span class="muted">{signed(overall)} per trade</span></summary>',
             table(["Timeframe", "Trades", "Win rate", "Avg per trade", "Change", "Stocks in profit"], lines)]
    mine = combos[combos["strategy"] == name]
    for timeframe in ORDER:
        best = mine[(mine["timeframe"] == timeframe) & (mine["total_net"] > 0)].nlargest(TOP, "total_net")
        if best.empty:
            continue
        parts.append(f"<h3>{timeframe}: top {len(best)} profitable stocks</h3>")
        parts.append(table(["Stock", "Trades", "Win rate", "Total", "Avg per trade"],
                           [[html.escape(r.symbol), int(r.trades), f"{r.win_pct:.1f}%", signed(r.total_net, 1), signed(r.avg_net)]
                            for r in best.itertuples()]))
    return "".join(parts) + "</details>"


def first_two_hours(today: str) -> str:
    """The main section: both strategies, EOD exit, signals of the first two hours only."""
    result = pd.read_csv(RESULTS / "first_two_hours.csv")
    stocks = pd.read_csv(RESULTS / "first_two_hours_stocks.csv")
    folder = HISTORY / "first_two_hours"
    folder.mkdir(parents=True, exist_ok=True)
    earlier = sorted(path for path in folder.glob("*.csv") if path.stem < today)
    old = pd.read_csv(earlier[-1]) if earlier else None
    result.to_csv(folder / f"{today}.csv", index=False)
    parts = ["<h2>Main: first two hours only (signals 9:15 to 11:15 AM, EOD exit)</h2>"
             "<p class='muted'>Entered at the trigger price without waiting for a candle to close, and squared off at 3:15 PM. A stock that opens past its trigger, "
             "up or down, is skipped on that side for the day. Trigger fills are estimated from hourly bars.</p>"]
    for strategy, group in result.groupby("strategy", sort=False):
        lines = []
        for r in group.sort_values(["timeframe", "side"]).itertuples():
            change = ""
            if old is not None:
                before = old[(old["strategy"] == strategy) & (old["timeframe"] == r.timeframe) & (old["side"] == r.side)]
                if not before.empty:
                    change = signed(r.avg_net - before.iloc[0]["avg_net"], 3)
            lines.append([f"{r.timeframe} {r.side}", f"{int(r.trades):,}", f"{r.win_pct:.1f}%", signed(r.avg_gross), signed(r.avg_net), change,
                          signed(r.avg_net_at_close), signed(r.edge_first_half), signed(r.edge_second_half), f"{r.stocks_positive_pct:.0f}%"])
        parts.append(f"<h3>{html.escape(strategy)}</h3>")
        parts.append(table(["Timeframe", "Trades", "Win rate", "Before costs", "After costs", "Change", "If waiting for close", "Edge, first half", "Edge, second half", "Stocks positive"], lines))
        for (timeframe, side), rows in stocks[(stocks["strategy"] == strategy) & (stocks["total_net"] > 0) & (stocks["trades"] >= 5)].groupby(["timeframe", "side"]):
            best = rows.nlargest(TOP, "total_net")
            body = table(["Stock", "Trades", "Win rate", "Total", "Avg per trade"],
                         [[html.escape(r.symbol), int(r.trades), f"{r.win_pct:.1f}%", signed(r.total_net, 1), signed(r.avg_net)] for r in best.itertuples()])
            parts.append(f"<details><summary>{timeframe} {side}: top {len(best)} of {len(rows)} profitable stocks</summary>{body}</details>")
    return "".join(parts)


def filters_section() -> str:
    path = RESULTS / "sma_rsi_filters.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    best = frame[frame["trades"] >= 300].nlargest(10, "avg_net")
    return ("<h2>SMA RSI EOD with RVOL and ATR filters</h2><p class='muted'>Best 10 settings with 300+ trades. RVOL skips Yahoo's unreliable first-hour volume.</p>"
            + table(["Timeframe", "Side", "RVOL", "ATR %", "Trades", "Win rate", "Avg per trade"],
                    [[r.timeframe, r.side, r.rvol, r.atr, int(r.trades), f"{r.win_pct:.1f}%", signed(r.avg_net)] for r in best.itertuples()]))


def build(today: str, fresh: int, count: int, last_day: str) -> Path:
    combos = pd.read_csv(RESULTS / "all_combinations.csv")
    rows = summarise(combos)
    old, old_day = previous_snapshot(today)
    HISTORY.mkdir(parents=True, exist_ok=True)
    rows.to_csv(HISTORY / f"{today}.csv", index=False)

    overall = (rows.groupby(["family", "strategy"]).agg(trades=("trades", "sum"), total=("total", "sum"), wins=("wins", "sum")).reset_index())
    overall["avg_net"] = overall["total"] / overall["trades"]
    overall["win_pct"] = overall["wins"] / overall["trades"] * 100
    overall = overall.sort_values("avg_net", ascending=False)
    pivot = rows.pivot(index="strategy", columns="timeframe", values="avg_net")
    league = [[html.escape(r.strategy), f"{int(r.trades):,}", f"{r.win_pct:.1f}%", signed(r.avg_net),
               *[signed(pivot.loc[r.strategy].get(t)) for t in ORDER]] for r in overall.itertuples()]
    profit = combos[combos["total_net"] > 0]
    share = len(profit) / len(combos) * 100
    main_names = [name for names in MAIN.values() for name in names]
    main_part = []
    for title, names in MAIN.items():
        main_part.append(f"<h2>{html.escape(title)}</h2>")
        main_part.append(table(["Version", "Trades", "Win rate", "Avg per trade", *ORDER],
                               [row for row in league if html.unescape(row[0]) in names]))
        for name in names:
            main_part.append(strategy_section(name, rows[rows["strategy"] == name], combos, old, opened=True))
    sections = []
    for family, group in overall[~overall["strategy"].isin(main_names)].groupby("family", sort=False):
        sections.append(f"<h3>{html.escape(family)}</h3>")
        for name in group["strategy"]:
            sections.append(strategy_section(name, rows[rows["strategy"] == name], combos, old))
    compared = f"Change compares with {old_day}." if old_day else "This is the first report, so there are no changes to show yet."
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy Report {today}</title><style>{STYLE}</style></head><body>
<h1>Strategy Report</h1><p class="muted">Run {today} &middot; prices through {last_day} &middot; {fresh} of {count} stocks refreshed</p>
<div class="box">{len(combos):,} stock, strategy and timeframe combinations tested, {len(profit):,} profitable ({share:.1f}%). {compared}</div>
<p class="muted">Backtests from Nov 2023, after estimated charges and slippage, on Yahoo cash-market prices. Totals add each trade's % return without compounding,
and profitable stocks are ranked after the fact, so a past winner is not a prediction.</p>
{first_two_hours(today)}
<h1 style="margin-top:36px">Good to have</h1>
{"".join(main_part)}{filters_section()}
<h2>Other strategies</h2><p class="muted">Everything tested, best first, including the two main ones.</p>
{table(["Strategy", "Trades", "Win rate", "Avg per trade", *ORDER], league)}{"".join(sections)}</body></html>"""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"report_{today}.html"
    path.write_text(page, encoding="utf-8")
    (REPORTS / "latest.html").write_text(page, encoding="utf-8")
    return path


def main() -> int:
    today = f"{dt.date.today():%Y-%m-%d}"
    LOGS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:  # rebuilds today's page from the latest results
        fresh, count, last_day = data_status(0)
        print(build(today, fresh, count, last_day))
        return 0
    with (LOGS / f"run_{today}.log").open("a", encoding="utf-8") as log:
        try:
            started = time.time()
            run(DOWNLOAD, log)
            fresh, count, last_day = data_status(started)
            if fresh < MIN_UPDATED:
                raise RuntimeError(f"only {fresh} of {count} stocks refreshed")
            for script in BACKTESTS:
                run(script, log)
            path = build(today, fresh, count, last_day)
            log.write(f"\nReport written: {path}\n")
            print(path)
            return 0
        except Exception as error:
            log.write(f"\nFAILED: {error}\n")
            print("FAILED:", error)
            return 1


if __name__ == "__main__":
    sys.exit(main())
