"""
Do the simulator's entry conditions pick the stocks Chartink lists? The widgets of both dashboards
now, against auto_trader_backtest's signals on the last session's close in data/.
"""
import json
import re

import numpy as np
import requests
from bs4 import BeautifulSoup

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import auto_trader_backtest as sim
import backtest as bt

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def widget_lists(dashboard: int, system: str) -> dict[tuple[str, int], set[str]]:
    found = {}
    with requests.Session() as session:
        session.headers.update(HEADERS)
        url = f"https://chartink.com/dashboard/{dashboard}"
        page = BeautifulSoup(session.get(url, timeout=30).text, "html.parser")
        token = page.find("meta", attrs={"name": "csrf-token"})["content"]
        for widget in json.loads(page.find("dashboard")[":widgets"]):
            name = widget.get("name") or ""
            match = re.search(r"\b(2H|4H|DAILY)\b", name, re.I)
            if not match or name.lower().startswith("futures"):
                continue
            timeframe = {"2H": "2H", "4H": "4H", "DAILY": "Daily"}[match.group(1).upper()]
            side = -1 if re.search("bear", name, re.I) else 1
            response = session.post("https://chartink.com/widget/process", timeout=30,
                                    data={"query": widget["query"], "use_live": "1", "limit": 500, "size": 1, "widget_id": widget["id"]},
                                    headers={"x-csrf-token": token, "X-Requested-With": "XMLHttpRequest", "Referer": url})
            found[(timeframe, side)] = {group["name"] for group in response.json().get("groupData") or []}
    return found


def main() -> None:
    mine = {}
    for path in sorted((bt.DATA / "60m").glob("*.csv")):
        if not (bt.DATA / "1d" / path.name).exists():
            continue
        hourly, daily = sim.load(path)
        if len(hourly) < 100:
            continue
        charts = sim.build_charts(hourly, daily)
        for timeframe in sim.TIMEFRAMES:
            for side in (1, -1):
                if sim.rsi_signal(charts[timeframe], timeframe, side)[-1]:
                    mine.setdefault(("RSI", timeframe, side), set()).add(path.stem)
                if sim.ha_signal(charts[timeframe], side)[-1]:
                    mine.setdefault(("HA", timeframe, side), set()).add(path.stem)
    for dashboard, system in ((501779, "RSI"), (502621, "HA")):
        for (timeframe, side), theirs in sorted(widget_lists(dashboard, system).items()):
            ours = mine.get((system, timeframe, side), set())
            print(f"{system} {timeframe} {'Long' if side == 1 else 'Short'}: Chartink {len(theirs)}, simulator {len(ours)}, both {len(theirs & ours)}"
                  f" | only Chartink {sorted(theirs - ours)[:12]} | only simulator {sorted(ours - theirs)[:12]}")


if __name__ == "__main__":
    main()
