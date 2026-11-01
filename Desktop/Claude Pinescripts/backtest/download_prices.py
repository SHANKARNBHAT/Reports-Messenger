"""
Download prices for the futures stocks from Yahoo's chart address into data/, one CSV per
interval and stock. Certificates are checked against the Windows store (yfinance fails them
on this PC). Prices are adjusted for splits, not dividends, as TradingView shows by default.
"""
import concurrent.futures
import csv
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

FUTURES_CSV = Path(r"C:\Users\shank\Report_Messenger_claude\futures stocks.csv")
DATA = Path(__file__).with_name("data")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}.NS?{}"
DAILY_START = 1420070400  # 1 Jan 2015
WORKERS = 5
RETRIES = 4
CONTEXT = ssl.create_default_context()


def symbols() -> list[str]:
    """The stocks of the list, without its index and commodity rows."""
    with FUTURES_CSV.open(encoding="utf-8-sig", newline="") as file:
        names = [row["Symbol"].strip() for row in csv.DictReader(file)]
    return [n for n in names if n and not set(n) & set("^=.") and "NIFTY" not in n]


def fetch(symbol: str, query: dict) -> pd.DataFrame | None:
    url = CHART_URL.format(urllib.parse.quote(symbol, safe=""), urllib.parse.urlencode(query))
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, context=CONTEXT, timeout=40) as response:
                result = json.load(response)["chart"]["result"][0]
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    else:
        return None
    if not result.get("timestamp"):
        return None
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({"Open": quote["open"], "High": quote["high"], "Low": quote["low"],
                          "Close": quote["close"], "Volume": quote["volume"]},
                         index=pd.to_datetime(result["timestamp"], unit="s", utc=True))
    frame.index.name = "Datetime"
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def merged(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """The new bars over the saved ones, so history stays once it falls out of Yahoo's window."""
    if not path.exists():
        return frame
    saved = pd.read_csv(path, index_col=0)
    saved.index = pd.to_datetime(saved.index, utc=True)
    saved.index.name = frame.index.name
    both = pd.concat([saved, frame])
    return both[~both.index.duplicated(keep="last")].sort_index()


def save(symbol: str) -> tuple[str, int, int]:
    counts = []
    for folder, query in (("60m", {"interval": "60m", "range": "730d"}),
                          ("1d", {"interval": "1d", "period1": DAILY_START, "period2": int(time.time())})):
        frame = fetch(symbol, query)
        if frame is None or frame.empty:
            counts.append(0)
            continue
        path = DATA / folder / f"{symbol}.csv"
        frame = merged(frame, path)
        frame.to_csv(path)
        counts.append(len(frame))
    return symbol, *counts


def main() -> None:
    (DATA / "60m").mkdir(parents=True, exist_ok=True)
    (DATA / "1d").mkdir(parents=True, exist_ok=True)
    names = symbols()
    print(len(names), "stocks", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for done, result in enumerate(pool.map(save, names), 1):
            results.append(result)
            if done % 25 == 0:
                print(f"{done}/{len(names)}", flush=True)
    print("no 60m data:", [r[0] for r in results if not r[1]])
    print("no daily data:", [r[0] for r in results if not r[2]])
    print("median bars, 60m:", sorted(r[1] for r in results)[len(results) // 2],
          "daily:", sorted(r[2] for r in results)[len(results) // 2])


if __name__ == "__main__":
    main()
