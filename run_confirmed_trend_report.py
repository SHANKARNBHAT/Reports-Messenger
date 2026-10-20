import os
import sqlite3
import pandas as pd
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "stock_market_data.db")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('expand_frame_repr', False)

def generate_unified_confirmed_report():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stock_ohlc_indicators WHERE symbol != '^NSEI';")
    symbols = [row[0] for row in cursor.fetchall()]

    records = []

    print(f"Analyzing {len(symbols)} stocks for confirmed trends...")

    for symbol in symbols:
        query = """
            SELECT date, close, sma_20, sma_50, sma_200
            FROM stock_ohlc_indicators 
            WHERE symbol = ? 
            ORDER BY date ASC;
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if len(df) < 200:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest['close']
        sma20 = latest['sma_20']
        sma50 = latest['sma_50']
        sma200 = latest['sma_200']
        prev_close = prev['close']
        prev_sma20 = prev['sma_20']
        prev_sma50 = prev['sma_50']

        # 3-Month Return (~63 trading days)
        idx_3m = -min(len(df), 63)
        close_3m_ago = df.iloc[idx_3m]['close']
        return_3m = ((close - close_3m_ago) / close_3m_ago) * 100

        # Filter strictly for positive 3-month return stocks (> 0%)
        if return_3m <= 0:
            continue

        # SMA Trend Classification (Priority)
        sma_trend = "Undetermined"
        if pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200):
            if (sma20 > sma50) and (close > sma200):
                sma_trend = "Bullish"
            elif (sma20 < sma50) and (close < sma200):
                sma_trend = "Bearish"

        # Strategy Signal Bias Evaluation
        is_bullish_trend = (sma20 > sma50) and (close > sma200)
        is_bearish_trend = (sma20 < sma50) and (close < sma200)
        cross_above_20_50 = (prev_close <= prev_sma20 and close > sma20) or (prev_close <= prev_sma50 and close > sma50)
        cross_below_20_50 = (prev_close >= prev_sma20 and close < sma20) or (prev_close >= prev_sma50 and close < sma50)

        signal_status = "Neutral / Flat"
        if is_bullish_trend:
            if cross_above_20_50:
                signal_status = "Bullish Entry Trigger"
            else:
                signal_status = "Bullish Trend (In Trade / Holding)"
        elif is_bearish_trend:
            if cross_below_20_50:
                signal_status = "Bearish Entry Trigger"
            else:
                signal_status = "Bearish Trend (In Trade / Holding)"

        signal_bias = "Undetermined"
        if "Bullish" in signal_status:
            signal_bias = "Bullish"
        elif "Bearish" in signal_status:
            signal_bias = "Bearish"

        # Conflict resolution: SMA trend takes priority
        if signal_bias in ["Bullish", "Bearish"] and signal_bias == sma_trend:
            resolved_trend = f"Confirmed {sma_trend}"
        else:
            resolved_trend = sma_trend

        # Keep ONLY strictly Bullish or Bearish confirmed trends (Drop Undetermined)
        if sma_trend in ["Bullish", "Bearish"]:
            records.append({
                'symbol': symbol,
                'return_3m': return_3m,
                'confirmed_trend': resolved_trend
            })

    conn.close()

    if not records:
        print("No matching bullish/bearish stocks found.")
        return

    df_res = pd.DataFrame(records)
    # Sort from highest to lowest 3-month performance
    df_sorted = df_res.sort_values(by='return_3m', ascending=False).copy()

    # Final columns: Symbol and Confirmed Trend only
    df_final = df_sorted[['symbol', 'confirmed_trend']]

    # Save output file for GitHub actions
    report_file = os.path.join(SCRIPT_DIR, "confirmed_bullish_bearish_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(" CONFIRMED BULLISH & BEARISH STOCKS (3M+)\n")
        f.write(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=========================================\n\n")
        f.write(df_final.to_string(index=False))
        f.write("\n=========================================\n")

    print(f"\nReport successfully saved to: {report_file}\n")
    print(df_final.to_string(index=False))

if __name__ == "__main__":
    generate_unified_confirmed_report()
