import streamlit as st
import os

st.set_page_config(page_title="Market Intelligence Hub", layout="wide")
st.title("📊 NSE Futures & Strategy Market Intelligence Hub")

# Create tabs for each report
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Consolidated Report", 
    "🔍 RVOL Active Signals", 
    "📈 Bullish Report", 
    "📉 Bearish Report", 
    "⚡ Active Stocks"
])

def load_report_text(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return "Report not generated yet. Run the GitHub pipeline to populate."

with tab1:
    st.subheader("Consolidated Strategy & Activity Report")
    st.code(load_report_text("consolidated_strategy_report.txt"), language="text")

with tab2:
    st.subheader("Weekly Signals Matched with Active Stocks (RVOL > 1)")
    st.code(load_report_text("rvol_active_signals_report.txt"), language="text")

with tab3:
    st.subheader("Bullish Strategy Report (Last 7 Days)")
    st.code(load_report_text("bullish_report.txt"), language="text")

with tab4:
    st.subheader("Bearish Strategy Report (Last 7 Days)")
    st.code(load_report_text("bearish_report.txt"), language="text")

with tab5:
    st.subheader("Top Active Stocks Report")
    st.code(load_report_text("active_stocks_report.txt"), language="text")
