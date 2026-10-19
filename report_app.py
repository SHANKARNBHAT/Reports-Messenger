import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sqlite3

# --- Page Configuration ---
st.set_page_config(
    page_title="NSE Futures Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NSE Stock Futures - Buildup & Analysis Dashboard")
st.markdown("Automated market insights powered by your GitHub pipeline (`stock_market_data.db` & `futures stocks.csv`).")

# --- Data Loading Function ---
@st.cache_data(ttl=300)
def load_data():
    # 1. Try loading from your SQLite database: stock_market_data.db
    if os.path.exists("stock_market_data.db"):
        try:
            conn = sqlite3.connect("stock_market_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if tables:
                table_name = tables[0][0]  # Take the first table found
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                conn.close()
                if not df.empty:
                    return df
            conn.close()
        except Exception as e:
            st.warning(f"Could not read from stock_market_data.db: {e}")

    # 2. Fallback to your CSV file: futures stocks.csv
    if os.path.exists("futures stocks.csv"):
        try:
            df = pd.read_csv("futures stocks.csv")
            if not df.empty:
                return df
        except Exception as e:
            st.warning(f"Could not read from futures stocks.csv: {e}")

    # 3. Fallback Mock Data (if files are missing or empty)
    data = {
        'Symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'TATAMOTORS', 'SBIN', 'LT'],
        'Price_Chg_%': [1.2, -0.8, -1.5, 0.9, -0.3, 2.4, -1.1, 0.4],
        'OI_Chg_%': [4.5, 3.2, -2.1, -1.8, 5.1, 6.2, -3.4, -0.5],
        'Volume': [150000, 98000, 110000, 210000, 180000, 320000, 140000, 85000]
    }
    return pd.DataFrame(data)

df = load_data()

# --- Deduplicate Columns to Prevent PyArrow Errors ---
df = df.loc[:, ~df.columns.duplicated()]

# --- Normalize Column Names (Safeguard) ---
column_mapping = {}
for col in df.columns:
    col_lower = str(col).lower()
    if 'symbol' in col_lower or 'stock' in col_lower:
        column_mapping[col] = 'Symbol'
    elif 'price' in col_lower and ('chg' in col_lower or 'change' in col_lower):
        column_mapping[col] = 'Price_Chg_%'
    elif 'oi' in col_lower and ('chg' in col_lower or 'change' in col_lower):
        column_mapping[col] = 'OI_Chg_%'
    elif 'vol' in col_lower:
        column_mapping[col] = 'Volume'

df = df.rename(columns=column_mapping)

# Re-check duplicates after renaming
df = df.loc[:, ~df.columns.duplicated()]

# --- Validation and Rendering ---
required_cols = {'Symbol', 'Price_Chg_%', 'OI_Chg_%', 'Volume'}
if not required_cols.issubset(df.columns):
    st.warning("⚠️ Column names in your database/CSV don't match the standard F&O template. Here is your raw data preview:")
    st.dataframe(df, use_container_width=True)
else:
    # Classify Buildup Regimes
    def categorize(row):
        p, oi = row['Price_Chg_%'], row['OI_Chg_%']
        if p > 0 and oi > 0: 
            return 'Long Buildup'
        elif p < 0 and oi > 0: 
            return 'Short Buildup'
        elif p < 0 and oi < 0: 
            return 'Long Unwinding'
        else: 
            return 'Short Covering'

    df['Buildup'] = df.apply(categorize, axis=1)

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Options")
    selected_buildup = st.sidebar.multiselect(
        "Select Buildup Type:",
        options=df['Buildup'].unique(),
        default=df['Buildup'].unique()
    )

    filtered_df = df[df['Buildup'].isin(selected_buildup)]

    # --- Key Metrics Overview ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Long Buildup", len(df[df['Buildup'] == 'Long Buildup']))
    col2.metric("Short Buildup", len(df[df['Buildup'] == 'Short Buildup']))
    col3.metric("Short Covering", len(df[df['Buildup'] == 'Short Covering']))
    col4.metric("Long Unwinding", len(df[df['Buildup'] == 'Long Unwinding']))

    st.divider()

    # --- Plotly Scatter Plot (Buildup Matrix) ---
    st.subheader("Futures Buildup Matrix (Price Chg % vs OI Chg %)")
    fig = px.scatter(
        filtered_df,
        x="Price_Chg_%",
        y="OI_Chg_%",
        color="Buildup",
        size="Volume",
        hover_name="Symbol",
        color_discrete_map={
            'Long Buildup': '#00CC96',
            'Short Buildup': '#EF553B',
            'Long Unwinding': '#FFA15A',
            'Short Covering': '#636EFA'
        },
        title="Price vs Open Interest Dynamics"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    # --- Raw Data Table ---
    st.subheader("Filtered Futures Data Table")
    st.dataframe(filtered_df.sort_values(by="OI_Chg_%", ascending=False), use_container_width=True)
