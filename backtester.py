import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="Quant Backtester", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c1f2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3250;
    }
    h1 { color: #00d4aa; font-family: 'Courier New', monospace; }
    .stMetric label { color: #8888aa; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Algorithmic Trading Backtester")
st.markdown("*Moving Average Crossover Strategy — Built with Python & Streamlit*")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    ticker     = st.text_input("Stock Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date   = st.date_input("End Date",   datetime(2024, 1, 1))
    sma_short  = st.slider("Short SMA (days)", 5,  50,  20)
    sma_long   = st.slider("Long SMA (days)",  20, 200, 50)
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[['Close']].copy()
    df.columns = ['Close']
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found. Check the ticker symbol.")
    st.stop()

# ── Strategy Logic ────────────────────────────────────────────────────────────
df[f'SMA_{sma_short}'] = df['Close'].rolling(sma_short).mean()
df[f'SMA_{sma_long}']  = df['Close'].rolling(sma_long).mean()

df['Signal']   = 0
df.loc[df[f'SMA_{sma_short}'] > df[f'SMA_{sma_long}'], 'Signal'] = 1
df['Position'] = df['Signal'].diff()

# ── Returns ───────────────────────────────────────────────────────────────────
df['Market_Return']   = df['Close'].pct_change()
df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)

df['Cumulative_Market']   = (1 + df['Market_Return']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

df['Portfolio_Value'] = initial_capital * df['Cumulative_Strategy']

df.dropna(inplace=True)

# ── Risk Metrics ──────────────────────────────────────────────────────────────
sharpe = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * np.sqrt(252)

rolling_max  = df['Cumulative_Strategy'].cummax()
drawdown     = (df['Cumulative_Strategy'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

total_return_strategy = df['Cumulative_Strategy'].iloc[-1] - 1
total_return_market   = df['Cumulative_Market'].iloc[-1] - 1

buy_signals  = df[df['Position'] ==  1]
sell_signals = df[df['Position'] == -1]
num_trades   = len(buy_signals)

win_trades  = df[df['Strategy_Return'] > 0]
win_rate    = len(win_trades) / max(len(df[df['Strategy_Return'] != 0]), 1)

final_value = initial_capital * df['Cumulative_Strategy'].iloc[-1]

# ── Metrics Row ───────────────────────────────────────────────────────────────
st.subheader(f"📊 Results for {ticker}")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Strategy Return", f"{total_return_strategy:.1%}",
            delta=f"{(total_return_strategy - total_return_market):.1%} vs Buy & Hold")
col2.metric("Sharpe Ratio",   f"{sharpe:.2f}")
col3.metric("Max Drawdown",   f"{max_drawdown:.1%}")
col4.metric("Total Trades",   num_trades)
col5.metric("Final Portfolio", f"${final_value:,.0f}",
            delta=f"${final_value - initial_capital:,.0f}")

st.divider()

# ── Chart 1: Price + Signals ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                         facecolor='#0e1117', gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

ax1, ax2, ax3 = axes
for ax in axes:
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='#888')
    ax.spines[:].set_color('#2d3250')

# Price & MAs
ax1.plot(df.index, df['Close'],           color='#aaaacc', lw=1,   label='Price',          alpha=0.8)
ax1.plot(df.index, df[f'SMA_{sma_short}'], color='#00d4aa', lw=1.5, label=f'SMA {sma_short}')
ax1.plot(df.index, df[f'SMA_{sma_long}'],  color='#ff6b6b', lw=1.5, label=f'SMA {sma_long}')
ax1.scatter(buy_signals.index,  buy_signals['Close'],  marker='^', color='#00ff88', s=80, zorder=5, label='Buy')
ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='#ff4444', s=80, zorder=5, label='Sell')
ax1.set_title(f'{ticker} — MA Crossover Strategy', color='white', fontsize=14, pad=10)
ax1.legend(facecolor='#1c1f2e', labelcolor='white', fontsize=8)
ax1.set_ylabel('Price ($)', color='#888')

# Cumulative Returns
ax2.plot(df.index, df['Cumulative_Strategy'] * 100 - 100, color='#00d4aa', lw=2, label='Strategy')
ax2.plot(df.index, df['Cumulative_Market']   * 100 - 100, color='#ff6b6b', lw=2, label='Buy & Hold', alpha=0.7)
ax2.axhline(0, color='#444', lw=0.8, linestyle='--')
ax2.fill_between(df.index, df['Cumulative_Strategy'] * 100 - 100, 0,
                 where=df['Cumulative_Strategy'] >= 1, alpha=0.15, color='#00d4aa')
ax2.set_ylabel('Return (%)', color='#888')
ax2.legend(facecolor='#1c1f2e', labelcolor='white', fontsize=8)
ax2.set_title('Cumulative Returns', color='white', fontsize=11)

# Drawdown
ax3.fill_between(df.index, drawdown * 100, 0, color='#ff4444', alpha=0.4)
ax3.plot(df.index, drawdown * 100, color='#ff4444', lw=1)
ax3.set_ylabel('Drawdown (%)', color='#888')
ax3.set_title('Drawdown', color='white', fontsize=11)

plt.tight_layout(pad=2)
st.pyplot(fig)

# ── Trade Log ─────────────────────────────────────────────────────────────────
with st.expander("📋 View Trade Log"):
    trades = df[df['Position'] != 0][['Close', 'Position']].copy()
    trades['Action'] = trades['Position'].map({1: '🟢 BUY', -1: '🔴 SELL'})
    trades = trades.rename(columns={'Close': 'Price'})
    st.dataframe(trades[['Price', 'Action']].tail(30), use_container_width=True)

st.caption("Built with Python · yfinance · Streamlit | Data from Yahoo Finance")
