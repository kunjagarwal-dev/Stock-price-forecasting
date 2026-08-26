import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import data_stock_loader
from preprocessing import train_test_split_series
from utils import calculate_rmse

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="MSFT Stock Forecast | Kunj Agarwal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CUSTOM CSS — fonts, colors, spacing
# ==================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1f77b4, #2ca02c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #8a8f98;
            margin-top: 0px;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #161a23;
            border: 1px solid #2a2f3a;
            border-radius: 14px;
            padding: 18px 22px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #9aa0aa;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f2f2f2;
        }
        .badge-best {
            background-color: #1e3d2f;
            color: #4ade80;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .section-divider {
            border-top: 1px solid #2a2f3a;
            margin: 2rem 0 1.5rem 0;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==================================================================
# HEADER
# ==================================================================
st.markdown('<p class="main-title">📈 MSFT Stock Price Forecasting</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comparing ARIMA, Prophet, and LSTM on 5 years of Microsoft closing price data</p>', unsafe_allow_html=True)

# ==================================================================
# LOAD DATA
# ==================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "MSFT.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

@st.cache_data
def load_data():
    return data_stock_loader(DATA_PATH)

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

df = load_data()
arima_results = load_json(os.path.join(MODELS_DIR, "arima_model_results.json"))
prophet_results = load_json(os.path.join(MODELS_DIR, "prophet_results.json"))

lstm_path = os.path.join(PROCESSED_DIR, "lstm_predictions.csv")
lstm_df = pd.read_csv(lstm_path) if os.path.exists(lstm_path) else None

# ==================================================================
# SIDEBAR
# ==================================================================
st.sidebar.header("⚙️ Controls")
date_range = st.sidebar.slider(
    "Select date range",
    min_value=df.index.min().to_pydatetime(),
    max_value=df.index.max().to_pydatetime(),
    value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
)
show_ma30 = st.sidebar.checkbox("Show 30-day Moving Average", value=True)
show_ma200 = st.sidebar.checkbox("Show 200-day Moving Average", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project by:** Kunj Agarwal")
st.sidebar.markdown("[GitHub](https://github.com/kunjagarwal-dev) · [LinkedIn](https://www.linkedin.com/in/kunjagarwal)")

filtered_df = df.loc[date_range[0]:date_range[1]]

# ==================================================================
# TABS
# ==================================================================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Model Comparison", "📉 Forecast Detail"])

# ------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    latest_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    pct_change = (price_change / df['Close'].iloc[-2]) * 100

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Latest Close</div>
            <div class="metric-value">${latest_price:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        color = "#4ade80" if price_change >= 0 else "#f87171"
        arrow = "▲" if price_change >= 0 else "▼"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Daily Change</div>
            <div class="metric-value" style="color:{color}">{arrow} {pct_change:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">52-Week High</div>
            <div class="metric-value">${df['Close'].tail(252).max():,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">52-Week Low</div>
            <div class="metric-value">${df['Close'].tail(252).min():,.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Historical Closing Price")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(filtered_df.index, filtered_df['Close'], label="Close", color="#1f77b4", linewidth=1.3)
    if show_ma30:
        ax.plot(filtered_df.index, filtered_df['Close'].rolling(30).mean(), label="MA 30", color="#ff7f0e", linewidth=1.1)
    if show_ma200:
        ax.plot(filtered_df.index, filtered_df['Close'].rolling(200).mean(), label="MA 200", color="#2ca02c", linewidth=1.1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    with st.expander("📌 30-Day Rolling Volatility"):
        returns = filtered_df['Close'].pct_change()
        vol = returns.rolling(30).std()
        fig2, ax2 = plt.subplots(figsize=(13, 3.5))
        ax2.plot(filtered_df.index, vol, color="#e74c3c", linewidth=1.1)
        ax2.set_ylabel("Volatility")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

# ------------------------------------------------------------------
# TAB 2: MODEL COMPARISON
# ------------------------------------------------------------------
with tab2:
    st.subheader("RMSE Comparison Across Models")

    results = {}
    for k, v in arima_results.items():
        results[k] = v
    for k, v in prophet_results.items():
        results[k] = v
    if lstm_df is not None:
        results["LSTM"] = calculate_rmse(lstm_df["actual"], lstm_df["predicted"])

    if results:
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "RMSE"]).sort_values("RMSE")
        best_model = results_df.iloc[0]["Model"]

        colA, colB = st.columns([1, 1.3])

        with colA:
            st.markdown("#### Results Table")
            for _, row in results_df.iterrows():
                badge = '<span class="badge-best">BEST</span>' if row["Model"] == best_model else ""
                st.markdown(f"""<div class="metric-card" style="text-align:left; margin-bottom:10px;">
                    <span style="font-weight:600; font-size:1.05rem;">{row['Model']}</span> {badge}
                    <div class="metric-value" style="font-size:1.4rem;">{row['RMSE']:.2f}</div>
                </div>""", unsafe_allow_html=True)

        with colB:
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            colors = ["#2ca02c" if m == best_model else "#4a5568" for m in results_df["Model"]]
            bars = ax3.bar(results_df["Model"], results_df["RMSE"], color=colors)
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.2f}",
                          ha='center', fontsize=10, color="#e2e2e2")
            ax3.set_ylabel("RMSE (lower is better)")
            ax3.set_title("Model Performance")
            fig3.patch.set_alpha(0)
            ax3.patch.set_alpha(0)
            st.pyplot(fig3)

        st.info(f"🏆 **{best_model}** achieved the lowest RMSE, making it the best-performing model for this dataset.")
    else:
        st.warning("No saved results found. Run the model notebooks first to generate `arima_model_results.json`, `prophet_results.json`, and `lstm_predictions.csv`.")

# ------------------------------------------------------------------
# TAB 3: FORECAST DETAIL
# ------------------------------------------------------------------
with tab3:
    st.subheader("Predicted vs Actual (Test Period)")

    model_choice = st.selectbox("Select a model to inspect", ["LSTM", "ARIMA", "Prophet"])

    if model_choice == "LSTM" and lstm_df is not None:
        fig4, ax4 = plt.subplots(figsize=(13, 5))
        ax4.plot(lstm_df["actual"].values, label="Actual", color="#1f77b4")
        ax4.plot(lstm_df["predicted"].values, label="Predicted", color="#ff7f0e")
        ax4.set_title("LSTM: Predicted vs Actual")
        ax4.legend()
        ax4.grid(alpha=0.2)
        st.pyplot(fig4)
        st.caption("Note: LSTM predictions tend to lag slightly behind sharp reversals, since it partially tracks recent trend momentum.")

    elif model_choice == "ARIMA":
        try:
            arima_model = joblib.load(os.path.join(MODELS_DIR, "arima_model.pkl"))
            train, test = train_test_split_series(df["Close"])
            forecast = arima_model.forecast(steps=len(test))

            fig5, ax5 = plt.subplots(figsize=(13, 5))
            ax5.plot(test.index, test.values, label="Actual", color="#1f77b4")
            ax5.plot(test.index, forecast.values, label="Forecast", color="#2ca02c")
            ax5.set_title("ARIMA(2,1,2): Forecast vs Actual")
            ax5.legend()
            ax5.grid(alpha=0.2)
            st.pyplot(fig5)
        except Exception as e:
            st.warning(f"Could not load ARIMA model: {e}")

    elif model_choice == "Prophet":
        try:
            prophet_model = joblib.load(os.path.join(MODELS_DIR, "prophet_model.pkl"))
            prophet_df = df.reset_index()[["Date", "Close"]]
            prophet_df.columns = ["ds", "y"]
            train_size = int(len(prophet_df) * 0.9)
            test_df = prophet_df[train_size:]

            future = prophet_model.make_future_dataframe(periods=len(test_df))
            forecast = prophet_model.predict(future)
            predicted = forecast["yhat"].iloc[-len(test_df):].values

            fig6, ax6 = plt.subplots(figsize=(13, 5))
            ax6.plot(test_df["ds"], test_df["y"], label="Actual", color="#1f77b4")
            ax6.plot(test_df["ds"], predicted, label="Predicted", color="#e74c3c")
            ax6.set_title("Prophet: Predicted vs Actual")
            ax6.legend()
            ax6.grid(alpha=0.2)
            st.pyplot(fig6)
        except Exception as e:
            st.warning(f"Could not load Prophet model: {e}")

# ==================================================================
# FOOTER
# ==================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Built with Streamlit · Data via yfinance · Models: ARIMA, Prophet, LSTM")