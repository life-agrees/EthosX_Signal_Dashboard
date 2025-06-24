# dashboard.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from models.predict import load_model, make_prediction
from sentiment import get_token_sentiment
from alerts import add_subscriber, get_subscribers, maybe_alert, CONFIDENCE_THRESHOLD


st.set_page_config(page_title="EthosX OPerps Dashboard", layout="wide")

FEATURE_COLUMNS = [
    "close_return", "volume_change", "sentiment",
    "funding_rate", "open_interest", "return_15m", "volume_15m",
    "rsi_14", "macd_diff"
]

LABEL_MAP = {-1: "SELL", 0: "HOLD", 1: "BUY"}

MODEL_OPTIONS = {
    "RandomForest": "models/model.pkl",
    "XGBoost": "models/xgb_model.pkl"
}


def show_dashboard():
    st.title("📈 EthosX OPerps Predictive Market Signal Dashboard")

    # Email subscription
    st.sidebar.header("🔔 Subscribe for Alerts")
    email = st.sidebar.text_input("Enter your email:")
    if st.sidebar.button("Subscribe"):
        if email:
            added = add_subscriber(email)
            if added:
                st.sidebar.success("Subscribed successfully!")
            else:
                st.sidebar.info("You're already subscribed.")
        else:
            st.sidebar.error("Please enter a valid email.")
    st.sidebar.markdown(f"**Subscribers:** {len(get_subscribers())}")

    # Model selection
    model_choice = st.sidebar.selectbox("Choose Model", list(MODEL_OPTIONS.keys()))
    os.environ["MODEL_PATH"] = MODEL_OPTIONS[model_choice]
    model = load_model()

    # Token selection
    token = st.selectbox("Select Token:", ["BTC", "SOL", "DOGE", "FART"])

    # Market inputs
    st.subheader("Market Inputs")
    price = st.number_input("Current Price (USDT-Perp)", value=30000.0)
    volume = st.number_input("24h Volume", value=150_000_000.0)
    open_interest = st.number_input("Open Interest (contracts)", value=5_000_000.0)

    # Technical & sentiment
    st.subheader("Technical Indicators & Sentiment")
    sentiment_score = get_token_sentiment(token)
    st.metric("Sentiment", f"{sentiment_score:.3f}")
    st.write("(Replace placeholders with live per-token metrics)")
    return_15m = st.number_input("15m Return", value=0.0)
    volume_15m = st.number_input("15m Volume", value=0.0)
    rsi_14 = st.number_input("RSI (14)", value=50.0)
    macd_diff = st.number_input("MACD Diff", value=0.0)
    funding_rate = st.number_input("Funding Rate", value=0.0)

    # Prediction block
    if st.button("🔮 Predict Market Signal"):
        features = {
            "close_return": 0.0,  # replace with real-minute return
            "volume_change": 0.0,
            "sentiment": sentiment_score,
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "return_15m": return_15m,
            "volume_15m": volume_15m,
            "rsi_14": rsi_14,
            "macd_diff": macd_diff
        }
        label, confidence = make_prediction(model, token, features)
        st.success(f"Predicted Signal: **{LABEL_MAP[label]}**")
        st.info(f"Model: {model_choice} | Confidence: {confidence*100:.1f}%")

        # Trigger email alerts
        if confidence >= CONFIDENCE_THRESHOLD:
            maybe_alert(confidence, token, LABEL_MAP[label])
            st.warning("High-confidence alert sent to subscribers!")

    # Historical price placeholder
    st.subheader("Historical Trends (Placeholder)")
    dummy = pd.DataFrame({
        "time": pd.date_range(start="2023-01-01", periods=50, freq="H"),
        "price": pd.Series(range(50)) * 100 + price
    })
    fig = px.line(dummy, x="time", y="price", title=f"{token} Price Trend")
    st.plotly_chart(fig)


if __name__ == "__main__":
    show_dashboard()
