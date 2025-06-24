#config.py
# Configuration file for the trading bot
# General settings
import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables
from dotenv import load_dotenv
# Ensure .env file is in the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# API keys and secrets
if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
    logging.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file")
    sys.exit(1)
API_KEY    = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
SMTP_SERVER          = os.getenv("SMTP_SERVER")
SMTP_PORT            = int(os.getenv("SMTP_PORT", 587))
EMAIL_USER           = os.getenv("EMAIL_USER")
EMAIL_PASSWORD       = os.getenv("EMAIL_PASSWORD")
MODEL_PATH           = os.getenv("MODEL_PATH", "models/model.pkl")


SUPPORTED_TOKENS = ["BTC", "SOL", "DOGE", "FART"]
PERP_SYMBOLS = {
    "BTC": "BTCUSDT",   # Bitcoin USDT perpetual
    "SOL": "SOLUSDT",   # Solana USDT perpetual
    "DOGE": "DOGEUSDT", # Dogecoin USDT perpetual
    "FART": "FARTCOINUSDT"  # Fartcoin USDT perpetual
}

BYBIT_BASE_URL = "https://api.bybit.com"

TWITTER_QUERY_MAPPING = {
    "BTC": "bitcoin",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "FART": "fartcoin"
}

# Features and Labels
LOOKBACK_PERIOD = 60  # in minutes
LABEL_HORIZON = 15     # minutes into the future

# ML config
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

# Sentiment
NUM_TWEETS = 100
SENTIMENT_SCORER = "vader"  # Can add more later

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"