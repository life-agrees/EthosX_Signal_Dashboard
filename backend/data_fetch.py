# data_fetch_bybit.py

import os
import requests
from dotenv import load_dotenv
import pandas as pd
import socket
from datetime import datetime, timedelta
import ta
import time
import hashlib, hmac, time
import warnings
warnings.filterwarnings("ignore")


from config import (
    SUPPORTED_TOKENS,
    PERP_SYMBOLS,
    LOOKBACK_PERIOD,
    LABEL_HORIZON,
    DATA_DIR,
)
from sentiment import get_token_sentiment

load_dotenv()   # ← this reads your .env into os.environ
print("KEY:", os.getenv("BYBIT_API_KEY"))
print("SECRET:", os.getenv("BYBIT_API_SECRET"))


# Set up request headers for Bybit API calls
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

API_KEY    = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

def sign(params: dict) -> str:
    to_sign = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), to_sign.encode(), hashlib.sha256).hexdigest()

def public_get(path: str, params: dict) -> requests.Response:
    params.update({
        "apiKey":    API_KEY,
        "timestamp": int(time.time() * 1000),
    })
    params["sign"] = sign(params)
    return requests.get(BYBIT_BASE_URL + path, params=params, timeout=10)

# Bybit API configuration
BYBIT_BASE_URL = "https://api.bybit.com"

def fetch_funding_rate(symbol: str) -> float:
    """
    Fetch the most recent funding rate for a USDT perpetual on Bybit.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 1
    }
    
    try:
        resp = public_get("/v5/market/funding/history", params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            return float(data["result"]["list"][0]["fundingRate"])
        return 0.0
    except Exception as e:
        print(f"[WARN] Failed to fetch funding rate for {symbol}: {e}")
        return 0.0


def fetch_open_interest(symbol: str) -> float:
    """
    Fetch the current open interest for a USDT perpetual on Bybit.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/open-interest"
    params = {
        "category": "linear",
        "symbol": symbol
    }
    
    try:
        resp = public_get("/v5/market/open-interest", params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            return float(data["result"]["list"][0]["openInterest"])
        return 0.0
    except Exception as e:
        print(f"[WARN] Failed to fetch open interest for {symbol}: {e}")
        return 0.0


def get_bybit_klines(symbol: str, interval="1", limit=None, max_retries=3) -> pd.DataFrame:
    """
    Fetch 1-minute klines from Bybit.
    Returns a DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval ('1' for 1 minute)
        limit: Number of klines to fetch (max 1000 for Bybit)
    """
    if limit is None:
        limit = LOOKBACK_PERIOD + LABEL_HORIZON
    
    # Handle string limit parameter (from your original code)
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100  # Default fallback
    
    # Bybit has a max limit of 1000 klines per request
    limit = min(limit, 1000)
    
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": str(limit)  # Ensure limit is passed as string to API
    }
    
    try:
        # Add retry logic for 403 errors
        for attempt in range(max_retries):
            try:
                response = public_get("/v5/market/kline", params)
                
                # Handle 403 specifically
                if response.status_code == 403:
                    print(f"[WARN] 403 Forbidden for {symbol} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Progressive backoff: 5s, 10s, 15s
                        print(f"[INFO] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[ERROR] Max retries reached for {symbol}. Skipping.")
                        return pd.DataFrame()
                
                response.raise_for_status()
                break  # Success, exit retry loop
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 403 and attempt < max_retries - 1:
                    continue  # Will retry
                else:
                    raise e  # Re-raise if not 403 or max retries reached
        
        data = response.json()
        
        if data.get("retCode") != 0:
            print(f"[ERROR] Bybit API error for {symbol}: {data.get('retMsg', 'Unknown error')}")
            return pd.DataFrame()
        
        klines = data.get("result", {}).get("list", [])
        if not klines:
            print(f"[WARN] No kline data available for {symbol}")
            return pd.DataFrame()
        
        # Bybit returns data in reverse chronological order, so we need to reverse it
        klines.reverse()
        
        # Bybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # FIX 1: Ensure all data is properly typed before any operations
        # Convert timestamp from string milliseconds to proper datetime
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors='coerce')
        
        # FIX 2: Convert all OHLCV data to numeric, handling any string values
        numeric_columns = ["open", "high", "low", "close", "volume", "turnover"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # FIX 3: Remove any rows with invalid data (NaN timestamps or prices)
        df = df.dropna(subset=["timestamp", "close"])
        
        # FIX 4: Sort by timestamp to ensure proper chronological order
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # FIX 5: Return ALL OHLCV columns, not just close and volume
        # This matches what your API server expects
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        
        # Set timestamp as index AFTER all cleaning is done
        df.set_index("timestamp", inplace=True)
        
        return df
        
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, socket.gaierror) as e:
        print(f"[ERROR] Network error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error for {symbol}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"[ERROR] Data parsing error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Unexpected error for {symbol}: {e}")
        return pd.DataFrame()


def get_fallback_data(symbol: str, limit=100) -> pd.DataFrame:
    """
    Fallback function to get basic price data if Bybit API fails.
    This creates dummy data to keep the system running during API issues.
    """
    print(f"[INFO] Using fallback data for {symbol}")
    
    # Create timestamps for the last 'limit' minutes
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=i) for i in range(limit, 0, -1)]
    
    # Generate realistic-looking dummy data based on symbol
    base_price = 50000 if 'BTC' in symbol else 100  # Rough price estimates
    
    data = []
    current_price = base_price
    
    for ts in timestamps:
        # Small random price movements (±0.5%)
        import numpy as np
        change = current_price * (0.005 * (2 * np.random.random() - 1))
        current_price += change
        
        # Create OHLC data
        high = current_price * (1 + abs(np.random.random() * 0.002))
        low = current_price * (1 - abs(np.random.random() * 0.002))
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': ts,
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def build_token_features(token: str) -> pd.DataFrame:
    """
    Build features for a single token:
      - price_return (pct change of close),
      - volume_change (pct change of volume),
      - sentiment (Twitter),
      - target (future 5-min return),
      - label (1/0/-1).
    Returns a DataFrame with these columns.
    """
    symbol = PERP_SYMBOLS[token].upper()
    df = get_bybit_klines(symbol)
    
    # If Bybit data fails, use fallback
    if df.empty:
        print(f"[WARN] No data available for {token} from Bybit. Using fallback data.")
        df = get_fallback_data(symbol)
        
    if df.empty:
        print(f"[ERROR] No data available for {token} at all. Skipping feature build.")
        return df

    try:
        # FIX 6: Ensure we have enough data before calculating indicators
        if len(df) < 30:  # Need at least 30 points for reliable indicators
            print(f"[WARN] Insufficient data for {token} ({len(df)} points). Skipping.")
            return pd.DataFrame()

        # Basic price and volume features with error handling
        df["close_return"] = df["close"].pct_change(periods=min(15, len(df)-1))
        df["volume_change"] = df["volume"].pct_change()
        df["return_15m"] = df["close"].pct_change(periods=min(15, len(df)-1))
        df["volume_15m"] = df["volume"].rolling(window=min(15, len(df)), min_periods=1).sum()
        
        # Fetch funding rate and open interest (these return floats)
        funding_rate_val = fetch_funding_rate(symbol)
        open_interest_val = fetch_open_interest(symbol)
        
        df["funding_rate"] = float(funding_rate_val)
        df["open_interest"] = float(open_interest_val)

        # Fetch sentiment (cached internally)
        sentiment_score = get_token_sentiment(token)
        df["sentiment"] = float(sentiment_score)

        # FIX 7: Technical indicators with proper error handling
        try:
            # 14-period RSI (need at least 14 points)
            if len(df) >= 14:
                rsi_indicator = ta.momentum.RSIIndicator(df["close"], window=14)
                df["rsi_14"] = rsi_indicator.rsi()
            else:
                df["rsi_14"] = 50.0  # Neutral RSI
        except Exception as e:
            print(f"[WARN] RSI calculation failed for {token}: {e}")
            df["rsi_14"] = 50.0

        try:
            # MACD difference (need at least 26 points)
            if len(df) >= 26:
                macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
                df["macd_diff"] = macd.macd_diff()
            else:
                df["macd_diff"] = 0.0
        except Exception as e:
            print(f"[WARN] MACD calculation failed for {token}: {e}")
            df["macd_diff"] = 0.0

        # FIX 8: Safe target and label calculation
        try:
            # Create target: future return over LABEL_HORIZON minutes
            if len(df) > LABEL_HORIZON:
                df["target"] = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1
                
                # Discrete label: 1 (up), -1 (down), 0 (hold)
                def safe_label(x):
                    if pd.isna(x):
                        return 0
                    return 1 if x > 0.001 else -1 if x < -0.001 else 0
                
                df["label"] = df["target"].apply(safe_label)
            else:
                df["target"] = 0.0
                df["label"] = 0
        except Exception as e:
            print(f"[WARN] Target/label calculation failed for {token}: {e}")
            df["target"] = 0.0
            df["label"] = 0

        # FIX 9: Final cleanup - remove NaN values safely
        df = df.replace([float('inf'), float('-inf')], 0.0)  # Replace infinity values
        df = df.fillna(0.0)  # Fill remaining NaN with neutral values
        
        return df

    except Exception as e:
        print(f"[ERROR] Feature building failed for {token}: {e}")
        return pd.DataFrame()
    """
    Build features for a single token:
      - price_return (pct change of close),
      - volume_change (pct change of volume),
      - sentiment (Twitter),
      - target (future 5-min return),
      - label (1/0/-1).
    Returns a DataFrame with these columns.
    """
    symbol = PERP_SYMBOLS[token].upper()
    df = get_bybit_klines(symbol)
    
    if df.empty:
        print(f"[WARN] No data available for {token}. Skipping feature build.")
        return df

    try:
        # FIX 6: Ensure we have enough data before calculating indicators
        if len(df) < 30:  # Need at least 30 points for reliable indicators
            print(f"[WARN] Insufficient data for {token} ({len(df)} points). Skipping.")
            return pd.DataFrame()

        # Basic price and volume features with error handling
        df["close_return"] = df["close"].pct_change(periods=min(15, len(df)-1))
        df["volume_change"] = df["volume"].pct_change()
        df["return_15m"] = df["close"].pct_change(periods=min(15, len(df)-1))
        df["volume_15m"] = df["volume"].rolling(window=min(15, len(df)), min_periods=1).sum()
        
        # Fetch funding rate and open interest (these return floats)
        funding_rate_val = fetch_funding_rate(symbol)
        open_interest_val = fetch_open_interest(symbol)
        
        df["funding_rate"] = float(funding_rate_val)
        df["open_interest"] = float(open_interest_val)

        # Fetch sentiment (cached internally)
        sentiment_score = get_token_sentiment(token)
        df["sentiment"] = float(sentiment_score)

        # FIX 7: Technical indicators with proper error handling
        try:
            # 14-period RSI (need at least 14 points)
            if len(df) >= 14:
                rsi_indicator = ta.momentum.RSIIndicator(df["close"], window=14)
                df["rsi_14"] = rsi_indicator.rsi()
            else:
                df["rsi_14"] = 50.0  # Neutral RSI
        except Exception as e:
            print(f"[WARN] RSI calculation failed for {token}: {e}")
            df["rsi_14"] = 50.0

        try:
            # MACD difference (need at least 26 points)
            if len(df) >= 26:
                macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
                df["macd_diff"] = macd.macd_diff()
            else:
                df["macd_diff"] = 0.0
        except Exception as e:
            print(f"[WARN] MACD calculation failed for {token}: {e}")
            df["macd_diff"] = 0.0

        # FIX 8: Safe target and label calculation
        try:
            # Create target: future return over LABEL_HORIZON minutes
            if len(df) > LABEL_HORIZON:
                df["target"] = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1
                
                # Discrete label: 1 (up), -1 (down), 0 (hold)
                def safe_label(x):
                    if pd.isna(x):
                        return 0
                    return 1 if x > 0.001 else -1 if x < -0.001 else 0
                
                df["label"] = df["target"].apply(safe_label)
            else:
                df["target"] = 0.0
                df["label"] = 0
        except Exception as e:
            print(f"[WARN] Target/label calculation failed for {token}: {e}")
            df["target"] = 0.0
            df["label"] = 0

        # FIX 9: Final cleanup - remove NaN values safely
        df = df.replace([float('inf'), float('-inf')], 0.0)  # Replace infinity values
        df = df.fillna(0.0)  # Fill remaining NaN with neutral values
        
        return df

    except Exception as e:
        print(f"[ERROR] Feature building failed for {token}: {e}")
        return pd.DataFrame()


def save_token_data(token: str):
    """
    Build features for a token and save to CSV.
    """
    df = build_token_features(token)
    if df.empty:
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = f"{DATA_DIR}/{token}_features.csv"
    df.to_csv(file_path)
    print(f"Saved feature data for {token} to {file_path}")


def save_all():
    """
    Iterate through all supported tokens and save their feature CSVs.
    """
    for i, token in enumerate(SUPPORTED_TOKENS):
        try:
            print(f"[INFO] Processing {token} ({i+1}/{len(SUPPORTED_TOKENS)})")
            save_token_data(token)
            # Add longer delay between tokens to avoid rate limiting
            if i < len(SUPPORTED_TOKENS) - 1:  # Don't sleep after the last token
                print(f"[INFO] Waiting 2s before next token...")
                time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Failed for {token}: {e}")
            # Continue with next token even if one fails


def test_bybit_connection():
    """
    Test connection to Bybit API and display available symbols.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear", "limit": 10}
    
    try:
        resp = public_get("/v5/market/instruments-info", params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") == 0:
            print("[INFO] Successfully connected to Bybit API")
            symbols = [item["symbol"] for item in data.get("result", {}).get("list", [])]
            print(f"[INFO] Sample available symbols: {symbols[:5]}")
            return True
        else:
            print(f"[ERROR] Bybit API error: {data.get('retMsg', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to connect to Bybit API: {e}")
        return False


if __name__ == "__main__":
    print("Testing Bybit API connection...")
    if test_bybit_connection():
        print("Starting data collection...")
        save_all()
    else:
        print("Failed to connect to Bybit API. Please check your internet connection.")

# This script fetches and processes trading data from Bybit API for supported tokens,
# building features like price returns, volume changes, and sentiment scores.