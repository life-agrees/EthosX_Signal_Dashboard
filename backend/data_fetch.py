# data_fetch_bybit.py

import os
import requests
from dotenv import load_dotenv
import pandas as pd
import socket
from datetime import datetime, timedelta
import ta
import httpx
from httpx import HTTPStatusError
import asyncio
import time
import logging
from typing import Optional
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


# fallback to official Bybit API if proxy fails
# This is the official Bybit API endpoint:
OFFICIAL_BYBIT = "https://api.bybit.com"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Bybit API configuration
BYBIT_BASE_URL = "https://ethosx-bybit-proxy.ethosx-bybit.workers.dev"


load_dotenv()   # ← this reads your .env into os.environ
print("KEY:", os.getenv("BYBIT_API_KEY"))
print("SECRET:", os.getenv("BYBIT_API_SECRET"))


# Set up request headers for Bybit API calls
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":       "application/json",
}

API_KEY    = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

def sign(params: dict) -> str:
    to_sign = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), to_sign.encode(), hashlib.sha256).hexdigest()



async def public_get(path: str, params: dict) -> dict:
    """
    1) Try the proxy.
    2) If it returns 403, retry against the official Bybit API.
    Raises on any other HTTP error.
    """
    proxy_url    = BYBIT_BASE_URL + path
    official_url = OFFICIAL_BYBIT    + path

    logger.info(f"→ GET {proxy_url} params={params}")    

    async with httpx.AsyncClient(timeout=30) as client:
        # 1) proxy
        resp = await client.get(proxy_url, params=params, headers=HEADERS)
        logger.info(f"[data_fetch] proxy {proxy_url} → {resp.status_code}")
        try:
            resp.raise_for_status()
        except HTTPStatusError as e:
            if resp.status_code == 403:
                # fallback to official
                logger.warning(f"[data_fetch] proxy 403, retrying official {official_url}")
                resp = await client.get(official_url, params=params, headers=HEADERS)
                logger.info(f"[data_fetch] official {official_url} → {resp.status_code}")
                resp.raise_for_status()
            else:
                raise
        return resp.json()
async def fetch_funding_rate(symbol: str) -> float:
    params = {"category": "linear", "symbol": symbol, "limit": 1}
    data = await public_get("/v5/market/funding/history", params)
    if data.get("retCode") == 0:
        lst = data.get("result", {}).get("list", [])
        return float(lst[0]["fundingRate"]) if lst else 0.0
    logger.error(f"Funding API error: {data.get('retMsg')}")
    return 0.0

async def fetch_open_interest(symbol: str) -> float:
    params = {"category": "linear", "symbol": symbol}
    data = await public_get("/v5/market/open-interest", params)
    lst = data.get("result", {}).get("list", [])
    return float(lst[0]["openInterest"]) if lst else 0.0

async def get_bybit_klines(
    symbol: str,
    interval: str = "1",
    limit: Optional[int] = 100,
    max_retries: int = 3
) -> pd.DataFrame:
    limit = min(int(limit or 100), 1000)
    params = {"category": "linear", "symbol": symbol.upper(), "interval": interval, "limit": str(limit)}

    for attempt in range(max_retries):
        data = await public_get("/v5/market/kline", params)
        if data.get("retCode") == 0:
            break
        logger.warning(f"Error {data.get('retCode')}, retry {attempt+1}")
        await asyncio.sleep((attempt + 1) * 5)
    else:
        return pd.DataFrame()

    klines = data.get("result", {}).get("list", [])
    if not klines:
        return pd.DataFrame()

    klines.reverse()
    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    )

    # Safely convert weird timestamps to numeric, then to datetime
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().set_index("timestamp")[['open','high','low','close','volume']]
    return df

async def build_token_features(token: str) -> pd.DataFrame:
    symbol = PERP_SYMBOLS[token]
    df = await get_bybit_klines(symbol)
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df['close_return'] = df['close'].pct_change(15)
    df['volume_change'] = df['volume'].pct_change()
    df['return_15m'] = df['close'].rolling(15, min_periods=1).apply(
        lambda x: (x[-1] / x[0]) - 1 if len(x) > 1 else 0.0
    )
    df['volume_15m'] = df['volume'].rolling(15, min_periods=1).sum()

    # async calls for rates and interest
    funding, oi = await asyncio.gather(
        fetch_funding_rate(symbol),
        fetch_open_interest(symbol)
    )
    df['funding_rate'] = funding
    df['open_interest'] = oi

    # sentiment placeholder
    df['sentiment'] = get_token_sentiment(token)

    # indicators
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()

    # target and label
    df['target'] = df['close'].shift(-LABEL_HORIZON) / df['close'] - 1
    df['label'] = df['target'].apply(lambda x: 1 if x>0.001 else -1 if x<-0.001 else 0)

    return df.fillna(0.0)

async def save_token_data(token: str):
    df = await build_token_features(token)
    if df.empty:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{token}_features.csv")
    df.to_csv(path)
    logger.info(f"Saved {path}")

async def main():
    # test connection
    sample = await public_get("/v5/market/instruments-info", {"category": "linear", "limit": 5})
    logger.info(sample)

    # save all tokens
    for token in SUPPORTED_TOKENS:
        await save_token_data(token)
        await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(main())
