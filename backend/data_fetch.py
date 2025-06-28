# data_fetch.py

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import ta
import numpy as np
import aiohttp
from aiolimiter import AsyncLimiter
import asyncio
import time
import logging
from typing import Optional, Dict, List
import warnings
import random
from urllib.parse import urljoin
warnings.filterwarnings("ignore")

from config import (
    SUPPORTED_TOKENS,
    PERP_SYMBOLS,
    COINGECKO_SYMBOLS,
    LABEL_HORIZON,
    DATA_DIR,
)
from sentiment import get_token_sentiment

# API configurations
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_BASE_URL = "https://fapi.binance.com/fapi/v1"
BYBIT_BASE_URL = "https://api.bybit.com/v5"

# Alternative endpoints for better reliability
ALTERNATIVE_ENDPOINTS = {
    "binance": [
        "https://fapi.binance.com/fapi/v1",
    ],
    "bybit": [
        "https://api.bybit.com/v5",
        "https://api.bybit.com/v2/public",  # Legacy API as fallback
    ]
}

def choose_url(service: str, path: str) -> List[str]:
    """Construct full URLs for a service’s path using its fallbacks."""
    bases = ALTERNATIVE_ENDPOINTS.get(service, [])
    return [urljoin(base, path) for base in bases]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Symbol mappings for different exchanges
COINGECKO =COINGECKO_SYMBOLS 

BINANCE_PERP_SYMBOLS = PERP_SYMBOLS

BYBIT_PERP_SYMBOLS = PERP_SYMBOLS

# Improved headers with rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "DNT": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
    }

#  Use a token-bucket limiter instead of manual sleeps
#    Here we allow:   Coingecko → 5 calls/minute,  Binance → 20/min,  Bybit → 30/min
LIMITERS = {
    "coingecko": AsyncLimiter(5, 60),
    "binance":  AsyncLimiter(20, 60),
    "bybit":    AsyncLimiter(30, 60),
}

async def make_request_with_retry(url: str, params: dict = None, max_retries: int = 3, service: str = "default") -> Optional[dict]:
    """Enhanced request function with aiohttp for better connection handling"""
    
    #  Wait for token from the bucket
    limiter = LIMITERS.get(service)
    if limiter:
        await limiter.acquire()

    
    # Use aiohttp instead of httpx for better connection handling
    timeout = aiohttp.ClientTimeout(total=30, connect=15)
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, 
                headers=get_headers(),
                connector=aiohttp.TCPConnector(ssl=False)  # fallback  
            ) as session:
                async with session.get(url, params=params) as response:
                    
                    if response.status == 429:  # Rate limited
                        wait_time = min(30 * (2 ** attempt), 120)
                        logger.warning(f"Rate limited on {url}. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status == 403:  # Forbidden
                        logger.error(f"Forbidden (403) for {url} - possible IP block")
                        await asyncio.sleep(60)
                        continue
                    
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                    
                    return await response.json()
                    
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"Connection error for {url} (attempt {attempt + 1}/{max_retries}): {e}")
        except aiohttp.ClientTimeout:
            logger.warning(f"Timeout for {url} (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            wait_time = min(15 * (2 ** attempt), 90)  # Longer waits
            logger.info(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    logger.error(f"All retry attempts failed for {url}")
    return None
# Enhanced CoinGecko functions with better error handling
async def get_coingecko_spot_data(token: str, days: int = 30) -> pd.DataFrame:
    """Fetch spot OHLCV data from CoinGecko with improved error handling"""
    if token not in COINGECKO_SYMBOLS:
        logger.warning(f"Token {token} not found in CoinGecko IDs")
        return pd.DataFrame()
    
    coin_id = COINGECKO_SYMBOLS[token]
    
    # Get OHLC data first
    ohlc_url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    ohlc_params = {"vs_currency": "usd", "days": str(days)}
    
    ohlc_data = await make_request_with_retry(ohlc_url, ohlc_params, service="coingecko")
    if not ohlc_data or len(ohlc_data) < 10:
        logger.error(f"Insufficient OHLC data for {token}")
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get volume data with additional delay to avoid rate limits
    await asyncio.sleep(2)  # Extra delay between CoinGecko requests
    
    market_url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
    market_params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    
    market_data = await make_request_with_retry(market_url, market_params, service="coingecko")
    if market_data and "total_volumes" in market_data:
        volumes = market_data["total_volumes"]
        volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms")
        volume_df["volume"] = pd.to_numeric(volume_df["volume"], errors='coerce')
        
        # Merge volumes with OHLC data
        df = pd.merge_asof(
            df.sort_values('timestamp'), 
            volume_df.sort_values('timestamp'), 
            on='timestamp', 
            direction='nearest'
        )
    else:
        logger.warning(f"No volume data available for {token}, using estimated volume")
        # Estimate volume based on price volatility if real volume unavailable
        df["volume"] = (df["high"] - df["low"]) / df["close"] * 1000000  # Rough estimate
    
    return df.dropna().set_index("timestamp")

# Enhanced derivatives data functions with multiple endpoints
async def get_binance_funding_rate(token: str) -> float:
    """Get current funding rate from Binance with fallback endpoints."""
    if token not in BINANCE_PERP_SYMBOLS:
        return 0.0

    symbol = BINANCE_PERP_SYMBOLS[token]
    params = {"symbol": symbol}

    for url in choose_url("binance", "/premiumIndex"):
        data = await make_request_with_retry(url, params=params, service="binance")
        if data and "lastFundingRate" in data:
            rate = float(data["lastFundingRate"]) * 100
            logger.info(f"✅ Got Binance funding rate for {token}: {rate:.4f}%")
            return rate

    logger.warning(f"⚠️  Failed to get Binance funding rate for {token}")
    return 0.0

async def get_binance_open_interest(token: str) -> float:
    """Get open interest from Binance with fallback endpoints."""
    if token not in BINANCE_PERP_SYMBOLS:
        return 0.0

    symbol = BINANCE_PERP_SYMBOLS[token]
    params = {"symbol": symbol}

    for url in choose_url("binance", "/openInterest"):
        data = await make_request_with_retry(url, params=params, service="binance")
        if data and "openInterest" in data:
            oi = float(data["openInterest"])
            logger.info(f"✅ Got Binance open interest for {token}: {oi:,.0f}")
            return oi

    logger.warning(f"⚠️  Failed to get Binance open interest for {token}")
    return 0.0

async def get_bybit_funding_rate(token: str) -> float:
    """Get funding rate from Bybit with improved error handling."""
    if token not in BYBIT_PERP_SYMBOLS:
        return 0.0

    symbol = BYBIT_PERP_SYMBOLS[token]
    params = {"category": "linear", "symbol": symbol}

    # try each base_url in turn
    for url in choose_url("bybit", "/market/tickers"):
        data = await make_request_with_retry(url, params=params, service="bybit")
        if not data:
            continue
        result = data.get("result", {})
        tickers = result.get("list", [])
        if tickers and "fundingRate" in tickers[0]:
            rate = float(tickers[0]["fundingRate"]) * 100
            logger.info(f"✅ Got Bybit funding rate for {token}: {rate:.4f}%")
            return rate

    logger.warning(f"⚠️  Unable to fetch Bybit funding rate for {token}")
    return 0.0

async def get_bybit_open_interest(token: str) -> float:
    """Get open interest from Bybit with improved error handling."""
    if token not in BYBIT_PERP_SYMBOLS:
        return 0.0

    symbol = BYBIT_PERP_SYMBOLS[token]
    params = {"category": "linear", "symbol": symbol}

    # try each base_url in turn
    for url in choose_url("bybit", "/market/open-interest"):
        data = await make_request_with_retry(url, params=params, service="bybit")
        if not data:
            continue
        result = data.get("result", {})
        oi_list = result.get("list", [])
        if oi_list and "openInterest" in oi_list[0]:
            oi = float(oi_list[0]["openInterest"])
            logger.info(f"✅ Got Bybit open interest for {token}: {oi:,.0f}")
            return oi

    logger.warning(f"⚠️  Unable to fetch Bybit open interest for {token}")
    return 0.0

async def fetch_derivatives_data(token: str) -> Dict[str, float]:
    """Simplified derivatives data fetch with better error handling and fallbacks"""
    
    logger.info(f"Fetching derivatives data for {token}")
    
    # Initialize default values
    funding_rate = 0.0
    open_interest = 0.0
    
    # Strategy 1: Try Binance with single request and immediate fallback
    if token in BINANCE_PERP_SYMBOLS:
        symbol = BINANCE_PERP_SYMBOLS[token]
        
        try:
            # Test connectivity first with a simple ping
            ping_url = "https://fapi.binance.com/fapi/v1/ping"
            ping_result = await make_request_with_retry(ping_url, service="binance")
            
            if ping_result is not None:
                # If ping works, try funding rate
                funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex"
                funding_params = {"symbol": symbol}
                
                funding_data = await make_request_with_retry(funding_url, funding_params, max_retries=2, service="binance")
                if funding_data and "lastFundingRate" in funding_data:
                    funding_rate = float(funding_data["lastFundingRate"]) * 100
                    logger.info(f"✅ Got Binance funding rate for {token}: {funding_rate:.4f}%")
                    
                    # If funding rate works, try open interest
                    await asyncio.sleep(2)  # Wait between requests
                    oi_url = f"https://fapi.binance.com/fapi/v1/openInterest"
                    oi_params = {"symbol": symbol}
                    
                    oi_data = await make_request_with_retry(oi_url, oi_params, max_retries=2, service="binance")
                    if oi_data and "openInterest" in oi_data:
                        open_interest = float(oi_data["openInterest"])
                        logger.info(f"✅ Got Binance open interest for {token}: {open_interest:,.0f}")
            
        except Exception as e:
            logger.debug(f"Binance derivatives failed for {token}: {e}")
    
    # Strategy 2: Try Bybit only if Binance completely failed
    if funding_rate == 0.0 and token in BYBIT_PERP_SYMBOLS:
        symbol = BYBIT_PERP_SYMBOLS[token]
        
        try:
            await asyncio.sleep(2)  # Wait before trying Bybit
            
            bybit_url = f"https://api.bybit.com/v5/market/tickers"
            bybit_params = {"category": "linear", "symbol": symbol}
            
            bybit_data = await make_request_with_retry(bybit_url, bybit_params, max_retries=2, service="bybit")
            if bybit_data and "result" in bybit_data and "list" in bybit_data["result"]:
                tickers = bybit_data["result"]["list"]
                if tickers and "fundingRate" in tickers[0]:
                    funding_rate = float(tickers[0]["fundingRate"]) * 100
                    logger.info(f"✅ Got Bybit funding rate for {token}: {funding_rate:.4f}%")
                    
        except Exception as e:
            logger.debug(f"Bybit derivatives failed for {token}: {e}")
    
    # Strategy 3: Use synthetic data if all APIs fail
    if funding_rate == 0.0 and open_interest == 0.0:
        # Generate reasonable synthetic values based on token
        synthetic_funding = {
            "BTC": 0.01,
            "SOL": 0.05,
            "DOGE": 0.02,
            "FART": 0.10,  # Higher for meme coins
        }
        
        synthetic_oi = {
            "BTC": 50000000,
            "SOL": 10000000,
            "DOGE": 5000000,
            "FART": 1000000,
        }
        
        funding_rate = synthetic_funding.get(token, 0.03)
        open_interest = synthetic_oi.get(token, 1000000)
        
        logger.warning(f"⚠️  Using synthetic derivatives data for {token}: funding={funding_rate:.4f}%, OI={open_interest:,.0f}")
    
    return {
        "funding_rate": funding_rate,
        "open_interest": open_interest
    }
async def build_token_features(token: str) -> pd.DataFrame:
    """Build comprehensive features with improved error handling and data validation"""
    
    logger.info(f"Starting feature building for {token}")
    
    # Get spot market data from CoinGecko
    df = await get_coingecko_spot_data(token, days=30)
    
    if df.empty or len(df) < 20:
        logger.error(f"Insufficient spot data for {token}: {len(df)} rows")
        return pd.DataFrame()

    logger.info(f"Building features for {token} with {len(df)} data points")

    # Calculate basic features with error handling
    try:
        df['close_return'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Ensure we have valid window size
        window_size = min(15, max(5, len(df) // 3))
        df['return_15m'] = df['close'].rolling(window_size, min_periods=1).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else 0.0
        )
        df['volume_15m'] = df['volume'].rolling(window_size, min_periods=1).sum()
    except Exception as e:
        logger.warning(f"Error calculating basic features for {token}: {e}")
        df['close_return'] = 0.0
        df['volume_change'] = 0.0
        df['return_15m'] = 0.0
        df['volume_15m'] = df.get('volume', 0.0)

    # Get derivatives data with timeout
    try:
        derivatives_task = asyncio.wait_for(fetch_derivatives_data(token), timeout=30)
        derivatives_data = await derivatives_task
    except asyncio.TimeoutError:
        logger.warning(f"Derivatives data timeout for {token}")
        derivatives_data = {"funding_rate": 0.0, "open_interest": 0.0}
    except Exception as e:
        logger.warning(f"Error fetching derivatives data for {token}: {e}")
        derivatives_data = {"funding_rate": 0.0, "open_interest": 0.0}
    
    df['funding_rate'] = derivatives_data['funding_rate']
    df['open_interest'] = derivatives_data['open_interest']

    # Get sentiment with error handling
    try:
        sentiment = get_token_sentiment(token)
        df['sentiment'] = sentiment if isinstance(sentiment, (int, float)) else 0.0
    except Exception as e:
        logger.warning(f"Error fetching sentiment for {token}: {e}")
        df['sentiment'] = 0.0

    # Technical indicators with robust error handling
    try:
        if len(df) >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            df['rsi_14'] = rsi.rsi()
        else:
            df['rsi_14'] = 50.0
            
        macd = ta.trend.MACD(df['close'])
        df['macd_diff'] = macd.macd_diff()
        
        # Additional technical indicators
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
    except Exception as e:
        logger.warning(f"Error calculating technical indicators for {token}: {e}")
        df['rsi_14'] = 50.0
        df['macd_diff'] = 0.0
        df['bb_width'] = 0.0

    # Target and labels with validation
    try:
        if len(df) > LABEL_HORIZON:
            df['target'] = df['close'].shift(-LABEL_HORIZON) / df['close'] - 1
            df['label'] = df['target'].apply(lambda x: 1 if x > 0.001 else -1 if x < -0.001 else 0)
            df = df[:-LABEL_HORIZON]  # Remove rows without future data
        else:
            df['target'] = 0.0
            df['label'] = 0
    except Exception as e:
        logger.warning(f"Error calculating targets for {token}: {e}")
        df['target'] = 0.0
        df['label'] = 0

    # Fill NaN values and validate data
    df = df.fillna(0.0)
    
    # Data validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0)
    
    logger.info(f"Features built for {token}: {len(df)} rows, funding_rate={derivatives_data['funding_rate']:.4f}%, OI={derivatives_data['open_interest']:.0f}")
    
    return df

async def save_token_data(token: str):
    """Save comprehensive token data with validation"""
    try:
        df = await build_token_features(token)
        if df.empty:
            logger.error(f"No data to save for {token}")
            return False
        
        # Validate data quality
        if len(df) < 10:
            logger.warning(f"Insufficient data quality for {token}: only {len(df)} rows")
            return False
        
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{token}_features.csv")
        df.to_csv(path)
        
        # Log data summary
        logger.info(f"Saved {path} with {len(df)} rows")
        logger.info(f"Data summary for {token}: price_range=${df['close'].min():.4f}-${df['close'].max():.4f}, avg_volume=${df['volume'].mean():.0f}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data for {token}: {e}")
        return False

async def test_network_connectivity():
    """Test network connectivity to crypto APIs"""
    logger.info("🔍 Testing network connectivity...")
    
    test_urls = [
        ("CoinGecko", "https://api.coingecko.com/api/v3/ping"),
        ("Binance", "https://fapi.binance.com/fapi/v1/ping"),
        ("Bybit", "https://api.bybit.com/v5/announcements"),
    ]
    
    for name, url in test_urls:
        try:
            result = await make_request_with_retry(url, max_retries=1, service=name.lower())
            status = "✅ Connected" if result else "❌ Failed"
            logger.info(f"{name}: {status}")
        except Exception as e:
            logger.info(f"{name}: ❌ Error - {e}")

async def main():
    """Main execution function with improved error handling"""
    # Test network connectivity first
    await test_network_connectivity()
    logger.info("Starting enhanced multi-source crypto data fetching...")
    logger.info(f"Supported tokens: {SUPPORTED_TOKENS}")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, token in enumerate(SUPPORTED_TOKENS):
        try:
            logger.info(f"Processing {token} ({i+1}/{len(SUPPORTED_TOKENS)})...")
            token_start_time = time.time()
            
            success = await save_token_data(token)
            
            token_duration = time.time() - token_start_time
            
            if success:
                successful += 1
                logger.info(f"✅ {token} completed in {token_duration:.1f}s")
            else:
                failed += 1
                logger.error(f"❌ {token} failed after {token_duration:.1f}s")
                
            # Rate limiting between tokens - increased for better reliability
            await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"Critical error processing {token}: {e}")
            failed += 1
    
    total_duration = time.time() - start_time
    logger.info(f"Multi-source data fetching complete in {total_duration:.1f}s")
    logger.info(f"Results: {successful} successful, {failed} failed")
    
    # Enhanced summary of data sources used
    logger.info("=== DATA SOURCES SUMMARY ===")
    logger.info("✓ Spot prices & volume: CoinGecko API")
    logger.info("✓ Funding rates & OI: Binance/Bybit derivatives (with fallbacks)")
    logger.info("✓ Technical indicators: TA-Lib (RSI, MACD, Bollinger Bands)")
    logger.info("✓ Sentiment: Custom sentiment function")
    logger.info("✓ Rate limiting: Adaptive with exponential backoff")
    logger.info("✓ Error handling: Multi-tier fallbacks and retries")

if __name__ == '__main__':
    # Run the main function with asyncio
    logging.getLogger().setLevel(logging.INFO)  # Set global logging level
    asyncio.run(main())