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
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import warnings
import json
from urllib.parse import urljoin
import hashlib
import random
warnings.filterwarnings("ignore")

from .config import (
    SUPPORTED_TOKENS,
    PERP_SYMBOLS,
    COINGECKO_SYMBOLS,
    LABEL_HORIZON,
    DATA_DIR,
)
from .sentiment import get_token_sentiment

# API configurations
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_BASE_URL = "https://fapi.binance.com/fapi/v1"
BYBIT_BASE_URL = "https://api.bybit.com/v5"

# Alternative endpoints for better reliability (from File 1)
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
    """Construct full URLs for a service's path using its fallbacks."""
    bases = ALTERNATIVE_ENDPOINTS.get(service, [])
    return [urljoin(base, path) for base in bases]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Symbol mappings for different exchanges
COINGECKO = COINGECKO_SYMBOLS 
BINANCE_PERP_SYMBOLS = PERP_SYMBOLS
BYBIT_PERP_SYMBOLS = PERP_SYMBOLS

# User agents rotation (from File 1)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

# ===== ENHANCED RATE LIMITING IMPLEMENTATION =====

class RateLimitTier(Enum):
    FREE = "free"
    PAID = "paid"
    PREMIUM = "premium"

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    burst_limit: int
    backoff_multiplier: float
    max_backoff: int
    tier: RateLimitTier

class AdaptiveRateLimiter:
    """Enhanced rate limiter with dynamic adjustment and tier management"""
    
    def __init__(self):
        self.configs = {
            "coingecko": {
                RateLimitTier.FREE: RateLimitConfig(30, 5, 1.5, 300, RateLimitTier.FREE),
                RateLimitTier.PAID: RateLimitConfig(100, 20, 1.2, 120, RateLimitTier.PAID),
            },
            "binance": {
                RateLimitTier.FREE: RateLimitConfig(1200, 50, 2.0, 300, RateLimitTier.FREE),
                RateLimitTier.PAID: RateLimitConfig(2400, 100, 1.5, 180, RateLimitTier.PAID),
            },
            "bybit": {
                RateLimitTier.FREE: RateLimitConfig(600, 30, 1.8, 240, RateLimitTier.FREE),
                RateLimitTier.PAID: RateLimitConfig(1200, 60, 1.3, 120, RateLimitTier.PAID),
            }
        }
        
        self.current_tiers = {
            "coingecko": RateLimitTier.FREE,
            "binance": RateLimitTier.FREE,
            "bybit": RateLimitTier.FREE,
        }
        
        self.limiters = {}
        self.rate_limit_history = {}
        self.adaptive_delays = {}
        self.last_request_time = {}
        
        self._initialize_limiters()
    
    def _initialize_limiters(self):
        """Initialize rate limiters based on current tiers"""
        for service, tier in self.current_tiers.items():
            config = self.configs[service][tier]
            self.limiters[service] = AsyncLimiter(
                config.requests_per_minute, 
                60
            )
            self.adaptive_delays[service] = 0
            self.rate_limit_history[service] = []
    
    async def acquire(self, service: str, endpoint: str = "default") -> bool:
        """Acquire rate limit token with adaptive delays"""
        if service not in self.limiters:
            logger.warning(f"No rate limiter configured for {service}")
            return True
        
        # Apply adaptive delay if needed
        if service in self.adaptive_delays and self.adaptive_delays[service] > 0:
            logger.debug(f"Applying adaptive delay of {self.adaptive_delays[service]}s for {service}")
            await asyncio.sleep(self.adaptive_delays[service])
        
        # Wait for rate limit token
        await self.limiters[service].acquire()
        
        # Track request timing
        self.last_request_time[service] = time.time()
        
        return True
    
    def record_rate_limit(self, service: str, retry_after: Optional[int] = None):
        """Record a rate limit hit and adjust strategy"""
        current_time = time.time()
        
        # Add to history
        if service not in self.rate_limit_history:
            self.rate_limit_history[service] = []
        
        self.rate_limit_history[service].append(current_time)
        
        # Clean old entries (keep last hour)
        self.rate_limit_history[service] = [
            t for t in self.rate_limit_history[service] 
            if current_time - t < 3600
        ]
        
        # Increase adaptive delay
        config = self.configs[service][self.current_tiers[service]]
        
        if retry_after:
            self.adaptive_delays[service] = min(retry_after, config.max_backoff)
        else:
            current_delay = self.adaptive_delays.get(service, 0)
            new_delay = min(
                max(1, current_delay * config.backoff_multiplier),
                config.max_backoff
            )
            self.adaptive_delays[service] = new_delay
        
        logger.warning(f"Rate limit hit for {service}. New adaptive delay: {self.adaptive_delays[service]}s")
        
        # If too many rate limits, consider downgrading tier
        recent_limits = len([
            t for t in self.rate_limit_history[service]
            if current_time - t < 600  # last 10 minutes
        ])
        
        if recent_limits >= 3:
            self._adjust_tier(service, more_conservative=True)
    
    def record_success(self, service: str):
        """Record successful request and potentially reduce delays"""
        current_time = time.time()
        
        # Gradually reduce adaptive delay on success
        if service in self.adaptive_delays and self.adaptive_delays[service] > 0:
            self.adaptive_delays[service] = max(0, self.adaptive_delays[service] * 0.9)
        
        # If we haven't hit rate limits recently, consider upgrading tier
        if service in self.rate_limit_history:
            recent_limits = len([
                t for t in self.rate_limit_history[service]
                if current_time - t < 1800  # last 30 minutes
            ])
            
            if recent_limits == 0 and self.adaptive_delays[service] == 0:
                self._adjust_tier(service, more_conservative=False)
    
    def _adjust_tier(self, service: str, more_conservative: bool):
        """Adjust rate limiting tier based on performance"""
        if service not in self.configs:
            return
        
        available_tiers = list(self.configs[service].keys())
        current_tier = self.current_tiers[service]
        current_index = available_tiers.index(current_tier)
        
        if more_conservative and current_index > 0:
            # Move to more conservative tier
            new_tier = available_tiers[current_index - 1]
            self.current_tiers[service] = new_tier
            logger.info(f"Downgrading {service} rate limit tier to {new_tier.value}")
            
        elif not more_conservative and current_index < len(available_tiers) - 1:
            # Move to more aggressive tier
            new_tier = available_tiers[current_index + 1]
            self.current_tiers[service] = new_tier
            logger.info(f"Upgrading {service} rate limit tier to {new_tier.value}")
        
        # Reinitialize limiter with new tier
        self._initialize_limiters()
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get current rate limiting statistics"""
        stats = {}
        current_time = time.time()
        
        for service in self.limiters:
            recent_limits = len([
                t for t in self.rate_limit_history.get(service, [])
                if current_time - t < 3600
            ])
            
            stats[service] = {
                "tier": self.current_tiers[service].value,
                "adaptive_delay": self.adaptive_delays.get(service, 0),
                "rate_limits_last_hour": recent_limits,
                "last_request": self.last_request_time.get(service, 0)
            }
        
        return stats

adaptive_rate_limiter = AdaptiveRateLimiter()


# In-memory cache with TTL 
class DataCache:
    def __init__(self):
        self.cache = {}
        self.cache_times = {}
        self.ttl = 300  # 5 minutes TTL
    
    def get_cache_key(self, url: str, params: dict = None) -> str:
        """Generate cache key from URL and parameters"""
        key_data = f"{url}_{params or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache:
            if time.time() - self.cache_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.cache_times[key]
        return None
    
    def set(self, key: str, data: Any):
        """Set cached data with timestamp"""
        self.cache[key] = data
        self.cache_times[key] = time.time()
    
    def clear_expired(self):
        """Clear all expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, cache_time in self.cache_times.items()
            if current_time - cache_time >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.cache_times[key]

# Global cache instance
cache = DataCache()

# Circuit breaker for API endpoints 
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.blocked_until = {}
    
    def is_blocked(self, service: str) -> bool:
        """Check if service is currently blocked"""
        if service in self.blocked_until:
            if time.time() < self.blocked_until[service]:
                return True
            else:
                # Recovery time passed, reset
                del self.blocked_until[service]
                self.failure_count[service] = 0
        return False
    
    def record_failure(self, service: str):
        """Record a failure for the service"""
        self.failure_count[service] = self.failure_count.get(service, 0) + 1
        self.last_failure_time[service] = time.time()
        
        if self.failure_count[service] >= self.failure_threshold:
            self.blocked_until[service] = time.time() + self.recovery_timeout
            logger.warning(f"🚨 Circuit breaker activated for {service} - blocked for {self.recovery_timeout}s")
    
    def record_success(self, service: str):
        """Record a successful request"""
        if service in self.failure_count:
            self.failure_count[service] = 0

# Global circuit breaker
circuit_breaker = CircuitBreaker()

def get_headers():
    """Get headers with user agent rotation"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
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

async def make_request_with_retry(url: str, params: dict = None, max_retries: int = 3, service: str = "default") -> Optional[dict]:
    """Enhanced request function with caching, circuit breaker, and retry logic - merged from both files"""
    
    # Check cache first 
    cache_key = cache.get_cache_key(url, params)
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.debug(f"Cache hit for {url}")
        return cached_data
    
    # Check circuit breaker status
    if circuit_breaker.is_blocked(service):
        logger.warning(f"Circuit breaker active for {service}, skipping request to {url}")
        return None
    

    await adaptive_rate_limiter.acquire(service)
    # Enhanced timeout and connection settings
    timeout = aiohttp.ClientTimeout(total=45, connect=20)
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, 
                headers=get_headers(),
                connector=aiohttp.TCPConnector(ssl=False, limit=10, limit_per_host=5)
            ) as session:
                async with session.get(url, params=params) as response:
                    
                    if response.status == 429:  # Rate limited
                        # Extract retry-after header if available
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                retry_after = int(retry_after)
                            except ValueError:
                                retry_after = None

                        # Record rate limit hit
                        adaptive_rate_limiter.record_rate_limit(service, retry_after)

                        wait_time = min(60 * (2 ** attempt), 240)  # Conservative backoff
                        logger.warning(f"Rate limited on {url}. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        circuit_breaker.record_failure(service)
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status == 403:  # Forbidden
                        logger.error(f"Forbidden (403) for {url} - possible IP block")
                        circuit_breaker.record_failure(service)
                        await asyncio.sleep(120)  # Longer wait for 403
                        continue
                    
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        circuit_breaker.record_failure(service)
                        if response.status >= 500:  # Server errors
                            await asyncio.sleep(30)
                        continue
                    
                    data = await response.json()
                    
                    # Success! Cache the result and record success
                    cache.set(cache_key, data)
                    circuit_breaker.record_success(service)
                    
                    return data
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url} (attempt {attempt + 1}/{max_retries})")
            circuit_breaker.record_failure(service)
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"Connection error for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            circuit_breaker.record_failure(service)
        except Exception as e:
            logger.warning(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            circuit_breaker.record_failure(service)
        
        if attempt < max_retries - 1:
            wait_time = min(30 * (2 ** attempt), 120)  # Conservative backoff
            logger.info(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    circuit_breaker.record_failure(service)
    logger.error(f"All retry attempts failed for {url}")
    return None

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
    
    # Conservative delay between CoinGecko requests
    #await asyncio.sleep(5)  # Extra delay to avoid rate limits
    
    # Get volume data
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

# Individual exchange functions from File 1 - restored
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
    """Enhanced derivatives data fetch combining both approaches"""
    
    logger.info(f"Fetching derivatives data for {token}")
    
    # Check cache first (from File 2)
    cache_key = f"derivatives_{token}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached derivatives data for {token}")
        return cached_data
    
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
    
    result = {
        "funding_rate": funding_rate,
        "open_interest": open_interest
    }
    
    # Cache the result for 5 minutes
    cache.set(cache_key, result)
    
    return result

async def build_token_features(token: str) -> pd.DataFrame:
    """Build comprehensive features with improved error handling and data validation"""
    
    logger.info(f"Starting feature building for {token}")
    
    # Clear expired cache entries
    cache.clear_expired()
    
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
        derivatives_task = asyncio.wait_for(fetch_derivatives_data(token), timeout=45)
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
        else:
            df['bb_width'] = 0.0
        
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
    """Test network connectivity to crypto APIs - restored from File 1"""
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
                
            # Conservative delay between tokens to prevent rate limiting
            await asyncio.sleep(10)  # Conservative delay from File 2
            
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
    logger.info("✓ Caching: 5-minute TTL with automatic cleanup")
    logger.info("✓ Circuit breaker: Automatic API failure protection")

if __name__ == '__main__':
    # Run the main function with asyncio
    logging.getLogger().setLevel(logging.INFO)  # Set global logging level
    asyncio.run(main())