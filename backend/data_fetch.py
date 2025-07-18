# data_fetch_ccxt.py - Enhanced version with CCXT integration

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import ta
import numpy as np
import asyncio
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union
import warnings
import json
import hashlib
import random
import ccxt.pro as ccxt
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

from config import (
    SUPPORTED_TOKENS,
    PERP_SYMBOLS,
    COINGECKO_SYMBOLS,
    LABEL_HORIZON,
    DATA_DIR,
)
from sentiment import get_token_sentiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ===== ENHANCED RATE LIMITING WITH CCXT INTEGRATION =====

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
    """Enhanced rate limiter that works alongside CCXT's built-in rate limiting"""
    
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
        
        self.rate_limit_history = {}
        self.adaptive_delays = {}
        self.last_request_time = {}
        self.failure_counts = {}
        
    async def acquire(self, service: str, endpoint: str = "default") -> bool:
        """Acquire rate limit token with adaptive delays"""
        
        # Apply adaptive delay if needed
        if service in self.adaptive_delays and self.adaptive_delays[service] > 0:
            logger.debug(f"Applying adaptive delay of {self.adaptive_delays[service]}s for {service}")
            await asyncio.sleep(self.adaptive_delays[service])
        
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
                max(5, current_delay * config.backoff_multiplier),
                config.max_backoff
            )
            self.adaptive_delays[service] = new_delay
        
        logger.warning(f"Rate limit hit for {service}. New adaptive delay: {self.adaptive_delays[service]}s")
    
    def record_success(self, service: str):
        """Record successful request and potentially reduce delays"""
        # Gradually reduce adaptive delay on success
        if service in self.adaptive_delays and self.adaptive_delays[service] > 0:
            self.adaptive_delays[service] = max(0, self.adaptive_delays[service] * 0.9)
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get current rate limiting statistics"""
        stats = {}
        current_time = time.time()
        
        for service in ["coingecko", "binance", "bybit"]:
            recent_limits = len([
                t for t in self.rate_limit_history.get(service, [])
                if current_time - t < 3600
            ])
            
            stats[service] = {
                "tier": self.current_tiers[service].value,
                "adaptive_delay": self.adaptive_delays.get(service, 0),
                "rate_limits_last_hour": recent_limits,
                "last_request": self.last_request_time.get(service, 0),
                "failure_count": self.failure_counts.get(service, 0)
            }
        
        return stats

# ===== CACHING SYSTEM =====

class DataCache:
    def __init__(self):
        self.cache = {}
        self.cache_times = {}
        self.ttl = 300  # 5 minutes TTL
    
    def get_cache_key(self, service: str, method: str, symbol: str = "", params: dict = None) -> str:
        """Generate cache key from service, method, symbol and parameters"""
        key_data = f"{service}_{method}_{symbol}_{params or {}}"
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

# ===== CIRCUIT BREAKER =====

class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=600):
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

# ===== CCXT EXCHANGE MANAGER =====

class ExchangeManager:
    """Manages CCXT exchange instances with robust error handling"""
    
    def __init__(self, cache: DataCache, rate_limiter: AdaptiveRateLimiter, circuit_breaker: CircuitBreaker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.exchanges = {}
        self.coingecko_api = None
        self.initialize_exchanges()
    
    def initialize_exchanges(self):
        """Initialize CCXT exchange instances"""
        
        # Binance futures
        self.exchanges['binance'] = ccxt.binance({
            'sandbox': False,
            'options': {
                'defaultType': 'future',  # Use futures/derivatives
            },
            'rateLimit': 50,  # Conservative rate limiting
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # Bybit
        self.exchanges['bybit'] = ccxt.bybit({
            'sandbox': False,
            'options': {
                'defaultType': 'linear',  # Linear perpetual contracts
            },
            'rateLimit': 100,  # Conservative rate limiting
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        logger.info("✅ CCXT exchanges initialized")
    
    async def close_exchanges(self):
        """Close all exchange connections"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed {exchange_name} connection")
            except Exception as e:
                logger.warning(f"Error closing {exchange_name}: {e}")
    
    @asynccontextmanager
    async def get_exchange(self, exchange_name: str):
        """Context manager for exchange operations"""
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        try:
            yield exchange
        except Exception as e:
            self.circuit_breaker.record_failure(exchange_name)
            raise e
        finally:
            # Record success if no exception
            self.circuit_breaker.record_success(exchange_name)
    
    async def ccxt_request_with_retry(self, exchange_name: str, method: str, symbol: str = "", params: dict = None, max_retries: int = 3) -> Optional[Any]:
        """Make CCXT request with caching and retry logic"""
        
        # Check cache first
        cache_key = self.cache.get_cache_key(exchange_name, method, symbol, params)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {exchange_name}.{method}({symbol})")
            return cached_data
        
        # Check circuit breaker
        if self.circuit_breaker.is_blocked(exchange_name):
            logger.warning(f"Circuit breaker active for {exchange_name}")
            return None
        
        # Apply rate limiting
        await self.rate_limiter.acquire(exchange_name, method)
        
        for attempt in range(max_retries):
            try:
                async with self.get_exchange(exchange_name) as exchange:
                    
                    # Dynamic method calling
                    if hasattr(exchange, method):
                        method_func = getattr(exchange, method)
                        
                        if symbol and params:
                            result = await method_func(symbol, params)
                        elif symbol:
                            result = await method_func(symbol)
                        elif params:
                            result = await method_func(params)
                        else:
                            result = await method_func()
                        
                        # Cache successful result
                        self.cache.set(cache_key, result)
                        self.rate_limiter.record_success(exchange_name)
                        
                        return result
                    else:
                        logger.error(f"Method {method} not found on {exchange_name}")
                        return None
                        
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded for {exchange_name}.{method} (attempt {attempt + 1}/{max_retries})")
                self.rate_limiter.record_rate_limit(exchange_name)
                if attempt < max_retries - 1:
                    wait_time = min(120 * (2 ** attempt), 600)
                    await asyncio.sleep(wait_time)
                
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                logger.warning(f"Network error for {exchange_name}.{method} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(30 * (2 ** attempt), 120)
                    await asyncio.sleep(wait_time)
                
            except ccxt.BaseError as e:
                logger.warning(f"CCXT error for {exchange_name}.{method}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Unexpected error for {exchange_name}.{method}: {e}")
                break
        
        logger.error(f"All retry attempts failed for {exchange_name}.{method}")
        return None

# ===== COINGECKO INTEGRATION =====

class CoinGeckoAPI:
    """CoinGecko API integration with caching and rate limiting"""
    
    def __init__(self, cache: DataCache, rate_limiter: AdaptiveRateLimiter, circuit_breaker: CircuitBreaker):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.base_url = "https://api.coingecko.com/api/v3"
        
    async def get_ohlcv_data(self, token: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLCV data from CoinGecko"""
        if token not in COINGECKO_SYMBOLS:
            logger.warning(f"Token {token} not found in CoinGecko symbols")
            return pd.DataFrame()
        
        coin_id = COINGECKO_SYMBOLS[token]
        
        # Check cache
        cache_key = self.cache.get_cache_key("coingecko", "ohlcv", coin_id, {"days": days})
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for CoinGecko OHLCV {token}")
            return cached_data
        
        # Check circuit breaker
        if self.circuit_breaker.is_blocked("coingecko"):
            logger.warning("Circuit breaker active for CoinGecko")
            return pd.DataFrame()
        
        try:
            await self.rate_limiter.acquire("coingecko", "ohlcv")
            
            # Get OHLC data
            ohlc_url = f"{self.base_url}/coins/{coin_id}/ohlc"
            ohlc_params = {"vs_currency": "usd", "days": str(days)}
            
            # Here you would use aiohttp as in your original code
            # For brevity, I'll show the structure
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(ohlc_url, params=ohlc_params) as response:
                    if response.status == 200:
                        ohlc_data = await response.json()
                        
                        if not ohlc_data or len(ohlc_data) < 10:
                            logger.error(f"Insufficient OHLC data for {token}")
                            return pd.DataFrame()
                        
                        df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        
                        for col in ["open", "high", "low", "close"]:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Get volume data
                        await asyncio.sleep(5)  # Rate limit spacing
                        
                        market_url = f"{self.base_url}/coins/{coin_id}/market_chart"
                        market_params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
                        
                        async with session.get(market_url, params=market_params) as vol_response:
                            if vol_response.status == 200:
                                market_data = await vol_response.json()
                                
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
                                    # Estimate volume if not available
                                    df["volume"] = (df["high"] - df["low"]) / df["close"] * 1000000
                            else:
                                df["volume"] = (df["high"] - df["low"]) / df["close"] * 1000000
                        
                        df = df.dropna().set_index("timestamp")
                        
                        # Cache the result
                        self.cache.set(cache_key, df)
                        self.rate_limiter.record_success("coingecko")
                        
                        return df
                    elif response.status == 429:
                        logger.error(f"CoinGecko API error 429 for {token}")
                        retry_after = response.headers.get('Retry-After', 120)
                        self.rate_limiter.record_rate_limit("coingecko", int(retry_after))
                        return pd.DataFrame()
                    else:
                        logger.error(f"CoinGecko API error {response.status} for {token}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data for {token}: {e}")
            self.circuit_breaker.record_failure("coingecko")
            return pd.DataFrame()

# ===== ENHANCED DATA FETCHER =====

class EnhancedDataFetcher:
    """Main data fetcher combining CCXT with robust error handling"""
    
    def __init__(self):
        self.cache = DataCache()
        self.rate_limiter = AdaptiveRateLimiter()
        self.circuit_breaker = CircuitBreaker()
        self.exchange_manager = ExchangeManager(self.cache, self.rate_limiter, self.circuit_breaker)
        self.coingecko_api = CoinGeckoAPI(self.cache, self.rate_limiter, self.circuit_breaker)
        
    async def get_derivatives_data(self, token: str) -> Dict[str, float]:
        """Fetch derivatives data using CCXT"""
        
        logger.info(f"Fetching derivatives data for {token}")
        
        # Check cache first
        cache_key = f"derivatives_{token}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached derivatives data for {token}")
            return cached_data
        
        funding_rate = 0.0
        open_interest = 0.0
        
        # Try Binance first
        if token in PERP_SYMBOLS:
            symbol = PERP_SYMBOLS[token]
            
            # Get funding rate
            try:
                funding_data = await self.exchange_manager.ccxt_request_with_retry(
                    'binance', 'fetch_funding_rates', [symbol]  # Note: plural and array
                    )
                if funding_data and 'info' in funding_data:
                    funding_rate = float(funding_data['info'].get('lastFundingRate', 0)) * 100
                    logger.info(f"✅ Got Binance funding rate for {token}: {funding_rate:.4f}%")
            except Exception as e:
                logger.debug(f"Binance funding rate failed for {token}: {e}")
            
            # Get open interest
            try:
                # Note: CCXT doesn't have a standard fetch_open_interest method
                # So we'll use the custom method approach
                oi_data = await self.exchange_manager.ccxt_request_with_retry(
                    'binance', 'fetch_open_interests', [symbol]  # Note: plural and array
                    )
                if oi_data and 'info' in oi_data:
                    open_interest = float(oi_data['info'].get('openInterest', 0))
                    logger.info(f"✅ Got Binance open interest for {token}: {open_interest:,.0f}")
            except Exception as e:
                logger.debug(f"Binance open interest failed for {token}: {e}")
        
        # Try Bybit if Binance failed
        if funding_rate == 0.0 and token in PERP_SYMBOLS:
            symbol = PERP_SYMBOLS[token]
            
            try:
                await asyncio.sleep(2)  # Rate limit spacing
                
                funding_data = await self.exchange_manager.ccxt_request_with_retry(
                    'bybit', 'fetch_funding_rate', symbol
                )
                if funding_data and 'info' in funding_data:
                    funding_rate = float(funding_data['info'].get('fundingRate', 0)) * 100
                    logger.info(f"✅ Got Bybit funding rate for {token}: {funding_rate:.4f}%")
            except Exception as e:
                logger.debug(f"Bybit funding rate failed for {token}: {e}")
        
        # Fallback to synthetic data if all APIs fail
        if funding_rate == 0.0 and open_interest == 0.0:
            synthetic_funding = {
                "BTC": 0.01, "SOL": 0.05, "DOGE": 0.02, "FART": 0.10,
            }
            synthetic_oi = {
                "BTC": 50000000, "SOL": 10000000, "DOGE": 5000000, "FART": 1000000,
            }
            
            funding_rate = synthetic_funding.get(token, 0.03)
            open_interest = synthetic_oi.get(token, 1000000)
            
            logger.warning(f"⚠️ Using synthetic derivatives data for {token}")
        
        result = {
            "funding_rate": funding_rate,
            "open_interest": open_interest
        }
        
        # Cache the result
        self.cache.set(cache_key, result)
        return result
    
    async def build_token_features(self, token: str) -> pd.DataFrame:
        """Build comprehensive features with CCXT integration"""
        
        logger.info(f"Starting feature building for {token}")
        
        # Clear expired cache entries
        self.cache.clear_expired()
        
        # Get spot market data from CoinGecko
        df = await self.coingecko_api.get_ohlcv_data(token, days=30)
        
        if df.empty or len(df) < 20:
            logger.error(f"Insufficient spot data for {token}: {len(df)} rows")
            return pd.DataFrame()
        
        logger.info(f"Building features for {token} with {len(df)} data points")
        
        # Calculate basic features
        try:
            df['close_return'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
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
        
        # Get derivatives data
        try:
            derivatives_task = asyncio.wait_for(self.get_derivatives_data(token), timeout=60)
            derivatives_data = await derivatives_task
        except asyncio.TimeoutError:
            logger.warning(f"Derivatives data timeout for {token}")
            derivatives_data = {"funding_rate": 0.0, "open_interest": 0.0}
        except Exception as e:
            logger.warning(f"Error fetching derivatives data for {token}: {e}")
            derivatives_data = {"funding_rate": 0.0, "open_interest": 0.0}
        
        df['funding_rate'] = derivatives_data['funding_rate']
        df['open_interest'] = derivatives_data['open_interest']
        
        # Get sentiment
        try:
            sentiment = get_token_sentiment(token)
            df['sentiment'] = sentiment if isinstance(sentiment, (int, float)) else 0.0
        except Exception as e:
            logger.warning(f"Error fetching sentiment for {token}: {e}")
            df['sentiment'] = 0.0
        
        # Technical indicators
        try:
            if len(df) >= 14:
                rsi = ta.momentum.RSIIndicator(df['close'], window=14)
                df['rsi_14'] = rsi.rsi()
            else:
                df['rsi_14'] = 50.0
                
            macd = ta.trend.MACD(df['close'])
            df['macd_diff'] = macd.macd_diff()
            
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
        
        # Target and labels
        try:
            if len(df) > LABEL_HORIZON:
                df['target'] = df['close'].shift(-LABEL_HORIZON) / df['close'] - 1
                df['label'] = df['target'].apply(lambda x: 1 if x > 0.001 else -1 if x < -0.001 else 0)
                df = df[:-LABEL_HORIZON]
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
    
    async def save_token_data(self, token: str) -> bool:
        """Save token data with validation"""
        try:
            df = await self.build_token_features(token)
            if df.empty:
                logger.error(f"No data to save for {token}")
                return False
            
            if len(df) < 10:
                logger.warning(f"Insufficient data quality for {token}: only {len(df)} rows")
                return False
            
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, f"{token}_features.csv")
            df.to_csv(path)
            
            logger.info(f"Saved {path} with {len(df)} rows")
            logger.info(f"Data summary for {token}: price_range=${df['close'].min():.4f}-${df['close'].max():.4f}, avg_volume=${df['volume'].mean():.0f}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {token}: {e}")
            return False
    
    async def test_connectivity(self):
        """Test connectivity to all data sources"""
        logger.info("🔍 Testing connectivity...")
        
        # Test CCXT exchanges
        for exchange_name in self.exchange_manager.exchanges:
            try:
                async with self.exchange_manager.get_exchange(exchange_name) as exchange:
                    # Test with a simple method call
                    result = await exchange.load_markets()
                    if result:
                        logger.info(f"✅ {exchange_name}: Connected")
                    else:
                        logger.info(f"❌ {exchange_name}: Failed")
            except Exception as e:
                logger.info(f"❌ {exchange_name}: Failed - {e}")
        
        # Test CoinGecko
        try:
            test_df = await self.coingecko_api.get_ohlcv_data("BTC", days=1)
            if not test_df.empty:
                logger.info("✅ CoinGecko: Connected")
            else:
                logger.info("❌ CoinGecko: Failed")
        except Exception as e:
            logger.info(f"❌ CoinGecko: Failed - {e}")
        
        # Show rate limiting stats
        stats = self.rate_limiter.get_stats()
        logger.info(f"Rate limiting stats: {stats}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exchange_manager.close_exchanges()
        logger.info("Cleanup completed")

# ===== NETWORK CONNECTIVITY TEST =====

async def test_network_connectivity():
    """Test basic network connectivity"""
    import aiohttp
    
    test_urls = [
        "https://api.coingecko.com/api/v3/ping",
        "https://fapi.binance.com/fapi/v1/ping",
        "https://api.bybit.com/v5/market/time"
    ]
    
    logger.info("Testing network connectivity...")
    
    async with aiohttp.ClientSession() as session:
        for url in test_urls:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"✅ {url}: Connected")
                    else:
                        logger.info(f"❌ {url}: Status {response.status}")
            except Exception as e:
                logger.info(f"❌ {url}: {e}")


# ===== STANDALONE WRAPPER FUNCTIONS =====
# These functions provide a simple interface for api_server.py

async def get_coingecko_spot_data(token: str, days: int = 30) -> pd.DataFrame:
    """Standalone function to get CoinGecko spot data"""
    fetcher = EnhancedDataFetcher()
    try:
        result = await fetcher.coingecko_api.get_ohlcv_data(token, days)
        return result
    finally:
        await fetcher.cleanup()

async def fetch_derivatives_data(token: str) -> Dict[str, float]:
    """Standalone function to fetch derivatives data"""
    fetcher = EnhancedDataFetcher()
    try:
        result = await fetcher.get_derivatives_data(token)
        return result
    finally:
        await fetcher.cleanup()

async def build_token_features(token: str) -> pd.DataFrame:
    """Standalone function to build token features"""
    fetcher = EnhancedDataFetcher()
    try:
        result = await fetcher.build_token_features(token)
        return result
    finally:
        await fetcher.cleanup()

async def make_request_with_retry(url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
    """Standalone function for making HTTP requests with retry logic"""
    import aiohttp
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"All retry attempts failed for {url}")
    return None

# ===== MAIN EXECUTION =====

async def main():
    """Main execution function"""
    
    # Test network connectivity first
    await test_network_connectivity()
    
    logger.info("Starting enhanced multi-source crypto data fetching...")
    logger.info(f"Supported tokens: {SUPPORTED_TOKENS}")
    
    # Initialize the enhanced data fetcher
    fetcher = EnhancedDataFetcher()
    
    try:
        # Test connectivity
        await fetcher.test_connectivity()
        
        successful = 0
        failed = 0
        start_time = time.time()
        
        priority_tokens = ["BTC", "SOL", "DOGE", "FART"]  # BTC works best
        for i, token in enumerate(priority_tokens):
            try:
                logger.info(f"Processing {token} ({i+1}/{len(SUPPORTED_TOKENS)})...")
                token_start_time = time.time()
                
                success = await fetcher.save_token_data(token)
                token_duration = time.time() - token_start_time
                
                if success:
                    successful += 1
                    logger.info(f"✅ {token} completed in {token_duration:.1f}s")
                else:
                    failed += 1
                    logger.error(f"❌ {token} failed after {token_duration:.1f}s")
                
                # Conservative delay between tokens to prevent rate limiting
                await asyncio.sleep(30)  # Conservative delay
                
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
        
        # Show final rate limiting stats
        stats = fetcher.rate_limiter.get_stats()
        logger.info(f"Final rate limiting stats: {stats}")
        
    finally:
        # Clean up resources
        await fetcher.cleanup()

if __name__ == '__main__':
    # Run the main function with asyncio
    logging.getLogger().setLevel(logging.INFO)  # Set global logging level
    asyncio.run(main())