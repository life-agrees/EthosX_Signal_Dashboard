# sentiment.py

import os
import re
import time
import tweepy
from textblob import TextBlob
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Tuple, Optional

# Environment variables
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER_TOKEN:
    raise RuntimeError("Please set the TWITTER_BEARER_TOKEN environment variable.")

twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Enhanced rate limiting settings
MAX_RETRIES = 2
INITIAL_BACKOFF = 15       # seconds
CACHE_REFRESH_MINUTES = 30  # Increased from 5 to 30 minutes
REQUEST_DELAY = 60         # Minimum 60 seconds between any API calls
MAX_REQUESTS_PER_WINDOW = 250  # Conservative limit (Twitter allows 300)
WINDOW_DURATION = 15 * 60  # 15 minutes in seconds

# Global rate limiting state
class RateLimitManager:
    def __init__(self):
        self.request_times = []
        self.last_request_time = 0
        
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        now = time.time()
        
        # Remove requests older than 15 minutes
        self.request_times = [t for t in self.request_times if now - t < WINDOW_DURATION]
        
        # Check if we're under the request limit
        if len(self.request_times) >= MAX_REQUESTS_PER_WINDOW:
            return False
            
        # Check minimum delay between requests
        if now - self.last_request_time < REQUEST_DELAY:
            return False
            
        return True
    
    def record_request(self):
        """Record that we made a request"""
        now = time.time()
        self.request_times.append(now)
        self.last_request_time = now
    
    def time_until_next_request(self) -> float:
        """Calculate how long to wait before next request"""
        now = time.time()
        
        # Time until minimum delay is satisfied
        delay_wait = max(0, REQUEST_DELAY - (now - self.last_request_time))
        
        # Time until we're under rate limit
        if len(self.request_times) >= MAX_REQUESTS_PER_WINDOW:
            oldest_request = min(self.request_times)
            rate_wait = max(0, WINDOW_DURATION - (now - oldest_request))
        else:
            rate_wait = 0
            
        return max(delay_wait, rate_wait)

rate_limiter = RateLimitManager()

# Enhanced caching
# token -> (timestamp_last_fetched, sentiment_score, fetch_count)
sentiment_cache: Dict[str, Tuple[datetime, float, int]] = {}

def clean_text(text: str) -> str:
    """Clean tweet text for sentiment analysis"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?\\'\\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def get_sentiment(text: str) -> float:
    """Get sentiment score from text using TextBlob"""
    return TextBlob(text).sentiment.polarity

def fetch_twitter_sentiment_sync(token_symbol: str, max_tweets: int = 10) -> float:
    """Synchronous version of Twitter sentiment fetching with better rate limiting"""
    query = f"{token_symbol} OR {token_symbol}USDC -is:retweet lang:en"
    
    # Wait if we need to respect rate limits
    wait_time = rate_limiter.time_until_next_request()
    if wait_time > 0:
        print(f"[INFO] Waiting {wait_time:.1f}s for rate limit for {token_symbol}")
        time.sleep(wait_time)
    
    if not rate_limiter.can_make_request():
        print(f"[WARN] Rate limit exceeded for {token_symbol}, returning cached/default value")
        return 0.0
    
    backoff = INITIAL_BACKOFF
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.record_request()
            resp = twitter_client.search_recent_tweets(query=query, max_results=max_tweets)
            tweets = resp.data or []
            
            if not tweets:
                print(f"[INFO] No tweets found for {token_symbol}")
                return 0.0

            scores = []
            for tweet in tweets:
                cleaned = clean_text(tweet.text)
                sentiment_score = get_sentiment(cleaned)
                scores.append(sentiment_score)
            
            final_score = float(sum(scores) / len(scores)) if scores else 0.0
            print(f"[INFO] Fetched sentiment for {token_symbol}: {final_score:.3f} from {len(tweets)} tweets")
            return final_score

        except tweepy.errors.TooManyRequests:
            print(f"[WARN] Rate limited. Sleeping {backoff}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(backoff)
            backoff *= 2

        except Exception as e:
            print(f"[ERROR] Error fetching sentiment for {token_symbol}: {e}")
            return 0.0

    # If still rate limited after retries, wait longer
    print(f"[WARN] Exceeded retries for {token_symbol}. Will wait before next attempt.")
    return 0.0

async def fetch_twitter_sentiment_async(token_symbol: str, max_tweets: int = 10) -> float:
    """Async version of Twitter sentiment fetching with better rate limiting"""
    query = f"{token_symbol} OR {token_symbol}USDC -is:retweet lang:en"
    
    # Wait if we need to respect rate limits
    wait_time = rate_limiter.time_until_next_request()
    if wait_time > 0:
        print(f"[INFO] Waiting {wait_time:.1f}s for rate limit for {token_symbol}")
        await asyncio.sleep(wait_time)
    
    if not rate_limiter.can_make_request():
        print(f"[WARN] Rate limit exceeded for {token_symbol}, returning cached/default value")
        return 0.0
    
    backoff = INITIAL_BACKOFF
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.record_request()
            resp = twitter_client.search_recent_tweets(query=query, max_results=max_tweets)
            tweets = resp.data or []
            
            if not tweets:
                print(f"[INFO] No tweets found for {token_symbol}")
                return 0.0

            scores = []
            for tweet in tweets:
                cleaned = clean_text(tweet.text)
                sentiment_score = get_sentiment(cleaned)
                scores.append(sentiment_score)
            
            final_score = float(sum(scores) / len(scores)) if scores else 0.0
            print(f"[INFO] Fetched sentiment for {token_symbol}: {final_score:.3f} from {len(tweets)} tweets")
            return final_score

        except tweepy.errors.TooManyRequests:
            print(f"[WARN] Rate limited. Sleeping {backoff}s (attempt {attempt}/{MAX_RETRIES})")
            await asyncio.sleep(backoff)
            backoff *= 2

        except Exception as e:
            print(f"[ERROR] Error fetching sentiment for {token_symbol}: {e}")
            return 0.0

    # If still rate limited after retries, wait longer
    print(f"[WARN] Exceeded retries for {token_symbol}. Will wait before next attempt.")
    return 0.0

def fetch_twitter_sentiment(token_symbol: str, max_tweets: int = 10) -> float:
    """Synchronous wrapper for async function"""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we're in a loop, we can't use run_until_complete
        # Instead, we'll use the sync version
        return fetch_twitter_sentiment_sync(token_symbol, max_tweets)
    except RuntimeError:
        # No event loop running, safe to create one
        return asyncio.run(fetch_twitter_sentiment_async(token_symbol, max_tweets))

def get_token_sentiment(token_symbol: str) -> float:
    """Get sentiment with enhanced caching and rate limiting"""
    now = datetime.utcnow()
    cache_entry = sentiment_cache.get(token_symbol)

    # Check cache first
    if cache_entry:
        last_time, cached_score, fetch_count = cache_entry
        time_since_fetch = now - last_time
        
        # Use cached value if within refresh window
        if time_since_fetch < timedelta(minutes=CACHE_REFRESH_MINUTES):
            print(f"[INFO] Using cached sentiment for {token_symbol}: {cached_score:.3f}")
            return cached_score
        
        # For frequently failing tokens, extend cache time
        if fetch_count > 5 and time_since_fetch < timedelta(hours=1):
            print(f"[INFO] Using extended cache for {token_symbol}: {cached_score:.3f}")
            return cached_score

    # Check if we can make a request
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.time_until_next_request()
        print(f"[WARN] Rate limit active. Next request possible in {wait_time:.1f}s")
        
        # Return cached value if available, otherwise return neutral
        if cache_entry:
            return cache_entry[1]
        return 0.0

    # Fetch fresh sentiment
    try:
        new_score = fetch_twitter_sentiment(token_symbol)
        fetch_count = cache_entry[2] + 1 if cache_entry else 1
        sentiment_cache[token_symbol] = (now, new_score, fetch_count)
        return new_score
    except Exception as e:
        print(f"[ERROR] Failed to fetch sentiment for {token_symbol}: {e}")
        # Return cached value if available
        if cache_entry:
            return cache_entry[1]
        return 0.0

def get_sentiment_status() -> Dict:
    """Get status of sentiment system"""
    return {
        "cached_tokens": list(sentiment_cache.keys()),
        "requests_in_window": len(rate_limiter.request_times),
        "time_until_next_request": rate_limiter.time_until_next_request(),
        "can_make_request": rate_limiter.can_make_request()
    }

# Batch processing for multiple tokens
async def get_multiple_sentiments(tokens: list, delay_between: int = 60) -> Dict[str, float]:
    """Get sentiment for multiple tokens with proper delays"""
    results = {}
    
    for i, token in enumerate(tokens):
        if i > 0:  # Add delay between tokens (except first)
            await asyncio.sleep(delay_between)
        
        results[token] = get_token_sentiment(token)
    
    return results

if __name__ == "__main__":
    tokens = ["BTC", "SOL", "DOGE", "FART"]
    
    print("Fetching sentiments with rate limiting...")
    print(f"Status: {get_sentiment_status()}")
    
    # Use synchronous processing for direct script execution
    for token in tokens:
        sentiment = get_token_sentiment(token)
        print(f"{token} sentiment: {sentiment:.3f}")
        # Add delay between tokens to respect rate limits
        if token != tokens[-1]:  # Don't sleep after last token
            print(f"Waiting 60 seconds before next token...")
            time.sleep(60)