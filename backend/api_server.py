# api_server_bybit.py - Updated for Bybit API
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from httpx import HTTPStatusError
import joblib
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules - UPDATE THIS IMPORT
from models.predict import load_model, make_prediction
from sentiment import get_token_sentiment, get_multiple_sentiments, get_sentiment_status
from alerts import add_subscriber, get_subscribers, maybe_alert, CONFIDENCE_THRESHOLD
# CHANGED: Import from new Bybit data_fetch module
from data_fetch import fetch_funding_rate, fetch_open_interest, get_bybit_klines,BYBIT_BASE_URL,public_get
from config import SUPPORTED_TOKENS, PERP_SYMBOLS, LABEL_HORIZON

# Simplified Pydantic models - NO model selection exposed to users
class PredictionRequest(BaseModel):  
    token: str

class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    timestamp: str
    token: str
    # Removed 'model' field - users don't need to know
    features: Dict[str, float]

class SubscriptionRequest(BaseModel):
    email: EmailStr

class MarketDataResponse(BaseModel):
    price: float
    change_24h: float
    volume_24h: float
    open_interest: float
    funding_rate: float
    timestamp: str

class TechnicalIndicatorsResponse(BaseModel):
    rsi_14: float
    macd_diff: float
    sentiment: float
    funding_rate: float
    timestamp: str

# Smart Model Manager - handles all model complexity internally
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_priority = ["RandomForest", "XGBoost", #"LogisticRegression"
        ]
        self.primary_model = None
        self.primary_model_name = None
    
    def load_models(self):
        """Load models in priority order"""
        loaded_models = []
        
        # Try to load models in priority order
        model_paths = {
            "RandomForest": "models/model.pkl",
            "XGBoost": "models/xgb_model.pkl",
            #"LogisticRegression": "models/lr_model.pkl"
        }
        
        for model_name in self.model_priority:
            if model_name in model_paths:
                try:
                    self.models[model_name] = load_model(model_paths[model_name])
                    loaded_models.append(model_name)
                    print(f"✓ Loaded {model_name} model")
                    
                    # Set first successfully loaded model as primary
                    if self.primary_model is None:
                        self.primary_model = self.models[model_name]
                        self.primary_model_name = model_name
                        print(f"🎯 Using {model_name} as primary model")
                        
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
        
        if not self.models:
            print("⚠️  WARNING: No models loaded! Predictions will not be available.")
            return False
        
        print(f"📊 Model system ready: {len(loaded_models)} models loaded")
        return True
    
    def get_prediction_model(self):
        """Get the best model for predictions"""
        return self.primary_model, self.primary_model_name
    
    def is_available(self):
        """Check if prediction service is available"""
        return self.primary_model is not None
    
    def get_status(self):
        """Get model system status for health checks"""
        return {
            "available": self.is_available(),
            "primary_model": self.primary_model_name,
            "total_models": len(self.models),
            "loaded_models": list(self.models.keys())
        }

# Global instances
model_manager = ModelManager()
market_data_cache = {}
prediction_cache = {}
sentiment_data = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Background tasks (updated for Bybit)
async def sentiment_manager():
    """Improved sentiment manager with exponential backoff"""
    print("[INFO] Starting sentiment manager...")
    
    # Rate limiting state
    last_successful_request = {}
    consecutive_failures = {}
    
    while True:
        try:
            for token in SUPPORTED_TOKENS:
                try:
                    current_time = datetime.utcnow()
                    
                    # Check if we should skip this token due to recent failures
                    if token in consecutive_failures:
                        failure_count = consecutive_failures[token]
                        if failure_count >= 3:
                            # Exponential backoff: wait longer after multiple failures
                            wait_time = min(2 ** failure_count * 60, 3600)  # Max 1 hour
                            last_attempt = last_successful_request.get(token, current_time - timedelta(hours=2))
                            if (current_time - last_attempt).total_seconds() < wait_time:
                                print(f"[INFO] Skipping {token} due to backoff (failures: {failure_count})")
                                continue
                    
                    print(f"[INFO] Attempting sentiment update for {token}")
                    
                    # Try to get sentiment with timeout
                    sentiment_score = get_token_sentiment(token)
                    
                    # Success - reset failure count
                    consecutive_failures.pop(token, None)
                    last_successful_request[token] = current_time
                    
                    sentiment_data[token] = {
                        'score': sentiment_score,
                        'last_updated': current_time,
                        'status': 'success'
                    }
                    print(f"[INFO] Updated sentiment for {token}: {sentiment_score:.3f}")
                    
                    # Longer delay between successful requests to avoid rate limits
                    await asyncio.sleep(90)  # 1.5 minutes between tokens
                    
                except Exception as e:
                    print(f"[ERROR] Failed to update sentiment for {token}: {e}")
                    
                    # Track consecutive failures
                    consecutive_failures[token] = consecutive_failures.get(token, 0) + 1
                    
                    # Set error state
                    if token not in sentiment_data:
                        sentiment_data[token] = {
                            'score': 0.0,
                            'last_updated': current_time,
                            'status': 'error'
                        }
                    else:
                        sentiment_data[token]['status'] = 'error'
                    
                    # Wait longer after errors
                    await asyncio.sleep(120)  # 2 minutes after error
            
            print("[INFO] Sentiment cycle complete. Waiting 45 minutes for next cycle...")
            await asyncio.sleep(45 * 60)  # 45 minutes between full cycles
            
        except Exception as e:
            print(f"[ERROR] Sentiment manager error: {e}")
            await asyncio.sleep(300)  # 5 minutes on major error

async def update_market_data():
    """Background task to update market data every 15 seconds - FIXED for type errors"""
    while True:
        try:
            for token in SUPPORTED_TOKENS:
                try:
                    symbol = PERP_SYMBOLS[token].upper()
                    
                    # CHANGED: Use Bybit function instead of Binance
                    df = await get_bybit_klines(symbol, limit=100)
                    print(f"Fetched data for {token}: {len(df)} rows")
                    if df.empty:
                        print(f"[WARN] No data available for {token} from Bybit")
                        continue
                    
                    # FIX 1: Ensure all price/volume columns are properly converted to float
                    # Convert ALL relevant columns to numeric, including timestamp if present
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # FIX: Also convert timestamp column if it exists and is string-based
                    if 'timestamp' in df.columns:
                        # If timestamp is string, convert to datetime then to numeric timestamp
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                            # Sort by timestamp to ensure proper order
                            df = df.sort_values('timestamp')
                        except Exception as e:
                            print(f"[WARN] Timestamp conversion failed for {token}: {e}")
                    
                    # Remove any rows with NaN values that could cause comparison issues
                    df = df.dropna(subset=['close'])
                    
                    # FIX: Reset index after dropping rows to avoid index comparison issues
                    df = df.reset_index(drop=True)
                    
                    if df.empty:
                        print(f"[WARN] No valid data after cleaning for {token}")
                        continue
                    
                    current_price = float(df['close'].iloc[-1])
                    
                    # Calculate 24h change (using available data points)
                    if len(df) >= 2:
                        previous_price = float(df['close'].iloc[-2])
                        change_24h = ((current_price - previous_price) / previous_price) * 100
                    else:
                        change_24h = 0.0
                    
                    # FIX 2: Ensure volume is numeric before summing
                    if 'volume' in df.columns:
                        volume_24h = float(df['volume'].sum())
                    else:
                        volume_24h = 0.0
                    
                    # CHANGED: Use Bybit functions with error handling
                    try:
                        funding_rate = await fetch_funding_rate(symbol)
                        # Ensure funding_rate is a number
                        funding_rate = float(funding_rate) if funding_rate is not None else 0.0
                    except Exception as e:
                        print(f"[WARN] Failed to fetch funding rate for {symbol}: {e}")
                        funding_rate = 0.0
                    
                    try:
                        open_interest = await fetch_open_interest(symbol)
                        # Ensure open_interest is a number
                        open_interest = float(open_interest) if open_interest is not None else 0.0
                    except Exception as e:
                        print(f"[WARN] Failed to fetch open interest for {symbol}: {e}")
                        open_interest = 0.0
                    
                    # FIX 3: Technical indicators with proper type handling
                    if len(df) >= 14:
                        try:
                            # Create a clean Series with proper numeric index
                            close_prices = pd.Series(
                                pd.to_numeric(df['close'], errors='coerce').values,
                                index=range(len(df))
                            ).dropna()
                            
                            if len(close_prices) >= 14:
                                # Simple RSI calculation with type safety
                                delta = close_prices.diff()
                                gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
                                
                                # Avoid division by zero
                                rs = gain / loss.replace(0, 1e-10)
                                rsi_series = 100 - (100 / (1 + rs))
                                
                                rsi_14 = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
                                
                                # Clamp RSI to valid range
                                rsi_14 = max(0.0, min(100.0, rsi_14))
                            else:
                                rsi_14 = 50.0
                        except Exception as e:
                            print(f"[WARN] RSI calculation failed for {token}: {e}")
                            rsi_14 = 50.0
                    else:
                        rsi_14 = 50.0
                    
                    # FIX 4: MACD calculation with type safety
                    if len(df) >= 26:
                        try:
                            # Create a clean Series with proper numeric index
                            close_prices = pd.Series(
                                pd.to_numeric(df['close'], errors='coerce').values,
                                index=range(len(df))
                            ).dropna()
                            
                            if len(close_prices) >= 26:
                                ema_12 = close_prices.ewm(span=12, min_periods=12).mean()
                                ema_26 = close_prices.ewm(span=26, min_periods=26).mean()
                                
                                # Ensure both EMAs have values before subtracting
                                if not ema_12.empty and not ema_26.empty and len(ema_12) > 0 and len(ema_26) > 0:
                                    macd_diff = float(ema_12.iloc[-1] - ema_26.iloc[-1])
                                    
                                    # Handle NaN values
                                    if pd.isna(macd_diff):
                                        macd_diff = 0.0
                                else:
                                    macd_diff = 0.0
                            else:
                                macd_diff = 0.0
                        except Exception as e:
                            print(f"[WARN] MACD calculation failed for {token}: {e}")
                            macd_diff = 0.0
                    else:
                        macd_diff = 0.0
                    
                    # Get sentiment score safely
                    sentiment_score = sentiment_data.get(token, {}).get('score', 0.0)
                    
                    # FIX 5: Ensure all values are proper types before storing
                    market_data_cache[token] = {
                        'price': float(current_price),
                        'change_24h': float(change_24h),
                        'volume_24h': float(volume_24h),
                        'open_interest': float(open_interest),
                        'funding_rate': float(funding_rate),
                        'rsi_14': float(rsi_14),
                        'macd_diff': float(macd_diff),
                        'sentiment': float(sentiment_score),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    await manager.broadcast({
                        'type': 'market_data',
                        'token': token,
                        'data': market_data_cache[token]
                    })
                    
                    # Small delay between tokens to avoid rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    print(f"Error updating market data for {token}: {e}")
                    # Continue with next token instead of breaking the entire loop
                    continue
                
        except Exception as e:
            print(f"Error in market data update cycle: {e}")
        
        await asyncio.sleep(15)

async def generate_predictions():
    """Background task to generate predictions every 15 minutes"""
    while True:
        try:
            # Skip if no models available
            if not model_manager.is_available():
                print("[WARNING] No models available for predictions")
                await asyncio.sleep(LABEL_HORIZON * 60)
                continue
            
            for token in SUPPORTED_TOKENS:
                if token not in market_data_cache:
                    continue
                
                data = market_data_cache[token]
                sentiment_score = sentiment_data.get(token, {}).get('score', 0.0)
                
                features = {'close_return':  data.get('close_return', 0.0),
                'volume_change': data.get('volume_change', 0.0),
                'return_15m':    data.get('return_15m', 0.0),
                'volume_15m':    data.get('volume_15m', 0.0),
                'funding_rate':  data['funding_rate'],
                'open_interest': data['open_interest'],
                'rsi_14':        data['rsi_14'],
                'macd_diff':     data['macd_diff'],
                'sentiment':     sentiment_score}
                
                # Use model manager - users never see which model is used
                model_to_use, _ = model_manager.get_prediction_model()
                
                if model_to_use is not None:
                    label, confidence = make_prediction(model_to_use, token, features)
                    
                    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                    signal = signal_map.get(label, "HOLD")
                    
                    prediction = {
                        'signal': signal,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat(),
                        'token': token,
                        'features': features
                        # No model name exposed to users
                    }
                    
                    prediction_cache[token] = prediction
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        try:
                            await maybe_alert(confidence, token, signal)
                        except Exception as e:
                            print(f"[WARN] Alert failed: {e}")
                    
                    await manager.broadcast({
                        'type': 'prediction',
                        'token': token,
                        'data': prediction
                    })
                    
        except Exception as e:
            print(f"Error generating predictions: {e}")
        
        await asyncio.sleep(LABEL_HORIZON * 60)

# Startup event with simplified model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("🚀 Starting EthosX API Server with Bybit integration...")
    success = model_manager.load_models()
    
    if not success:
        print("⚠️  Server starting without prediction capabilities")
    
    # Start background tasks
    sentiment_task = asyncio.create_task(sentiment_manager())
    market_task = asyncio.create_task(update_market_data())
    prediction_task = asyncio.create_task(generate_predictions())
    
    print("✅ Server ready with Bybit data feed!")
    yield
    
    # Cleanup
    sentiment_task.cancel()
    market_task.cancel()
    prediction_task.cancel()

# Create FastAPI app
app = FastAPI(
    title="EthosX API",
    description="Real-time trading signals and market data API powered by Bybit",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes (same as before, but now powered by Bybit data)

@app.get("/")
async def root():
    return {"message": "EthosX API Server", "status": "running", "data_source": "Bybit"}

@app.get("/tokens")
async def get_supported_tokens():
    """Get list of supported tokens"""
    return {"tokens": SUPPORTED_TOKENS}

@app.get("/market/{token}", response_model=MarketDataResponse)
async def get_market_data(token: str):
    if token not in SUPPORTED_TOKENS:
        raise HTTPException(404, "Token not supported")

    try:
        raw = await public_get(
            path="/v5/market/tickers",
            params={"category": "linear", "symbol": f"{token}USDT"}
        )
    except HTTPStatusError as e:
        # both proxy & official failed
        logger.error(f"[api] market/{token} fetch failed: {e}")
        raise HTTPException(503, "Unable to fetch market data right now")

    # parse...
    item = raw["result"]["list"][0]
    data = MarketDataResponse(
        price       = float(item["lastPrice"]),
        change_24h  = float(item["price24hPcnt"]),
        volume_24h  = float(item["volume24h"]),
        open_interest = float(item["openInterestValue"]),
        funding_rate  = float(item["fundingRate"]),
        timestamp     = datetime.now().isoformat()
    )
    market_data_cache[token] = data.dict()
    return data

@app.get("/technical/{token}", response_model=TechnicalIndicatorsResponse)
async def get_technical_indicators(token: str):
    if token not in SUPPORTED_TOKENS:
        raise HTTPException(404, "Token not supported")

    data = market_data_cache.get(token)
    if not data:
        # fallback: neutral indicators + timestamp
        return TechnicalIndicatorsResponse(
            rsi_14=50.0,
            macd_diff=0.0,
            sentiment=0.0,
            funding_rate=0.0,
            timestamp=datetime.utcnow().isoformat()
        )

    return TechnicalIndicatorsResponse(
        rsi_14=data['rsi_14'],
        macd_diff=data['macd_diff'],
        sentiment=data['sentiment'],
        funding_rate=data['funding_rate'],
        timestamp=data['timestamp']
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(request: PredictionRequest):
    """Generate a trading signal prediction - model selection handled automatically"""
    if request.token not in SUPPORTED_TOKENS:
        raise HTTPException(status_code=404, detail="Token not supported")
    
    if not model_manager.is_available():
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    if request.token not in market_data_cache:
        raise HTTPException(status_code=503, detail="Market data not available")
    
    try:
        data = market_data_cache[request.token]
        sentiment_score = sentiment_data.get(request.token, {}).get('score', 0.0)
        
        features = {
            'close_return': 0.001,
            'volume_change': 0.05,
            'sentiment': sentiment_score,
            'funding_rate': data['funding_rate'],
            'open_interest': data['open_interest'],
            'return_15m': 0.01,
            'volume_15m': data['volume_24h'] / 96,
            'rsi_14': data['rsi_14'],
            'macd_diff': data['macd_diff']
        }
        
        # Model selection is completely hidden from user
        model_to_use, _ = model_manager.get_prediction_model()
        label, confidence = make_prediction(model_to_use, request.token, features)
        
        signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        signal = signal_map.get(label, "HOLD")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            maybe_alert(confidence, request.token, signal)
        
        return PredictionResponse(
            signal=signal,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            token=request.token,
            features=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/subscribe")
async def subscribe_email(request: SubscriptionRequest):
    """Subscribe to email alerts"""
    try:
        added = add_subscriber(str(request.email))
        if added:
            return {"message": "Successfully subscribed", "email": request.email}
        else:
            return {"message": "Already subscribed", "email": request.email}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")

@app.get("/subscribers")
async def get_subscriber_count():
    """Get number of subscribers"""
    try:
        subscribers = get_subscribers()
        return {"count": len(subscribers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get subscribers: {str(e)}")

@app.get("/predictions/{token}")
async def get_latest_prediction(token: str):
    if token not in SUPPORTED_TOKENS:
        raise HTTPException(404, "Token not supported")

    pred = prediction_cache.get(token)
    if not pred:
        # fallback prediction
        return PredictionResponse(
            signal="HOLD",
            confidence=0.0,
            timestamp=datetime.utcnow().isoformat(),
            token=token,
            features={}
        )

    return pred

@app.get("/sentiment/status")
async def get_sentiment_system_status():
    """Get status of sentiment analysis system"""
    try:
        system_status = get_sentiment_status()
        token_status = {}
        
        for token in SUPPORTED_TOKENS:
            if token in sentiment_data:
                token_status[token] = {
                    'score': sentiment_data[token]['score'],
                    'last_updated': sentiment_data[token]['last_updated'].isoformat(),
                    'status': sentiment_data[token]['status']
                }
            else:
                token_status[token] = {'status': 'not_loaded'}
        
        return {
            'system': system_status,
            'tokens': token_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment status: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        sentiment_status = get_sentiment_status()
        model_status = model_manager.get_status()
        
        return {
            "status": "healthy" if model_status["available"] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "data_source": "Bybit API",
            "prediction_service": model_status,
            "active_connections": len(manager.active_connections),
            "cached_tokens": list(market_data_cache.keys()),
            "sentiment_system": {
                "cached_tokens": len(sentiment_data),
                "can_make_request": sentiment_status["can_make_request"],
                "requests_in_window": sentiment_status["requests_in_window"],
                "time_until_next_request": sentiment_status["time_until_next_request"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )