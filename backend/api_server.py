# api_server.py - Aligned with data_fetch.py
import os
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import pandas as pd
import joblib
from contextlib import asynccontextmanager
import ta
import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules - ALIGNED WITH data_fetch.py
from .models.predict import load_model, make_prediction
from .sentiment import get_token_sentiment, get_sentiment_status
from .alerts import add_subscriber, get_subscribers, maybe_alert, CONFIDENCE_THRESHOLD
# ALIGNED: Import actual functions from data_fetch.py
from .data_fetch import (
    get_coingecko_spot_data, 
    fetch_derivatives_data,
    build_token_features,
    make_request_with_retry
)
from .config import SUPPORTED_TOKENS, PERP_SYMBOLS, LABEL_HORIZON, COINGECKO_SYMBOLS

# Pydantic models remain the same...
class PredictionRequest(BaseModel):  
    token: str

class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    timestamp: str
    token: str
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

# Model Manager remains the same...
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_priority = ["XGBoost", "RandomForest"]
        self.primary_model = None
        self.primary_model_name = None
    
    def load_models(self):
        """Load models in priority order"""
        loaded_models = []
        
        model_paths = {
            "XGBoost": "models/xgb_model.pkl",
            "RandomForest": "models/model.pkl",
        }
        
        for model_name in self.model_priority:
            if model_name in model_paths:
                try:
                    self.models[model_name] = load_model(model_paths[model_name])
                    loaded_models.append(model_name)
                    print(f"✓ Loaded {model_name} model")
                    
                    if self.primary_model is None:
                        self.primary_model = self.models[model_name]
                        self.primary_model_name = model_name
                        print(f"🎯 Using {model_name} as primary model")
                        
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
        
        if not self.models:
            print("⚠️  WARNING: No models loaded!")
            return False
        
        print(f"📊 Model system ready: {len(loaded_models)} models loaded")
        return True
    
    def get_prediction_model(self):
        return self.primary_model, self.primary_model_name
    
    def is_available(self):
        return self.primary_model is not None
    
    def get_status(self):
        return {
            "available": self.is_available(),
            "primary_model": self.primary_model_name,
            "total_models": len(self.models),
            "loaded_models": list(self.models.keys())
        }

# ALIGNED: Updated market data functions using data_fetch.py functions
async def get_current_price(token: str) -> Dict[str, float]:
    """Get current price using same method as data_fetch.py"""
    try:
        df = await get_coingecko_spot_data(token, days=1)
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        change_24h = ((latest['close'] - previous['close']) / previous['close'] * 100) if previous['close'] != 0 else 0.0
        
        return {
            "price": float(latest['close']),
            "change_24h": float(change_24h),
            "volume_24h": float(latest.get('volume', 0)),
            "high_24h": float(latest.get('high', latest['close'])),
            "low_24h": float(latest.get('low', latest['close']))
        }
    except Exception as e:
        logger.error(f"Error getting price for {token}: {e}")
        return {}

async def calculate_real_time_features(token: str) -> Dict[str, float]:
    """Calculate features using same method as data_fetch.py"""
    try:
        # Get recent spot data (same as data_fetch.py)
        df = await get_coingecko_spot_data(token, days=7)
        if df.empty or len(df) < 15:
            logger.warning(f"Insufficient data for {token}")
            return {}
        
        # Calculate features exactly like data_fetch.py
        df['close_return'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        window_size = min(15, max(5, len(df) // 3))
        df['return_15m'] = df['close'].rolling(window_size, min_periods=1).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else 0.0
        )
        df['volume_15m'] = df['volume'].rolling(window_size, min_periods=1).sum()
        
        # Technical indicators (same as data_fetch.py)
        if len(df) >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            df['rsi_14'] = rsi.rsi()
        else:
            df['rsi_14'] = 50.0
            
        macd = ta.trend.MACD(df['close'])
        df['macd_diff'] = macd.macd_diff()
        
        # Get derivatives data (same as data_fetch.py)
        derivatives_data = await fetch_derivatives_data(token)
        
        # Get sentiment
        sentiment_score = sentiment_data.get(token, {}).get('score', 0.0)
        
        latest = df.iloc[-1]
        
        return {
            'close_return': float(latest.get('close_return', 0.0)) if not pd.isna(latest.get('close_return', 0.0)) else 0.0,
            'volume_change': float(latest.get('volume_change', 0.0)) if not pd.isna(latest.get('volume_change', 0.0)) else 0.0,
            'return_15m': float(latest.get('return_15m', 0.0)) if not pd.isna(latest.get('return_15m', 0.0)) else 0.0,
            'volume_15m': float(latest.get('volume_15m', 0.0)) if not pd.isna(latest.get('volume_15m', 0.0)) else 0.0,
            'rsi_14': float(latest.get('rsi_14', 50.0)) if not pd.isna(latest.get('rsi_14', 50.0)) else 50.0,
            'macd_diff': float(latest.get('macd_diff', 0.0)) if not pd.isna(latest.get('macd_diff', 0.0)) else 0.0,
            'funding_rate': derivatives_data.get('funding_rate', 0.0),
            'open_interest': derivatives_data.get('open_interest', 0.0),
            'sentiment': sentiment_score
        }
        
    except Exception as e:
        logger.error(f"Error calculating features for {token}: {e}")
        return {}

# Global instances
model_manager = ModelManager()
market_data_cache = {}
prediction_cache = {}
sentiment_data = {}

# WebSocket connection manager (same as before)
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

# ALIGNED: Updated background tasks
async def sentiment_manager():
    """Same sentiment manager as before"""
    print("[INFO] Starting sentiment manager...")
    
    last_successful_request = {}
    consecutive_failures = {}
    
    while True:
        try:
            for token in SUPPORTED_TOKENS:
                try:
                    current_time = datetime.utcnow()
                    
                    if token in consecutive_failures:
                        failure_count = consecutive_failures[token]
                        if failure_count >= 3:
                            wait_time = min(2 ** failure_count * 60, 3600)
                            last_attempt = last_successful_request.get(token, current_time - timedelta(hours=2))
                            if (current_time - last_attempt).total_seconds() < wait_time:
                                continue
                    
                    sentiment_score = get_token_sentiment(token)
                    
                    consecutive_failures.pop(token, None)
                    last_successful_request[token] = current_time
                    
                    sentiment_data[token] = {
                        'score': sentiment_score,
                        'last_updated': current_time,
                        'status': 'success'
                    }
                    
                    await asyncio.sleep(90)
                    
                except Exception as e:
                    consecutive_failures[token] = consecutive_failures.get(token, 0) + 1
                    if token not in sentiment_data:
                        sentiment_data[token] = {
                            'score': 0.0,
                            'last_updated': current_time,
                            'status': 'error'
                        }
                    await asyncio.sleep(120)
            
            await asyncio.sleep(45 * 60)
            
        except Exception as e:
            print(f"[ERROR] Sentiment manager error: {e}")
            await asyncio.sleep(300)

async def market_data_updater():
    """ALIGNED: Updated to use data_fetch.py functions"""
    print("[INFO] Starting market data updater (aligned with data_fetch.py)...")
    
    while True:
        try:
            for token in SUPPORTED_TOKENS:
                try:
                    # Use aligned functions
                    price_data = await get_current_price(token)
                    features = await calculate_real_time_features(token)
                    
                    if price_data and features:
                        # Build market data cache entry
                        market_data_cache[token] = {
                            **price_data,
                            **features,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # Broadcast to WebSocket clients
                        await manager.broadcast({
                            "type": "market_data",
                            "token": token,
                            "data": market_data_cache[token]
                        })
                        
                        print(f"[INFO] Updated market data for {token}: ${price_data.get('price', 0):.4f}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to update market data for {token}: {e}")
                
                await asyncio.sleep(5)  # Rate limiting
            
            print("[INFO] Market data cycle complete. Waiting 2 minutes...")
            await asyncio.sleep(120)
            
        except Exception as e:
            print(f"[ERROR] Market data updater error: {e}")
            await asyncio.sleep(300)

async def generate_predictions():
    """ALIGNED: Updated to use proper features"""
    while True:
        try:
            if not model_manager.is_available():
                await asyncio.sleep(LABEL_HORIZON * 60)
                continue
            
            for token in SUPPORTED_TOKENS:
                if token not in market_data_cache:
                    continue
                
                # Use the properly calculated features
                data = market_data_cache[token]
                
                # Extract features in the exact format expected by XGBoost model
                features = {
                    'close_return': data.get('close_return', 0.0),
                    'volume_change': data.get('volume_change', 0.0),
                    'sentiment': data.get('sentiment', 0.0),
                    'funding_rate': data.get('funding_rate', 0.0),
                    'open_interest': data.get('open_interest', 0.0),
                    'return_15m': data.get('return_15m', 0.0),
                    'volume_15m': data.get('volume_15m', 0.0),
                    'rsi_14': data.get('rsi_14', 50.0),
                    'macd_diff': data.get('macd_diff', 0.0)
                }
                
                model_to_use, _ = model_manager.get_prediction_model()
                
                if model_to_use is not None:
                    label, confidence = make_prediction(model_to_use, token, features)
                    
                    signal_map = {0: "Long PUT", 1: "Long CALL"}
                    signal = signal_map.get(label, "Long PUT")
                    
                    prediction = {
                        'signal': signal,
                        'confidence': float(confidence),
                        'timestamp': datetime.now().isoformat(),
                        'token': token,
                        'features': features,
                        'features': {k: float(v) for k, v in features.items()}  # Convert all numpy types
                        #'model': model_manager.primary_model_name if model_manager.primary_model else "None"
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
                        'data': prediction,

                    })
                    
        except Exception as e:
            print(f"Error generating predictions: {e}")
        
        await asyncio.sleep(LABEL_HORIZON * 60)

# Startup and FastAPI app setup (same as before)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting EthosX API Server (aligned with data_fetch.py)...")
    success = model_manager.load_models()
    
    if not success:
        print("⚠️  Server starting without prediction capabilities")
    
    sentiment_task = asyncio.create_task(sentiment_manager())
    market_task = asyncio.create_task(market_data_updater())
    prediction_task = asyncio.create_task(generate_predictions())
    
    print("✅ Server ready with aligned data pipeline!")
    yield
    
    sentiment_task.cancel()
    market_task.cancel()
    prediction_task.cancel()

app = FastAPI(
    title="EthosX API",
    description="Real-time trading signals and market data API (aligned with data_fetch.py)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ethosx-signal-dashboard.fly.dev",  # Your frontend domain
        "http://localhost:3000",  # For local development
        "http://localhost:5173",  # For Vite dev server
        "http://localhost:8080",  # Alternative local port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# API Routes (updated for alignment)
@app.get("/")
async def root():
    return {"message": "EthosX API Server", "status": "running", "aligned": "data_fetch.py"}

@app.get("/tokens")
async def get_supported_tokens():
    return {"tokens": SUPPORTED_TOKENS}

@app.get("/market/{token}", response_model=MarketDataResponse)
async def get_market_data(token: str):
    if token not in SUPPORTED_TOKENS:
        raise HTTPException(404, "Token not supported")

    try:
        price_data = await get_current_price(token)
        
        if not price_data:
            raise HTTPException(503, "Unable to fetch market data")
        
        # Get derivatives data using data_fetch.py function
        derivatives_data = await fetch_derivatives_data(token)
        
        data = MarketDataResponse(
            price=price_data.get("price", 0.0),
            change_24h=price_data.get("change_24h", 0.0),
            volume_24h=price_data.get("volume_24h", 0.0),
            open_interest=derivatives_data.get("open_interest", 0.0),
            funding_rate=derivatives_data.get("funding_rate", 0.0),
            timestamp=datetime.now().isoformat()
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Market data fetch failed for {token}: {e}")
        raise HTTPException(503, "Unable to fetch market data right now")

@app.get("/technical/{token}", response_model=TechnicalIndicatorsResponse)
async def get_technical_indicators(token: str):
    if token not in SUPPORTED_TOKENS:
        raise HTTPException(404, "Token not supported")

    data = market_data_cache.get(token)
    if not data:
        return TechnicalIndicatorsResponse(
            rsi_14=50.0,
            macd_diff=0.0,
            sentiment=0.0,
            funding_rate=0.0,
            timestamp=datetime.utcnow().isoformat()
        )

    return TechnicalIndicatorsResponse(
        rsi_14=data.get('rsi_14', 50.0),
        macd_diff=data.get('macd_diff', 0.0),
        sentiment=data.get('sentiment', 0.0),
        funding_rate=data.get('funding_rate', 0.0),
        timestamp=data['timestamp']
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(request: PredictionRequest):
    """Generate prediction using aligned features"""
    if request.token not in SUPPORTED_TOKENS:
        raise HTTPException(status_code=404, detail="Token not supported")
    
    if not model_manager.is_available():
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    try:
        # Calculate real-time features using same method as data_fetch.py
        features = await calculate_real_time_features(request.token)
        
        if not features:
            raise HTTPException(status_code=503, detail="Unable to calculate features")
        
        model_to_use, _ = model_manager.get_prediction_model()
        label, confidence = make_prediction(model_to_use, request.token, features)
        
        signal_map = {0: "Long PUT", 1: "Long CALL"}
        signal = signal_map.get(label, "Long PUT")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            await maybe_alert(confidence, request.token, signal)
        
        return PredictionResponse(
            signal=signal,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            token=request.token,
            features=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Rest of the endpoints remain the same...
@app.post("/subscribe")
async def subscribe_email(request: SubscriptionRequest):
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
        return PredictionResponse(
            signal="HOLD",
            confidence=0.0,
            timestamp=datetime.utcnow().isoformat(),
            token=token,
            features={}
        )
    # Convert numpy types to Python types before returning
    safe_pred = {
        "signal": pred["signal"],
        "confidence": float(pred["confidence"]),  # Convert numpy.float32 to float
        "timestamp": pred["timestamp"],
        "token": pred["token"],
        "features": {k: float(v) for k, v in pred["features"].items()}  # Convert all feature values
    }

    return safe_pred

@app.get("/sentiment/status")
async def get_sentiment_system_status():
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
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    try:
        sentiment_status = get_sentiment_status()
        model_status = model_manager.get_status()
        
        return {
            "status": "healthy" if model_status["available"] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "data_source": "Aligned with data_fetch.py (CoinGecko + Binance/Bybit)",
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
        "backend.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )