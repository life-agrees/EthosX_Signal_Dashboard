import requests
import asyncio
import websockets
import json

BASE_URL = "https://ethosx-signal-dashboard.onrender.com"
WS_URL = "wss://ethosx-signal.onrender.com/ws"  # fixed double https typo
TEST_TOKEN = "BTC"

def seed_test_data():
    print("[SETUP] Seeding test data")
    try:
        response = requests.post(f"{BASE_URL}/__test_seed", json={
            "token": TEST_TOKEN,
            "market_data": {
                "funding_rate": 0.01,
                "open_interest": 5000000,
                "volume_24h": 1000000,
                "rsi_14": 50.0,
                "macd_diff": 0.2
            },
            "model_available": True
        })
        assert response.status_code == 200
        print("✅ Test data seeded successfully")
    except Exception as e:
        print("❌ Failed to seed test data:", e)

def test_predict_endpoint():
    print("[TEST] /predict endpoint")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={"token": TEST_TOKEN})
        assert response.status_code == 200
        data = response.json()
        assert 'signal' in data and 'confidence' in data
        print("✅ /predict passed:", data)
    except Exception as e:
        print("❌ /predict failed:", e)

def test_latest_prediction():
    print("[TEST] /predictions/{token} endpoint")
    try:
        response = requests.get(f"{BASE_URL}/predictions/{TEST_TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert 'signal' in data and 'confidence' in data
        print("✅ /predictions passed:", data)
    except Exception as e:
        print("❌ /predictions failed:", e)

async def test_websocket():
    print("[TEST] WebSocket connection")
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            print("✅ WebSocket connected and ping sent")
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=10)
                print("✅ WebSocket message received:", message)
            except asyncio.TimeoutError:
                print("⚠️ WebSocket connected, but no message received (might be idle)")
    except Exception as e:
        print("❌ WebSocket failed:", e)

def run_all_tests():
    seed_test_data()
    test_predict_endpoint()
    test_latest_prediction()
    asyncio.run(test_websocket())

if __name__ == "__main__":
    run_all_tests()
