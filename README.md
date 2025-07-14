# EthosX OPerps Predictive Dashboard

Hey there! This repo powers a live trading signal platform for EthosX OPerps, fusing on-chain perp data with off-chain sentiment and ML. Think of it as your 15-minute cycle market “radar” – complete with email alerts when we get juicy Long CALL or Long PUT signals.

---

## 🚀 Features

- **Real-time data**: 15 s updates of price, volume, open interest, funding rate, RSI, MACD from Bybit.  
- **Sentiment score**: pulls recent Tweets (once per 15 min) to gauge mood.  
- **ML signals**: XGBoost model predicts BUY/HOLD/SELL every cycle.  
- **Email alerts**: subscribers get notified if confidence ≥ your threshold.  
- **WebSocket + REST API**: power the front end with live pushes or on-demand calls.  
- **React dashboard**: sleek UI with charts, metrics, alerts, and subscriptions.

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn, Pandas, ta, Tweepy, joblib  
- **ML**: scikit-learn (RandomForest), XGBoost  
- **Email**: SMTP via Python’s `smtplib`  
- **Frontend**: React + Vite, Recharts, Lucide-React icons, Axios  
- **Containerization**: Docker + Docker-Compose (optional)  

---

## ⚙️ Quick Setup

### 1. Clone & venv

```bash
git clone <https://github.com/life-agrees/EthosX_Signal_Dashboard.git>
cd ethosX
# activate your existing venv or:
python3 -m venv .venv && source .venv/bin/activate
```
### 2.  Backend
```bash
cd backend
pip install -r requirements.txt
#Create a .env in backend/ with:
TWITTER_BEARER_TOKEN=…
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
EMAIL_USER=you@example.com
EMAIL_PASSWORD=…
MODEL_PATH=models/xgb_model.pkl
#Run it:
uvicorn api_server:app --reload --port 8000
REST API: http://localhost:8000/docs
WebSocket: ws://localhost:8000/ws
```
### 3. Frontend
```bash
cd ../frontend
npm install
#Edit frontend/.env:
VITE_API_BASE_URL=http://localhost:8000
#Start:
npm run dev
Visit http://localhost:5173 to see your dashboard.
