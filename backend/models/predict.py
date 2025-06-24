# predict.py

import os
import joblib
import pandas as pd

# Default model path (change to xgb_model.pkl if desired)
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")


def load_model(model_path=MODEL_PATH):
    """
    Load a trained model from disk.
    """
    return joblib.load(model_path)


def make_prediction(model, token: str, features: dict):
    """
    Prepare input DataFrame and return (label, confidence).
    - model: trained sklearn/XGB model
    - token: one of 'BTC','SOL','DOGE','FART'
    - features: dict with keys matching your feature columns
    """
    # Build DataFrame: ensure correct feature order
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    # Map class index back to original labels: if model uses 0,1,2 for -1,0,1
    if hasattr(model, 'classes_') and set(model.classes_) == {0,1,2}:
        label_map = {0: -1, 1: 0, 2: 1}
        pred = label_map.get(pred, pred)
        # confidence of predicted class
        confidence = proba.max()
    else:
        confidence = proba.max()
    return pred, confidence


if __name__ == '__main__':
    # Example usage
    model = load_model()
    sample_features = {
        'close_return': 0.002,
        'volume_change': 0.05,
        'sentiment': 0.1,
        'funding_rate': 0.0005,
        'open_interest': 12345,
        'return_15m': 0.01,
        'volume_15m': 500000,
        'rsi_14': 65.0,
        'macd_diff': 0.0003
    }
    label, conf = make_prediction(model, 'BTC', sample_features)
    print(f"Predicted label: {label}, confidence: {conf:.2f}")
