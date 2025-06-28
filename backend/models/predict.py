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
    Ensures feature order matches the trained model.
    """
    # Ensure feature order matches the model's expected input
    if hasattr(model, 'feature_names_in_'):
        ordered_keys = list(model.feature_names_in_)
    else:
        # Fallback: manually define the order
        ordered_keys = [
            "close_return", "volume_change", "sentiment",
            "funding_rate", "open_interest", "return_15m", "volume_15m",
            "rsi_14", "macd_diff"
        ]

    # Reorder the features accordingly
    ordered_features = {k: features[k] for k in ordered_keys if k in features}
    df = pd.DataFrame([ordered_features])

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    # Decode label if needed
    if hasattr(model, 'classes_') and set(model.classes_) == {0, 1, 2}:
        label_map = {0: -1, 1: 0, 2: 1}
        pred = label_map.get(pred, pred)

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
