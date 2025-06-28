import os
import glob
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

# Directories
DATA_DIR   = "data"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")


def load_and_combine_data():
    files = glob.glob(os.path.join(DATA_DIR, "*_features.csv"))
    if not files:
        raise FileNotFoundError("No feature CSVs found in data/")
    df_list = [pd.read_csv(f, index_col="timestamp", parse_dates=["timestamp"]) for f in files]
    combined = pd.concat(df_list, axis=0)
    combined = combined.dropna(subset=["label"])
    return combined


def train_xgboost():
    # 1) Load data
    df = load_and_combine_data()

    # 2) Define feature columns
    feature_cols = [
        "close_return", "volume_change", "sentiment",
        "funding_rate", "open_interest", "return_15m", "volume_15m",
        "rsi_14", "macd_diff"
    ]

    # 3) Filter to only SELL(-1) or BUY(1), then build X, y in lock-step
    mask = df['label'].isin([-1, 1])
    df_filtered = df.loc[mask]

    X = df_filtered[feature_cols]
    y = (
        df_filtered['label']
                   .map({-1: 0, 1: 1})
                   .astype(int)
    )

    # Sanity-check: feature and label row counts
    print(f"Feature rows: {len(X)}, Label rows: {len(y)}")

    # 4) Check new distribution
    counts = y.value_counts().sort_index()
    print("Binary label distribution:")
    print(f"  SELL (0): {counts.get(0, 0)}")
    print(f"  BUY  (1): {counts.get(1, 0)}")

    # 5) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 6) Compute scale_pos_weight = (# negative) / (# positive)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # 7) Set up XGBoost binary classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # 8) Hyperparameter search
    param_dist = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "n_estimators": [100, 200, 300]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # 9) Evaluate
    y_pred = model.predict(X_test)
    print("\n===== XGBoost Binary Model Evaluation (BUY vs SELL) =====\n")
    print(classification_report(
        y_test.map({0: 'SELL', 1: 'BUY'}),
        pd.Series(y_pred).map({0: 'SELL', 1: 'BUY'})
    ))

    # 10) Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nBinary XGBoost model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_xgboost()
