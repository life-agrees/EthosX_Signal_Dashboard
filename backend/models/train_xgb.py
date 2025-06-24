# models/train_xgb.py

import os
import glob
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    return combined.dropna(subset=["label"])


def train_xgboost():
    # 1) Load data
    df = load_and_combine_data()

    # 2) Features & target
    feature_cols = [
        "close_return","volume_change","sentiment",
        "funding_rate","open_interest","return_15m","volume_15m",
        "rsi_14","macd_diff"
    ]
    X = df[feature_cols]

    # 2.1) Check original label distribution
    orig_counts = df["label"].value_counts().sort_index()
    print("Original label distribution:")
    for lbl, cnt in orig_counts.items():
        print(f"  {lbl}: {cnt}")

    # Map labels from -1,0,1 to 0,1,2 for XGBoost
    y = df["label"].map({-1: 0, 0: 1, 1: 2}).astype(int)

    # 3) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4) Compute scale_pos_weight for imbalance (negative vs positive+neutral)
    neg = (y_train == 0).sum()
    pos = len(y_train) - neg
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # 5) Set up XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    # Quick hyperparameter search
    param_dist = {
        "max_depth": [4,6,8],
        "learning_rate": [0.01,0.05,0.1],
        "subsample": [0.6,0.8,1.0],
        "colsample_bytree": [0.6,0.8,1.0],
        "n_estimators": [100,200,300]
    }
    search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # 6) Evaluate
    y_pred = model.predict(X_test)
    # Map back labels for reporting
    y_test_orig = y_test.map({0: -1, 1: 0, 2: 1})
    y_pred_orig = pd.Series(y_pred).map({0: -1, 1: 0, 2: 1})
    print("\n===== XGBoost Model Evaluation =====\n")
    print(classification_report(y_test_orig, y_pred_orig))

    # 7) Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nXGBoost model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost()
