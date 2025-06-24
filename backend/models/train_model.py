# train_model.py

import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_and_combine_data(data_dir=DATA_DIR):
    """
    Read all *_features.csv files from the data directory,
    concatenate into a single DataFrame, and return it.
    """
    pattern = os.path.join(data_dir, "*_features.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No feature CSVs found in {data_dir}")

    df_list = []
    for fpath in files:
        df = pd.read_csv(fpath, index_col="timestamp", parse_dates=["timestamp"])
        expected_cols = {
            "close", "volume", "close_return", "volume_change", 
            "sentiment", "label", "target",
            "funding_rate", "open_interest",
            "return_15m", "volume_15m"
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {fpath}: {missing}")
        df_list.append(df)

    combined = pd.concat(df_list, axis=0)
    combined = combined.dropna(subset=["label"])
    return combined


def train_model(model_path=MODEL_PATH):
    # 1) Load and combine data
    combined_df = load_and_combine_data()

    # 2) Define features and target
    features = [
        "close_return","volume_change","sentiment",
        "funding_rate","open_interest","return_15m","volume_15m",
        "rsi_14","macd_diff"
    ]
    X = combined_df[features]
    y = combined_df["label"].astype(int)

    # 2.1) Check label distribution
    label_counts = y.value_counts().sort_index()
    print("Label distribution before split:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt}")

    # 3) Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4) Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    print("\n===== Model Evaluation =====\n")
    print(classification_report(y_test, y_pred))

    # 6) Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_model()
