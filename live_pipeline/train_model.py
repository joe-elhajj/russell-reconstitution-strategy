import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error

DATA_PATH = "data/master_features.csv"
MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "lgbm_classifier.pkl")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "lgbm_regressor.pkl")

DIRECTION_CLASS_MAP = {-1: 0, 0: 1, 1: 2, 2: 3}
CLASS_TO_DIRECTION = {v: k for k, v in DIRECTION_CLASS_MAP.items()}

FEATURES = [
    "sp500_close",
    "vix_close",
    "treasury_10y_close",
    "treasury_2y_close",
    "dollar_index_close",
    "gold_close",
    "oil_close",
    "rut_close_return",
    "rut_close_ma_5",
    "rut_close_ma_20",
    "rut_momentum_5",
    "rut_momentum_20",
    "yield_curve_spread",
    "vix_ma_5",
    "rut_rsi_14",
    "rut_bb_upper",
    "rut_bb_lower",
    "rut_macd",
    "rut_macd_signal",
    "rut_volume_ma_5",
    "rut_volume_ma_20",
]


def get_direction_label(next_return: float) -> int:
    if next_return >= 0.004:
        return 2
    if next_return > 0:
        return 1
    if next_return <= -0.004:
        return -1
    return 0


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["next_rut_close"] = df["rut_close"].shift(-1)
    df["next_return"] = df["next_rut_close"] / df["rut_close"] - 1
    df["direction"] = df["next_return"].apply(get_direction_label)

    df = df.dropna(subset=["next_rut_close", "next_return"]).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    train_mask = (df["date"].dt.year >= 2010) & (df["date"].dt.year <= 2024)

    X_train = df.loc[train_mask, FEATURES]
    y_train_clf = df.loc[train_mask, "direction"].map(DIRECTION_CLASS_MAP)
    y_train_reg = df.loc[train_mask, "next_return"]

    X_val = None
    y_val_clf = None
    y_val_reg = None

    return X_train, X_val, y_train_clf, y_val_clf, y_train_reg, y_val_reg


def train_classifier(X_train, y_train):
    clf = LGBMClassifier(random_state=42, objective="multiclass", num_class=4)
    clf.fit(X_train, y_train)
    return clf


def train_regressor(X_train, y_train):
    reg = LGBMRegressor(random_state=42)
    reg.fit(X_train, y_train)
    return reg


def evaluate_classifier(model, X_val, y_val):
    raw_preds = model.predict(X_val)
    preds = np.array([CLASS_TO_DIRECTION[int(p)] for p in raw_preds])
    truth = np.array([CLASS_TO_DIRECTION[int(p)] for p in y_val])

    acc = accuracy_score(truth, preds)
    prec = precision_score(truth, preds, average="macro", zero_division=0)
    rec = recall_score(truth, preds, average="macro", zero_division=0)
    cm = confusion_matrix(truth, preds, labels=[-1, 0, 1, 2])

    print("Classifier evaluation")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro precision: {prec:.4f}")
    print(f"Macro recall: {rec:.4f}")
    print("Confusion matrix (rows=true, cols=predicted) for labels [-1,0,1,2]:")
    print(cm)


def evaluate_regressor(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print("Regressor evaluation")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")


def print_feature_importance(model, feature_names, title: str):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    print(f"{title} feature importance:")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]}: {importances[idx]}")


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def main() -> None:
    print(f"Loading data from {DATA_PATH}...")
    df = load_data()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X_train, X_val, y_train_clf, y_val_clf, y_train_reg, y_val_reg = split_data(df)
    print(f"Train shape: {X_train.shape}")

    classifier = train_classifier(X_train, y_train_clf)
    print("Classifier trained on full 2010-2024 dataset")

    regressor = train_regressor(X_train, y_train_reg)
    print("Regressor trained on full 2010-2024 dataset")

    print_feature_importance(classifier, FEATURES, "Classifier")
    print_feature_importance(regressor, FEATURES, "Regressor")

    ensure_model_dir()
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(regressor, REGRESSOR_PATH)
    print(f"Saved classifier to {CLASSIFIER_PATH}")
    print(f"Saved regressor to {REGRESSOR_PATH}")


if __name__ == "__main__":
    main()
