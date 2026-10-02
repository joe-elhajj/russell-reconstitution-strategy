import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

DATA_PATH = 'data/master_features.csv'
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'lgbm_classifier_v2.pkl')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'lgbm_regressor_v2.pkl')

DIRECTION_CLASS_MAP = {-1: 0, 0: 1, 1: 2, 2: 3}
CLASS_TO_DIRECTION = {v: k for k, v in DIRECTION_CLASS_MAP.items()}

FEATURES = [
    'sp500_close',
    'vix_close',
    'treasury_10y_close',
    'treasury_2y_close',
    'dollar_index_close',
    'gold_close',
    'oil_close',
    'rut_close_return',
    'rut_close_ma_5',
    'rut_close_ma_20',
    'rut_momentum_5',
    'rut_momentum_20',
    'yield_curve_spread',
    'vix_ma_5',
    'rut_rsi_14',
    'rut_bb_upper',
    'rut_bb_lower',
    'rut_macd',
    'rut_macd_signal',
    'rut_volume_ma_5',
    'rut_volume_ma_20',
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
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['next_rut_close'] = df['rut_close'].shift(-1)
    df['next_return'] = df['next_rut_close'] / df['rut_close'] - 1
    df['direction'] = df['next_return'].apply(get_direction_label)
    df = df.dropna(subset=['next_rut_close', 'next_return']).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    train_mask = (df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2022)
    val_mask = (df['date'].dt.year >= 2023) & (df['date'].dt.year <= 2024)

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, 'direction'].map(DIRECTION_CLASS_MAP)
    X_val = df.loc[val_mask, FEATURES]
    y_val = df.loc[val_mask, 'direction'].map(DIRECTION_CLASS_MAP)

    if X_train.empty or X_val.empty:
        raise RuntimeError('Training or validation set is empty. Check date ranges and data coverage.')

    print(f'Training rows: {X_train.shape[0]}')
    print(f'Validation rows: {X_val.shape[0]}')

    return X_train, X_val, y_train, y_val


def find_best_classifier_params(X_train, y_train):
    param_grid = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
    }
    clf = LGBMClassifier(random_state=42, objective='multiclass', num_class=4)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


def baseline_accuracy(X_train, y_train, X_val, y_val):
    baseline = LGBMClassifier(random_state=42, objective='multiclass', num_class=4)
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_val)
    return accuracy_score(y_val, preds)


def train_final_models(X_full, y_full_clf, y_full_reg, best_params):
    clf = LGBMClassifier(
        random_state=42,
        objective='multiclass',
        num_class=4,
        **best_params,
    )
    clf.fit(X_full, y_full_clf)

    reg = LGBMRegressor(
        random_state=42,
        **best_params,
    )
    reg.fit(X_full, y_full_reg)

    return clf, reg


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def main() -> None:
    print(f'Loading data from {DATA_PATH}...')
    df = load_data()
    print(f'Data loaded: {df.shape[0]} rows, {df.shape[1]} columns')

    X_train, X_val, y_train, y_val = split_data(df)
    print('Training on 2010-2022 and validating on 2023-2024.')

    print('Computing default baseline accuracy...')
    baseline_acc = baseline_accuracy(X_train, y_train, X_val, y_val)
    print(f'Baseline 2024 accuracy: {baseline_acc:.4f}')

    print('Searching hyperparameters with cross-validation...')
    best_params, cv_score = find_best_classifier_params(X_train, y_train)
    print(f'Best params from CV: {best_params}')
    print(f'CV accuracy for best params: {cv_score:.4f}')

    best_clf = LGBMClassifier(
        random_state=42,
        objective='multiclass',
        num_class=4,
        **best_params,
    )
    best_clf.fit(X_train, y_train)
    best_val_preds = best_clf.predict(X_val)
    best_val_acc = accuracy_score(y_val, best_val_preds)
    improvement = best_val_acc - baseline_acc

    print(f'Validation 2024 accuracy with best params: {best_val_acc:.4f}')
    print(f'Accuracy improvement vs default: {improvement:.4f}')

    print('Retraining on full 2010-2024 with best hyperparameters...')
    full_mask = (df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2024)
    X_full = df.loc[full_mask, FEATURES]
    y_full_clf = df.loc[full_mask, 'direction'].map(DIRECTION_CLASS_MAP)
    y_full_reg = df.loc[full_mask, 'next_return']

    clf_v2, reg_v2 = train_final_models(X_full, y_full_clf, y_full_reg, best_params)

    ensure_model_dir()
    joblib.dump(clf_v2, CLASSIFIER_PATH)
    joblib.dump(reg_v2, REGRESSOR_PATH)

    print(f'Saved classifier to {CLASSIFIER_PATH}')
    print(f'Saved regressor to {REGRESSOR_PATH}')


if __name__ == '__main__':
    main()
