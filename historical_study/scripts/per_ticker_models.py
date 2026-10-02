#!/usr/bin/env python
"""
Train per-ticker LightGBM models for Russell 2026 candidate tickers.
Designed to run on a high-CPU Vast.ai instance with joblib Parallel.
"""
import os
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ADDITIONS = ['CRWV','CHYM','FIG','KVYO','HNGE','GLXY','TTAN',
             'ULS','RJET','AXTI','FCEL','LPTH','UMAC','WOLF',
             'GPRO','CHPT','SPCE','OPEN','SYM','FROG','DLO',
             'ASND','ODD','TBLA','IREN','CSIQ']

DELETIONS = ['HAIN','SNBR','DCGO','BYND','VRM','FFAI','SPWR',
             'GETY','SKIL','ALIT','DH','HCAT','NFE','EXFY',
             'MVIS','TR','ATYR','FATE','ALDX','PRLD']

ALL_TICKERS = ADDITIONS + DELETIONS

# Feature helpers (shared with backtest)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(prices, period=20, num_std=2):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def build_features(df, rut_series=None):
    df = df.copy()
    close = df['Close']
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    features = pd.DataFrame(index=df.index)
    features['price_return_1d'] = close.pct_change()
    features['price_return_5d'] = close.pct_change(5)
    features['price_return_20d'] = close.pct_change(20)

    features['price_ma_5'] = close.rolling(5).mean()
    features['price_ma_20'] = close.rolling(20).mean()
    features['price_ma_50'] = close.rolling(50).mean()

    features['rsi_14'] = calculate_rsi(close, 14)
    bb_up, bb_low = bollinger_bands(close, 20, 2)
    features['bb_upper'] = bb_up
    features['bb_lower'] = bb_low
    features['bb_position'] = (close - bb_low) / (bb_up - bb_low)

    features['volume_ma_5'] = vol.rolling(5).mean()
    features['volume_ma_20'] = vol.rolling(20).mean()
    features['volume_ratio'] = vol / (features['volume_ma_20'].replace(0, np.nan))

    macd_line, macd_signal, macd_hist = macd(close)
    features['macd'] = macd_line
    features['macd_signal'] = macd_signal
    features['macd_histogram'] = macd_hist

    # Year-to-date return
    features['ytd_return'] = close / close.groupby(close.index.year).transform(lambda x: x.iloc[0]) - 1

    # vs_rut: 20d return difference
    if rut_series is not None and len(rut_series) == len(close):
        features['vs_rut'] = close.pct_change(20) - rut_series.pct_change(20)
    else:
        features['vs_rut'] = 0

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    return features


def prepare_labels(df_close):
    # Label = 1 if price higher 5 trading days from now
    future = df_close.shift(-5)
    label = (future > df_close).astype(int)
    label = label.loc[~label.isna()]
    return label


def download_ticker(ticker):
    start = '2021-01-01'
    end = '2025-12-31'
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        # Handle MultiIndex (rare for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                df = df['Close'].to_frame(name='Close') if isinstance(df['Close'], pd.Series) else df
        # Ensure Close and Volume columns exist
        if 'Close' not in df.columns:
            # try flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1)
        df.to_csv(DATA_DIR / f'ticker_{ticker}.csv')
        return df
    except Exception as e:
        print(f'Error downloading {ticker}: {e}')
        return None


def process_ticker(ticker):
    try:
        df = download_ticker(ticker)
        if df is None or df.empty:
            return {'ticker': ticker, 'status': 'download_failed'}

        # Build features
        features = build_features(df)
        labels = prepare_labels(df['Close'])

        # align
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]

        if len(X) < 200:
            return {'ticker': ticker, 'status': 'insufficient_data'}

        # Train/validate split: train on 2021-2024, validate 2025
        train_idx = X.index.year <= 2024
        val_idx = X.index.year == 2025
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]

        if len(X_val) < 10 or len(X_train) < 50:
            return {'ticker': ticker, 'status': 'insufficient_split'}

        model = lgb.LGBMClassifier(n_estimators=200, n_jobs=1)
        model.fit(X_train, y_train)

        # Validation
        preds = model.predict(X_val)
        probas = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, preds)

        # Save model
        with open(MODELS_DIR / f'lgbm_{ticker}.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Save accuracy
        acc_path = RESULTS_DIR / 'ticker_accuracy.csv'
        df_acc = pd.DataFrame([{'ticker': ticker, 'accuracy': acc}])
        if acc_path.exists():
            df_acc.to_csv(acc_path, mode='a', header=False, index=False)
        else:
            df_acc.to_csv(acc_path, index=False)

        # Generate 2026 prediction: use most recent available row
        latest = build_features(df.tail(60)).tail(1)
        if latest.empty:
            signal = 'HOLD'
            probability = 0.5
        else:
            prob = model.predict_proba(latest)[0, 1]
            probability = float(prob)
            if probability > 0.6:
                signal = 'BUY'
            elif probability < 0.4:
                signal = 'SELL'
            else:
                signal = 'HOLD'

        confidence = int(round(probability * 10))
        current_price = float(df['Close'].iloc[-1])
        ytd = float((current_price / df['Close'].loc[df.index.year == df.index[-1].year][0] - 1) if True else 0)

        pred_row = {
            'ticker': ticker,
            'category': 'ADDITION' if ticker in ADDITIONS else 'DELETION',
            'signal': signal,
            'probability': probability,
            'confidence': confidence,
            'current_price': current_price,
            'ytd_return': ytd,
            'recommendation': signal
        }

        preds_path = RESULTS_DIR / 'ticker_predictions.csv'
        pd.DataFrame([pred_row]).to_csv(preds_path, mode='a', header=not preds_path.exists(), index=False)

        return {'ticker': ticker, 'status': 'trained', 'accuracy': acc, 'confidence': confidence}

    except Exception as e:
        return {'ticker': ticker, 'status': f'error: {e}'}


def main():
    print('Training per-ticker LightGBM models (parallel)...')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_ticker)(t) for t in ALL_TICKERS)

    # Summary
    rows = [r for r in results if isinstance(r, dict) and r.get('status') == 'trained']
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        df_rows = df_rows.sort_values('confidence', ascending=False)
        top_buys = df_rows[df_rows['confidence'] > 5].head(5)
        top_sells = df_rows[df_rows['confidence'] > 5].tail(5)

        summary_path = RESULTS_DIR / 'ticker_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('Top buys (by confidence)\n')
            f.write(top_buys.to_string(index=False))
            f.write('\n\nTop sells (by confidence)\n')
            f.write(top_sells.to_string(index=False))

    print('Per-ticker modeling complete.')


if __name__ == '__main__':
    main()
