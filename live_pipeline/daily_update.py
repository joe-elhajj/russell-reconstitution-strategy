import os
import datetime
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Daily update constants
LOOKBACK_DAYS = 45
CLASSIFIER_PATH = 'models/lgbm_classifier.pkl'
OUTPUT_PATH = 'backtests/predictions_2026.csv'

TICKERS = {
    'rut': '^RUT',
    'sp500': '^GSPC',
    'vix': '^VIX',
    'treasury_10y': '^TNX',
    'treasury_2y': '^IRX',
    'dollar_index': 'DX-Y.NYB',
    'gold': 'GC=F',
    'oil': 'CL=F',
}

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


def download_and_clean(symbol: str, name: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f'No data downloaded for {symbol}')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = 'date'
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={'close': f'{name}_close', 'volume': f'{name}_volume'})
    df['date'] = pd.to_datetime(df['date'])
    keep = ['date', f'{name}_close'] + ([f'{name}_volume'] if f'{name}_volume' in df.columns else [])
    return df[keep].dropna()


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(start: str, end: str) -> pd.DataFrame:
    frames = {}
    for name, symbol in TICKERS.items():
        frames[name] = download_and_clean(symbol, name, start, end)

    if 'rut' not in frames:
        raise RuntimeError('Missing RUT data; cannot build features')

    merged = frames['rut']
    for name, frame in frames.items():
        if name == 'rut':
            continue
        merged = merged.merge(frame, on='date', how='inner')

    merged = merged.sort_values('date').reset_index(drop=True)
    merged['rut_close_return'] = merged['rut_close'].pct_change()
    merged['rut_close_ma_5'] = merged['rut_close'].rolling(5).mean()
    merged['rut_close_ma_20'] = merged['rut_close'].rolling(20).mean()
    merged['rut_momentum_5'] = merged['rut_close'] - merged['rut_close_ma_5']
    merged['rut_momentum_20'] = merged['rut_close'] - merged['rut_close_ma_20']

    if 'rut_volume' in merged.columns:
        merged['rut_volume_ma_5'] = merged['rut_volume'].rolling(5).mean()
        merged['rut_volume_ma_20'] = merged['rut_volume'].rolling(20).mean()
    else:
        merged['rut_volume_ma_5'] = np.nan
        merged['rut_volume_ma_20'] = np.nan

    merged['rut_rsi_14'] = compute_rsi(merged['rut_close'], 14)
    merged['rut_bb_upper'] = merged['rut_close_ma_20'] + 2 * merged['rut_close'].rolling(20).std()
    merged['rut_bb_lower'] = merged['rut_close_ma_20'] - 2 * merged['rut_close'].rolling(20).std()
    ema12 = merged['rut_close'].ewm(span=12, adjust=False).mean()
    ema26 = merged['rut_close'].ewm(span=26, adjust=False).mean()
    merged['rut_macd'] = ema12 - ema26
    merged['rut_macd_signal'] = merged['rut_macd'].ewm(span=9, adjust=False).mean()
    merged['yield_curve_spread'] = merged['treasury_10y_close'] - merged['treasury_2y_close']
    merged['vix_ma_5'] = merged['vix_close'].rolling(5).mean()

    merged = merged.dropna().reset_index(drop=True)
    return merged


def build_signal(df: pd.DataFrame, classifier) -> pd.DataFrame:
    X = df[FEATURES]
    proba = classifier.predict_proba(X)
    bull_prob = proba[:, 2] + proba[:, 3]
    df = df.copy()
    df['bull_prob'] = bull_prob
    df['signal'] = np.where(bull_prob > 0.50, 2, np.where(bull_prob > 0.35, 1, -1))
    df['position_size'] = np.where(df['signal'] >= 1, 1.0, 0.0)
    return df


def interpret_signal(signal: int) -> str:
    if signal == 2:
        return 'Strong bullish: full position is recommended.'
    if signal == 1:
        return 'Bullish: a full long position is reasonable.'
    return 'Bearish: no position is recommended today.'


def append_today_prediction(today_row: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    today = pd.to_datetime(today_row['date'].iloc[0]).date()

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=['date'])
        if today in existing['date'].dt.date.values:
            print(f'Prediction for {today} already exists in {OUTPUT_PATH}. Skipping append.')
            return
        today_row.to_csv(OUTPUT_PATH, index=False, mode='a', header=False)
    else:
        today_row.to_csv(OUTPUT_PATH, index=False, mode='w', header=True)

    print(f'Appended today\'s prediction to {OUTPUT_PATH}')


def run() -> None:
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = (today + datetime.timedelta(days=1)).isoformat()

    print(f'Building features from {start} to {end}...')
    df = build_features(start, end)
    if df.empty:
        raise RuntimeError('No feature rows were generated for the lookback window.')

    classifier = joblib.load(CLASSIFIER_PATH)
    df = build_signal(df, classifier)

    latest = df.iloc[-1]
    report_date = latest['date'].date()
    signal = int(latest['signal'])
    position_size = float(latest['position_size'])
    english = interpret_signal(signal)

    print(f"\nTODAY: {report_date}")
    print(f"Signal: {signal}")
    print(f"Position size: {position_size:.1f}")
    print(f"Interpretation: {english}\n")

    context = df[['date', 'signal', 'position_size', 'bull_prob']].tail(5)
    print('Last 5 trading days:')
    print(context.to_string(index=False))

    append_today_prediction(df[['date', 'signal', 'position_size', 'bull_prob']].tail(1))


if __name__ == '__main__':
    run()
