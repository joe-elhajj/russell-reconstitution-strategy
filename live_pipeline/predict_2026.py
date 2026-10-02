import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Constants
START = '2026-01-01'
END = '2026-05-23'
CLASSIFIER_PATH = 'models/lgbm_classifier.pkl'
REGRESSOR_PATH = 'models/lgbm_regressor.pkl'
OUTPUT_PATH = 'backtests/predictions_2026.csv'
CHART_PATH = 'backtests/predictions_2026.png'

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


def download_and_clean(symbol, name):
    print(f"Downloading {name} ({symbol}) {START} to {END}...")
    df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
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


def build_features():
    frames = {}
    for name, symbol in TICKERS.items():
        df = download_and_clean(symbol, name)
        if df is None:
            print(f"Ticker {name} failed; skipping.")
        else:
            frames[name] = df

    if 'rut' not in frames:
        raise RuntimeError('Missing RUT data; cannot build features')

    merged = frames['rut']
    for k, v in frames.items():
        if k == 'rut':
            continue
        merged = merged.merge(v, on='date', how='inner')

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
    print(f"Built features with shape: {merged.shape}")
    return merged


def run():
    df = build_features()

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    DIRECTION_CLASS_MAP = {-1: 0, 0: 1, 1: 2, 2: 3}
    CLASS_TO_DIRECTION = {v: k for k, v in DIRECTION_CLASS_MAP.items()}

    X = df[FEATURES]
    proba = classifier.predict_proba(X)

    print('\nDEBUG: raw class probability scores for the last 10 rows:')
    last_n = min(10, len(df))
    proba_cols = [f'class_{i}_prob' for i in range(proba.shape[1])]
    debug_proba = pd.DataFrame(proba[-last_n:], columns=proba_cols, index=df['date'].iloc[-last_n:].dt.date)
    print(debug_proba.to_string())

    print('\nDEBUG: raw predicted class numbers for the last 10 rows:')
    raw_preds = classifier.predict(X)
    debug_preds = pd.DataFrame({'date': df['date'].iloc[-last_n:].dt.date, 'raw_class': raw_preds[-last_n:]})
    print(debug_preds.to_string(index=False))

    print('\nDEBUG: using bullish probability thresholds to build signals:')
    bull_prob = proba[:, 2] + proba[:, 3]
    df['signal'] = np.where(bull_prob > 0.50, 2, np.where(bull_prob > 0.35, 1, -1))
    df['bull_prob'] = bull_prob

    print('\nDEBUG: actual RUT return over the last 30 trading days:')
    if len(df) >= 31:
        recent_returns = df[['date', 'rut_close']].copy()
        recent_returns['return_30d'] = recent_returns['rut_close'].pct_change(periods=30)
        print(recent_returns[['date', 'rut_close', 'return_30d']].tail(30).to_string(index=False))
        total_30d = recent_returns['return_30d'].iloc[-1]
        print(f"\nDEBUG: total RUT return from {recent_returns['date'].iloc[-31].date()} to {recent_returns['date'].iloc[-1].date()} = {total_30d:.4%}")
    else:
        print('Not enough data to compute 30-day RUT return.')

    df['position_size'] = 0.0
    df.loc[df['signal'].isin([1, 2]), 'position_size'] = 1.0
    df.loc[df['signal'] == 0, 'position_size'] = 0.5
    df.loc[df['signal'] == -1, 'position_size'] = 0.0

    print('\nSignal distribution for 2026:')
    print(df['signal'].value_counts().sort_index().to_string())

    print('\nMay 2026 and later signals:')
    may_start = pd.to_datetime('2026-05-01')
    for _, row in df[df['date'] >= may_start].iterrows():
        marker = ' ***' if row['signal'] >= 1 else ''
        print(f"{row['date'].date()} | signal={int(row['signal'])} | position_size={row['position_size']:.1f}{marker}")

    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to {OUTPUT_PATH}")

    # Plot RUT price with signal dots
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['rut_close'], label='RUT Close', color='black', linewidth=1.5)
    colors = { -1: 'red', 1: 'yellow', 2: 'green' }
    plot_df = df[df['signal'].isin([-1, 1, 2])]  # only plot signal markers for bearish/bullish
    plt.scatter(plot_df['date'], plot_df['rut_close'],
                c=plot_df['signal'].map(colors),
                s=40,
                edgecolor='black',
                label='Signal')
    plt.title('RUT Price with 2026 Signal Markers')
    plt.xlabel('Date')
    plt.ylabel('RUT Close Price')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Signal -1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, markeredgecolor='black', label='Signal 1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Signal 2'),
    ])
    plt.tight_layout()
    os.makedirs(os.path.dirname(CHART_PATH) if os.path.dirname(CHART_PATH) else '.', exist_ok=True)
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()
    print(f"Saved chart to {CHART_PATH}")


if __name__ == '__main__':
    run()
