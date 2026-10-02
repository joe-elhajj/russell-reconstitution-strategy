import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Constants
START = '2025-01-01'
END = '2025-12-31'
CLASSIFIER_PATH = 'models/lgbm_classifier.pkl'
REGRESSOR_PATH = 'models/lgbm_regressor.pkl'
EQUITY_PATH = 'backtests/equity_curve_v3_2025_oos.csv'
CHART_PATH = 'backtests/equity_curve_v3_2025_oos.png'
START_CAPITAL = 100_000.0

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
    raw_preds = classifier.predict(X)
    df['signal'] = [CLASS_TO_DIRECTION.get(int(p), 0) for p in raw_preds]

    df['position_pct'] = 0.0
    df.loc[df['signal'].isin([1, 2]), 'position_pct'] = 1.0
    df.loc[df['signal'] == 0, 'position_pct'] = 0.5
    df.loc[df['signal'] == -1, 'position_pct'] = 0.0

    df['next_return'] = df['rut_close'].shift(-1) / df['rut_close'] - 1
    df['next_return'] = df['next_return'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df['strategy_return'] = df['position_pct'] * df['next_return']
    df['portfolio_value'] = START_CAPITAL * (1 + df['strategy_return']).cumprod()
    df['benchmark_value'] = START_CAPITAL * df['rut_close'] / df['rut_close'].iloc[0]
    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0.0)

    strategy_return = df['portfolio_value'].iloc[-1] / START_CAPITAL - 1
    bh_return = df['benchmark_value'].iloc[-1] / START_CAPITAL - 1
    trade_mask = df['position_pct'] > 0
    wins = (df.loc[trade_mask, 'next_return'] > 0).sum()
    trade_count = trade_mask.sum()
    win_rate = wins / trade_count if trade_count > 0 else 0.0
    sharpe = (
        df['daily_return'].mean() / df['daily_return'].std(ddof=0) * np.sqrt(252)
        if df['daily_return'].std(ddof=0) != 0
        else 0.0
    )

    print('\n2025 Out-of-Sample Results (v3)')
    print(f"Strategy Return: {strategy_return*100:.2f}%")
    print(f"Buy & Hold Return: {bh_return*100:.2f}%")
    print(f"Win Rate: {win_rate*100:.2f}% ({int(wins)}/{int(trade_count)})")
    print(f"Sharpe Ratio: {sharpe:.4f}")

    os.makedirs(os.path.dirname(EQUITY_PATH) if os.path.dirname(EQUITY_PATH) else '.', exist_ok=True)
    df[['date', 'portfolio_value', 'benchmark_value', 'signal', 'position_pct', 'strategy_return']].to_csv(
        EQUITY_PATH, index=False
    )
    print(f"Saved equity curve to {EQUITY_PATH}")

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['portfolio_value'], label='Strategy Portfolio')
    plt.plot(df['date'], df['benchmark_value'], label='Buy & Hold RUT', linestyle='--')
    plt.title('2025 Out-of-Sample v3: Strategy vs Buy & Hold RUT')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(CHART_PATH) if os.path.dirname(CHART_PATH) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    print(f"Saved chart to {CHART_PATH}")
    plt.close()


if __name__ == '__main__':
    run()
