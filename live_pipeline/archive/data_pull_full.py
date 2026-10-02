import yfinance as yf
import pandas as pd
import numpy as np

DATA_FOLDER = "data"
DATE_START = "2010-01-01"
DATE_END = "2024-12-31"
OUTPUT_FILE = f"{DATA_FOLDER}/master_features.csv"

EQUITY_TICKERS = {
    "rut": "^RUT",
    "sp500": "^GSPC",
    "vix": "^VIX",
}

MACRO_TICKERS = {
    "treasury_10y": "^TNX",
    "treasury_2y": "^IRX",
    "dollar_index": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
}

ALL_TICKERS = {**EQUITY_TICKERS, **MACRO_TICKERS}


def download_and_clean(symbol: str, name: str) -> pd.DataFrame:
    print(f"Downloading {name} ({symbol}) from {DATE_START} to {DATE_END}...")
    df = yf.download(symbol, start=DATE_START, end=DATE_END, progress=False)
    print(f"Downloaded {name}: {len(df)} rows")

    df = df.dropna().reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip().lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    close_cols = [col for col in df.columns if "close" in col]
    volume_cols = [col for col in df.columns if "volume" in col]

    keep_cols = ["date", close_cols[0]]
    if volume_cols and name == "rut":
        keep_cols.append(volume_cols[0])

    df = df[keep_cols]
    df.columns = ["date"] + [f"{name}_close"] + ([f"{name}_volume"] if name == "rut" else [])
    print(f"Cleaned {name}: {len(df)} rows")
    return df


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def main() -> None:
    print("Downloading all assets...")
    data_frames = [download_and_clean(symbol, name) for name, symbol in ALL_TICKERS.items()]

    print("Merging datasets on date...")
    merged = data_frames[0]
    for df in data_frames[1:]:
        merged = merged.merge(df, on="date", how="inner")
    print(f"Merged dataset shape: {merged.shape}")

    print("Engineering features...")
    merged = merged.sort_values("date").reset_index(drop=True)

    asset_closes = [f"{name}_close" for name in ALL_TICKERS.keys()]
    for close_col in asset_closes:
        merged[f"{close_col}_return"] = merged[close_col].pct_change()

    merged["rut_close_ma_5"] = merged["rut_close"].rolling(5).mean()
    merged["rut_close_ma_20"] = merged["rut_close"].rolling(20).mean()
    merged["rut_momentum_5"] = merged["rut_close"] - merged["rut_close_ma_5"]
    merged["rut_momentum_20"] = merged["rut_close"] - merged["rut_close_ma_20"]

    merged["rut_volume_ma_5"] = merged["rut_volume"].rolling(5).mean()
    merged["rut_volume_ma_20"] = merged["rut_volume"].rolling(20).mean()

    merged["rut_rsi_14"] = compute_rsi(merged["rut_close"], 14)
    merged["rut_bb_upper"] = merged["rut_close_ma_20"] + 2 * merged["rut_close"].rolling(20).std()
    merged["rut_bb_lower"] = merged["rut_close_ma_20"] - 2 * merged["rut_close"].rolling(20).std()

    macd_short = merged["rut_close"].ewm(span=12, adjust=False).mean()
    macd_long = merged["rut_close"].ewm(span=26, adjust=False).mean()
    merged["rut_macd"] = macd_short - macd_long
    merged["rut_macd_signal"] = merged["rut_macd"].ewm(span=9, adjust=False).mean()

    merged["yield_curve_spread"] = merged["treasury_10y_close"] - merged["treasury_2y_close"]
    merged["vix_ma_5"] = merged["vix_close"].rolling(5).mean()

    merged = merged.dropna().reset_index(drop=True)
    print(f"Final feature dataset shape after dropping nulls: {merged.shape}")

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved master features to {OUTPUT_FILE}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
