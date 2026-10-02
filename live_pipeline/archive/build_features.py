import pandas as pd

DATA_FOLDER = "data"
INPUT_FILES = {
    "rut": "rut.csv",
    "sp500": "sp500.csv",
    "vix": "vix.csv",
    "treasury_10y": "treasury_10y.csv",
    "treasury_2y": "treasury_2y.csv",
    "dollar_index": "dollar_index.csv",
    "gold": "gold.csv",
    "oil": "oil.csv",
}


def load_dataset(name: str, filename: str) -> pd.DataFrame:
    path = f"{DATA_FOLDER}/{filename}"
    print(f"Loading {name} from {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.index.name is None:
        df.index.name = "date"
    df = df.reset_index()

    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    close_cols = [col for col in df.columns if "close" in str(col).lower()]
    if not close_cols:
        raise ValueError(f"No close-like column found in {path}. Columns: {list(df.columns)}")
    close_col = close_cols[0]

    columns = ["date", close_col]
    volume_cols = [col for col in df.columns if "volume" in str(col).lower()]
    if volume_cols:
        columns.append(volume_cols[0])

    rename_map = {close_col: f"{name}_close"}
    if volume_cols:
        rename_map[volume_cols[0]] = f"{name}_volume"

    df = df[columns].rename(columns=rename_map)
    print(f"Loaded {name}: {len(df)} rows using close column '{close_col}'")
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
    sample_path = f"{DATA_FOLDER}/rut.csv"
    print(f"Inspecting sample CSV: {sample_path}")
    sample_df = pd.read_csv(sample_path, index_col=0, parse_dates=True)
    print(f"Sample columns: {list(sample_df.columns)}")
    print("Sample first 2 rows:")
    print(sample_df.head(2))

    data_frames = [load_dataset(name, filename) for name, filename in INPUT_FILES.items()]

    print("Merging datasets on date...")
    merged = data_frames[0]
    for df in data_frames[1:]:
        merged = merged.merge(df, on="date", how="inner")
    print(f"Merged dataset shape: {merged.shape}")

    print("Engineering features...")
    asset_closes = [f"{name}_close" for name in INPUT_FILES.keys()]
    for close_col in asset_closes:
        merged[f"{close_col}_return"] = merged[close_col].pct_change()

    merged["rut_close_ma_5"] = merged["rut_close"].rolling(5).mean()
    merged["rut_close_ma_20"] = merged["rut_close"].rolling(20).mean()
    merged["rut_momentum_5"] = merged["rut_close"] - merged["rut_close_ma_5"]
    merged["rut_momentum_20"] = merged["rut_close"] - merged["rut_close_ma_20"]
    if "rut_volume" not in merged.columns:
        raise ValueError("rut_volume is required for volume-based features")
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

    output_path = f"{DATA_FOLDER}/master_features.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved master features to {output_path}")
    print("First 5 rows of final dataset:")
    print(merged.head())


if __name__ == "__main__":
    main()
