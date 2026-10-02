import yfinance as yf
import pandas as pd

DATA_FOLDER = "data"
DATE_START = "2010-01-01"
DATE_END = "2023-12-31"

TICKERS = {
    "treasury_10y": "^TNX",
    "treasury_2y": "^IRX",
    "dollar_index": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
}


def download_and_save(symbol: str, filename: str) -> None:
    print(f"Downloading {symbol} from {DATE_START} to {DATE_END}...")
    df = yf.download(symbol, start=DATE_START, end=DATE_END, progress=False)
    print(f"Downloaded {symbol}: {len(df)} rows")

    print(f"Cleaning {symbol} data...")
    df = df.dropna().reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip().lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    print(f"Cleaned {symbol} data: {len(df)} rows after dropping nulls")

    output_path = f"{DATA_FOLDER}/{filename}"
    df.to_csv(output_path, index=False)
    print(f"Saved {symbol} data to {output_path}\n")


if __name__ == "__main__":
    for filename, symbol in TICKERS.items():
        download_and_save(symbol, f"{filename}.csv")
    print("All macro downloads complete.")
