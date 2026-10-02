import pandas as pd
import numpy as np

def load(path, close_col, date_col="date_"):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", close_col: "close"})
    return df[["date", "close"]].dropna()

rut = pd.read_csv("data/rut_2024_2025.csv", parse_dates=["date_"])
rut = rut.rename(columns={"date_": "date", "close_^rut": "rut_close", "volume_^rut": "rut_volume"})
rut = rut[["date", "rut_close", "rut_volume"]].dropna()
sp500 = load("data/sp500_2024_2025.csv", "close_^gspc"); sp500.columns = ["date", "sp500_close"]
vix = load("data/vix_2024_2025.csv", "close_^vix"); vix.columns = ["date", "vix_close"]
t10 = load("data/treasury_10y_2024_2025.csv", "close_^tnx"); t10.columns = ["date", "treasury_10y_close"]
t2 = load("data/treasury_2y_2024_2025.csv", "close_^irx"); t2.columns = ["date", "treasury_2y_close"]
dxy = load("data/dollar_index_2024_2025.csv", "close_dx-y.nyb"); dxy.columns = ["date", "dollar_index_close"]
gold = load("data/gold_2024_2025.csv", "close_gc=f"); gold.columns = ["date", "gold_close"]
oil = load("data/oil_2024_2025.csv", "close_cl=f"); oil.columns = ["date", "oil_close"]
df = rut.merge(sp500,on="date").merge(vix,on="date").merge(t10,on="date").merge(t2,on="date").merge(dxy,on="date").merge(gold,on="date").merge(oil,on="date")
df = df.sort_values("date").reset_index(drop=True)
df["rut_close_return"] = df["rut_close"].pct_change()
df["rut_close_ma_5"] = df["rut_close"].rolling(5).mean()
df["rut_close_ma_20"] = df["rut_close"].rolling(20).mean()
df["rut_momentum_5"] = df["rut_close"].pct_change(5)
df["rut_momentum_20"] = df["rut_close"].pct_change(20)
df["yield_curve_spread"] = df["treasury_10y_close"] - df["treasury_2y_close"]
df["vix_ma_5"] = df["vix_close"].rolling(5).mean()
delta = df["rut_close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["rut_rsi_14"] = 100 - (100 / (1 + gain / loss))
df["rut_bb_upper"] = df["rut_close"].rolling(20).mean() + 2 * df["rut_close"].rolling(20).std()
df["rut_bb_lower"] = df["rut_close"].rolling(20).mean() - 2 * df["rut_close"].rolling(20).std()
ema12 = df["rut_close"].ewm(span=12).mean()
ema26 = df["rut_close"].ewm(span=26).mean()
df["rut_macd"] = ema12 - ema26
df["rut_macd_signal"] = df["rut_macd"].ewm(span=9).mean()
df["rut_volume_ma_5"] = df["rut_volume"].rolling(5).mean()
df["rut_volume_ma_20"] = df["rut_volume"].rolling(20).mean()
df = df.dropna()
df2024 = df[df["date"] <= "2024-12-31"]
df2024.to_csv("data/features_2024.csv", index=False)
print(f"Saved {len(df2024)} rows to data/features_2024.csv")
print(df2024["date"].min(), "to", df2024["date"].max())
