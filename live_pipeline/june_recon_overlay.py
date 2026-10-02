"""
June 2026 Russell Reconstitution - Macro Regime Overlay (fixed feature alignment)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

FETCH_START = "2025-01-01"
FETCH_END   = "2026-06-27"

FEATURE_COLS = [
    'sp500_close','vix_close','treasury_10y_close','treasury_2y_close',
    'dollar_index_close','gold_close','oil_close','rut_close_return',
    'rut_close_ma_5','rut_close_ma_20','rut_momentum_5','rut_momentum_20',
    'yield_curve_spread','vix_ma_5','rut_rsi_14','rut_bb_upper',
    'rut_bb_lower','rut_macd','rut_macd_signal','rut_volume_ma_5',
    'rut_volume_ma_20'
]

TRADES = {
    "Big-YTD-Gainer Adds (AXTI,WOLF,FCEL,SPCE...)": {"direction":"short_bias","base":2.0},
    "General R2000 Additions (RJET,UMAC,LPTH...)":  {"direction":"long",       "base":1.0},
    "Deletion Basket PRE-recon (flat)":              {"direction":"flat",       "base":0.0},
    "Deletion Basket POST-recon (SNBR,ALDX,DH...)": {"direction":"long",       "base":1.0},
    "Promotion SHORT (FROG,ULS,ASND,TTAN,SYM)":     {"direction":"short",      "base":0.5},
    "Promotion LONG bounce post-recon":              {"direction":"long",       "base":0.5},
}

print("=" * 60)
print("JUNE 2026 RECON MACRO REGIME OVERLAY")
print("=" * 60)

# 1. Download
print("\n[1] Downloading data...")
tickers = {
    "rut":"^RUT","sp500":"^GSPC","vix":"^VIX",
    "treasury_10y":"^TNX","treasury_2y":"^IRX",
    "dollar_index":"DX-Y.NYB","gold":"GC=F","oil":"CL=F"
}
frames = {}
for name, tkr in tickers.items():
    df = yf.download(tkr, start=FETCH_START, end=FETCH_END,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    frames[name] = df
    print(f"    ✓ {name}  ({len(df)} rows)")

# 2. Build the exact 21 features the model was trained on
rut = frames["rut"].copy()
rut.index = pd.to_datetime(rut.index)

def safe_close(name):
    df = frames[name]
    df.index = pd.to_datetime(df.index)
    return df["Close"] if "Close" in df.columns else df.iloc[:,0]

feat = pd.DataFrame(index=rut.index)
feat["sp500_close"]        = safe_close("sp500")
feat["vix_close"]          = safe_close("vix")
feat["treasury_10y_close"] = safe_close("treasury_10y")
feat["treasury_2y_close"]  = safe_close("treasury_2y")
feat["dollar_index_close"] = safe_close("dollar_index")
feat["gold_close"]         = safe_close("gold")
feat["oil_close"]          = safe_close("oil")

rut_close = rut["Close"] if "Close" in rut.columns else rut.iloc[:,0]
feat["rut_close_return"]   = rut_close.pct_change(1)
feat["rut_close_ma_5"]     = rut_close.rolling(5).mean()
feat["rut_close_ma_20"]    = rut_close.rolling(20).mean()
feat["rut_momentum_5"]     = rut_close.pct_change(5)
feat["rut_momentum_20"]    = rut_close.pct_change(20)
feat["yield_curve_spread"] = feat["treasury_10y_close"] - feat["treasury_2y_close"]
feat["vix_ma_5"]           = feat["vix_close"].rolling(5).mean()

# RSI 14
delta = rut_close.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feat["rut_rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Bollinger Bands
bb_mid   = rut_close.rolling(20).mean()
bb_std   = rut_close.rolling(20).std()
feat["rut_bb_upper"] = bb_mid + 2 * bb_std
feat["rut_bb_lower"] = bb_mid - 2 * bb_std

# MACD
ema12 = rut_close.ewm(span=12).mean()
ema26 = rut_close.ewm(span=26).mean()
macd  = ema12 - ema26
feat["rut_macd"]        = macd
feat["rut_macd_signal"] = macd.ewm(span=9).mean()

# Volume MAs (use 0 if no volume)
vol = rut["Volume"] if "Volume" in rut.columns else pd.Series(0, index=rut.index)
feat["rut_volume_ma_5"]  = vol.rolling(5).mean()
feat["rut_volume_ma_20"] = vol.rolling(20).mean()

feat = feat[FEATURE_COLS].dropna()
print(f"\n[2] Features built: {feat.shape[0]} rows × {feat.shape[1]} cols ✓")

# 3. Load & predict
clf = joblib.load("models/lgbm_classifier.pkl")
reg = joblib.load("models/lgbm_regressor.pkl")
print("[3] Models loaded ✓")

proba    = clf.predict_proba(feat.values)
raw_pred = clf.predict(feat.values)
class_map = {0:-1, 1:1, 2:2, 3:-1}
signals  = [class_map.get(int(p), -1) for p in raw_pred]

pred_df = pd.DataFrame({
    "rut_close": rut_close.reindex(feat.index),
    "signal":    signals,
    "bear_prob": proba[:,0],
    "bull_prob": proba[:,2],
}, index=feat.index)

# 4. June window table
window = pred_df[pred_df.index >= pd.Timestamp("2026-05-22")]
print(f"\n[4] SIGNAL TABLE — May 22 through latest available")
print("-" * 58)
print(f"{'Date':<12} {'RUT':>7} {'Signal':>7} {'Bear%':>7} {'Bull%':>7}  Regime")
print("-" * 58)
for dt, row in window.iterrows():
    sig = int(row["signal"])
    regime = "STRONG BULL 🟢" if sig==2 else ("BULL 🟡" if sig==1 else "BEAR 🔴")
    print(f"{str(dt.date()):<12} {row['rut_close']:>7.0f} {sig:>7} "
          f"{row['bear_prob']:>6.1%} {row['bull_prob']:>6.1%}  {regime}")

# 5. Regime summary
bull_days  = sum(1 for s in window["signal"] if s >= 1)
total_days = len(window)
bull_pct   = bull_days / total_days if total_days else 0

if bull_pct >= 0.60:
    regime_label, size_mult = "BULLISH",  1.25
elif bull_pct >= 0.40:
    regime_label, size_mult = "NEUTRAL",  1.00
else:
    regime_label, size_mult = "BEARISH",  0.50

print(f"\n[5] REGIME SUMMARY ({total_days} days in window)")
print(f"    Bullish: {bull_days} days ({bull_pct:.0%})  |  "
      f"Bearish: {total_days-bull_days} days ({1-bull_pct:.0%})")
print(f"    ► JUNE REGIME: {regime_label}  →  size multiplier {size_mult}x")

# 6. Adjusted sizing
print(f"\n[6] ADJUSTED POSITION SIZING")
print("-" * 60)
print(f"{'Trade':<44} {'Base':>6} {'Adj':>7}")
print("-" * 60)
for trade, info in TRADES.items():
    base = info["base"]
    adj  = base * size_mult if info["direction"] != "flat" else 0.0
    warn = ""
    if info["direction"] == "short_bias" and regime_label == "BULLISH": warn = " ⚠"
    if info["direction"] == "long"       and regime_label == "BEARISH": warn = " ⚠"
    print(f"{trade[:43]:<44} {base:>5.1f}%  {adj:>5.2f}%{warn}")

total_base = sum(v["base"] for v in TRADES.values())
total_adj  = sum(v["base"]*size_mult if v["direction"]!="flat" else 0 for v in TRADES.values())
print("-" * 60)
print(f"{'TOTAL GROSS EXPOSURE':<44} {total_base:>5.1f}%  {total_adj:>5.2f}%")

# 7. Hard stop check
rut_series = pred_df["rut_close"].dropna()
if len(rut_series) > 21:
    latest   = rut_series.iloc[-1]
    ago_30   = rut_series.iloc[-21]
    ret_30   = (latest - ago_30) / ago_30
    print(f"\n[7] HARD STOP CHECK")
    print(f"    RUT now: {latest:.1f}  |  30 days ago: {ago_30:.1f}  |  return: {ret_30:+.2%}")
    if ret_30 < -0.08:
        print("    ► ⛔ HARD STOP — market down >8% in 30 days. Close all trades.")
    else:
        print("    ► ✅ No hard stop triggered")

print("\n" + "=" * 60)
print("Overlay complete.")
