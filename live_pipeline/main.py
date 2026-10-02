"""
main.py — Master pipeline for the Russell 2026 reconstitution daily brief.

Pipeline steps
--------------
1. Pull fresh market data (all 8 macro tickers, 60-day rolling window)
2. Build the exact 21-column feature set the models were trained on
3. Run lgbm_classifier.pkl + lgbm_regressor.pkl
4. Compute macro regime + hard-stop (june_recon_overlay logic)
5. Score Russell additions / deletions / promotions (russell_candidates_2026)
6. Write backtests/daily_brief.txt

Run with:
    python main.py
"""

from __future__ import annotations
import datetime
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# Reuse feature-building and signal logic from daily_update
from daily_update import FEATURES, build_features, build_signal

# Reuse Russell candidate lists and scoring from russell_candidates_2026
from russell_candidates_2026 import (
    ADDITIONS,
    DELETIONS,
    EFFECTIVE_DATE,
    RUSSELL1000_MCAP_THRESHOLD,
    fetch_data,
    fmt_mcap,
    fmt_pct,
    fmt_price,
    score_additions,
    score_deletions,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASSIFIER_PATH = "models/lgbm_classifier.pkl"
REGRESSOR_PATH  = "models/lgbm_regressor.pkl"
BRIEF_PATH      = "backtests/daily_brief.txt"
LOOKBACK_DAYS   = 60   # 20-day rolling windows need ≥20 rows after dropna

# Trade categories and base sizes (from june_recon_overlay)
TRADE_CATEGORIES: list[dict] = [
    {"label": "Big-YTD-Gainer Adds (short_bias)",  "direction": "short_bias", "base": 2.0},
    {"label": "General R2000 Additions (long)",     "direction": "long",       "base": 1.0},
    {"label": "Deletion Basket PRE-recon (flat)",   "direction": "flat",       "base": 0.0},
    {"label": "Deletion Basket POST-recon (long)",  "direction": "long",       "base": 1.0},
    {"label": "Promotion SHORT",                    "direction": "short",      "base": 0.5},
    {"label": "Promotion LONG bounce",              "direction": "long",       "base": 0.5},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}")
    print(f"     {'─' * 56}")


def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def compute_regime(signal_df: pd.DataFrame, window: int = 30) -> tuple[str, float]:
    """Return (label, size_multiplier) based on last `window` days of signals."""
    recent   = signal_df.tail(window)
    bull_days = int((recent["signal"] >= 1).sum())
    total     = len(recent)
    bull_pct  = bull_days / total if total > 0 else 0.0
    if bull_pct >= 0.60:
        return "BULLISH", 1.25
    if bull_pct >= 0.40:
        return "NEUTRAL", 1.00
    return "BEARISH", 0.50


def check_hard_stop(signal_df: pd.DataFrame) -> tuple[bool, float]:
    """Return (stop_triggered, 30d_rut_return). Stop fires if RUT down >8% in 30 days."""
    prices = signal_df["rut_close"].dropna()
    if len(prices) < 22:
        return False, 0.0
    ret = float(prices.iloc[-1]) / float(prices.iloc[-22]) - 1.0
    return ret < -0.08, ret


# ---------------------------------------------------------------------------
# Brief builder
# ---------------------------------------------------------------------------

def build_brief(
    today: datetime.date,
    days_left: int,
    signal_df: pd.DataFrame,
    latest: pd.Series,
    reg_pred: float,
    regime_label: str,
    size_mult: float,
    hard_stop: bool,
    ret_30: float,
    df_add: pd.DataFrame,
    df_del: pd.DataFrame,
) -> str:
    lines: list[str] = []
    ln = lines.append

    paradox = df_add[df_add["market_cap"] > RUSSELL1000_MCAP_THRESHOLD]
    paradox_tickers = list(paradox["ticker"]) if not paradox.empty else []

    # ── Header ───────────────────────────────────────────────────────────────
    ln(_sep("="))
    ln("RUSSELL 2026 RECONSTITUTION — DAILY BRIEF")
    ln(f"Run date   : {today.strftime('%Y-%m-%d (%A)')}")
    ln(f"Effective  : {EFFECTIVE_DATE}  |  {days_left} days away")
    ln(_sep("="))

    # ── Market snapshot ───────────────────────────────────────────────────────
    ln("")
    ln("MARKET SNAPSHOT")
    ln(_sep())
    ln(f"  RUT (Russell 2000) : {float(latest.get('rut_close', float('nan'))):>9.2f}")
    ln(f"  S&P 500            : {float(latest.get('sp500_close', float('nan'))):>9.2f}")
    ln(f"  VIX                : {float(latest.get('vix_close', float('nan'))):>9.2f}")
    ln(f"  10Y-2Y spread      : {float(latest.get('yield_curve_spread', float('nan'))):>+9.3f}")
    ln(f"  RSI-14 (RUT)       : {float(latest.get('rut_rsi_14', float('nan'))):>9.1f}")
    ln(f"  MACD               : {float(latest.get('rut_macd', float('nan'))):>+9.4f}")
    ln(f"  Regressor forecast : {reg_pred:>+9.4f}  (predicted next-day RUT return)")

    # ── Signal table (last 5 trading days) ────────────────────────────────────
    ln("")
    ln("MODEL SIGNALS — LAST 5 TRADING DAYS")
    ln(_sep())
    tail5 = signal_df[["date", "rut_close", "signal", "bull_prob"]].tail(5)
    for i, (_, row) in enumerate(tail5.iterrows()):
        dt_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        sig    = int(row["signal"])
        bp     = float(row["bull_prob"])
        label  = {2: "STRONG BULL", 1: "BULL       ", -1: "BEAR       "}.get(sig, "NEUTRAL    ")
        today_marker = "  ← TODAY" if i == len(tail5) - 1 else ""
        ln(f"  {dt_str}  RUT {float(row['rut_close']):>8.1f}  sig={sig}  bull={bp:>5.1%}  {label}{today_marker}")

    # ── Macro regime ──────────────────────────────────────────────────────────
    ln("")
    ln("MACRO REGIME  (june_recon_overlay)")
    ln(_sep())
    recent30  = signal_df.tail(30)
    bull_days = int((recent30["signal"] >= 1).sum())
    bear_days = len(recent30) - bull_days
    ln(f"  Regime       : {regime_label}")
    ln(f"  Window       : last {len(recent30)} trading days")
    ln(f"  Bull days    : {bull_days}  ({bull_days/len(recent30):.0%})")
    ln(f"  Bear days    : {bear_days}  ({bear_days/len(recent30):.0%})")
    ln(f"  Size mult    : {size_mult:.2f}x")

    # ── Hard stop ─────────────────────────────────────────────────────────────
    ln("")
    ln("HARD STOP CHECK")
    ln(_sep())
    ln(f"  30-day RUT return : {ret_30:>+.2%}  (threshold: -8.00%)")
    if hard_stop:
        ln("  STATUS  : *** HARD STOP TRIGGERED — market down >8% in 30 days ***")
        ln("            CLOSE ALL OPEN RECON TRADES IMMEDIATELY")
    else:
        ln("  STATUS  : No hard stop triggered")

    # ── Adjusted position sizing ───────────────────────────────────────────────
    ln("")
    ln("ADJUSTED POSITION SIZING")
    ln(_sep())
    ln(f"  {'Category':<43} {'Base':>6}  {'Adj':>6}  Flags")
    ln("  " + _sep("─", 62))
    total_base = 0.0
    total_adj  = 0.0
    for cat in TRADE_CATEGORIES:
        base = cat["base"]
        adj  = round(base * size_mult, 2) if cat["direction"] != "flat" else 0.0
        total_base += base
        total_adj  += adj
        flags: list[str] = []
        if hard_stop:
            flags.append("⛔ HARD STOP")
        elif cat["direction"] == "short_bias" and regime_label == "BULLISH":
            flags.append("⚠ short_bias in BULLISH mkt")
        elif cat["direction"] == "long" and regime_label == "BEARISH":
            flags.append("⚠ long in BEARISH mkt")
        flag_str = "  " + " | ".join(flags) if flags else ""
        ln(f"  {cat['label']:<43} {base:>5.1f}%  {adj:>5.2f}%{flag_str}")
    ln("  " + _sep("─", 62))
    ln(f"  {'TOTAL GROSS EXPOSURE':<43} {total_base:>5.1f}%  {total_adj:>5.2f}%")

    # ── Top 10 additions ──────────────────────────────────────────────────────
    ln("")
    ln("TOP 10 ADDITIONS TO BUY  (40% momentum · 40% mcap pressure · 20% vol impact)")
    ln(_sep())
    ln(f"  {'#':<4} {'Ticker':<7} {'Score':>5}  {'Price':>8}  {'Mkt Cap':>10}  {'YTD Ret':>8}  {'30d Mom':>8}")
    ln("  " + _sep("─", 62))
    for rank, (_, row) in enumerate(df_add.head(10).iterrows(), start=1):
        flag = " *" if row["ticker"] in paradox_tickers else "  "
        ln(
            f"  {rank:<4} {row['ticker']:<7} {row['final_score']:>5.2f}  "
            f"{fmt_price(row['price']):>8}  {fmt_mcap(row['market_cap']):>10}  "
            f"{fmt_pct(row['ytd_return']):>8}  {fmt_pct(row['momentum_30d']):>8}"
            f"{flag}"
        )
    if any(t in paradox_tickers for t in df_add.head(10)["ticker"]):
        ln("  * = Promotion paradox — see section below")

    # ── Top 10 deletions ──────────────────────────────────────────────────────
    ln("")
    ln("TOP 10 DELETIONS TO SHORT  (60% decline severity · 40% mcap pressure)")
    ln(_sep())
    ln(f"  {'#':<4} {'Ticker':<7} {'Score':>5}  {'Price':>8}  {'Mkt Cap':>10}  {'YTD Ret':>8}")
    ln("  " + _sep("─", 62))
    for rank, (_, row) in enumerate(df_del.head(10).iterrows(), start=1):
        ln(
            f"  {rank:<4} {row['ticker']:<7} {row['final_score']:>5.2f}  "
            f"{fmt_price(row['price']):>8}  {fmt_mcap(row['market_cap']):>10}  "
            f"{fmt_pct(row['ytd_return']):>8}"
        )

    # ── Promotion paradox ─────────────────────────────────────────────────────
    ln("")
    ln("PROMOTION PARADOX  (additions >$5.7B entering Russell 1000 = net SELL)")
    ln(_sep())
    ln("  IWM (R2000) must SELL their ~9% float stake; R1000 trackers only add ~4%")
    ln("  Net result: ~5% of float forced into the market as sellers on June 26")
    ln("  " + _sep("─", 62))
    if paradox.empty:
        ln("  None detected — no additions above the $5.7B Russell 1000 breakpoint.")
    else:
        ln(f"  {'Ticker':<7} {'Price':>8}  {'Mkt Cap':>10}  {'YTD Ret':>8}  Action")
        ln("  " + _sep("─", 62))
        for _, row in paradox.iterrows():
            ln(
                f"  {row['ticker']:<7} {fmt_price(row['price']):>8}  "
                f"{fmt_mcap(row['market_cap']):>10}  {fmt_pct(row['ytd_return']):>8}  "
                "SELL / SHORT into June 26 close"
            )

    # ── Full ticker lists by category ─────────────────────────────────────────
    ln("")
    ln("FULL TICKER LISTS BY CATEGORY")
    ln(_sep())
    buy_tickers = [t for t in df_add["ticker"] if t not in paradox_tickers]
    del_tickers = list(df_del["ticker"])
    ln(f"  ADDITIONS — BUY  ({len(buy_tickers)} tickers):")
    ln("    " + ", ".join(buy_tickers))
    ln(f"  PROMOTION PARADOX — SELL/SHORT  ({len(paradox_tickers)} tickers):")
    ln("    " + (", ".join(paradox_tickers) or "none"))
    ln(f"  DELETIONS — SHORT  ({len(del_tickers)} tickers):")
    ln("    " + ", ".join(del_tickers))

    # ── Flags & warnings ──────────────────────────────────────────────────────
    ln("")
    ln("FLAGS AND WARNINGS")
    ln(_sep())
    issued: list[str] = []

    if hard_stop:
        issued.append("  ⛔ HARD STOP ACTIVE — close all open recon trades immediately")

    if regime_label == "BEARISH":
        issued.append("  ⚠  BEARISH regime — cut long sizes to 0.50x; avoid new longs")

    if regime_label == "BULLISH" and any(c["direction"] == "short_bias" for c in TRADE_CATEGORIES):
        issued.append("  ⚠  Short-bias trade active in a BULLISH regime — monitor closely")

    vix_now = float(latest.get("vix_close", 0) or 0)
    if vix_now > 30:
        issued.append(f"  ⚠  VIX elevated at {vix_now:.1f} — high volatility; reduce position sizes")

    if days_left <= 5:
        issued.append(f"  ⚠  ONLY {days_left} DAYS TO EFFECTIVE DATE — final positioning window")
    elif days_left <= 10:
        issued.append(f"  ℹ  {days_left} days to effective date — begin scaling into positions")

    if paradox_tickers:
        issued.append(f"  ℹ  Promotion paradox: {', '.join(paradox_tickers)} — net sell pressure on June 26")

    if float(latest.get("rut_rsi_14", 50) or 50) > 70:
        issued.append("  ⚠  RUT RSI-14 > 70 — overbought; additions may already be front-run")

    if issued:
        for flag in issued:
            ln(flag)
    else:
        ln("  None.")

    # ── Footer ───────────────────────────────────────────────────────────────
    ln("")
    ln(_sep("="))
    ln(f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ln(f"  Output    : {BRIEF_PATH}")
    ln(_sep("="))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    today     = datetime.date.today()
    days_left = max((EFFECTIVE_DATE - today).days, 0)

    print(_sep("="))
    print("RUSSELL 2026 RECONSTITUTION — MASTER PIPELINE")
    print(f"Date   : {today}")
    print(f"T-minus: {days_left} days to June 26 effective date")
    print(_sep("="))

    os.makedirs("backtests", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Step 1: Pull fresh market data ────────────────────────────────────────
    _step(1, "Pulling fresh market data (8 macro tickers, 60-day window)...")
    start_dt = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_dt   = (today + datetime.timedelta(days=1)).isoformat()
    print(f"     Window: {start_dt} → {end_dt}")

    # ── Step 2: Build 21-column feature set ──────────────────────────────────
    _step(2, "Building 21-column feature set...")
    df = build_features(start_dt, end_dt)
    if df.empty:
        print("ERROR: No feature rows returned. Check network / data sources.")
        sys.exit(1)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        sys.exit(1)
    print(f"     {len(df)} rows × {len(df.columns)} columns  |  {len(FEATURES)} model features confirmed")

    # ── Step 3: Run predictions ───────────────────────────────────────────────
    _step(3, "Loading models and running predictions...")
    for path in (CLASSIFIER_PATH, REGRESSOR_PATH):
        if not os.path.exists(path):
            print(f"ERROR: Model not found: {path}")
            print("       Run train_model.py first to generate the model files.")
            sys.exit(1)

    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    print(f"     Classifier  : {CLASSIFIER_PATH}")
    print(f"     Regressor   : {REGRESSOR_PATH}")

    signal_df   = build_signal(df, clf)
    latest      = signal_df.iloc[-1]
    X_latest    = df[FEATURES].iloc[[-1]]
    reg_pred    = float(reg.predict(X_latest)[0])

    print(f"     Today signal: {int(latest['signal'])}  |  bull_prob={float(latest['bull_prob']):.1%}")
    print(f"     Regressor   : {reg_pred:+.4f}  (next-day RUT return forecast)")

    # ── Step 4: Regime overlay (june_recon_overlay logic) ─────────────────────
    _step(4, "Computing macro regime (june_recon_overlay logic)...")
    regime_label, size_mult = compute_regime(signal_df)
    hard_stop, ret_30       = check_hard_stop(signal_df)

    recent30  = signal_df.tail(30)
    bull_days = int((recent30["signal"] >= 1).sum())
    print(f"     Window     : last {len(recent30)} trading days")
    print(f"     Bull days  : {bull_days}  ({bull_days/len(recent30):.0%})")
    print(f"     Regime     : {regime_label}  →  {size_mult:.2f}x size multiplier")
    print(f"     30d return : {ret_30:+.2%}  |  Hard stop: {'*** YES ***' if hard_stop else 'No'}")

    # ── Step 5: Russell candidates ────────────────────────────────────────────
    _step(5, f"Downloading Russell candidates ({len(ADDITIONS)} additions, {len(DELETIONS)} deletions)...")
    print("     Additions:")
    df_add_raw = fetch_data(ADDITIONS)
    print("\n     Deletions:")
    df_del_raw = fetch_data(DELETIONS)

    print("\n     Scoring...")
    df_add = score_additions(df_add_raw)
    df_del = score_deletions(df_del_raw)

    paradox_count = int((df_add["market_cap"] > RUSSELL1000_MCAP_THRESHOLD).sum())
    print(f"     Top add : {df_add['ticker'].iloc[0]}  (score {df_add['final_score'].iloc[0]:.2f})")
    print(f"     Top del : {df_del['ticker'].iloc[0]}  (score {df_del['final_score'].iloc[0]:.2f})")
    print(f"     Promotion paradox: {paradox_count} ticker(s) above $5.7B threshold")

    # ── Step 6: Write daily brief ─────────────────────────────────────────────
    _step(6, f"Writing daily brief → {BRIEF_PATH}")
    brief = build_brief(
        today        = today,
        days_left    = days_left,
        signal_df    = signal_df,
        latest       = latest,
        reg_pred     = reg_pred,
        regime_label = regime_label,
        size_mult    = size_mult,
        hard_stop    = hard_stop,
        ret_30       = ret_30,
        df_add       = df_add,
        df_del       = df_del,
    )

    with open(BRIEF_PATH, "w", encoding="utf-8") as fh:
        fh.write(brief)

    print(f"     Saved: {BRIEF_PATH}")

    # Print the brief to terminal too
    print()
    print(brief)

    print(_sep("="))
    print("Pipeline complete.")
    print(_sep("="))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise
