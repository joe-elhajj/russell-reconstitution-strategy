"""
dashboard.py — Professional quant terminal brief for Russell 2026 reconstitution.
Runs the full pipeline and writes a box-bordered text report.

Override tickers by creating data/candidates_override.csv with columns:
    ticker, category   (category: addition | deletion | promotion)

Run with:
    python dashboard.py
"""

from __future__ import annotations
import contextlib
import datetime
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from daily_update import FEATURES, build_features, build_signal
from main import (
    BRIEF_PATH,
    CLASSIFIER_PATH,
    LOOKBACK_DAYS,
    REGRESSOR_PATH,
    TRADE_CATEGORIES,
    check_hard_stop,
    compute_regime,
)
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
# Layout constants
# ---------------------------------------------------------------------------

W = 76
OVERRIDE_PATH = "data/candidates_override.csv"

KEY_DATES = [
    (datetime.date(2026, 5, 29), "Weekly update #1"),
    (datetime.date(2026, 6, 5),  "Weekly update #2"),
    (datetime.date(2026, 6, 8),  "Lockdown — no further index changes"),
    (datetime.date(2026, 6, 12), "Weekly update #3"),
    (datetime.date(2026, 6, 18), "Weekly update #4 (final)"),
    (datetime.date(2026, 6, 26), "EFFECTIVE DATE — rebalance close"),
]
WEEKLY_UPDATE_DATES = [dt for dt, lbl in KEY_DATES if "Weekly update" in lbl]

RATIONALE = {
    "Big-YTD-Gainer Adds (short_bias)":  "YTD gainers face mean-reversion once forced buying peaks at close",
    "General R2000 Additions (long)":    "Forced passive buying (IWM ~$75B AUM) creates temporary premium",
    "Deletion Basket PRE-recon (flat)":  "Hold off — forced selling not yet locked in before June 26",
    "Deletion Basket POST-recon (long)": "Buy the dip; forced selling exhausts itself post-June 26",
    "Promotion SHORT":                   "Net ~5% float for sale: R2000 exits, R1000 only adds ~4%",
    "Promotion LONG bounce":             "Post-recon recovery as new R1000 passive buyers absorb supply",
}


# ---------------------------------------------------------------------------
# Box-drawing helpers
# ---------------------------------------------------------------------------

def heavy(title: str = "") -> str:
    if not title:
        return "═" * W
    inner = f" {title} "
    pad = W - len(inner)
    left = pad // 2
    return "═" * left + inner + "═" * (pad - left)


def light() -> str:
    return "─" * W


def sec(title: str) -> str:
    slug = f"── {title} "
    return slug + "─" * max(0, W - len(slug))


def col(s: str, width: int, align: str = "<") -> str:
    s = str(s)[:width]
    return format(s, f"{align}{width}")


def bar(score: float, width: int = 10) -> str:
    n = max(0, min(width, round(score / 10.0 * width)))
    return "█" * n + "░" * (width - n)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def conviction_score(bull_pct: float, sig: int, hard_stop: bool, days_left: int) -> float:
    s = bull_pct * 4.0
    if sig == 1:          # as spec'd: +2 for signal == 1 (BULL)
        s += 2.0
    if not hard_stop:
        s += 2.0
    if 5 <= days_left <= 25:
        s += 2.0
    return round(min(s, 10.0), 1)


def conviction_label(score: float) -> str:
    if score >= 8.0:
        return "VERY HIGH"
    if score >= 6.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def reg_direction(pred: float) -> str:
    if pred > 0.003:  return "BULLISH"
    if pred > 0.0:    return "SLIGHTLY BULLISH"
    if pred < -0.003: return "BEARISH"
    if pred < 0.0:    return "SLIGHTLY BEARISH"
    return "FLAT"


def setup_label(ytd, action: str) -> str:
    if action == "SELL":
        return "PARADOX"
    if action == "SHORT":
        return "FLOW"
    if pd.notna(ytd) and float(ytd) > 1.0:
        return "MOMENTUM"
    return "FLOW"


def next_weekly(today: datetime.date) -> tuple[datetime.date, int] | None:
    for dt in WEEKLY_UPDATE_DATES:
        if dt >= today:
            return dt, (dt - today).days
    return None


# ---------------------------------------------------------------------------
# Candidate loader — supports data/candidates_override.csv
# ---------------------------------------------------------------------------

def load_candidates() -> tuple[list[str], list[str], list[str] | None, str]:
    """Return (additions, deletions, promotions_override, source_label).
    promotions_override=None → use market-cap threshold for paradox detection.
    """
    if os.path.exists(OVERRIDE_PATH):
        try:
            ov = pd.read_csv(OVERRIDE_PATH)
            if not ov.empty and {"ticker", "category"}.issubset(ov.columns):
                def pick(cat: str) -> list[str]:
                    return list(
                        ov[ov["category"] == cat]["ticker"]
                        .str.strip().str.upper()
                    )
                adds, dels, promo = pick("addition"), pick("deletion"), pick("promotion")
                msg = (f"OVERRIDE  {OVERRIDE_PATH}  "
                       f"({len(adds)} adds / {len(dels)} dels / {len(promo)} promo)")
                print(f"    [candidates] {msg}")
                return adds, dels, promo, msg
        except Exception as exc:
            print(f"    [candidates] Override load failed ({exc}) — using hardcoded lists")

    msg = f"HARDCODED  ({len(ADDITIONS)} adds / {len(DELETIONS)} dels)"
    print(f"    [candidates] {msg}")
    return list(ADDITIONS), list(DELETIONS), None, msg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    today     = datetime.date.today()
    days_left = max((EFFECTIVE_DATE - today).days, 0)

    # ── Step 1: fresh market data ─────────────────────────────────────────
    print("[1] Pulling fresh market data...")
    start_dt = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_dt   = (today + datetime.timedelta(days=1)).isoformat()
    df = build_features(start_dt, end_dt)
    if df.empty:
        raise RuntimeError("No feature rows returned — check network / data sources")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing model features: {missing}")
    print(f"    {len(df)} rows × {len(df.columns)} cols  |  21 features confirmed")

    # ── Step 2: models ────────────────────────────────────────────────────
    print("[2] Loading models...")
    for path in (CLASSIFIER_PATH, REGRESSOR_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}  (run train_model.py first)")
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)

    # ── Step 3: predictions ───────────────────────────────────────────────
    print("[3] Running predictions...")
    signal_df = build_signal(df, clf)
    latest    = signal_df.iloc[-1]
    X_latest  = df[FEATURES].iloc[[-1]]
    reg_pred  = float(reg.predict(X_latest)[0])
    print(f"    signal={int(latest['signal'])}  bull_prob={float(latest['bull_prob']):.1%}"
          f"  reg={reg_pred:+.4f}")

    # ── Step 4: regime overlay ────────────────────────────────────────────
    print("[4] Computing macro regime...")
    regime_label, size_mult = compute_regime(signal_df)
    hard_stop, ret_30       = check_hard_stop(signal_df)
    recent30  = signal_df.tail(30)
    bull_days = int((recent30["signal"] >= 1).sum())
    print(f"    {regime_label}  mult={size_mult:.2f}x  "
          f"bull={bull_days}/{len(recent30)}  hard_stop={hard_stop}  30d={ret_30:+.2%}")

    # ── Step 5: Russell candidates ────────────────────────────────────────
    print("[5] Loading candidate lists...")
    adds, dels, promo_override, source_msg = load_candidates()

    print(f"    Fetching {len(adds)} additions...")
    with contextlib.redirect_stdout(io.StringIO()):
        df_add_raw = fetch_data(adds)
    print(f"    Fetching {len(dels)} deletions...")
    with contextlib.redirect_stdout(io.StringIO()):
        df_del_raw = fetch_data(dels)

    df_add = score_additions(df_add_raw)
    df_del = score_deletions(df_del_raw)

    paradox = (
        df_add[df_add["ticker"].isin(promo_override)].copy()
        if promo_override is not None
        else df_add[df_add["market_cap"] > RUSSELL1000_MCAP_THRESHOLD].copy()
    )

    # 1-day RUT change
    rut_delta, rut_pct = 0.0, 0.0
    if len(signal_df) >= 2:
        prev_rut  = float(signal_df["rut_close"].iloc[-2])
        curr_rut  = float(latest["rut_close"])
        if prev_rut > 0:
            rut_delta = curr_rut - prev_rut
            rut_pct   = rut_delta / prev_rut

    return dict(
        today=today, days_left=days_left, signal_df=signal_df,
        latest=latest, reg_pred=reg_pred, regime_label=regime_label,
        size_mult=size_mult, hard_stop=hard_stop, ret_30=ret_30,
        df_add=df_add, df_del=df_del, paradox=paradox,
        rut_delta=rut_delta, rut_pct=rut_pct, source_msg=source_msg,
    )


# ---------------------------------------------------------------------------
# Brief builder
# ---------------------------------------------------------------------------

def build_dashboard(ctx: dict) -> str:
    lines: list[str] = []
    ln = lines.append

    today         = ctx["today"]
    days_left     = ctx["days_left"]
    signal_df     = ctx["signal_df"]
    latest        = ctx["latest"]
    reg_pred      = ctx["reg_pred"]
    regime_label  = ctx["regime_label"]
    size_mult     = ctx["size_mult"]
    hard_stop     = ctx["hard_stop"]
    ret_30        = ctx["ret_30"]
    df_add        = ctx["df_add"]
    df_del        = ctx["df_del"]
    paradox       = ctx["paradox"]
    rut_delta     = ctx["rut_delta"]
    rut_pct       = ctx["rut_pct"]
    source_msg    = ctx["source_msg"]

    paradox_tickers = list(paradox["ticker"]) if not paradox.empty else []
    sig             = int(latest["signal"])
    bull_prob       = float(latest["bull_prob"])
    recent30        = signal_df.tail(30)
    bull_days       = int((recent30["signal"] >= 1).sum())
    bull_pct        = bull_days / max(len(recent30), 1)
    cvs             = conviction_score(bull_pct, sig, hard_stop, days_left)
    cvl             = conviction_label(cvs)

    # ─────────────────────────────────────────────────────────────────────
    # 1. HEADER
    # ─────────────────────────────────────────────────────────────────────
    ln(heavy())
    ln(f"  RUSSELL 2026 RECON — DAILY BRIEF")
    ln(f"  {today.strftime('%A, %B %d %Y')}   │   T-{days_left} to June 26 effective date")
    ln(heavy())

    # ─────────────────────────────────────────────────────────────────────
    # 2. MARKET SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("MARKET SNAPSHOT"))

    rut_px = float(latest.get("rut_close", 0) or 0)
    spx_px = float(latest.get("sp500_close", 0) or 0)
    vix_px = float(latest.get("vix_close", 0) or 0)
    ycs    = float(latest.get("yield_curve_spread", 0) or 0)
    rsi    = float(latest.get("rut_rsi_14", 0) or 0)

    ln(f"  RUT   {rut_px:>9,.2f}   {rut_delta:>+8.2f} ({rut_pct:>+6.2%})"
       f"   │   VIX  {vix_px:>6.2f}   │   SPX  {spx_px:>9,.2f}")
    ln(f"  10Y-2Y spread: {ycs:>+.3f}   │   RSI-14: {rsi:.1f}"
       f"   │   Days to recon: {days_left}")

    # ─────────────────────────────────────────────────────────────────────
    # 3. REGIME BLOCK
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("MACRO REGIME"))

    sig_label = {2: "STRONG BULL", 1: "BULL", -1: "BEAR"}.get(sig, str(sig))
    rd        = reg_direction(reg_pred)

    ln(f"  REGIME: {regime_label:<11}│   Bull {bull_days}/{len(recent30)}d"
       f" ({bull_pct:.0%})   │   Multiplier: {size_mult:.2f}×")
    ln(f"  Signal: {sig} – {sig_label:<13}│   Bull prob: {bull_prob:.1%}"
       f"   │   Reg forecast: {reg_pred:>+.4f} ({rd})")
    ln("")
    ln(f"  CONVICTION   {bar(cvs)}   {cvs:.1f}/10   {cvl}")

    # ─────────────────────────────────────────────────────────────────────
    # 4. HARD STOP
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("HARD STOP"))

    stop_str = "⛔ TRIGGERED — CLOSE ALL TRADES" if hard_stop else "✓  CLEAR"
    gap      = ret_30 - (-0.08)
    nxt      = next_weekly(today)
    nxt_str  = (f"Next weekly update: {nxt[0].strftime('%b %d')} ({nxt[1]}d)"
                if nxt else "No more weekly updates scheduled")

    ln(f"  Status : {stop_str}")
    ln(f"  30d RUT: {ret_30:>+.2%}   │   Trigger: -8.00%"
       f"   │   Gap to trigger: {gap:>+.2%}")
    ln(f"  {nxt_str}")

    # ─────────────────────────────────────────────────────────────────────
    # 5. POSITION SIZING TABLE
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("POSITION SIZING"))

    big_ytd_set     = set(df_add.nlargest(5, "ytd_return")["ticker"])
    paradox_set     = set(paradox_tickers)
    non_promo_non_big = df_add[~df_add["ticker"].isin(paradox_set | big_ytd_set)]

    def bucket_top3(label: str) -> str:
        if "Big-YTD-Gainer" in label:
            t = list(df_add[df_add["ticker"].isin(big_ytd_set)].nlargest(3, "ytd_return")["ticker"])
        elif "General R2000" in label:
            t = list(non_promo_non_big.head(3)["ticker"])
        elif "PRE-recon" in label:
            return "—"
        elif "POST-recon" in label:
            t = list(df_del.head(3)["ticker"])
        elif "Promotion" in label:
            t = paradox_tickers[:3]
        else:
            t = []
        return ", ".join(t) if t else "—"

    total_base, total_adj = 0.0, 0.0
    for cat in TRADE_CATEGORIES:
        base = cat["base"]
        adj  = round(base * size_mult, 2) if cat["direction"] != "flat" else 0.0
        total_base += base
        total_adj  += adj

        flags: list[str] = []
        if hard_stop and cat["direction"] not in ("flat",):
            flags.append("⛔")
        elif cat["direction"] == "short_bias" and regime_label == "BULLISH":
            flags.append("⚠")
        elif cat["direction"] == "long" and regime_label == "BEARISH":
            flags.append("⚠")
        flag_str = "  " + " ".join(flags) if flags else ""

        tickers_str = bucket_top3(cat["label"])
        adj_str     = f"{adj:.2f}%" if cat["direction"] != "flat" else " flat"

        ln(f"  {cat['label']:<35}  {cat['direction']:<11}  {base:.1f}% → {adj_str}{flag_str}")
        ln(f"    » {tickers_str}")
        ln(f"    {RATIONALE.get(cat['label'], '')}")
        ln("")

    ln(light())
    ln(f"  {'TOTAL GROSS EXPOSURE':<35}  {'':11}  {total_base:.1f}% → {total_adj:.2f}%")

    # ─────────────────────────────────────────────────────────────────────
    # 6. RANKED TRADE IDEAS  (top 5)
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("RANKED TRADE IDEAS  (top 5)"))

    ideas: list[dict] = []
    for _, row in df_add.iterrows():
        action = "SELL" if row["ticker"] in paradox_set else "BUY"
        ytd    = row.get("ytd_return")
        ideas.append(dict(
            action=action, ticker=row["ticker"],
            price=row.get("price"), ytd=ytd,
            score=row.get("final_score", 0),
            setup=setup_label(ytd, action),
        ))
    for _, row in df_del.iterrows():
        ytd = row.get("ytd_return")
        ideas.append(dict(
            action="SHORT", ticker=row["ticker"],
            price=row.get("price"), ytd=ytd,
            score=row.get("final_score", 0),
            setup=setup_label(ytd, "SHORT"),
        ))
    ideas.sort(key=lambda x: float(x["score"]) if pd.notna(x["score"]) else 0.0, reverse=True)

    ln(f"  {'#':<3} │ {'Action':<5} │ {'Ticker':<7} │ {'Price':>8} │"
       f" {'YTD':>8} │ {'Score':>5} │ Setup")
    ln(f"  ────┼───────┼─────────┼──────────┼"
       f"──────────┼───────┼──────────")
    for i, idea in enumerate(ideas[:5], start=1):
        p_str = fmt_price(idea["price"])
        y_str = fmt_pct(idea["ytd"]) if pd.notna(idea.get("ytd")) else "   N/A"
        score = idea["score"]
        s_str = f"{score:.2f}" if pd.notna(score) else "  N/A"
        ln(f"  {i:<3} │ {idea['action']:<5} │ {idea['ticker']:<7} │"
           f" {p_str:>8} │ {y_str:>8} │ {s_str:>5} │ {idea['setup']}")

    # ─────────────────────────────────────────────────────────────────────
    # 7. RISK FLAGS  (only printed if triggered)
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("RISK FLAGS"))

    risk_flags: list[str] = []

    # DANGER — YTD > 200% in long bucket
    danger = [
        row["ticker"] for _, row in df_add.iterrows()
        if pd.notna(row.get("ytd_return"))
        and float(row["ytd_return"]) > 2.0
        and row["ticker"] not in paradox_set
    ]
    if danger:
        risk_flags.append(
            f"  DANGER    │ {', '.join(danger)}"
            f" — YTD >200% in long bucket; forced buy may be priced in"
        )

    # CONFLICT — ticker in both a LONG bucket and a SHORT bucket simultaneously
    # long  = general additions (bought) + post-recon deletions (bought)
    # short = big-YTD additions (short_bias) + promotion paradox (sold)
    long_tickers  = set(non_promo_non_big["ticker"]) | set(df_del["ticker"])
    short_tickers = big_ytd_set | paradox_set
    conflict      = sorted(long_tickers & short_tickers)
    if conflict:
        risk_flags.append(
            f"  CONFLICT  │ {', '.join(conflict)}"
            f" — appears in both a LONG and SHORT bucket simultaneously"
        )

    # CAUTION — bearish regime with long positions
    if regime_label == "BEARISH":
        risk_flags.append(
            "  CAUTION   │ BEARISH regime active — long positions carry elevated"
            " market risk; consider 0.50× scale"
        )

    # CLOSE SOON — days < 5
    if 0 < days_left < 5:
        risk_flags.append(
            f"  CLOSE SOON │ Only {days_left} days to effective date"
            f" — begin closing positions before June 26 open"
        )

    # WARNING — hard stop within 3% of trigger
    if not hard_stop and ret_30 < -0.05:
        risk_flags.append(
            f"  WARNING   │ 30d RUT {ret_30:+.2%}"
            f" — within 3% of hard-stop trigger (-8.00%); reduce size now"
        )

    # Hard stop itself
    if hard_stop:
        risk_flags.append(
            "  ⛔ HARD STOP │ All recon trades must be closed immediately"
        )

    if risk_flags:
        for f in risk_flags:
            ln(f)
    else:
        ln("  ✓  No flags triggered")

    # ─────────────────────────────────────────────────────────────────────
    # 8. WEEKLY CALENDAR
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(sec("WEEKLY CALENDAR  (remaining)"))

    future = [(dt, lbl) for dt, lbl in KEY_DATES if dt >= today]
    if future:
        for dt, lbl in future:
            delta  = (dt - today).days
            is_eff = dt == EFFECTIVE_DATE
            marker = "★ " if is_eff else "  "
            ln(f"  {marker}{dt.strftime('%b %d')}  (+{delta:>2}d)  {lbl}")
    else:
        ln("  Effective date has passed.")

    # ─────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────
    ln("")
    ln(heavy())
    ln(f"  Candidates : {source_msg}")
    ln(f"  Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ln(f"  Next run   : daily before market open  │  python dashboard.py")
    ln(heavy())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(heavy("DASHBOARD PIPELINE"))
    print("")
    os.makedirs("backtests", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    ctx   = run_pipeline()
    brief = build_dashboard(ctx)

    with open(BRIEF_PATH, "w", encoding="utf-8") as fh:
        fh.write(brief)

    print("")
    print(brief)
    print(f"\n  ✓ Saved → {BRIEF_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise
