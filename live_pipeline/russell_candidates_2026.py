"""
Russell 3000 Reconstitution Screener — June 26, 2026
Based on the confirmed FTSE Russell May 22, 2026 preliminary list.

Usage:
    python russell_candidates_2026.py [--refresh]

Output:
    data/russell_candidates_2026.csv
"""

from __future__ import annotations
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

# --- Key dates ---
EFFECTIVE_DATE = datetime.date(2026, 6, 26)
RUSSELL1000_MCAP_THRESHOLD = 5_700_000_000  # $5.7B breakpoint

# --- Confirmed May 22 2026 FTSE Russell preliminary list ---

ADDITIONS = [
    'CRWV', 'CHYM', 'FIG',  'KVYO', 'HNGE', 'GLXY', 'TTAN', 'ULS',
    'RJET', 'AXTI', 'FCEL', 'LPTH', 'UMAC', 'WOLF', 'GPRO', 'CHPT',
    'SPCE', 'OPEN', 'SYM',  'FROG', 'DLO',  'ASND', 'ODD',  'TBLA',
    'IREN', 'CSIQ',
]

DELETIONS = [
    'HAIN', 'SNBR', 'DCGO', 'BYND', 'VRM',  'FFAI', 'SPWR', 'GETY',
    'SKIL', 'ALIT', 'DH',   'HCAT', 'NFE',  'EXFY', 'MVIS', 'TR',
    'ATYR', 'FATE', 'ALDX', 'PRLD',
]

OUTPUT_CSV = 'data/russell_candidates_2026.csv'
YTD_START   = datetime.date(2026, 1, 1)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_data(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker in tickers:
        print(f'  {ticker:<6}', end=' ', flush=True)
        row: dict = {'ticker': ticker}
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info or {}

            row['price']      = info.get('regularMarketPrice') or info.get('previousClose')
            row['market_cap'] = info.get('marketCap')

            hist = tk.history(period='ytd', auto_adjust=True)
            if hist.empty:
                hist = tk.history(period='6mo', auto_adjust=True)

            if not hist.empty and 'Close' in hist.columns:
                close = hist['Close'].dropna().sort_index()

                if len(close) > 0:
                    # Use last close as price if info didn't return one
                    if not row['price']:
                        row['price'] = float(close.iloc[-1])

                    # YTD return: Jan 1 2026 → today
                    ytd_slice = close[close.index.tz_localize(None) >= pd.Timestamp(YTD_START)]
                    first = float(ytd_slice.iloc[0]) if len(ytd_slice) > 0 else float(close.iloc[0])
                    last  = float(close.iloc[-1])
                    row['ytd_return'] = last / first - 1.0 if first else np.nan

                    # 30-day momentum (~22 trading days)
                    row['momentum_30d'] = (
                        last / float(close.iloc[-22]) - 1.0
                        if len(close) >= 22 else np.nan
                    )

                    # Average daily volume (20-day)
                    vol = hist.get('Volume')
                    row['avg_vol'] = float(vol.tail(20).mean()) if vol is not None else np.nan
                else:
                    row.update({'ytd_return': np.nan, 'momentum_30d': np.nan, 'avg_vol': np.nan})
            else:
                row.update({'ytd_return': np.nan, 'momentum_30d': np.nan, 'avg_vol': np.nan})

            print('OK')
        except Exception as exc:
            print(f'FAILED  ({exc})')
            row.update({'price': np.nan, 'market_cap': np.nan,
                        'ytd_return': np.nan, 'momentum_30d': np.nan, 'avg_vol': np.nan})

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ['price', 'market_cap', 'ytd_return', 'momentum_30d', 'avg_vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def rank_1_10(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Scale a series to 1–10 via percentile rank. NaN stays NaN."""
    out  = pd.Series(np.nan, index=series.index)
    mask = series.notna()
    if mask.sum() < 1:
        return out
    pct = series[mask].rank(pct=True, method='average')
    if not ascending:
        pct = 1.0 - pct
    out[mask] = (pct * 9.0 + 1.0).clip(1.0, 10.0)
    return out


def score_additions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['momentum_score'] = rank_1_10(df['ytd_return'],   ascending=True)   # higher YTD  → higher
    df['mcap_score']     = rank_1_10(df['market_cap'],   ascending=False)  # smaller cap → higher
    df['volume_score']   = rank_1_10(df['avg_vol'],      ascending=False)  # lower vol   → higher

    # Fill missing components with neutral midpoint before weighting
    m = df['momentum_score'].fillna(5.5)
    c = df['mcap_score'].fillna(5.5)
    v = df['volume_score'].fillna(5.5)

    df['final_score'] = (0.40 * m + 0.40 * c + 0.20 * v).round(2)
    return df.sort_values('final_score', ascending=False).reset_index(drop=True)


def score_deletions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['decline_score'] = rank_1_10(df['ytd_return'],  ascending=False)  # bigger loss  → higher
    df['mcap_score']    = rank_1_10(df['market_cap'],  ascending=False)  # smaller cap  → higher

    d = df['decline_score'].fillna(5.5)
    c = df['mcap_score'].fillna(5.5)

    df['final_score'] = (0.60 * d + 0.40 * c).round(2)
    return df.sort_values('final_score', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_price(v) -> str:
    return f'${v:.2f}' if pd.notna(v) else 'N/A'

def fmt_mcap(v) -> str:
    if pd.isna(v):
        return 'N/A'
    if v >= 1e9:
        return f'${v/1e9:.2f}B'
    return f'${v/1e6:.1f}M'

def fmt_pct(v) -> str:
    return f'{v*100:+.1f}%' if pd.notna(v) else 'N/A'


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_additions_table(df: pd.DataFrame) -> None:
    top = df.head(10).copy()
    top.insert(0, 'Rank', range(1, len(top) + 1))
    top['Price']        = top['price'].apply(fmt_price)
    top['Market Cap']   = top['market_cap'].apply(fmt_mcap)
    top['YTD Return']   = top['ytd_return'].apply(fmt_pct)
    top['30d Momentum'] = top['momentum_30d'].apply(fmt_pct)
    top['Score']        = top['final_score']
    print(tabulate(
        top[['Rank', 'ticker', 'Price', 'Market Cap', 'YTD Return', '30d Momentum', 'Score']],
        headers='keys', tablefmt='psql', showindex=False,
    ))


def print_deletions_table(df: pd.DataFrame) -> None:
    top = df.head(10).copy()
    top.insert(0, 'Rank', range(1, len(top) + 1))
    top['Price']      = top['price'].apply(fmt_price)
    top['Market Cap'] = top['market_cap'].apply(fmt_mcap)
    top['YTD Return'] = top['ytd_return'].apply(fmt_pct)
    top['Score']      = top['final_score']
    print(tabulate(
        top[['Rank', 'ticker', 'Price', 'Market Cap', 'YTD Return', 'Score']],
        headers='keys', tablefmt='psql', showindex=False,
    ))


def print_promotion_paradox(df_add: pd.DataFrame) -> None:
    paradox = df_add[df_add['market_cap'] > RUSSELL1000_MCAP_THRESHOLD].copy()
    if paradox.empty:
        print('  None detected (no additions above the $5.7B Russell 1000 breakpoint).')
        return
    paradox['Price']      = paradox['price'].apply(fmt_price)
    paradox['Market Cap'] = paradox['market_cap'].apply(fmt_mcap)
    paradox['YTD Return'] = paradox['ytd_return'].apply(fmt_pct)
    paradox['Action']     = 'SELL / SHORT into June 26 close'
    print(tabulate(
        paradox[['ticker', 'Price', 'Market Cap', 'YTD Return', 'Action']],
        headers='keys', tablefmt='psql', showindex=False,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Russell 3000 Reconstitution Screener 2026')
    parser.add_argument('--refresh', action='store_true', help='Ignore cached data')
    args = parser.parse_args()  # noqa: F841 (--refresh kept for future caching layer)

    today     = datetime.date.today()
    days_left = max((EFFECTIVE_DATE - today).days, 0)

    print('=' * 62)
    print('  FTSE Russell 3000 Reconstitution Screener — 2026')
    print(f'  Preliminary list date : May 22, 2026')
    print(f'  Today                 : {today}')
    print(f'  Days until June 26 effective date: {days_left}')
    print('=' * 62)
    print()

    os.makedirs('data', exist_ok=True)

    print(f'Downloading data for {len(ADDITIONS)} additions...')
    df_add_raw = fetch_data(ADDITIONS)

    print(f'\nDownloading data for {len(DELETIONS)} deletions...')
    df_del_raw = fetch_data(DELETIONS)

    print('\nScoring...')
    df_add = score_additions(df_add_raw)
    df_del = score_deletions(df_del_raw)

    # -----------------------------------------------------------------------
    print()
    print('=' * 62)
    print('  TOP 10 ADDITIONS TO BUY')
    print('  Buy before June 26 close; hold through passive rebalance')
    print('  Scoring: 40% YTD momentum | 40% market-cap pressure | 20% volume impact')
    print('=' * 62)
    print_additions_table(df_add)

    print()
    print('=' * 62)
    print('  TOP 10 DELETIONS TO SHORT')
    print('  Short before June 26 close; cover after forced selling exhausts')
    print('  Scoring: 60% decline severity | 40% market-cap pressure')
    print('=' * 62)
    print_deletions_table(df_del)

    print()
    print('=' * 62)
    print('  PROMOTION PARADOX FLAGS')
    print('  Additions with market cap > $5.7B → Russell 1000 (net sellers)')
    print('  IWM (R2000) must SELL; R1000 trackers hold much less → price headwind')
    print('=' * 62)
    print_promotion_paradox(df_add)

    # -----------------------------------------------------------------------
    # Save CSV
    df_add['list']              = 'ADDITION'
    df_del['list']              = 'DELETION'
    df_add['promotion_paradox'] = df_add['market_cap'] > RUSSELL1000_MCAP_THRESHOLD
    df_del['promotion_paradox'] = False

    out = pd.concat([df_add, df_del], sort=False, ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f'\nFull results saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(0)
    except Exception as exc:
        print(f'\nFatal error: {exc}')
        raise
