"""
Generate a standalone HTML trade report for the 2026 Russell reconstitution.

Downloads fresh data, scores, and writes data/russell_report_2026.html.

Usage:
    python generate_report_2026.py
"""

from __future__ import annotations
import datetime
import os
import sys

import pandas as pd

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

OUTPUT_HTML = 'data/russell_report_2026.html'


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def score_badge(score: float) -> str:
    if pd.isna(score):
        return '<span class="badge badge-na">N/A</span>'
    s = float(score)
    if s >= 7.5:
        css = 'badge-high'
    elif s >= 5.0:
        css = 'badge-mid'
    else:
        css = 'badge-low'
    return f'<span class="badge {css}">{s:.2f}</span>'


def pct_cell(raw_val) -> str:
    if pd.isna(raw_val):
        return '<td class="num">N/A</td>'
    pct = raw_val * 100
    css = 'pos' if pct >= 0 else 'neg'
    return f'<td class="num {css}">{pct:+.1f}%</td>'


def additions_rows(df: pd.DataFrame, n: int = 10) -> str:
    html = ''
    for rank, (_, row) in enumerate(df.head(n).iterrows(), start=1):
        is_paradox = pd.notna(row['market_cap']) and row['market_cap'] > RUSSELL1000_MCAP_THRESHOLD
        paradox_tag = ' <span class="paradox-flag" title="Promotion paradox — see Table 3">&#9888;</span>' if is_paradox else ''
        html += (
            f'<tr>'
            f'<td class="num">{rank}</td>'
            f'<td class="ticker">{row["ticker"]}{paradox_tag}</td>'
            f'<td class="num">{fmt_price(row["price"])}</td>'
            f'<td class="num">{fmt_mcap(row["market_cap"])}</td>'
            + pct_cell(row.get('ytd_return'))
            + pct_cell(row.get('momentum_30d'))
            + f'<td class="num">{score_badge(row["final_score"])}</td>'
            f'</tr>\n'
        )
    return html


def deletions_rows(df: pd.DataFrame, n: int = 10) -> str:
    html = ''
    for rank, (_, row) in enumerate(df.head(n).iterrows(), start=1):
        html += (
            f'<tr>'
            f'<td class="num">{rank}</td>'
            f'<td class="ticker">{row["ticker"]}</td>'
            f'<td class="num">{fmt_price(row["price"])}</td>'
            f'<td class="num">{fmt_mcap(row["market_cap"])}</td>'
            + pct_cell(row.get('ytd_return'))
            + f'<td class="num">{score_badge(row["final_score"])}</td>'
            f'</tr>\n'
        )
    return html


def paradox_rows(df: pd.DataFrame) -> str:
    paradox = df[df['market_cap'] > RUSSELL1000_MCAP_THRESHOLD].copy()
    if paradox.empty:
        return '<tr><td colspan="5" class="empty-row">No stocks currently above the $5.7B Russell 1000 breakpoint.</td></tr>\n'
    html = ''
    for _, row in paradox.iterrows():
        html += (
            f'<tr>'
            f'<td class="ticker">{row["ticker"]}</td>'
            f'<td class="num">{fmt_price(row["price"])}</td>'
            f'<td class="num">{fmt_mcap(row["market_cap"])}</td>'
            + pct_cell(row.get('ytd_return'))
            + f'<td><span class="action-sell">SELL / SHORT into close</span></td>'
            f'</tr>\n'
        )
    return html


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a2332;
    font-size: 14px;
    line-height: 1.6;
}

/* Header */
.site-header {
    background: linear-gradient(135deg, #0f1c2e 0%, #1a3050 60%, #0d2137 100%);
    color: #fff;
    padding: 36px 40px 28px;
    border-bottom: 4px solid #f59e0b;
}
.site-header h1 {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin-bottom: 6px;
}
.site-header .meta {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 16px;
}
.countdown {
    display: inline-block;
    background: #f59e0b;
    color: #0f1c2e;
    font-weight: 700;
    font-size: 15px;
    padding: 6px 18px;
    border-radius: 20px;
    letter-spacing: 0.2px;
}

/* Content wrapper */
.content {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 48px;
}

/* Section cards */
.section {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 32px;
    overflow: hidden;
}

.section-header {
    padding: 18px 24px 14px;
    border-bottom: 1px solid #e5e7eb;
}
.section-header h2 {
    font-size: 17px;
    font-weight: 700;
    color: #0f1c2e;
    margin-bottom: 6px;
}
.section-header .strategy-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 3px 10px;
    border-radius: 12px;
    margin-right: 6px;
}
.tag-buy    { background: #dcfce7; color: #15803d; }
.tag-short  { background: #fee2e2; color: #b91c1c; }
.tag-sell   { background: #fef3c7; color: #92400e; }
.tag-info   { background: #e0f2fe; color: #075985; }

.section-body { padding: 16px 24px 20px; }

.explain {
    font-size: 13.5px;
    color: #374151;
    line-height: 1.7;
    padding: 16px 24px 20px;
    border-top: 1px solid #f3f4f6;
    background: #fafafa;
}

/* Executive summary */
.exec-summary {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 22px 28px;
    margin-bottom: 32px;
    border-left: 4px solid #3b82f6;
    font-size: 13.5px;
    color: #374151;
    line-height: 1.8;
}
.exec-summary h2 {
    font-size: 14px;
    font-weight: 700;
    color: #0f1c2e;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    table-layout: fixed;
    word-wrap: break-word;
    page-break-inside: avoid;
}
tr { page-break-inside: avoid; page-break-after: auto; }
thead th {
    background: #1a2332;
    color: #e2e8f0;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
thead th.num { text-align: right; }

tbody tr:nth-child(even) { background: #f8fafc; }
tbody tr:hover           { background: #eff6ff; transition: background 0.15s; }
tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}

.num    { text-align: right; font-variant-numeric: tabular-nums; }
.ticker { font-weight: 700; font-size: 14px; color: #0f1c2e; letter-spacing: 0.3px; }
.pos    { color: #16a34a; font-weight: 600; }
.neg    { color: #dc2626; font-weight: 600; }
.empty-row { text-align: center; color: #9ca3af; padding: 20px; font-style: italic; }

/* Score badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    min-width: 44px;
    text-align: center;
}
.badge-high { background: #dcfce7; color: #15803d; }
.badge-mid  { background: #fef9c3; color: #854d0e; }
.badge-low  { background: #fee2e2; color: #991b1b; }
.badge-na   { background: #f1f5f9; color: #94a3b8; }

/* Promotion paradox action */
.action-sell {
    display: inline-block;
    background: #fff7ed;
    color: #c2410c;
    font-weight: 700;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid #fed7aa;
}

/* Warning triangle for paradox tickers in table 1 */
.paradox-flag {
    color: #f59e0b;
    font-size: 12px;
    margin-left: 4px;
    cursor: help;
}

/* Scoring legend */
.legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    padding: 12px 24px;
    border-top: 1px solid #f3f4f6;
    background: #fafafa;
    font-size: 12px;
    color: #6b7280;
    align-items: center;
}
.legend strong { color: #374151; margin-right: 4px; }

@media print {
    table   { page-break-inside: avoid; }
    tr      { page-break-inside: avoid; }
    .section { page-break-inside: avoid; }
    h2      { page-break-after: avoid; }
}

/* Footer */
.site-footer {
    text-align: center;
    padding: 24px 16px;
    color: #9ca3af;
    font-size: 12px;
    border-top: 1px solid #e5e7eb;
    background: #fff;
    margin-top: 16px;
}
"""


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

def build_html(df_add: pd.DataFrame, df_del: pd.DataFrame, today: datetime.date, days_left: int) -> str:
    add_rows_html = additions_rows(df_add)
    del_rows_html = deletions_rows(df_del)
    par_rows_html = paradox_rows(df_add)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Russell 2000 Reconstitution Trade Report — June 26, 2026</title>
<style>
{CSS}
</style>
</head>
<body>

<!-- ===== HEADER ===== -->
<header class="site-header">
  <h1>Russell 2000 Reconstitution Trade Report &mdash; June 26, 2026</h1>
  <div class="meta">
    FTSE Russell Preliminary List: May 22, 2026 &nbsp;|&nbsp;
    Report generated: {today.strftime("%B %d, %Y")}
  </div>
  <div class="countdown">&#9200; {days_left} days until effective date (June 26, 2026)</div>
</header>

<div class="content">

<!-- ===== EXECUTIVE SUMMARY ===== -->
<div class="exec-summary">
  <h2>Executive Summary</h2>
  Every year, FTSE Russell reconstitutes its index family — stocks that have grown or shrunk
  are added to or removed from the Russell 1000, Russell 2000, and Russell 3000 indexes.
  The critical moment is the <strong>effective date (June 26, 2026 after market close)</strong>,
  when every passive fund tracking these indexes must simultaneously rebalance.
  Combined, Russell-tracking ETFs and funds manage over <strong>$10 trillion</strong> in assets.
  The forced buying into additions and forced selling out of deletions creates predictable,
  size-dependent price pressure in the days leading up to and on the effective date.
  Smaller-cap additions face the most acute buying pressure relative to their float.
  Stocks being deleted — often already fundamentally weak — face compounded selling pressure.
  This report ranks both sides of the trade using momentum, market-cap pressure, and volume impact.
</div>

<!-- ===== TABLE 1: ADDITIONS ===== -->
<div class="section">
  <div class="section-header">
    <h2>Table 1 &mdash; Top 10 Additions to Buy</h2>
    <span class="strategy-tag tag-buy">BUY</span>
    <span class="strategy-tag tag-info">Before June 26 close</span>
  </div>
  <div class="section-body">
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th>Ticker</th>
          <th class="num">Price</th>
          <th class="num">Market Cap</th>
          <th class="num">YTD Return</th>
          <th class="num">30d Momentum</th>
          <th class="num">Score</th>
        </tr>
      </thead>
      <tbody>
{add_rows_html}      </tbody>
    </table>
  </div>
  <div class="legend">
    <strong>Scoring:</strong>
    40% YTD momentum &nbsp;|&nbsp;
    40% market-cap pressure (smaller = higher score) &nbsp;|&nbsp;
    20% volume impact (lower avg vol = more price impact) &nbsp;|&nbsp;
    &#9888; = Promotion paradox (see Table 3)
  </div>
  <div class="explain">
    These stocks are being <strong>added to the Russell 2000 index on June 26, 2026</strong>.
    Passive ETFs — led by iShares IWM (~$75B AUM) — are <strong>forced to buy</strong> these stocks
    before the close to match the new index composition. The smaller the market cap, the larger
    the forced buy relative to available float, creating more acute upward price pressure.
    Stocks with strong YTD momentum are also carried further by momentum-chasing active managers
    who front-run the rebalance. <strong>Higher score = stronger combination of forced-buying
    pressure and momentum tailwind.</strong>
  </div>
</div>

<!-- ===== TABLE 2: DELETIONS ===== -->
<div class="section">
  <div class="section-header">
    <h2>Table 2 &mdash; Top 10 Deletions to Short</h2>
    <span class="strategy-tag tag-short">SHORT</span>
    <span class="strategy-tag tag-info">Before June 26 close</span>
  </div>
  <div class="section-body">
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th>Ticker</th>
          <th class="num">Price</th>
          <th class="num">Market Cap</th>
          <th class="num">YTD Return</th>
          <th class="num">Score</th>
        </tr>
      </thead>
      <tbody>
{del_rows_html}      </tbody>
    </table>
  </div>
  <div class="legend">
    <strong>Scoring:</strong>
    60% decline severity (larger YTD loss = higher score) &nbsp;|&nbsp;
    40% market-cap pressure (smaller = more forced-sell impact)
  </div>
  <div class="explain">
    These stocks are being <strong>removed from the Russell 2000 on June 26, 2026</strong>.
    Passive funds are <strong>forced to sell</strong> their entire positions before the close.
    These stocks are typically already in decline — they are deleted because their market cap
    fell below Russell's reconstitution threshold — so the forced mechanical selling from
    index funds compounds existing fundamental weakness.
    Smaller-cap deletions face sharper price impact because the forced-sell volume is large
    relative to average daily trading volume. <strong>Cover shorts after the forced selling
    exhausts itself post-June 26.</strong>
  </div>
</div>

<!-- ===== TABLE 3: PROMOTION PARADOX ===== -->
<div class="section">
  <div class="section-header">
    <h2>Table 3 &mdash; Promotion Paradox</h2>
    <span class="strategy-tag tag-sell">SELL / SHORT</span>
    <span class="strategy-tag tag-info">Additions &gt; $5.7B moving to Russell 1000</span>
  </div>
  <div class="section-body">
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th class="num">Price</th>
          <th class="num">Market Cap</th>
          <th class="num">YTD Return</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
{par_rows_html}      </tbody>
    </table>
  </div>
  <div class="explain">
    These stocks appear on the <em>additions</em> list but are going into the
    <strong>Russell 1000</strong> (not the Russell 2000) because their market cap exceeds the
    ~$5.7B breakpoint. This sounds positive, but it is actually a <strong>net sell signal</strong>.
    Russell 2000 trackers currently hold ~9% of float in these names and are
    <strong>forced to sell their entire position</strong> as the stock leaves the R2000.
    Russell 1000 trackers, meanwhile, only need to add ~4% of float.
    The net result is a <strong>5-percentage-point forced sell of float</strong> concentrated into
    the June 26 close — a meaningful headwind that is frequently overlooked because the stock
    is technically an &ldquo;addition&rdquo; to a Russell index.
    <strong>Sell or short these into the June 26 close.</strong>
  </div>
</div>

</div><!-- /content -->

<!-- ===== FOOTER ===== -->
<footer class="site-footer">
  <p>
    <strong>Disclaimer:</strong>
    This report is for informational purposes only and does not constitute financial advice.
    All data sourced from public market data via yfinance. Past index reconstitution patterns
    are not a guarantee of future price movements. Trading around index events carries risk,
    including the risk of loss of principal.
  </p>
</footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    today     = datetime.date.today()
    days_left = max((EFFECTIVE_DATE - today).days, 0)

    print('=' * 62)
    print('  Russell 2026 HTML Report Generator')
    print(f'  Today        : {today}')
    print(f'  Days to June 26 close: {days_left}')
    print('=' * 62)

    os.makedirs('data', exist_ok=True)

    print(f'\nDownloading fresh data for {len(ADDITIONS)} additions...')
    df_add_raw = fetch_data(ADDITIONS)

    print(f'\nDownloading fresh data for {len(DELETIONS)} deletions...')
    df_del_raw = fetch_data(DELETIONS)

    print('\nScoring...')
    df_add = score_additions(df_add_raw)
    df_del = score_deletions(df_del_raw)

    print('Building HTML...')
    html = build_html(df_add, df_del, today, days_left)

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as fh:
        fh.write(html)

    print(f'\nReport saved to {OUTPUT_HTML}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(0)
    except Exception as exc:
        print(f'\nFatal error: {exc}')
        raise
