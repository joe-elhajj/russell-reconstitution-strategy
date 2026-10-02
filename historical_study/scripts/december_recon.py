#!/usr/bin/env python
"""
December 2026 Russell recon preview based on June patterns.

Context notes:
# December 2026 is the first semi-annual Russell recon ever.
# No historical December recon data exists.
# Strategy: use June recon patterns as training signal,
# adjust for December-specific macro factors (year-end flows, tax loss selling, window dressing).
"""
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Adjustment assumptions
TAX_LOSS_SELL = -0.02  # additional -2% to deletion rundown
WINDOW_DRESSING = 0.015  # +1.5% to addition runup
VOLUME_REDUCTION = 0.30  # 30% less volume

SCENARIOS = ['BULL', 'NEUTRAL', 'BEAR']


def load_june_patterns():
    path = RESULTS_DIR / 'pattern_measurements.csv'
    if not path.exists():
        raise FileNotFoundError('pattern_measurements.csv not found; run backtest_full.py first')
    df = pd.read_csv(path)
    return df


def build_adjusted_factors(df):
    # Compute regime averages for June
    avgs = df.groupby('regime').agg({
        'pattern_1_additions_runup': 'mean',
        'pattern_3_post_recon_reversal': 'mean',
        'pattern_2_vix_regime': 'mean'
    }).rename(columns={
        'pattern_1_additions_runup': 'add_runup',
        'pattern_3_post_recon_reversal': 'post_reversal',
        'pattern_2_vix_regime': 'vix'
    })

    # Apply December adjustments
    dec = avgs.copy()
    dec['add_runup'] = dec['add_runup'] + WINDOW_DRESSING
    dec['post_reversal'] = dec['post_reversal'] + TAX_LOSS_SELL
    dec['volume_adj'] = 1.0 - VOLUME_REDUCTION
    return dec


def simulate_scenarios(dec_factors):
    rows = []
    for regime in SCENARIOS:
        if regime not in dec_factors.index:
            base = dec_factors.mean()
        else:
            base = dec_factors.loc[regime]
        if regime == 'BULL':
            add = base['add_runup'] * 1.25
            delete = base['post_reversal'] * 1.25
            size = 1.0
        elif regime == 'NEUTRAL':
            add = base['add_runup']
            delete = base['post_reversal']
            size = 1.0
        else:  # BEAR
            add = base['add_runup'] * 0.5
            delete = base['post_reversal'] * 0.5
            size = 0.5
        rows.append({
            'scenario': regime,
            'expected_addition_runup_pct': add,
            'expected_deletion_bounce_pct': delete,
            'recommended_position_size_multiplier': size,
            'volume_multiplier': base['volume_adj'] if 'volume_adj' in base else 0.7
        })
    df_out = pd.DataFrame(rows)
    return df_out


def calendar_output():
    cal = {
        'rank_day_estimate': '2026-10-31',
        'preliminary_list_estimate': 'last Friday of November 2026',
        'weekly_updates': ['2026-12-04', '2026-12-11'],
        'effective_date': '2026-12-11',
        'post_recon_window': '2026-12-14 to 2026-12-31'
    }
    return cal


def main():
    df = load_june_patterns()
    dec_factors = build_adjusted_factors(df)
    scenarios = simulate_scenarios(dec_factors)

    # Save outputs
    scenarios.to_csv(RESULTS_DIR / 'december_scenarios.csv', index=False)

    preview_text = []
    preview_text.append('December 2026 Recon Preview')
    preview_text.append('--------------------------------')
    preview_text.append('Adjusted factors (based on June averages):')
    preview_text.append(dec_factors.to_string())
    preview_text.append('\nScenarios:')
    preview_text.append(scenarios.to_string(index=False))
    preview_text.append('\nCalendar:')
    cal = calendar_output()
    preview_text.append(str(cal))

    with open(RESULTS_DIR / 'december_recon_preview.txt', 'w') as f:
        f.write('\n'.join(preview_text))

    print('December recon preview saved to results/')

if __name__ == '__main__':
    main()
