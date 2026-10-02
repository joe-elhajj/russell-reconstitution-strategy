#!/usr/bin/env python
"""
Monte Carlo optimizer for position sizing.
Vectorized numpy operations to run 100k sims quickly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

TOTAL_CAPITAL = 1_000_000
N_SIMULATIONS = 100_000
TRADING_DAYS = 30
HARD_STOP_PCT = -0.08

trades = [
    {"name": "Big-YTD-Gainer Adds SHORT", "direction": -1, "expected_return": -0.02, "std_dev": 0.08, "base_size": 0.02},
    {"name": "General R2000 Additions LONG", "direction": 1, "expected_return": 0.015, "std_dev": 0.04, "base_size": 0.01},
    {"name": "Deletion Basket POST-recon LONG", "direction": 1, "expected_return": 0.02, "std_dev": 0.05, "base_size": 0.01},
    {"name": "Promotion SHORT", "direction": -1, "expected_return": -0.011, "std_dev": 0.03, "base_size": 0.005},
    {"name": "Promotion LONG bounce", "direction": 1, "expected_return": 0.02, "std_dev": 0.04, "base_size": 0.005},
]

names = [t['name'] for t in trades]
base_sizes = np.array([t['base_size'] for t in trades])
directions = np.array([t['direction'] for t in trades])
mu = np.array([t['expected_return'] for t in trades])
sigma = np.array([t['std_dev'] for t in trades])


def run_simulations(n_sims=N_SIMULATIONS, multiplier=1.0):
    # Draw one-period returns for each trade category per simulation
    rng = np.random.default_rng()
    # Shape: (n_sims, n_trades)
    sims = rng.normal(loc=mu, scale=sigma, size=(n_sims, len(trades)))
    # Apply hard stop per trade: cap losses at -15%
    sims = np.where(sims < -0.15, -0.15, sims)
    # Position sizes
    sizes = base_sizes * multiplier
    # P&L per trade per simulation (dollar)
    pnl = (sizes * TOTAL_CAPITAL) * (directions * sims)
    total_pnl = pnl.sum(axis=1)
    final_values = TOTAL_CAPITAL + total_pnl
    # Approx max drawdown approximation: worst single-trade loss proportion * capital
    max_drawdowns = np.min(pnl, axis=1)  # negative numbers
    # Sharpe proxy (mean / std of returns)
    returns = total_pnl / TOTAL_CAPITAL
    sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
    return {
        'final_values': final_values,
        'total_pnl': total_pnl,
        'max_drawdowns': max_drawdowns,
        'sharpe': sharpe,
    }


def analyze_results(sim_res):
    final = sim_res['final_values']
    pnl = sim_res['total_pnl']
    drawdowns = sim_res['max_drawdowns']
    sharpe = sim_res['sharpe']

    percentiles = np.percentile(final, [25, 50, 75])
    prob_profit = np.mean(final > TOTAL_CAPITAL)
    prob_lose_5 = np.mean(final < TOTAL_CAPITAL * 0.95)
    worst_99 = np.percentile(final, 1)

    summary = {
        '25pct': percentiles[0],
        '50pct': percentiles[1],
        '75pct': percentiles[2],
        'prob_profit': prob_profit,
        'prob_lose_5pct': prob_lose_5,
        'worst_1pct_value': worst_99,
        'sharpe': sharpe,
    }
    return summary


def find_optimal_multiplier():
    multipliers = np.linspace(0.25, 2.0, 8)
    best = None
    best_sharpe = -np.inf
    results_summary = []
    for m in multipliers:
        res = run_simulations(N_SIMULATIONS, multiplier=m)
        summary = analyze_results(res)
        results_summary.append((m, summary))
        if summary['sharpe'] > best_sharpe:
            best_sharpe = summary['sharpe']
            best = (m, summary)
    return best, results_summary


def generate_charts(final_values, drawdowns):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(final_values, bins=200, color='tab:blue', alpha=0.7)
    axs[0].set_title('Final Portfolio Values')

    axs[1].hist(drawdowns, bins=200, color='tab:orange', alpha=0.7)
    axs[1].set_title('Max Drawdowns (approx)')

    # P&L by trade category: sample boxplots by computing per-trade pnl across sims
    rng = np.random.default_rng()
    sims = rng.normal(loc=mu, scale=sigma, size=(N_SIMULATIONS, len(trades)))
    sims = np.where(sims < -0.15, -0.15, sims)
    pnl = (base_sizes * TOTAL_CAPITAL) * (directions * sims)
    axs[2].boxplot([pnl[:, i] for i in range(pnl.shape[1])], labels=names)
    axs[2].set_title('P&L by Trade Category')
    plt.tight_layout()
    out = RESULTS_DIR / 'monte_carlo_chart.png'
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    t0 = time.time()
    best, all_results = find_optimal_multiplier()
    best_mult, best_summary = best

    # Run full simulation with best multiplier for outputs
    sim_res = run_simulations(N_SIMULATIONS, multiplier=best_mult)
    summary = analyze_results(sim_res)

    # Save results csv (sampled subset for size)
    df_out = pd.DataFrame({
        'final_value': sim_res['final_values'],
        'total_pnl': sim_res['total_pnl'],
        'max_drawdown': sim_res['max_drawdowns'],
    })
    df_out.to_csv(RESULTS_DIR / 'monte_carlo_results.csv', index=False)

    # Charts
    generate_charts(sim_res['final_values'], sim_res['max_drawdowns'])

    # Summary text
    summary_txt = f"""
Monte Carlo Summary
Total simulations: {N_SIMULATIONS}
Best multiplier: {best_mult}
25/50/75 percentiles (final value): {best_summary['25pct']:.2f}, {best_summary['50pct']:.2f}, {best_summary['75pct']:.2f}
Probability of profit: {best_summary['prob_profit']:.3f}
Probability of losing >5%: {best_summary['prob_lose_5pct']:.3f}
Worst 1% value: {best_summary['worst_1pct_value']:.2f}
Sharpe proxy: {best_summary['sharpe']:.3f}
"""
    with open(RESULTS_DIR / 'monte_carlo_summary.txt', 'w') as f:
        f.write(summary_txt)

    t1 = time.time()
    print('Monte Carlo complete, time:', t1 - t0)

if __name__ == '__main__':
    main()
