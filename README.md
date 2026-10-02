# Russell Reconstitution Trading Strategy

A research project for a summer trading competition. Every June, FTSE Russell reshuffles its indexes and every fund that tracks the Russell 2000 has to trade by the effective date, whatever the price. The preliminary lists come out in late May, so the flow is visible weeks ahead. This repo asks whether that flow, and the market regime around it, can drive a position signal.

## What I built

- **Daily pipeline.** Eight market series from Yahoo Finance, 21 engineered features, and a LightGBM classifier for next-day Russell 2000 direction.
- **Regime filter.** Calls the market bullish, neutral or bearish and sizes the position full, half or flat, with a hard stop on an 8% drop over 30 days.
- **Candidate screen.** Scores the May 2026 preliminary additions and deletions on momentum, market cap and trading volume, and writes a daily brief.
- **Historical study.** 26 reconstitutions (2000 to 2025), a Monte Carlo on position sizing, and a December 2026 preview.

## Headline result

Held out on 2025, a year the model never saw in training, the strategy returned 14.9% gross against 9.3% for buy and hold, and 13.3% after a 2 bps cost per unit traded (+3.9 points, Sharpe 0.79). These are backtest outputs, not live trading.

![2025 out-of-sample equity curve](docs/oos_2025_equity_curve.png)

The chart is gross of costs. The strategy changed position 72 times in 230 trading days (about 79 units of one-way turnover), so I repriced the saved positions with a flat cost per unit traded, cash earning nothing:

| Cost per unit traded | Net return | Sharpe | Net vs buy and hold |
|---|---|---|---|
| 0 bps | 14.9% | 0.87 | +5.6 pts |
| 2 bps | 13.3% | 0.79 | +3.9 pts |
| 5 bps | 10.8% | 0.67 | +1.5 pts |
| 10 bps | 6.9% | 0.48 | -2.4 pts |

It stays ahead of buy and hold up to about 7 bps per unit traded and reaches zero return near 19 bps. The cost levels are assumptions, not measured spreads.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Each folder is meant to be run from inside itself, since paths are relative. No API keys are needed. Data and trained models are not stored in the repo, so you rebuild them first:

```bash
cd live_pipeline
python archive/data_pull_full.py   # writes data/master_features.csv (2010-2024)
python train_model.py              # writes models/lgbm_classifier.pkl and lgbm_regressor.pkl
python validate_2025.py            # the 2025 run shown above
python main.py                     # daily brief -> backtests/daily_brief.txt
```

```bash
cd historical_study
python run_all.py                  # backtest, per-ticker models, Monte Carlo, December preview
```

The candidate lists in `live_pipeline/russell_candidates_2026.py` are the 2026 preliminary lists, hard-coded. Swap them out to look at another year. `generate_report_2026.py` makes a PDF version of the brief using `reportlab`.

```
live_pipeline/
  daily_update.py             features and signal for the latest day
  train_model.py              trains the classifier and regressor
  validate_2025.py            out-of-sample run for 2025
  russell_candidates_2026.py  scores additions and deletions
  june_recon_overlay.py       regime and hard-stop logic
  main.py                     full pipeline and daily brief
  dashboard.py                terminal dashboard
  predict_2026.py             2026 predictions
  archive/                    earlier data pulls, feature builds and tuning
historical_study/
  scripts/                    backtest, per-ticker models, Monte Carlo, December preview
  run_all.py                  runs all four steps in order
  results/                    summaries and CSVs from the last run
docs/                       figures used in this README
```

## Takeaways

- The reconstitution gives a known date, a public list of names and a mechanical reason for price pressure. The code turns that into a repeatable routine: score the candidates, size by regime, cap the downside.
- The model works best as one input to the regime filter, not a standalone signal.
- Next: a walk-forward test over several years with costs inside the simulation, starting with the December 2026 reconstitution.

## Limitations

- One held-out year (2025). The classifier trains on 2010 to 2024 with no separate validation split. Costs above are a flat repricing, with no modeled spread, impact or borrow.
- The small-cap candidate trades have no backtest. Plausible round-trip costs of 1% to 3% are the same size as the 1.1% to 2.0% per-trade edges set by hand in the Monte Carlo.
- The historical study had a column-labeling bug (series named by position instead of ticker), now corrected. The rest of it is not audited. Its "edge" is the index return from preliminary list to effective date (0.76% average, t-statistic 0.9), not the added or deleted names.
- Candidate lists are the hard-coded June 26, 2026 lists.

## Related

I also have a separate project on the variance risk premium (VIX against forward realized volatility) that lives outside this repo.
