# Russell Reconstitution Trading Strategy

**What this is:** a research project on the Russell 2000 annual reconstitution. It combines a daily pipeline that screens the 2026 additions and deletions with a historical study of past reconstitutions.

**What it was built for:** to test whether the forced buying and selling around the June reconstitution is a measurable, tradeable pattern, and to produce a daily brief in the weeks before the June 26, 2026 effective date.

**What to take from it:** a working framework for turning the reconstitution calendar into a systematic process (screen, size, stop), plus a clear list of what must be fixed before any number from it is trusted. The last section covers how to use it going forward.

It is research code, not a product and not investment advice. Nothing here is a claim about live or future performance.

## The idea

Every June, FTSE Russell reshuffles its index family. Companies that have grown too big leave the Russell 2000, small companies move in, and every fund that tracks the index has to trade the changes before the effective date, no matter the price. The preliminary lists come out in May, so the flow is predictable weeks ahead. The question here is whether that flow leaves a pattern you can measure, and whether the market backdrop changes how much of it you should trust.

## How it works

The repo has two halves that look at the same event from different sides.

**A daily pipeline for the 2026 reconstitution (`live_pipeline/`).** Each day it pulls eight market series from Yahoo Finance (the Russell 2000, S&P 500, VIX, 10-year and short-term rates, the dollar, gold and oil) and turns them into 21 features. A LightGBM classifier, trained on 2010-2024, predicts the Russell 2000's next-day direction. That signal feeds a simple regime filter, which labels the market bull, neutral or bear and scales position size down when conditions are weak, and an 8% hard stop that fires if the index falls that far in 30 days. On top of that sits a candidate screen: the May 22, 2026 preliminary additions and deletions are scored on momentum, market-cap pressure and volume, since the smallest names have the most forced buying relative to how much stock trades. The output is a daily text brief.

**A historical study (`historical_study/`).** This half goes back through past reconstitutions and measures what happened around each one: the run-up in additions, the reversal afterward, the VIX and yield-curve backdrop, and how results split by bull, bear and neutral regimes. It runs in parallel with joblib, and adds a Monte Carlo run on position sizing and a December preview.

## A first look at the model

To check that the pipeline works end to end, the trained classifier was run over 2025, a year it never saw in training. Positions were 0, half or full size depending on the signal, with no trading costs.

![2025 out-of-sample equity curve](docs/oos_2025_equity_curve.png)

In that run the strategy ended near 114.9k from 100k, against about 109.3k for buying and holding the index. It is one year, tested once, without costs or slippage, so read it as a sanity check on the plumbing and not as evidence of an edge.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs from inside its own folder, because paths are relative. No API keys are needed. Data and trained models are not stored in the repo, so you rebuild them:

```bash
cd live_pipeline
python archive/data_pull_full.py   # writes data/master_features.csv (2010-2024)
python train_model.py              # writes models/lgbm_classifier.pkl and lgbm_regressor.pkl
python validate_2025.py            # the 2025 out-of-sample run above
python main.py                     # daily brief -> backtests/daily_brief.txt
```

```bash
cd historical_study
python run_all.py                  # backtest, per-ticker models, Monte Carlo, December preview
```

The candidate lists in `live_pipeline/russell_candidates_2026.py` are the 2026 preliminary lists, hard-coded as inputs. Change them to look at another year. `generate_report_2026.py` builds a PDF version of the brief and needs `reportlab`, which is in `requirements.txt`.

## What is where

```
live_pipeline/
  daily_update.py             features and signal for the latest day
  train_model.py              trains the classifier and regressor
  validate_2025.py            out-of-sample 2025 run
  russell_candidates_2026.py  scores additions and deletions
  june_recon_overlay.py       regime and hard-stop logic
  main.py                     full pipeline and daily brief
  dashboard.py                terminal dashboard
  predict_2026.py             2026 predictions
  archive/                    earlier data pulls, feature builds and tuning
historical_study/
  scripts/                    backtest_full, per_ticker_models, monte_carlo, december_recon
  run_all.py                  runs the four steps in order
  results/                    summaries and CSVs from the last run
docs/                         figures used in this README
```

## Takeaways and applying them forward

- **The framework is the useful part.** The calendar gives you a known date, a published candidate list and a mechanical reason for price pressure. The code turns that into a repeatable routine: score the candidates, size by market regime, cap the downside with a hard stop. It can be pointed at the next reconstitution by changing the candidate lists.
- **Small names carry the pressure.** The screen is built on the idea that the smallest additions face the most forced buying relative to how much stock trades. That is the working hypothesis, and testing it properly is the first job for the next run.
- **Treat the model as a regime input, not a stand-alone signal.** The next-day direction model is one input to the regime filter. The 2025 check shows the pipeline runs end to end; it does not show an edge.
- **Before using any of it again:** fix the historical study's data columns (see Caveats), rerun it, then validate the model walk-forward across several years with costs included. Only after that should the study's numbers be quoted.
- **Next event:** the December 2026 preview in `historical_study/` is the natural place to start, since December 2026 is the first semi-annual Russell reconstitution and the same screen applies.

## Caveats

These are the known rough edges. They are here so the numbers are not read as more than they are.

- **The historical study has data problems.** Its results cover 25 years (2000-2025, with gaps), not the 1989-2025 that some script headers say. A few columns hold values that make no sense: the VIX pattern column reaches the hundreds, and the yield-curve column subtracts `^IRX` (a 13-week bill rate, not a 2-year yield) from `^TNX`, and the two are quoted in different units. Until that is fixed, the study's summary statistics, including its "Sharpe proxy", should not be relied on.
- **The Monte Carlo is assumption-driven.** It draws returns from hand-set expected returns and volatilities for each trade bucket (see `monte_carlo.py`) on a notional $1,000,000. Its output reflects those assumptions, not realized trading.
- **Only one held-out check.** The classifier trains on 2010-2024 with no separate validation split, so the 2025 run is the single test.
- **No costs.** Transaction costs, borrow costs and market impact are not modeled.
- **Year-specific inputs.** The candidate lists are tied to the June 26, 2026 reconstitution.

## Related

A separate project on the variance risk premium (VIX against forward realized volatility) is kept outside this repository.
