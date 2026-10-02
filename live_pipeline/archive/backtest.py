import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/master_features.csv"
CLASSIFIER_PATH = "models/lgbm_classifier.pkl"
REGRESSOR_PATH = "models/lgbm_regressor.pkl"
EQUITY_PATH = "backtests/equity_curve.csv"
START_CAPITAL = 100_000.0

FEATURES = [
    "sp500_close",
    "vix_close",
    "treasury_10y_close",
    "treasury_2y_close",
    "dollar_index_close",
    "gold_close",
    "oil_close",
    "rut_close_return",
    "rut_close_ma_5",
    "rut_close_ma_20",
    "rut_momentum_5",
    "rut_momentum_20",
    "yield_curve_spread",
    "vix_ma_5",
    "rut_rsi_14",
    "rut_bb_upper",
    "rut_bb_lower",
    "rut_macd",
    "rut_macd_signal",
    "rut_volume_ma_5",
    "rut_volume_ma_20",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_models():
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    return classifier, regressor


def simulate(df: pd.DataFrame, classifier, regressor) -> pd.DataFrame:
    df = df.copy()
    mask = (df["date"] >= pd.Timestamp("2022-01-01")) & (df["date"] <= pd.Timestamp("2023-12-31"))
    df = df.loc[mask].reset_index(drop=True)

    X = df[FEATURES]
    df["signal"] = classifier.predict(X)

    df["position_pct"] = 0.0
    df.loc[df["signal"].isin([1, 2]), "position_pct"] = 1.0
    df.loc[df["signal"] == 0, "position_pct"] = 0.5
    df.loc[df["signal"] == -1, "position_pct"] = 0.0

    df["next_return"] = df["rut_close"].shift(-1).sub(df["rut_close"]).div(df["rut_close"])
    df["next_return"] = df["next_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["strategy_return"] = (df["position_pct"] * df["next_return"]).fillna(0.0)
    df["portfolio_value"] = START_CAPITAL * (1 + df["strategy_return"]).cumprod()
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)

    trade_rows = df[df["position_pct"] > 0]
    if not trade_rows.empty:
        print("Trades taken:")
        for _, row in trade_rows.iterrows():
            print(
                f"{row['date'].date()} | signal={int(row['signal'])} | position={row['position_pct']:.2f} | daily_return={row['daily_return']:.6f}"
            )

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    total_return = df["portfolio_value"].iloc[-1] / START_CAPITAL - 1
    returns = df["daily_return"]
    sharpe = returns.mean() / returns.std(ddof=0) * np.sqrt(252) if returns.std(ddof=0) != 0 else 0.0
    cumulative_max = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    trade_mask = df["position_pct"] > 0
    wins = (df.loc[trade_mask, "next_return"] > 0).sum()
    trade_count = trade_mask.sum()
    win_rate = wins / trade_count if trade_count > 0 else 0.0

    return {
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate * 100,
        "trade_count": int(trade_count),
    }


def save_equity_curve(df: pd.DataFrame):
    os.makedirs(os.path.dirname(EQUITY_PATH), exist_ok=True)
    df[["date", "portfolio_value", "signal", "position_pct", "strategy_return"]].to_csv(EQUITY_PATH, index=False)


def print_summary(metrics: dict):
    print("Backtest performance summary")
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Trades taken: {metrics['trade_count']}")


def main() -> None:
    print(f"Loading data from {DATA_PATH}...")
    df = load_data()
    print(f"Loading models from {CLASSIFIER_PATH} and {REGRESSOR_PATH}...")
    classifier, regressor = load_models()
    print("Starting backtest simulation...")
    result = simulate(df, classifier, regressor)
    save_equity_curve(result)
    metrics = compute_metrics(result)
    print_summary(metrics)
    print(f"Equity curve saved to {EQUITY_PATH}")


if __name__ == "__main__":
    main()
