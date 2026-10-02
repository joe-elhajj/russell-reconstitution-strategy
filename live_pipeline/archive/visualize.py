import os
import pandas as pd
import matplotlib.pyplot as plt

EQUITY_PATH = "backtests/equity_curve.csv"
RUT_PATH = "data/rut.csv"
OUTPUT_PATH = "backtests/equity_curve.png"


def _load_csv_with_date_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    return df


def load_equity_curve() -> pd.DataFrame:
    df = _load_csv_with_date_index(EQUITY_PATH)
    return df.sort_values("date").reset_index(drop=True)


def load_rut_benchmark(equity_df: pd.DataFrame) -> pd.DataFrame:
    rut = _load_csv_with_date_index(RUT_PATH)
    rut = rut.sort_values("date").reset_index(drop=True)

    benchmark = rut[rut["date"].isin(equity_df["date"])].copy()
    if benchmark.empty:
        raise ValueError("No overlapping dates found between equity curve and RUT data.")

    first_value = 100_000.0
    benchmark["benchmark_value"] = first_value * benchmark["close_^rut"] / benchmark["close_^rut"].iloc[0]
    return benchmark[["date", "benchmark_value"]]


def plot_equity_curve(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["portfolio_value"], label="Strategy Portfolio", linewidth=2)
    plt.plot(benchmark_df["date"], benchmark_df["benchmark_value"], label="Buy and Hold RUT", linewidth=2, linestyle="--")

    plt.title("Equity Curve vs. Buy and Hold RUT")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved chart to {OUTPUT_PATH}")
    plt.show()


def main() -> None:
    equity_df = load_equity_curve()
    benchmark_df = load_rut_benchmark(equity_df)
    plot_equity_curve(equity_df, benchmark_df)


if __name__ == "__main__":
    main()
