"""
Download historical prices, plot cumulative returns, correlation matrix,
and risk/return metrics for selected tickers.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

TICKERS = ["NVDA", "MU", "MRVL", "VRT", "SOXX"]
YEARS = 3
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # annualized, e.g. ~4% T-bill proxy


def download_prices(tickers: list[str], years: int) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all")
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return (1 + daily_returns(prices)).cumprod() - 1


def annualized_metrics(returns: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    mean_daily = returns.mean()
    std_daily = returns.std()
    ann_return = mean_daily * TRADING_DAYS
    ann_vol = std_daily * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol
    return pd.DataFrame(
        {
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
        }
    )


def plot_cumulative_returns(cum_returns: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in cum_returns.columns:
        ax.plot(cum_returns.index, cum_returns[ticker] * 100, label=ticker, linewidth=1.5)
    ax.set_title(f"Cumulative Returns — Last {YEARS} Years")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(returns: pd.DataFrame, output_path: str) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Daily Return Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Downloading {YEARS} years of data for: {', '.join(TICKERS)}")
    prices = download_prices(TICKERS, YEARS)
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Trading days: {len(prices)}\n")

    returns = daily_returns(prices)
    cum_ret = cumulative_returns(prices)
    metrics = annualized_metrics(returns, RISK_FREE_RATE)

    print("=" * 60)
    print("Risk / Return Metrics (annualized)")
    print(f"Risk-free rate assumption: {RISK_FREE_RATE:.1%}")
    print("=" * 60)
    formatted = metrics.copy()
    formatted["Annualized Return"] = formatted["Annualized Return"].map("{:.2%}".format)
    formatted["Annualized Volatility"] = formatted["Annualized Volatility"].map("{:.2%}".format)
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map("{:.2f}".format)
    print(formatted.to_string())
    print()

    print("=" * 60)
    print("Correlation Matrix")
    print("=" * 60)
    print(returns.corr().round(3).to_string())
    print()

    plot_cumulative_returns(cum_ret, "cumulative_returns.png")
    plot_correlation_matrix(returns, "correlation_matrix.png")
    print("Saved: cumulative_returns.png")
    print("Saved: correlation_matrix.png")


if __name__ == "__main__":
    main()
