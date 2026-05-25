import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def calculate_performance_metrics(
    equity_series: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculates key performance metrics for a portfolio equity curve.

    Args:
        equity_series (pd.Series): Portfolio equity values over time. Must have a datetime index.
        periods_per_year (int): Frequency of observations per year (default 252 for daily trading).
        risk_free_rate (float): Annualized risk-free rate (e.g. 0.02 = 2%).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "total_return": Total cumulative return (fraction).
            - "cagr": Compound Annual Growth Rate (fraction).
            - "sharpe_ratio": Annualized Sharpe Ratio.
            - "max_drawdown": Maximum Drawdown (fraction, typically negative).
    """
    if equity_series.empty or len(equity_series) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }

    # Total Return
    initial_equity = equity_series.iloc[0]
    final_equity = equity_series.iloc[-1]
    
    if initial_equity <= 0:
        raise ValueError("Initial equity must be greater than zero to calculate metrics.")
        
    total_return = (final_equity - initial_equity) / initial_equity

    # CAGR (Compound Annual Growth Rate)
    # Calculate years based on index dates if it is a datetime index, otherwise fallback to period count
    if isinstance(equity_series.index, pd.DatetimeIndex):
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
    else:
        years = len(equity_series) / periods_per_year

    if years > 0:
        cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Sharpe Ratio
    # Daily percentage returns of the equity
    returns = equity_series.pct_change().dropna()
    
    if len(returns) > 1 and returns.std() > 0:
        # Convert annualized risk-free rate to period risk-free rate
        period_rf = risk_free_rate / periods_per_year
        mean_excess_return = returns.mean() - period_rf
        sharpe_ratio = np.sqrt(periods_per_year) * (mean_excess_return / returns.std())
    else:
        sharpe_ratio = 0.0

    # Drawdowns and Maximum Drawdown
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = drawdowns.min()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown)
    }


def plot_performance(
    equity_series: pd.Series,
    log_scale: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a highly polished, professional 2-panel chart showing the
    cumulative equity curve and running drawdowns.

    Args:
        equity_series (pd.Series): The portfolio equity over time.
        log_scale (bool): Whether to plot the equity curve on a logarithmic y-axis scale.
        save_path (str): Optional file path to save the chart as an image (e.g. 'chart.png').
    """
    if equity_series.empty:
        raise ValueError("Cannot plot empty equity curve.")

    # Calculate Drawdown Curve
    running_max = equity_series.cummax()
    drawdown_series = (equity_series / running_max) - 1.0

    # Initialize professional plot styling
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # 1. Top Panel: Equity Curve
    color_equity = "#2E5BFF"  # Indigo/Royal Blue
    ax1.plot(equity_series.index, equity_series, color=color_equity, linewidth=2.0, label="Portfolio Equity")
    
    if log_scale:
        ax1.set_yscale("log")
        ax1.set_ylabel("Equity (Log Scale, $)", fontsize=11, fontweight="bold", labelpad=8)
    else:
        ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold", labelpad=8)
        
    ax1.set_title("Strategy Performance & Drawdown Analysis", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#D3D3D3")
    
    # Hide top/right boundaries for premium look
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", which="both", labelsize=10)

    # Add quick metrics summary box inside top panel
    metrics = calculate_performance_metrics(equity_series)
    summary_text = (
        f"Total Return: {metrics['total_return'] * 100:.2f}%\n"
        f"CAGR: {metrics['cagr'] * 100:.2f}%\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%"
    )
    
    ax1.text(
        0.02, 0.95, summary_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CFD8DC", alpha=0.9)
    )

    # 2. Bottom Panel: Drawdowns
    color_dd = "#E53935"  # Soft Solid Red
    ax2.plot(drawdown_series.index, drawdown_series * 100.0, color=color_dd, linewidth=1.2, label="Drawdown")
    ax2.fill_between(drawdown_series.index, drawdown_series * 100.0, 0.0, color=color_dd, alpha=0.25)
    
    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Timeline", fontsize=11, fontweight="bold", labelpad=8)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#D3D3D3")
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", which="both", labelsize=10)
    
    # Restrict drawdown graph to 0% ceiling and sensible floor
    max_dd_val = drawdown_series.min() * 100.0
    ax2.set_ylim(bottom=max(max_dd_val - 5.0, -100.0), top=2.0)

    # Align layout padding elegantly
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
