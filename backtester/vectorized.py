import pandas as pd
import numpy as np


class VectorizedBacktester:
    """
    VectorizedBacktester is a lightweight backtesting tool designed for rapid
    strategy prototyping using pandas and numpy vectorized operations.
    """

    def __init__(self, price_df: pd.DataFrame, price_col: str = "close"):
        """
        Initializes the vectorized backtester.

        Args:
            price_df (pd.DataFrame): DataFrame containing the price series.
            price_col (str): Column name containing the price data.
        """
        self.df = price_df.copy()
        self.price_col = price_col

        if price_col not in self.df.columns:
            raise KeyError(f"Price column '{price_col}' not found in the input DataFrame.")

        # Calculate standard asset daily percentage returns
        self.df["asset_returns"] = self.df[self.price_col].pct_change()

    def run_backtest(self, signals: pd.Series, tc: float = 0.001) -> pd.DataFrame:
        """
        Runs a vectorized backtest on the asset price series.

        Args:
            signals (pd.Series): Signals representing the position (1 for LONG, -1 for SHORT, 0 for CASH).
            tc (float): Percentage transaction cost per trade (e.g. 0.001 = 0.1%).

        Returns:
            pd.DataFrame: A DataFrame containing strategy holdings, returns, transaction costs, and equity curves.
        """
        results = self.df.copy()

        # Align signals to DataFrame index and fill missing values with 0.0 (no position)
        results["position"] = signals.reindex(results.index).fillna(0.0).astype(float)

        # Shift the signals by 1 period to prevent lookahead bias.
        # A signal generated at close of day T can only be executed at that close / next open,
        # meaning we capture returns of day T+1 using the position of day T.
        results["strategy_position"] = results["position"].shift(1).fillna(0.0)

        # Gross returns without trade commissions/costs
        results["gross_returns"] = results["strategy_position"] * results["asset_returns"]

        # Calculate position changes to apply transaction costs
        # The change in position is the absolute difference in target position from day to day
        results["position_diff"] = results["position"].diff().abs().fillna(0.0)

        # Vectorized transaction costs
        results["tc_costs"] = results["position_diff"] * tc

        # Net returns after transaction costs
        results["net_returns"] = results["gross_returns"] - results["tc_costs"]

        # Cumulative performance curves (equity curves starting at 1.0)
        results["cumulative_gross"] = (1.0 + results["gross_returns"].fillna(0.0)).cumprod()
        results["cumulative_net"] = (1.0 + results["net_returns"].fillna(0.0)).cumprod()

        return results
