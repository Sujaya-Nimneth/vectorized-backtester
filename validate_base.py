import os
import shutil
import pandas as pd
import numpy as np
from typing import List

from backtester import (
    HistoricCSVDataHandler,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    NaivePortfolio,
    SimulatedExecutionHandler,
    VectorizedBacktester,
    calculate_performance_metrics,
    plot_performance,
    EventType
)


def create_mock_csv_data(data_dir: str):
    """
    Creates mock CSV data for AAPL and MSFT with minor overlaps and gaps.
    """
    os.makedirs(data_dir, exist_ok=True)

    # AAPL has bars on Day 1, 2, 3, 5
    aapl_data = {
        "Date": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-05"],
        "Open": [150.0, 151.0, 152.0, 155.0],
        "High": [152.5, 153.0, 154.0, 156.5],
        "Low": [149.0, 150.5, 151.5, 154.0],
        "Close": [151.5, 152.0, 153.5, 156.0],
        "Volume": [1000, 1100, 1200, 1500],
    }
    df_aapl = pd.DataFrame(aapl_data)
    df_aapl.to_csv(os.path.join(data_dir, "AAPL.csv"), index=False)

    # MSFT has bars on Day 1, 2, 4, 5
    msft_data = {
        "Date": ["2026-05-01", "2026-05-02", "2026-05-04", "2026-05-05"],
        "Open": [250.0, 251.0, 253.0, 255.0],
        "High": [252.5, 253.0, 255.0, 257.5],
        "Low": [249.0, 250.5, 252.0, 254.0],
        "Close": [251.5, 252.0, 254.5, 256.0],
        "Volume": [2000, 2100, 2200, 2500],
    }
    df_msft = pd.DataFrame(msft_data)
    df_msft.to_csv(os.path.join(data_dir, "MSFT.csv"), index=False)


def run_event_driven_verification(mock_dir: str):
    print("--- 2. Instantiating Event-Driven Components ---")
    symbols = ["AAPL", "MSFT"]
    data_handler = HistoricCSVDataHandler(csv_dir=mock_dir, symbol_list=symbols)
    portfolio = NaivePortfolio(data_handler=data_handler, initial_capital=100000.0, fixed_quantity=100)
    
    # Execution handler with flat $5 commission and 0.1% fixed slippage
    execution_handler = SimulatedExecutionHandler(
        data_handler=data_handler,
        commission=5.0,
        slippage_type="fixed",
        fixed_slippage_pct=0.001  # 0.1% slippage
    )

    print(f"Combined Chronological Index: {data_handler.combined_index}")
    print("\n--- 3. Simulating Event Loop with Trading flow ---")
    
    events_count = 0
    expected_timeline = [
        pd.Timestamp("2026-05-01"),
        pd.Timestamp("2026-05-02"),
        pd.Timestamp("2026-05-03"),
        pd.Timestamp("2026-05-04"),
        pd.Timestamp("2026-05-05"),
    ]

    for event in data_handler.generate_events():
        current_dt = expected_timeline[events_count]
        print(f"\nStep {events_count + 1} - Current Datetime: {current_dt}")

        portfolio.update_timeindex()

        # Step 1: Generate a LONG signal on AAPL
        if current_dt == pd.Timestamp("2026-05-01"):
            print("  [SIGNAL] Strategy triggers LONG signal on AAPL")
            sig = SignalEvent(symbol="AAPL", datetime=str(current_dt), signal_type="LONG")
            
            orders = portfolio.update_signal(sig)
            assert len(orders) == 1
            order = orders[0]
            assert order.direction == "BUY"
            print(f"  [ORDER] Portfolio generates order: {order.direction} {order.quantity} {order.symbol}")

            fills = execution_handler.execute_order(order)
            fill = fills[0]
            # AAPL close on Day 1 is 151.5. Slippage is 0.1%. Price = 151.6515.
            expected_price = 151.5 * 1.001
            assert np.isclose(fill.fill_cost, expected_price)
            assert fill.commission == 5.0
            print(f"  [FILL] Execution executes fill: price={fill.fill_cost:.4f}, commission={fill.commission}")

            portfolio.update_fill(fill)
            print(f"  [PORTFOLIO] Position updated: {portfolio.current_positions['AAPL']} AAPL. Cash: {portfolio.current_cash:.2f}")
            expected_cash = 100000.0 - (100 * expected_price) - 5.0
            assert np.isclose(portfolio.current_cash, expected_cash)

        # Step 3: Exit position on AAPL
        if current_dt == pd.Timestamp("2026-05-03"):
            print("  [SIGNAL] Strategy triggers EXIT signal on AAPL")
            sig = SignalEvent(symbol="AAPL", datetime=str(current_dt), signal_type="EXIT")
            
            orders = portfolio.update_signal(sig)
            assert len(orders) == 1
            order = orders[0]
            assert order.direction == "SELL"
            print(f"  [ORDER] Portfolio generates order: {order.direction} {order.quantity} {order.symbol}")

            fills = execution_handler.execute_order(order)
            fill = fills[0]
            # AAPL close on Day 3 is 153.5. Slippage is 0.1%. Price = 153.3465.
            expected_price_sell = 153.5 * 0.999
            assert np.isclose(fill.fill_cost, expected_price_sell)
            print(f"  [FILL] Execution executes fill: price={fill.fill_cost:.4f}, commission={fill.commission}")

            portfolio.update_fill(fill)
            print(f"  [PORTFOLIO] Position updated: {portfolio.current_positions['AAPL']} AAPL. Cash: {portfolio.current_cash:.2f}")
            expected_cash_final = expected_cash + (100 * expected_price_sell) - 5.0
            assert np.isclose(portfolio.current_cash, expected_cash_final)
            assert portfolio.current_positions["AAPL"] == 0

        events_count += 1

    print("\n--- 4. Checking Equity Curve DataFrame Output ---")
    equity_df = portfolio.create_equity_curve_dataframe()
    print(equity_df)
    assert not equity_df.empty

    print("\n--- 5. Verifying Volatility Slippage Handler ---")
    data_handler_vol = HistoricCSVDataHandler(csv_dir=mock_dir, symbol_list=symbols)
    for _ in range(5):
        data_handler_vol.update_bars()
    
    execution_handler_vol = SimulatedExecutionHandler(
        data_handler=data_handler_vol,
        commission=2.0,
        slippage_type="volatility",
        volatility_window=10,
        volatility_multiplier=0.5
    )
    
    closes = [151.5, 152.0, 153.5, 156.0]
    expected_vol = np.std(closes)
    expected_slippage = expected_vol * 0.5
    
    order_vol = OrderEvent(symbol="AAPL", order_type="MKT", quantity=100, direction="BUY")
    fills_vol = execution_handler_vol.execute_order(order_vol)
    fill_vol = fills_vol[0]
    expected_price_vol = 156.0 + expected_slippage
    assert np.isclose(fill_vol.fill_cost, expected_price_vol)
    print(f"Volatility-based execution successfully verified: Exec price: {fill_vol.fill_cost:.4f} (std dev = {expected_vol:.4f})")


def run_vectorized_verification():
    print("\n--- 6. Verifying Lightweight Vectorized Backtester ---")
    dates = pd.date_range(start="2026-05-01", periods=5, freq="D")
    prices = [100.0, 102.0, 101.0, 105.0, 104.0]
    price_df = pd.DataFrame({"close": prices}, index=dates)
    
    backtester = VectorizedBacktester(price_df=price_df, price_col="close")
    signals = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0], index=dates)
    
    tc_rate = 0.001
    results = backtester.run_backtest(signals=signals, tc=tc_rate)
    
    print("Vectorized Results:")
    print(results[["close", "position", "strategy_position", "gross_returns", "tc_costs", "net_returns", "cumulative_net"]])
    assert np.isclose(results["asset_returns"].iloc[1], 0.02)
    assert results["strategy_position"].iloc[1] == 0.0
    assert results["strategy_position"].iloc[2] == 1.0
    assert np.isclose(results["tc_costs"].iloc[1], 0.001)

    print("Vectorized Backtester successfully verified!")


def run_performance_tracker_verification():
    print("\n--- 7. Verifying PerformanceTracker Module ---")
    
    # Simulate a full 1-year daily backtest equity curve (252 trading days)
    # Starts at 100,000, trends upward with some noise, encounters a major drawdown, and recovers.
    np.random.seed(42)
    t = pd.date_range(start="2025-01-01", periods=252, freq="B")  # Business days
    
    # Constructing deterministic trend + drawdown structure
    # Initial growth phase (first 100 days): grows from 100k to ~125k
    growth = np.linspace(100000, 125000, 100)
    
    # Drawdown phase (next 52 days): falls from 125k to ~110k (approx -12% drawdown)
    drawdown = np.linspace(125000, 110000, 52)
    
    # Recovery and final rally (remaining 100 days): recovers to 110k -> 130k
    recovery = np.linspace(110000, 130000, 100)
    
    raw_equity = np.concatenate([growth, drawdown, recovery])
    # Add minor daily percentage noise (daily return standard deviation of ~0.5%)
    noise_returns = np.random.normal(loc=0.0, scale=0.005, size=252)
    equity = np.zeros(252)
    equity[0] = 100000.0
    for i in range(1, 252):
        # Apply current baseline price + standard daily percentage returns noise
        base_return = (raw_equity[i] - raw_equity[i-1]) / raw_equity[i-1]
        equity[i] = equity[i-1] * (1.0 + base_return + noise_returns[i])
        
    equity_series = pd.Series(equity, index=t)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_series, periods_per_year=252, risk_free_rate=0.02)
    
    print("\nCalculated Metrics:")
    for k, v in metrics.items():
        if k in ["total_return", "cagr", "max_drawdown"]:
            print(f"  {k.replace('_', ' ').title()}: {v * 100:.2f}%")
        else:
            print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
            
    # Verification assertions:
    # 1. Total Return should be close to 30.0% (starts at 100,000, ends at ~130,000)
    assert 0.20 <= metrics["total_return"] <= 0.40, f"Total return {metrics['total_return']} not in expected range."
    # 2. CAGR should be in line with total return since duration is exactly ~1 year
    assert 0.20 <= metrics["cagr"] <= 0.40
    # 3. Max Drawdown should capture our programmed dip. High-watermark was ~125k, trough was ~110k, so it should be negative and around -10% to -15%
    assert -0.20 <= metrics["max_drawdown"] <= -0.05, f"Max Drawdown {metrics['max_drawdown']} not in expected range."
    # 4. Sharpe ratio should be positive
    assert metrics["sharpe_ratio"] > 0.0

    # Test visualization plot rendering
    chart_filename = "performance_chart.png"
    print(f"\nRendering and saving double-panel performance chart to: {chart_filename}")
    plot_performance(equity_series=equity_series, log_scale=False, save_path=chart_filename)
    
    # Confirm plot image file was generated successfully
    assert os.path.exists(chart_filename), f"Performance chart file '{chart_filename}' was not created."
    print("Performance plot successfully created and verified!")
    
    # Cleanup chart file
    if os.path.exists(chart_filename):
        os.remove(chart_filename)
        print("Cleaned up temporary plot file.")


def main():
    mock_dir = "./mock_data"
    try:
        print("--- 1. Creating Mock CSV Data ---")
        create_mock_csv_data(mock_dir)
        
        run_event_driven_verification(mock_dir)
        run_vectorized_verification()
        run_performance_tracker_verification()
    finally:
        print("\n--- 8. Cleaning up Mock Data ---")
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)
        print("All validations completed successfully!")


if __name__ == "__main__":
    main()
