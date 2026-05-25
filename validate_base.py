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
    
    # Create simple linear price history for rapid testing
    dates = pd.date_range(start="2026-05-01", periods=5, freq="D")
    prices = [100.0, 102.0, 101.0, 105.0, 104.0]
    price_df = pd.DataFrame({"close": prices}, index=dates)
    
    # Initialize backtester
    backtester = VectorizedBacktester(price_df=price_df, price_col="close")
    
    # Define trade signal series (1 for Long, -1 for Short, 0 for cash)
    # Day 1: Neutral, Day 2: Go Long, Day 3: Stay Long, Day 4: Go Short, Day 5: Exit (Neutral)
    signals = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0], index=dates)
    
    # Run backtest with 0.1% percentage transaction cost
    tc_rate = 0.001
    results = backtester.run_backtest(signals=signals, tc=tc_rate)
    
    print("\nVectorized Results DataFrame:")
    print(results[["close", "asset_returns", "position", "strategy_position", "gross_returns", "tc_costs", "net_returns", "cumulative_net"]])
    
    # 1. Check returns logic
    # Day 2 asset return = (102 - 100) / 100 = 0.02
    assert np.isclose(results["asset_returns"].iloc[1], 0.02)
    
    # 2. Check no-lookahead shift logic
    # Day 2 position = 1.0. Shifted strategy position for Day 2 must be 0.0 (from Day 1 position).
    # Day 3 position = 1.0. Shifted strategy position for Day 3 must be 1.0 (from Day 2 position).
    # Day 4 position = -1.0. Shifted strategy position for Day 4 must be 1.0 (from Day 3 position).
    assert results["strategy_position"].iloc[1] == 0.0
    assert results["strategy_position"].iloc[2] == 1.0
    assert results["strategy_position"].iloc[3] == 1.0
    assert results["strategy_position"].iloc[4] == -1.0
    
    # 3. Check gross returns
    # Day 3 gross return = strategy_position(1.0) * asset_return( (101-102)/102 = -0.0098 )
    expected_day3_gross = 1.0 * ((101.0 - 102.0) / 102.0)
    assert np.isclose(results["gross_returns"].iloc[2], expected_day3_gross)
    
    # 4. Check vectorized transaction costs (diff-based)
    # Position differences:
    # Day 1 -> Day 2: diff = |1.0 - 0.0| = 1.0. Cost = 1.0 * 0.001 = 0.001
    # Day 2 -> Day 3: diff = |1.0 - 1.0| = 0.0. Cost = 0.0
    # Day 3 -> Day 4: diff = |-1.0 - 1.0| = 2.0. Cost = 2.0 * 0.001 = 0.002 (switching from LONG to SHORT)
    # Day 4 -> Day 5: diff = |0.0 - (-1.0)| = 1.0. Cost = 1.0 * 0.001 = 0.001
    assert np.isclose(results["tc_costs"].iloc[1], 0.001)
    assert np.isclose(results["tc_costs"].iloc[2], 0.0)
    assert np.isclose(results["tc_costs"].iloc[3], 0.002)
    assert np.isclose(results["tc_costs"].iloc[4], 0.001)
    
    # 5. Check net returns
    # Day 3 net return = gross_returns - tc_costs
    expected_day3_net = expected_day3_gross - 0.0
    assert np.isclose(results["net_returns"].iloc[2], expected_day3_net)
    
    # Day 4 net return = strategy_position(1.0) * asset_return((105-101)/101 = 0.0396) - tc_cost(0.002) = 0.0396 - 0.002 = 0.0376
    expected_day4_gross = 1.0 * ((105.0 - 101.0) / 101.0)
    expected_day4_net = expected_day4_gross - 0.002
    assert np.isclose(results["net_returns"].iloc[3], expected_day4_net)

    print("\nVectorized Backtester successfully verified!")


def main():
    mock_dir = "./mock_data"
    try:
        print("--- 1. Creating Mock CSV Data ---")
        create_mock_csv_data(mock_dir)
        
        run_event_driven_verification(mock_dir)
        run_vectorized_verification()
    finally:
        print("\n--- 7. Cleaning up Mock Data ---")
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)
        print("All validations completed successfully!")


if __name__ == "__main__":
    main()
