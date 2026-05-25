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


def run_verification():
    mock_dir = "./mock_data"
    print("--- 1. Creating Mock CSV Data ---")
    create_mock_csv_data(mock_dir)

    print("--- 2. Instantiating Components ---")
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

        # Update portfolio holdings/cash snapshot on market bar arrival
        portfolio.update_timeindex()

        # Let's execute some trades!
        # Step 1: Generate a LONG signal on AAPL
        if current_dt == pd.Timestamp("2026-05-01"):
            print("  [SIGNAL] Strategy triggers LONG signal on AAPL")
            sig = SignalEvent(symbol="AAPL", datetime=str(current_dt), signal_type="LONG")
            
            # Portfolio processes signal -> produces order
            orders = portfolio.update_signal(sig)
            assert len(orders) == 1
            order = orders[0]
            assert isinstance(order, OrderEvent)
            assert order.symbol == "AAPL"
            assert order.quantity == 100
            assert order.direction == "BUY"
            print(f"  [ORDER] Portfolio generates order: {order.direction} {order.quantity} {order.symbol}")

            # Execution handler executes order -> produces fill
            fills = execution_handler.execute_order(order)
            assert len(fills) == 1
            fill = fills[0]
            assert isinstance(fill, FillEvent)
            assert fill.symbol == "AAPL"
            assert fill.quantity == 100
            assert fill.direction == "BUY"
            # Slippage check: execution price should be close * (1 + slippage_pct)
            # AAPL close on Day 1 is 151.5. Slippage is 0.1% = 0.1515. Execution price = 151.6515.
            expected_price = 151.5 * 1.001
            assert np.isclose(fill.fill_cost, expected_price), f"Execution price {fill.fill_cost} != expected {expected_price}"
            assert fill.commission == 5.0
            print(f"  [FILL] Execution executes fill: price={fill.fill_cost:.4f}, commission={fill.commission}")

            # Portfolio updates fill
            portfolio.update_fill(fill)
            print(f"  [PORTFOLIO] Position updated: {portfolio.current_positions['AAPL']} AAPL. Cash: {portfolio.current_cash:.2f}")
            # Cash should be initial (100000) - (100 * 151.6515) - 5.0 = 100000 - 15165.15 - 5.0 = 84829.85
            expected_cash = 100000.0 - (100 * expected_price) - 5.0
            assert np.isclose(portfolio.current_cash, expected_cash), f"Portfolio cash {portfolio.current_cash} != expected {expected_cash}"

        # Step 3: Exit position on AAPL
        if current_dt == pd.Timestamp("2026-05-03"):
            print("  [SIGNAL] Strategy triggers EXIT signal on AAPL")
            sig = SignalEvent(symbol="AAPL", datetime=str(current_dt), signal_type="EXIT")
            
            orders = portfolio.update_signal(sig)
            assert len(orders) == 1
            order = orders[0]
            assert order.direction == "SELL"
            assert order.quantity == 100
            print(f"  [ORDER] Portfolio generates order: {order.direction} {order.quantity} {order.symbol}")

            fills = execution_handler.execute_order(order)
            fill = fills[0]
            assert fill.direction == "SELL"
            # Slippage check on sell: execution price should be close * (1 - slippage_pct)
            # AAPL close on Day 3 is 153.5. Slippage is 0.1% = 0.1535. Execution price = 153.3465.
            expected_price_sell = 153.5 * 0.999
            assert np.isclose(fill.fill_cost, expected_price_sell)
            print(f"  [FILL] Execution executes fill: price={fill.fill_cost:.4f}, commission={fill.commission}")

            portfolio.update_fill(fill)
            print(f"  [PORTFOLIO] Position updated: {portfolio.current_positions['AAPL']} AAPL. Cash: {portfolio.current_cash:.2f}")
            # Cash should be previous cash (84829.85) + (100 * 153.3465) - 5.0 = 84829.85 + 15334.65 - 5 = 100159.50
            expected_cash_final = expected_cash + (100 * expected_price_sell) - 5.0
            assert np.isclose(portfolio.current_cash, expected_cash_final)
            assert portfolio.current_positions["AAPL"] == 0

        events_count += 1

    print("\n--- 4. Checking Equity Curve DataFrame Output ---")
    equity_df = portfolio.create_equity_curve_dataframe()
    print("Equity Curve DataFrame Header:")
    print(equity_df)

    assert not equity_df.empty, "Equity curve DataFrame should not be empty."
    assert "total" in equity_df.columns
    assert "returns" in equity_df.columns
    assert "equity_curve" in equity_df.columns
    # Check that returns are calculated correctly
    assert pd.isna(equity_df["returns"].iloc[0]), "First daily return must be NaN."
    
    print("\n--- 5. Verifying Volatility Slippage Handler ---")
    # Reset data handler and test volatility-based slippage
    data_handler_vol = HistoricCSVDataHandler(csv_dir=mock_dir, symbol_list=symbols)
    # Feed data until Day 5 so we have historical bars to calculate std dev
    for _ in range(5):
        data_handler_vol.update_bars()
    
    execution_handler_vol = SimulatedExecutionHandler(
        data_handler=data_handler_vol,
        commission=2.0,
        slippage_type="volatility",
        volatility_window=10,
        volatility_multiplier=0.5
    )
    
    # AAPL closes: Day 1: 151.5, Day 2: 152.0, Day 3: 153.5, Day 5: 156.0
    closes = [151.5, 152.0, 153.5, 156.0]
    expected_vol = np.std(closes)
    expected_slippage = expected_vol * 0.5
    
    order_vol = OrderEvent(symbol="AAPL", order_type="MKT", quantity=100, direction="BUY")
    fills_vol = execution_handler_vol.execute_order(order_vol)
    fill_vol = fills_vol[0]
    
    close_day5 = 156.0
    expected_price_vol = close_day5 + expected_slippage
    assert np.isclose(fill_vol.fill_cost, expected_price_vol), f"Volatility slippage price {fill_vol.fill_cost} != expected {expected_price_vol}"
    print(f"Volatility-based execution successfully verified: Exec price: {fill_vol.fill_cost:.4f} (std dev = {expected_vol:.4f}, slippage = {expected_slippage:.4f})")

    print("\n--- 6. Cleaning up Mock Data ---")
    shutil.rmtree(mock_dir)
    print("All component verifications completed successfully!")


if __name__ == "__main__":
    run_verification()
