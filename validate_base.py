import os
import shutil
import pandas as pd
from typing import List

from backtester import (
    HistoricCSVDataHandler,
    MarketEvent,
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

    print("--- 2. Instantiating HistoricCSVDataHandler ---")
    symbols = ["AAPL", "MSFT"]
    data_handler = HistoricCSVDataHandler(csv_dir=mock_dir, symbol_list=symbols)

    print(f"Combined Chronological Index: {data_handler.combined_index}")
    assert len(data_handler.combined_index) == 5, "Combined index should contain 5 unique days: 1, 2, 3, 4, 5."

    print("--- 3. Running Event loop & Verifying Zero Lookahead ---")
    
    events_count = 0
    expected_timeline = [
        pd.Timestamp("2026-05-01"),
        pd.Timestamp("2026-05-02"),
        pd.Timestamp("2026-05-03"),
        pd.Timestamp("2026-05-04"),
        pd.Timestamp("2026-05-05"),
    ]

    for event in data_handler.generate_events():
        assert isinstance(event, MarketEvent), "Event generated must be a MarketEvent."
        assert event.type == EventType.MARKET, "Event type must be MARKET."

        current_dt = expected_timeline[events_count]
        print(f"\nStep {events_count + 1} - Current Datetime: {current_dt}")

        # Verification at each time step
        for symbol in symbols:
            latest_bar = data_handler.get_latest_bar(symbol)
            if latest_bar:
                latest_dt = latest_bar[0]
                latest_close = data_handler.get_latest_bar_value(symbol, "close")
                print(f"  {symbol} latest bar datetime: {latest_dt}, close: {latest_close}")
                
                # Check for zero lookahead: the bar's timestamp must be <= current timeline timestamp
                assert latest_dt <= current_dt, f"Lookahead bias detected! {symbol} bar is from {latest_dt} but timeline is only at {current_dt}."
                
                # Verify standard index column values
                assert data_handler.get_latest_bar_value(symbol, "open") is not None
                assert data_handler.get_latest_bar_value(symbol, "high") is not None
                assert data_handler.get_latest_bar_value(symbol, "low") is not None
                assert data_handler.get_latest_bar_value(symbol, "volume") is not None
                assert data_handler.get_latest_bar_value(symbol, "adj_close") is not None
            else:
                print(f"  {symbol} has no bars loaded yet.")
                # Verify that AAPL should have no bars at day 4 (since it doesn't trade on day 4) but its latest_bar should be day 3
                if symbol == "AAPL" and current_dt == pd.Timestamp("2026-05-04"):
                    # Wait, AAPL should have day 3 data since that was loaded on day 3
                    pass

        # Specific assertions for each step to verify gaps are handled correctly:
        if current_dt == pd.Timestamp("2026-05-01"):
            # Both AAPL and MSFT have data
            assert data_handler.get_latest_bar_datetime("AAPL") == pd.Timestamp("2026-05-01")
            assert data_handler.get_latest_bar_datetime("MSFT") == pd.Timestamp("2026-05-01")
        elif current_dt == pd.Timestamp("2026-05-03"):
            # AAPL has a bar on Day 3, MSFT does not.
            # AAPL's latest bar should be Day 3. MSFT's latest bar should remain Day 2.
            assert data_handler.get_latest_bar_datetime("AAPL") == pd.Timestamp("2026-05-03")
            assert data_handler.get_latest_bar_datetime("MSFT") == pd.Timestamp("2026-05-02")
            assert len(data_handler.get_latest_bars("AAPL", N=3)) == 3
            assert len(data_handler.get_latest_bars("MSFT", N=3)) == 2
        elif current_dt == pd.Timestamp("2026-05-04"):
            # AAPL does NOT have a bar on Day 4, MSFT does.
            # AAPL's latest bar should remain Day 3. MSFT's latest bar should be Day 4.
            assert data_handler.get_latest_bar_datetime("AAPL") == pd.Timestamp("2026-05-03")
            assert data_handler.get_latest_bar_datetime("MSFT") == pd.Timestamp("2026-05-04")
            
        events_count += 1

    assert events_count == 5, f"Should run exactly 5 event steps, but got {events_count}."
    
    print("\n--- 4. Cleaning up Mock Data ---")
    shutil.rmtree(mock_dir)
    print("Verification completed successfully!")


if __name__ == "__main__":
    run_verification()
