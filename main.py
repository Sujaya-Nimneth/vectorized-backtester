import os
import shutil
import numpy as np
import pandas as pd

from backtester import (
    HistoricCSVDataHandler,
    MovingAverageCrossover,
    NaivePortfolio,
    SimulatedExecutionHandler,
    EventType,
    calculate_performance_metrics,
    plot_performance,
)


def generate_trending_mock_data(data_dir: str):
    """
    Generates 150 days of mock stock price data for AAPL.
    Features a strong upward trend with cyclical fluctuations to guarantee
    clear moving average crossovers.
    """
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)

    # 150 business days starting from Jan 1, 2026
    dates = pd.date_range(start="2026-01-01", periods=150, freq="B")
    
    # Start at 100
    price = 100.0
    prices = []
    
    # Generate prices using a random walk with strong positive drift and cyclical seasonality
    for i in range(150):
        drift = 0.25  # Upward bias
        cycle = 5.0 * np.sin(i / 8.0)  # Generates peaks and troughs for crossovers
        noise = np.random.normal(0.0, 1.0)
        
        # Base price changes
        price += drift + noise
        final_price = max(price + cycle, 10.0)  # Keep price positive
        prices.append(final_price)

    # Calculate Open, High, Low based on Close
    df = pd.DataFrame({"Close": prices}, index=dates)
    df["Open"] = df["Close"].shift(1).fillna(100.0) + np.random.normal(0, 0.5, 150)
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.uniform(0.1, 1.5, 150)
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.uniform(0.1, 1.5, 150)
    df["Volume"] = np.random.randint(100000, 500000, 150)
    df["Adj Close"] = df["Close"]

    # Reindex columns to standard names
    df = df.reset_index().rename(columns={"index": "Date"})
    csv_path = os.path.join(data_dir, "AAPL.csv")
    df.to_csv(csv_path, index=False)
    print(f"Mock trending market data successfully saved to: {csv_path}")


def main():
    data_dir = "./data"
    chart_output = "backtest_performance.png"

    # Step 1: Create mock price data
    print("=== Step 1: Generating Mock Market Data ===")
    generate_trending_mock_data(data_dir)

    # Step 2: Initialize event-driven backtesting components
    print("\n=== Step 2: Initializing Backtesting Components ===")
    symbols = ["AAPL"]
    
    data_handler = HistoricCSVDataHandler(csv_dir=data_dir, symbol_list=symbols)
    strategy = MovingAverageCrossover(data_handler=data_handler, short_window=5, long_window=20)
    portfolio = NaivePortfolio(data_handler=data_handler, initial_capital=100000.0, fixed_quantity=500)
    
    # Simulated execution with flat $10 commission and 0.05% slippage
    execution_handler = SimulatedExecutionHandler(
        data_handler=data_handler,
        commission=10.0,
        slippage_type="fixed",
        fixed_slippage_pct=0.0005,  # 0.05%
    )

    # Step 3: Run the queue-based event-driven loop
    print("\n=== Step 3: Executing Event-Driven Backtest Loop ===")
    events_queue = []
    trade_count = 0

    # generate_events() sequentially steps through time indexes
    for market_event in data_handler.generate_events():
        events_queue.append(market_event)

        # Process all triggered cascade events
        while len(events_queue) > 0:
            event = events_queue.pop(0)

            if event.type == EventType.MARKET:
                portfolio.update_timeindex()
                signals = strategy.calculate_signals(event)
                events_queue.extend(signals)

            elif event.type == EventType.SIGNAL:
                orders = portfolio.update_signal(event)
                events_queue.extend(orders)

            elif event.type == EventType.ORDER:
                fills = execution_handler.execute_order(event)
                events_queue.extend(fills)

            elif event.type == EventType.FILL:
                portfolio.update_fill(event)
                trade_count += 1
                fill_dir = event.direction
                print(
                    f"  [TRADE EXECUTION] Date: {event.datetime[:10]} | "
                    f"{fill_dir} {event.quantity} shares of {event.symbol} | "
                    f"Price: ${event.fill_cost:.2f} | Commission: ${event.commission:.2f}"
                )

    print(f"\nBacktest run completed. Total event steps: {len(data_handler.combined_index)}")
    print(f"Total simulated order fills: {trade_count}")

    # Step 4: Extract results and compute performance metrics
    print("\n=== Step 4: Compiling Performance Report ===")
    equity_df = portfolio.create_equity_curve_dataframe()

    if not equity_df.empty:
        equity_series = equity_df["total"]
        
        # Calculate key performance metrics (annualized at 252 business days, 2% risk-free rate)
        metrics = calculate_performance_metrics(
            equity_series=equity_series,
            periods_per_year=252,
            risk_free_rate=0.02
        )

        print("-" * 45)
        print(f"Backtesting Performance Summary (AAPL Crossover):")
        print("-" * 45)
        print(f"Initial Capital   : ${portfolio.initial_capital:,.2f}")
        print(f"Final Value       : ${equity_series.iloc[-1]:,.2f}")
        print(f"Total Return      : {metrics['total_return'] * 100:.2f}%")
        print(f"CAGR              : {metrics['cagr'] * 100:.2f}%")
        print(f"Sharpe Ratio      : {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown      : {metrics['max_drawdown'] * 100:.2f}%")
        print(f"Total Commissions : ${portfolio.current_commission:,.2f}")
        print("-" * 45)

        # Step 5: Save visual performance plot
        print(f"Generating and saving performance chart to: {chart_output}")
        plot_performance(
            equity_series=equity_series,
            log_scale=False,
            save_path=chart_output
        )
        print("Performance chart successfully saved.")
    else:
        print("Error: Equity curve is empty. Check data ingestion or timeline alignment.")

    # Step 6: Cleanup mock CSV directories
    print("\n=== Step 5: Cleaning Up Temporary Files ===")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("Mock CSV data folder removed successfully.")
    print("All tasks completed.")


if __name__ == "__main__":
    main()
