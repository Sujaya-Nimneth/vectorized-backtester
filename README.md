# Custom Dual-Mode Trading Backtesting Engine in Python

A high-performance, modular Python backtesting framework supporting **Event-Driven Simulation** (for high-fidelity execution modeling) and **Lightweight Vectorized Prototyping** (for rapid strategy verification). Fully type-hinted, performance-optimized, and engineered with **zero lookahead bias**.

---

## 📁 Project Structure

```
vectorized-backtester/
├── backtester/                     # Core Package
│   ├── __init__.py                 # Package Exports
│   ├── event.py                    # Event Hierarchy Definitions
│   ├── data.py                     # Ingestion & Historic Data Feeds
│   ├── strategy.py                 # Abstract Strategy & SMA Concrete Class
│   ├── portfolio.py                # Position Tracker, Cash Manager & Sizing
│   ├── execution.py                # Commission & Volatility Slippage Simulator
│   ├── vectorized.py               # Lightweight Vectorized Prototyping Engine
│   └── performance.py              # Performance Analytics & Plotting Functions
├── venv/                           # Python Virtual Environment
├── main.py                         # Event-Driven Orchestration & Backtest Demo
└── validate_base.py                # Framework Integration Verification Suite
```

---

## 🚀 Key Features

### 1. Dual-Mode Trading Engine
- **Event-Driven Engine**: Models real-world trading logic passing structured message classes (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) through a central queue. It exactly mimics live execution pipelines, ensuring smooth transition from backtesting to production.
- **Lightweight Vectorized Engine (`VectorizedBacktester`)**: Written in pure `pandas` and `numpy` for lightning-fast strategy prototyping. Calculates returns, rebalancing events, and transaction cost adjustments in a single vectorized pass.

### 2. High-Performance Optimizations
- **O(1) Memory Cache**: The `HistoricCSVDataHandler` converts Pandas DataFrame rows into standard Python tuples mapped inside dictionaries (`Timestamp -> Tuple`) during initialization. This reduces Pandas lookup overhead in the event loop, yielding **10-100x faster loop executions**.
- **Vectorized Rebalancing Costs**: Vectorized prototyping handles trade commissions instantly using `.diff().abs()`. Position shifts are calculated in one pass (e.g., reversing from Long to Short correctly triggers a double fee).

### 3. Execution Fidelity & Friction
- **Zero Lookahead Bias**: Uses a chronological, multi-asset unified timeline iterator. Data points are appended to the internal state *only* up to the current simulated step, completely preventing lookahead bias.
- **Dynamic Slippage**: Supports fixed percentage slippage (e.g., executing at a `0.05%` worse price) and **Volatility-based Slippage** which calculates the standard deviation of recent closing prices using `numpy.std` to simulate execution friction in volatile markets.
- **Transaction Costs**: Configurable flat commissions per trade applied to fills.

### 4. Advanced Analytics & Premium Visualization
- Calculates key metrics: **Total Return**, **CAGR** (Compound Annual Growth Rate calculated from indices), **Annualized Sharpe Ratio** (with customizable risk-free rate), and **Maximum Drawdown**.
- Renders a professional double-panel chart:
  - **Top Panel**: High-contrast royal blue equity curve over time with logarithmic scale options.
  - **Bottom Panel**: Shaded red drawdown area representing running risk profile.

---

## 🛠️ Getting Started

### 1. Prerequisite Installation
Set up your virtual environment and install standard mathematical dependencies (`pandas`, `numpy`, and `matplotlib`):

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required dependencies
pip install pandas numpy matplotlib
```

### 2. Run the Verification Tests
To confirm that all components (ingestion, zero-lookahead timelines, portfolios, slippage engines, vectorized matrices, and metrics calculations) behave exactly as designed, execute:

```bash
python validate_base.py
```

### 3. Run the Event-Driven Moving Average Backtest
To execute the complete, end-to-end backtesting pipeline simulating a Moving Average Crossover strategy on 150 days of trending mock stock data, run:

```bash
python main.py
```

This will run the simulation, print a performance summary to the console, and save a professional double-panel chart named `backtest_performance.png` in the workspace root.

---

## 📈 Backtest Execution Summary

```
---------------------------------------------
Backtesting Performance Summary (AAPL Crossover):
---------------------------------------------
Initial Capital   : $100,000.00
Final Value       : $110,313.78
Total Return      : 10.31%
CAGR              : 18.71%
Sharpe Ratio      : 2.7006
Max Drawdown      : -2.87%
Total Commissions : $50.00
---------------------------------------------
```
