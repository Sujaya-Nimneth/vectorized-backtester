from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .event import OrderEvent, FillEvent


class ExecutionHandler(ABC):
    """
    ExecutionHandler is an abstract base class representing a handler
    that receives OrderEvents and converts/fills them, generating FillEvents.
    
    This can be subclassed to model simulated executions (with slippage and commissions)
    or live broker connections (e.g. Interactive Brokers, Alpaca).
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> List[FillEvent]:
        """
        Takes an OrderEvent and executes it, returning a list of FillEvents.
        
        Args:
            event (OrderEvent): The order request to execute.
            
        Returns:
            List[FillEvent]: The resulting fill(s) representing executed trades.
        """
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    SimulatedExecutionHandler models order fills with configurable flat commissions
    and slippage models. It supports both fixed percentage slippage and
    volatility-based slippage calculated using rolling closing prices.
    """

    def __init__(
        self,
        data_handler,
        commission: float = 1.0,
        slippage_type: str = "fixed",
        fixed_slippage_pct: float = 0.0005,
        volatility_window: int = 10,
        volatility_multiplier: float = 0.1,
    ):
        """
        Initializes the execution simulator.
        
        Args:
            data_handler: The DataHandler instance for retrieving close prices/history.
            commission (float): Flat commission rate per trade.
            slippage_type (str): 'fixed' or 'volatility' slippage simulation.
            fixed_slippage_pct (float): Percentage of stock price for fixed slippage (e.g. 0.0005 = 0.05%).
            volatility_window (int): Rolling lookback window for volatility calculation.
            volatility_multiplier (float): Multiplier applied to standard deviation for slippage.
        """
        self.data_handler = data_handler
        self.commission = commission
        self.slippage_type = slippage_type.lower()
        self.fixed_slippage_pct = fixed_slippage_pct
        self.volatility_window = volatility_window
        self.volatility_multiplier = volatility_multiplier

        if self.slippage_type not in ["fixed", "volatility"]:
            raise ValueError("slippage_type must be either 'fixed' or 'volatility'.")

    def execute_order(self, event: OrderEvent) -> List[FillEvent]:
        """
        Executes an OrderEvent, applying commission and simulated slippage.
        Returns a List containing a single FillEvent.
        """
        if event.order_type != "MKT":
            raise NotImplementedError("Only Market Orders (MKT) are supported in this simulated handler.")

        symbol = event.symbol
        quantity = event.quantity
        direction = event.direction

        # Get latest close price
        close_price = self.data_handler.get_latest_bar_value(symbol, "close")
        if close_price is None:
            raise ValueError(f"No price data available to execute order for {symbol}.")

        # Get latest bar timestamp to record as execution datetime
        exec_dt = self.data_handler.get_latest_bar_datetime(symbol)
        exec_dt_str = str(exec_dt) if exec_dt else "unknown"

        # Calculate execution price incorporating slippage
        exec_price = self._calculate_execution_price(symbol, close_price, direction)

        # Create a single FillEvent
        fill_event = FillEvent(
            symbol=symbol,
            datetime=exec_dt_str,
            quantity=quantity,
            direction=direction,
            fill_cost=exec_price,
            commission=self.commission,
        )

        return [fill_event]

    def _calculate_execution_price(self, symbol: str, close_price: float, direction: str) -> float:
        """
        Helper method to calculate the fill price with slippage depending on the trade direction.
        For BUY: price is higher. For SELL: price is lower.
        """
        slippage = 0.0

        if self.slippage_type == "fixed":
            slippage = close_price * self.fixed_slippage_pct
        elif self.slippage_type == "volatility":
            # Retrieve historical close prices to calculate standard deviation
            bars = self.data_handler.get_latest_bars(symbol, N=self.volatility_window)
            if len(bars) >= 2:
                # Close price is stored at index 4 of the bar tuple
                closes = [bar[4] for bar in bars]
                volatility = float(np.std(closes))
                slippage = volatility * self.volatility_multiplier
            else:
                # Fallback to fixed slippage if there aren't enough historical data points
                slippage = close_price * self.fixed_slippage_pct

        # Apply slippage to trade execution price
        if direction == "BUY":
            return close_price + slippage
        elif direction == "SELL":
            return close_price - slippage
        else:
            raise ValueError(f"Unknown order direction '{direction}'. Must be BUY or SELL.")

