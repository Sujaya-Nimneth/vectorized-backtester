from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

from .event import SignalEvent, OrderEvent, FillEvent


class Portfolio(ABC):
    """
    Portfolio is an abstract base class that handles the positions,
    cash, holdings, and generates OrderEvents based on incoming SignalEvents.
    
    It keeps track of the portfolio's current state and makes trading decisions
    by turning signals into orders while considering risk management.
    """

    @abstractmethod
    def update_signal(self, event: SignalEvent) -> List[OrderEvent]:
        """
        Acts on a SignalEvent to generate new OrderEvents based on portfolio sizing
        and risk rules.
        
        Args:
            event (SignalEvent): The incoming trading signal.
            
        Returns:
            List[OrderEvent]: A list of orders to be sent to execution.
        """
        pass

    @abstractmethod
    def update_fill(self, event: FillEvent) -> None:
        """
        Updates the portfolio's positions, cash, and holdings based on a FillEvent
        received from the execution system.
        
        Args:
            event (FillEvent): The fill event representing the executed order details.
        """
        pass

    @abstractmethod
    def update_timeindex(self) -> None:
        """
        Adds a new record to the positions and holdings lists representing
        the current state of the portfolio at the latest datetime index.
        """
        pass


class NaivePortfolio(Portfolio):
    """
    NaivePortfolio tracks positions, cash balance, and updates an equity curve
    on every MarketEvent. It handles SignalEvents by generating OrderEvents with
    a simple fixed-quantity sizing logic.
    """

    def __init__(self, data_handler, initial_capital: float = 100000.0, fixed_quantity: int = 100):
        """
        Initializes the portfolio with a data handler, initial capital, and default order sizing.
        """
        self.data_handler = data_handler
        self.symbol_list = self.data_handler.symbol_list
        self.initial_capital = initial_capital
        self.fixed_quantity = fixed_quantity

        self.current_cash = initial_capital
        self.current_commission = 0.0
        self.current_positions = {symbol: 0 for symbol in self.symbol_list}
        self.current_holdings = {symbol: 0.0 for symbol in self.symbol_list}

        self.all_positions: List[Dict[str, Any]] = []
        self.all_holdings: List[Dict[str, Any]] = []

    def update_timeindex(self) -> None:
        """
        Adds a new record to the positions and holdings lists representing
        the current state of the portfolio at the latest datetime.
        """
        # Find the latest datetime from the active symbols
        latest_dt = None
        for symbol in self.symbol_list:
            dt = self.data_handler.get_latest_bar_datetime(symbol)
            if dt is not None:
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt

        if latest_dt is None:
            return

        # Snapshot current positions
        pos_record = {symbol: self.current_positions[symbol] for symbol in self.symbol_list}
        pos_record["datetime"] = latest_dt
        self.all_positions.append(pos_record)

        # Snapshot current holdings
        holdings_record = {symbol: 0.0 for symbol in self.symbol_list}
        holdings_record["datetime"] = latest_dt
        holdings_record["cash"] = self.current_cash
        holdings_record["commission"] = self.current_commission

        total_market_value = 0.0
        for symbol in self.symbol_list:
            qty = self.current_positions[symbol]
            close_price = self.data_handler.get_latest_bar_value(symbol, "close") or 0.0
            val = qty * close_price
            self.current_holdings[symbol] = val
            holdings_record[symbol] = val
            total_market_value += val

        holdings_record["total"] = self.current_cash + total_market_value
        self.all_holdings.append(holdings_record)

    def update_signal(self, event: SignalEvent) -> List[OrderEvent]:
        """
        Acts on a SignalEvent to generate a list of MKT (Market) OrderEvents
        using fixed-quantity order sizing rules.
        """
        orders: List[OrderEvent] = []
        symbol = event.symbol
        direction = None

        if event.signal_type == "LONG":
            direction = "BUY"
        elif event.signal_type == "SHORT":
            direction = "SELL"
        elif event.signal_type == "EXIT":
            # Exit position: close current position entirely
            cur_qty = self.current_positions[symbol]
            if cur_qty > 0:
                orders.append(
                    OrderEvent(
                        symbol=symbol,
                        order_type="MKT",
                        quantity=cur_qty,
                        direction="SELL",
                    )
                )
            elif cur_qty < 0:
                orders.append(
                    OrderEvent(
                        symbol=symbol,
                        order_type="MKT",
                        quantity=abs(cur_qty),
                        direction="BUY",
                    )
                )
            return orders

        if direction:
            orders.append(
                OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=self.fixed_quantity,
                    direction=direction,
                )
            )

        return orders

    def update_fill(self, event: FillEvent) -> None:
        """
        Updates the portfolio's cash, positions, and cumulative commission from a FillEvent.
        """
        symbol = event.symbol
        qty = event.quantity
        direction = event.direction
        fill_cost = event.fill_cost  # Executed price per share
        commission = event.commission or 0.0

        if direction == "BUY":
            self.current_positions[symbol] += qty
            self.current_cash -= (qty * fill_cost) + commission
        elif direction == "SELL":
            self.current_positions[symbol] -= qty
            self.current_cash += (qty * fill_cost) - commission

        self.current_commission += commission

    def create_equity_curve_dataframe(self) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from all_holdings with daily returns and
        a normalized equity curve.
        """
        df = pd.DataFrame(self.all_holdings)
        if not df.empty:
            df.set_index("datetime", inplace=True)
            df["returns"] = df["total"].pct_change()
            df["equity_curve"] = (1.0 + df["returns"].fillna(0.0)).cumprod()
        return df

