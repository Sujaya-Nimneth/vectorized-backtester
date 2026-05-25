from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class EventType(Enum):
    """
    Supported event types in the backtesting system.
    """
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

@dataclass
class Event:
    """
    Event is base class providing an interface for all subsequent
    (historic or live) events triggered in the backtester.
    """
    type: EventType

@dataclass
class MarketEvent(Event):
    """
    Handles the event of receiving a new market update (e.g. bar/tick).
    """
    type: EventType = field(init=False, default=EventType.MARKET)

@dataclass
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy to a Portfolio.
    """
    symbol: str
    datetime: str  # DateTime string or object representing the signal generation time
    signal_type: str  # 'LONG', 'SHORT', or 'EXIT'
    strength: float = 1.0
    type: EventType = field(init=False, default=EventType.SIGNAL)

@dataclass
class OrderEvent(Event):
    """
    Handles the event of sending an Order from a Portfolio to an Execution system.
    """
    symbol: str
    order_type: str  # 'MKT' (Market), 'LMT' (Limit)
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    type: EventType = field(init=False, default=EventType.ORDER)

@dataclass
class FillEvent(Event):
    """
    Handles the event of a Filled Order from an Execution system.
    Stores fill cost, execution quantity, direction, and commission.
    """
    symbol: str
    datetime: str  # Execution datetime
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    fill_cost: float  # Price at which the order was filled
    commission: Optional[float] = None
    type: EventType = field(init=False, default=EventType.FILL)
