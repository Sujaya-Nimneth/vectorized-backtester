from abc import ABC, abstractmethod
from typing import List

from .event import MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Strategy is an abstract base class providing an interface for all subsequent
    (historic or live) trading strategies.
    
    The goal of a Strategy object is to monitor market data and generate signals
    when predefined trading logic triggers them.
    """

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """
        Calculates the signals for the strategy based on the receipt of a MarketEvent.
        
        Args:
            event (MarketEvent): The event indicating new market data has arrived.
            
        Returns:
            List[SignalEvent]: A list of generated trading signals.
        """
        pass
