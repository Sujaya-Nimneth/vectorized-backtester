from abc import ABC, abstractmethod
from typing import List

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
