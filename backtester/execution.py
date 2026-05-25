from abc import ABC, abstractmethod
from typing import List

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
