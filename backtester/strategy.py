from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from .event import MarketEvent, SignalEvent, EventType


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


class MovingAverageCrossover(Strategy):
    """
    MovingAverageCrossover strategy is a classic momentum strategy that
    generates a LONG signal when a short-term simple moving average (SMA)
    crosses above a long-term SMA, and generates an EXIT signal to flatten
    the position when the short-term SMA crosses back below the long-term SMA.
    """

    def __init__(self, data_handler, short_window: int = 5, long_window: int = 20):
        """
        Initializes the crossover strategy.

        Args:
            data_handler: The DataHandler instance to fetch historic prices.
            short_window (int): Lookback period for the short moving average.
            long_window (int): Lookback period for the long moving average.
        """
        self.data_handler = data_handler
        self.symbol_list = self.data_handler.symbol_list
        self.short_window = short_window
        self.long_window = long_window

        # Map symbol -> invested state (True if holding position, False if flat)
        # Prevents redundant signal generation
        self.invested: Dict[str, bool] = {symbol: False for symbol in self.symbol_list}

    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """
        Calculates short-term and long-term SMAs for each symbol and
        emits LONG/EXIT signals on SMA crossovers.
        """
        signals: List[SignalEvent] = []

        if event.type != EventType.MARKET:
            return signals

        for symbol in self.symbol_list:
            # Fetch latest bars required to calculate the long moving average
            bars = self.data_handler.get_latest_bars(symbol, N=self.long_window)
            
            if len(bars) >= self.long_window:
                # Extracted closing prices (index 4 in the bar tuple)
                long_closes = [bar[4] for bar in bars]
                short_closes = long_closes[-self.short_window:]

                # Compute SMAs using numpy
                short_sma = float(np.mean(short_closes))
                long_sma = float(np.mean(long_closes))

                # Latest bar datetime for recording the signal
                latest_dt_str = str(bars[-1][0])

                # Check crossover logic
                if short_sma > long_sma and not self.invested[symbol]:
                    signals.append(
                        SignalEvent(
                            symbol=symbol,
                            datetime=latest_dt_str,
                            signal_type="LONG",
                            strength=1.0,
                        )
                    )
                    self.invested[symbol] = True

                elif short_sma < long_sma and self.invested[symbol]:
                    signals.append(
                        SignalEvent(
                            symbol=symbol,
                            datetime=latest_dt_str,
                            signal_type="EXIT",
                            strength=1.0,
                        )
                    )
                    self.invested[symbol] = False

        return signals

