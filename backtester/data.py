import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Generator, Optional
import pandas as pd

from .event import MarketEvent


class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for all subsequent
    (historic or live) data handlers.
    """

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Tuple]:
        """
        Returns the last bar updated for a symbol.
        """
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Tuple]:
        """
        Returns the last N bars updated for a symbol.
        """
        pass

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Returns the timestamp of the latest bar for a symbol.
        """
        pass

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """
        Returns one of open, high, low, close, volume, or adj_close of the latest bar.
        val_type should be one of 'open', 'high', 'low', 'close', 'volume', 'adj_close'.
        """
        pass

    @abstractmethod
    def update_bars(self) -> None:
        """
        Pushes the latest bar to the latest_symbol_data structure for all symbols.
        """
        pass

    @abstractmethod
    def generate_events(self) -> Generator[MarketEvent, None, None]:
        """
        Yields MarketEvents step-by-step as new market data arrives.
        """
        pass


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for a list of symbols
    from a directory and yield MarketEvents while maintaining latest symbol data.
    """

    def __init__(self, csv_dir: str, symbol_list: List[str]):
        """
        Initializes the historic CSV data handler by loading and formatting CSVs.
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.latest_symbol_data: Dict[str, List[Tuple]] = {symbol: [] for symbol in symbol_list}
        self.continue_backtest = True
        self.combined_index: List[pd.Timestamp] = []

        self._load_csv_data()
        self._create_datetime_index()

    def _load_csv_data(self) -> None:
        """
        Loads and formats the CSV files from the data directory.
        """
        combined_idx = None
        for symbol in self.symbol_list:
            file_path = os.path.join(self.csv_dir, f"{symbol}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found for symbol: {symbol} at {file_path}")

            # Load CSV with pandas
            df = pd.read_csv(
                file_path,
                header=0,
                index_col=0,
                parse_dates=True,
            )
            df.sort_index(inplace=True)

            # Standardize columns to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]

            # Verify required OHLC columns exist
            required = ["open", "high", "low", "close"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing in {file_path}")

            # Populate default volume/adj_close if missing
            if "volume" not in df.columns:
                df["volume"] = 0
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            self.symbol_data[symbol] = df

            # Form union of all datetimes across symbols
            if combined_idx is None:
                combined_idx = df.index
            else:
                combined_idx = combined_idx.union(df.index)

        self.combined_index = sorted(combined_idx)
        # Create an iterator over the index to simulate step updates
        self.datetime_iterator = iter(self.combined_index)

    def _create_datetime_index(self) -> None:
        """
        Pre-converts dataframe rows to lists of tuples for ultra-fast iteration.
        """
        self.symbol_bars: Dict[str, Dict[pd.Timestamp, Tuple]] = {}
        for symbol in self.symbol_list:
            df = self.symbol_data[symbol]
            bars_dict = {}
            for timestamp, row in df.iterrows():
                bars_dict[timestamp] = (
                    timestamp,
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    int(row["volume"]),
                    row["adj_close"],
                )
            self.symbol_bars[symbol] = bars_dict

    def get_latest_bar(self, symbol: str) -> Optional[Tuple]:
        """
        Returns the last bar updated for a symbol.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
            return bars_list[-1] if bars_list else None
        except KeyError:
            raise KeyError(f"Symbol '{symbol}' is not tracked by this DataHandler.")

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Tuple]:
        """
        Returns the last N bars updated for a symbol.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
            return bars_list[-N:]
        except KeyError:
            raise KeyError(f"Symbol '{symbol}' is not tracked by this DataHandler.")

    def get_latest_bar_datetime(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Returns the timestamp of the latest bar for a symbol.
        """
        bar = self.get_latest_bar(symbol)
        return bar[0] if bar else None

    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """
        Returns one of open, high, low, close, volume, or adj_close of the latest bar.
        """
        bar = self.get_latest_bar(symbol)
        if not bar:
            return None

        mapping = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
            "adj_close": 6,
        }
        try:
            col_idx = mapping[val_type.lower()]
            # Datetime is Timestamp, the rest are float/int
            return bar[col_idx] if col_idx != 0 else bar[col_idx]
        except KeyError:
            raise KeyError(f"Invalid column type '{val_type}'. Must be one of open, high, low, close, volume, adj_close.")

    def update_bars(self) -> None:
        """
        Pushes the latest bar to the latest_symbol_data structure for all symbols.
        """
        try:
            dt = next(self.datetime_iterator)
        except StopIteration:
            self.continue_backtest = False
            return

        for symbol in self.symbol_list:
            bar = self.symbol_bars[symbol].get(dt)
            if bar is not None:
                self.latest_symbol_data[symbol].append(bar)

    def generate_events(self) -> Generator[MarketEvent, None, None]:
        """
        Yields MarketEvents step-by-step as new market data arrives.
        """
        while self.continue_backtest:
            self.update_bars()
            if self.continue_backtest:
                yield MarketEvent()
