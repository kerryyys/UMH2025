from abc import ABC, abstractmethod


class DataFetcher(ABC):
    @abstractmethod
    def fetch_data(self):
        """
        Fetch data from a source.
        """
        pass
