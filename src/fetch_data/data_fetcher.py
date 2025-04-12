import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict
from data_structure import TopicMap, LinkBuilder

import pandas as pd
import cybotrade_datasource


class DataFetcher(ABC):
    """
    Abstract base class for data fetchers.
    """
    @abstractmethod
    async def run(self):
        """
        Run the data fetching process.
        """
        pass


@dataclass
class CybotradeDataFetcher:
    """
    A generic asynchronous fetcher that:
     - Builds topic URLs from a nested dict of {provider: {category: [endpoints]}}
     - Queries each topic via cybotrade_datasource.query_paginated
     - Saves each result to CSV under output_dir/provider_category_endpoint.csv
    """
    api_key: str
    start: datetime
    end: datetime
    links: TopicMap
    link_builder: LinkBuilder
    output_dir: Path = Path("data")
    concurrency_limit: int = 5

    # Initialized in __post_init__
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _topics: Dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        self._topics = self._build_topics()

    def _build_topics(self) -> Dict[str, str]:
        topics: Dict[str, str] = {}
        for provider, cats in self.links.items():
            for category, endpoints in cats.items():
                for ep in endpoints:
                    name = f"{provider}_{category}_{ep}"
                    topics[name] = self.link_builder(category, ep)
        return topics

    async def _fetch_and_save(self, name: str, topic: str):
        async with self._semaphore:
            try:
                data = await cybotrade_datasource.query_paginated(
                    api_key=self.api_key,
                    topic=topic,
                    start_time=self.start,
                    end_time=self.end,
                )
                df = pd.DataFrame(data)
                path = self.output_dir / f"{name}.csv"
                df.to_csv(path, index=False)
            except Exception:
                raise

    async def run(self):
        """Kick off all fetch tasks and wait for completion."""
        tasks = [
            asyncio.create_task(self._fetch_and_save(name, url))
            for name, url in self._topics.items()
        ]
        await asyncio.gather(*tasks)
