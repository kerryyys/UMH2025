import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict
from data_structure import TopicMap, LinkBuilder
from typing import List, Tuple, Callable
import pandas as pd
import cybotrade_datasource


@dataclass
class CybotradeDataFetcher:
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


@dataclass
class CybotradeCryptoDataFetcher:
    api_key: str
    crypto: str
    years: Dict[str, Tuple[datetime, datetime]]
    links: Dict[str, str] = field(default_factory=dict)
    output_dir: Path = Path("data")
    concurrency_limit: int = 5

    # Semaphore will be initialized in __post_init__
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)

    async def _fetch_and_save(self, name: str, provider: str, topic: str):
        async with self._semaphore:
            for year, (start, end) in self.years.items():
                try:
                    data = await cybotrade_datasource.query_paginated(
                        api_key=self.api_key,
                        topic=topic,
                        start_time=start,
                        end_time=end,
                    )
                    df = pd.DataFrame(data)
                    path = self.output_dir / \
                        f"{self.crypto}/{year}/{provider}/{name}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path, index=False)
                    print(f"Saved data for {name} for year {year} at {path}")
                except Exception as e:
                    print(f"Error fetching {name} for year {year}: {e}")

    async def run(self):
        """Kick off all fetch tasks and wait for their completion."""
        tasks = []
        # Iterate through each topic in the links dictionary
        for name, topic in self.links.items():
            # Assuming the topic name is in the format "provider_category_endpoint"
            provider = name.split('_')[0]
            print(provider)
            tasks.append(asyncio.create_task(
                self._fetch_and_save(name, provider, topic)))
        await asyncio.gather(*tasks)


def build_topics(links: Dict[str, Dict[str, list]], link_builder: Callable[[str, str], str]) -> Dict[str, str]:
    """
    Build a dictionary mapping topic names to URLs using the provided link builder function.

    Parameters:
        links (Dict[str, Dict[str, list]]): A nested dictionary where:
            - Keys are providers.
            - Values are dictionaries mapping categories to lists of endpoints.
        link_builder (Callable): A function that takes a category and an endpoint to return a link.

    Returns:
        Dict[str, str]: A dictionary where keys are in the format "provider_category_endpoint" and
                        values are the corresponding URLs.
    """
    topics = {}
    for provider, categories in links.items():
        for category, endpoints in categories.items():
            for ep in endpoints:
                name = f"{provider}_{category}_{ep}"
                topics[name] = link_builder(category, ep)
    return topics
