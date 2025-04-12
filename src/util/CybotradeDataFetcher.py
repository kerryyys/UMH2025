import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List
import cybotrade_datasource
import pandas as pd

TopicMap = Dict[str, Dict[str, List[str]]]
LinkBuilder = Callable[[str, str], str]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class CybotradeDataFetcher():
    api_key: str
    start: datetime
    end: datetime
    links: TopicMap
    link_builder: LinkBuilder
    output_dir: Path = Path("data")

    semaphore: asyncio.Semaphore = field(init=False, repr=False)
    topics: Dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.topics = self._generate_topics()

    def _generate_topics(self) -> Dict[str, str]:
        topics = {}
        for provider, categories in self.links.items():
            for category, endpoints in categories.items():
                for endpoint in endpoints:
                    name = f"{provider}_{category}_{endpoint}"
                    topics[name] = self.link_builder(category, endpoint)
        return topics

    async def fetch_and_save(self, name: str, topic: str):
        async with self.semaphore:
            try:
                logging.info(f"Starting fetch: {name}")
                data = await cybotrade_datasource.query_paginated(
                    api_key=self.api_key,
                    topic=topic,
                    start_time=self.start,
                    end_time=self.end,
                )
                df = pd.DataFrame(data)
                file_path = self.output_dir / f"{name}.csv"
                df.to_csv(file_path, index=False)
                logging.info(f"Completed fetch and saved: {name}")
            except Exception as e:
                logging.error(f"Error fetching {name}: {e}")

    async def run(self):
        tasks = [self.fetch_and_save(name, topic)
                 for name, topic in self.topics.items()]
        await asyncio.gather(*tasks)
