import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import cybotrade_datasource


# ──────────────────────────────────────────────────────────────────────────────
# 1) Shared Types & Helpers
# ──────────────────────────────────────────────────────────────────────────────

TopicMap = Dict[str, Dict[str, List[str]]]
LinkBuilder = Callable[[str, str], str]


# ──────────────────────────────────────────────────────────────────────────────
# 2) Generic Async DataFetcher
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataFetcher:
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
        # ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # limit concurrent requests
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        # build all topics once
        self._topics = self._build_topics()
        # basic logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

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
                logging.info(f"▶ Fetching {name}")
                data = await cybotrade_datasource.query_paginated(
                    api_key=self.api_key,
                    topic=topic,
                    start_time=self.start,
                    end_time=self.end,
                )
                df = pd.DataFrame(data)
                path = self.output_dir / f"{name}.csv"
                df.to_csv(path, index=False)
                logging.info(f"✔ Saved {name} → {path}")
            except Exception:
                logging.exception(f"✖ Error fetching {name}")

    async def run(self):
        """Kick off all fetch tasks and wait for completion."""
        tasks = [
            asyncio.create_task(self._fetch_and_save(name, url))
            for name, url in self._topics.items()
        ]
        await asyncio.gather(*tasks)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Provider‑specific Link Builders & Configs
# ──────────────────────────────────────────────────────────────────────────────

API_KEY = "p85PodLHYEP2zIfquwiYUgRejWRl40tnKsVFza6peLfju4eg"
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)


CRYPTOQUANT_PARAMS = {
    "PROVIDER":     "cryptoquant",
    "ON_CHAIN_COIN": "btc",
    "EXCHANGE":     "binance",
    "WINDOW":       "hour",
}


def make_cryptoquant_link(category: str, endpoint: str) -> str:
    p = CRYPTOQUANT_PARAMS
    return (
        f"{p['PROVIDER']}|{p['ON_CHAIN_COIN']}/"
        f"{category}/{endpoint}"
        f"?exchange={p['EXCHANGE']}&window={p['WINDOW']}"
    )


CRYPTOQUANT_LINKS: TopicMap = {
    "cryptoquant": {
        "market-data": [
            "price-ohlcv", "open-interest", "funding-rates",
            "taker-buy-sell-stats", "liquidations",
        ],
        "exchange-flows": [
            "reserve", "netflow", "inflow", "outflow",
            "transactions-count", "in-house-flow",
        ],
        "flow-indicator": [
            "exchange-shutdown-index",
            "exchange-whale-ratio",
            "exchange-supply-ratio",
        ],
    }
}


COINGLASS_PARAMS = {
    "PROVIDER":       "coinglass",
    "EXCHANGE":       "Binance",
    "WINDOW":         "1h",
    "ON_CHAIN_COIN": "BTCUSDT",
}


def make_coinglass_link(category: str, endpoint: str) -> str:
    p = COINGLASS_PARAMS
    base = f"{p['PROVIDER']}|futures/{category}/{endpoint}"
    qs = f"?exchange={p['EXCHANGE']}&symbol={p['ON_CHAIN_COIN']}&interval={p['WINDOW']}"
    return base + qs


COINGLASS_LINKS: TopicMap = {
    "coinglass": {
        "openInterest": [
            "ohlc-history", "ohlc-aggregated-history"
        ],
        "fundingRate": [
            "ohlc-history", "oi-weight-ohlc-history", "vol-weight-ohlc-history"
        ],
        "globalLongShortAccountRatio": ["history"],
        "topLongShortAccountRatio":    ["history"],
        "topLongShortPositionRatio":   ["history"],
        "aggregatedTakerBuySellVolumeRatio": ["history"],
        "takerBuySellVolume":          ["history"],
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# 4) Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    cq_fetcher = DataFetcher(
        api_key=API_KEY,
        start=START,
        end=END,
        links=CRYPTOQUANT_LINKS,
        link_builder=make_cryptoquant_link,
        output_dir=Path("data/cryptoquant"),
        concurrency_limit=5,
    )
    cg_fetcher = DataFetcher(
        api_key=API_KEY,
        start=START,
        end=END,
        links=COINGLASS_LINKS,
        link_builder=make_coinglass_link,
        output_dir=Path("data/coinglass"),
        concurrency_limit=5,
    )

    # run both in parallel
    await asyncio.gather(
        cq_fetcher.run(),
        cg_fetcher.run(),
    )


asyncio.run(main())
