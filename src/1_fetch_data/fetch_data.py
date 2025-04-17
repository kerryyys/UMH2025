import asyncio
from pathlib import Path
from api_config import *
from data_fetcher import CybotradeDataFetcher, CybotradeCryptoDataFetcher
from data_structure import LinkBuilder
from typing import Dict
import json


def build_topics(links: TopicMap, link_builder: LinkBuilder) -> Dict[str, str]:
    topics: Dict[str, str] = {}
    for provider, cats in links.items():
        for category, endpoints in cats.items():
            for ep in endpoints:
                name = f"{provider}_{category}_{ep}"
                topics[name] = link_builder(category, ep)
    return topics


async def main():
    # cq_fetcher = CybotradeDataFetcher(
    #     api_key=API_KEY,
    #     start=START,
    #     end=END,
    #     links=CRYPTOQUANT_LINKS,
    #     link_builder=make_cryptoquant_link,
    #     output_dir=Path("data/cryptoquant"),
    #     concurrency_limit=5,
    # )
    # cg_fetcher = CybotradeDataFetcher(
    #     api_key=API_KEY,
    #     start=START,
    #     end=END,
    #     links=COINGLASS_LINKS,
    #     link_builder=make_coinglass_link,
    #     output_dir=Path("data/coinglass"),
    #     concurrency_limit=5,
    # )

    # # run both in parallel
    # await asyncio.gather(
    #     cq_fetcher.run(),
    #     cg_fetcher.run(),
    # )

    btc_fetcher = CybotradeCryptoDataFetcher(
        api_key=API_KEY,
        crypto="btc",
        years=YEARS,
        concurrency_limit=5,
        links=LINKS
    )

    await asyncio.gather(
        btc_fetcher.run(),
    )

asyncio.run(main())
