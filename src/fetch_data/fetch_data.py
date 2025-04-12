import asyncio
from pathlib import Path
from api_config import *
from data_fetcher import CybotradeDataFetcher


async def main():
    cq_fetcher = CybotradeDataFetcher(
        api_key=API_KEY,
        start=START,
        end=END,
        links=CRYPTOQUANT_LINKS,
        link_builder=make_cryptoquant_link,
        output_dir=Path("data/cryptoquant"),
        concurrency_limit=5,
    )
    cg_fetcher = CybotradeDataFetcher(
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
