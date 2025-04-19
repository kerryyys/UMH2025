from pathlib import Path
from api_config import API_KEY, YEARS, LINKS
from data_structure import LinkBuilder, TopicMap
from data_fetcher import CybotradeCryptoDataFetcher
import asyncio
import cybotrade_datasource
from datetime import datetime, timezone
import pandas as pd

# if __name__ == "__main__":
#     btc_fetcher = CybotradeCryptoDataFetcher(
#         api_key=API_KEY,
#         crypto="btc",
#         years=YEARS,
#         links=LINKS,
#         output_dir=Path("data"),
#         concurrency_limit=5
#     )

#     data = asyncio.run(cybotrade_datasource.query_paginated(
#         api_key=API_KEY,
#         topic="cryptoquant|btc/market-data/price-ohlcv?exchange=binance&window=hour",
#         start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
#         end_time=datetime(2025, 1, 1, tzinfo=timezone.utc)
#     ))
#     df = pd.DataFrame(data)
#     path = Path("ohlcv.csv")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, index=False)

#     # btc_fetcher.run()


async def main():
    data = await cybotrade_datasource.query_paginated(
        api_key=API_KEY,
        topic="cryptoquant|btc/market-data/price-ohlcv?exchange=binance&window=hour",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    df = pd.DataFrame(data)
    path = Path("ohlcv.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    asyncio.run(main())
