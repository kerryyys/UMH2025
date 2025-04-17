from pathlib import Path
from api_config import API_KEY, YEARS, LINKS
from data_structure import LinkBuilder, TopicMap
from data_fetcher import CybotradeCryptoDataFetcher

if __name__ == "__main__":
    btc_fetcher = CybotradeCryptoDataFetcher(
        api_key=API_KEY,
        crypto="btc",
        years=YEARS,
        links=LINKS,
        output_dir=Path("data"),
        concurrency_limit=5
    )

    btc_fetcher.run()
