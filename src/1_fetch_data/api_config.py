from datetime import datetime, timezone
from data_structure import TopicMap
from typing import Dict
from data_structure import LinkBuilder
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Callable
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List

import httpx
from lxml import html
from tqdm.asyncio import tqdm_asyncio

# ──────────────────────────────────────────────────────────────────────────────
# 0) cybotrade_datasource API Configurations
# ──────────────────────────────────────────────────────────────────────────────

API_KEY = "p85PodLHYEP2zIfquwiYUgRejWRl40tnKsVFza6peLfju4eg"

YEARS = {
    "2022-2023": (datetime(2022, 4, 15, tzinfo=timezone.utc), datetime(2023, 4, 15, tzinfo=timezone.utc)),
    "2023-2024": (datetime(2023, 4, 16, tzinfo=timezone.utc), datetime(2024, 4, 16, tzinfo=timezone.utc)),
    "2024-2025": (datetime(2024, 4, 17, tzinfo=timezone.utc), datetime(2025, 4, 17, tzinfo=timezone.utc)),
}

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)

# ──────────────────────────────────────────────────────────────────────────────
# 1) CryptoQuant API Configurations
# ──────────────────────────────────────────────────────────────────────────────

# No need to change this section, it is already fully configured.
CRYPTOQUANT_PARAMS = {
    "PROVIDER":     "cryptoquant",
    "ON_CHAIN_COIN": "btc",
    "EXCHANGE":     "binance",
    "WINDOW":       "hour",
}

# No need to change this section, it is already fully configured.


def make_cryptoquant_link(category: str, endpoint: str) -> str:
    p = CRYPTOQUANT_PARAMS
    return (
        f"{p['PROVIDER']}|{p['ON_CHAIN_COIN']}/"
        f"{category}/{endpoint}"
        f"?exchange={p['EXCHANGE']}&window={p['WINDOW']}"
    )


# [WARNING]
# You can modify this as needed.
# Note that API endpoints that don't have `exchange` and `window` as query parameters are not allowed.
# An error will be raised if you attempt to use them.
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

# ──────────────────────────────────────────────────────────────────────────────
# 2) CryptoQuant API Configurations
# ──────────────────────────────────────────────────────────────────────────────

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


# [WARNING]
# You can modify this as needed.
# Note that API endpoints that don't have `exchange` and `window` as query parameters are not allowed.
# An error will be raised if you attempt to use them.
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
# 3) GlassNode API Configurations
# ──────────────────────────────────────────────────────────────────────────────

GLASSNODE_PARAMS = {
    "PROVIDER":       "glassnode",
    "WINDOW":         "1h",
    "CRYPTO": "BTC",
}


def make_glassnode_link(category: str, endpoint: str) -> str:
    p = GLASSNODE_PARAMS
    base = f"{p['PROVIDER']}|{category}/{endpoint}"
    qs = f"?a={p['CRYPTO']}&i={p['WINDOW']}"
    return base + qs


def fetch_html(url: str) -> str:
    """Fetch the HTML content from the given URL, raising an error on failure."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def fetch_data_from_glassnode():
    # Base URLs for the docs and the API endpoints.
    base_url = "https://docs.glassnode.com/basic-api/endpoints"
    docs_url = f"{base_url}/indicators"  # The docs page to scan for endpoints.
    base_filter_path = "https://api.glassnode.com/v1/metrics"

    # 1. Fetch and parse the indicators documentation page to extract endpoints.
    content = fetch_html(docs_url)
    soup = BeautifulSoup(content, "html.parser")

    base_path = "/basic-api/endpoints/"
    endpoints = set()

    # Extract endpoints from <a> tags with href starting with the expected base path.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(base_path):
            # Remove the base path and any leading/trailing slashes.
            endpoint = href[len(base_path):].strip("/")
            if endpoint:
                endpoints.add(endpoint)

    print("Endpoints found:")
    print(endpoints)
    # 2. For each endpoint, fetch its page and extract metric names.
    links: Dict[str, List[str]] = {}

    for endpoint in endpoints:
        print(f"Fetching metrics for endpoint: {endpoint}")
        endpoint_url = f"{base_url}/{endpoint}"
        filter_path = f"{base_filter_path}/{endpoint}/"

        endpoint_content = fetch_html(endpoint_url)
        endpoint_soup = BeautifulSoup(endpoint_content, "html.parser")

        formatted_html = endpoint_soup.prettify()

        # Split the formatted HTML into lines
        lines = formatted_html.splitlines()

        # Filter lines: keep only those that contain FILTER_STR and do not contain "{}"
        filtered = [
            line.strip()[len(filter_path):] for line in lines if filter_path in line and "{" not in line]

        links[endpoint] = filtered

    return links


GLASSNODE_LINKS: TopicMap = {
    "glassnode": fetch_data_from_glassnode()
}


def build_topics(links: TopicMap, link_builder: LinkBuilder) -> Dict[str, str]:
    topics: Dict[str, str] = {}
    for provider, cats in links.items():
        for category, endpoints in cats.items():
            for ep in endpoints:
                name = f"{provider}_{category}_{ep}"
                topics[name] = link_builder(category, ep)
    return topics


LINKS = {
    **build_topics(CRYPTOQUANT_LINKS, make_cryptoquant_link),
    **build_topics(COINGLASS_LINKS, make_coinglass_link),
    **build_topics(GLASSNODE_LINKS, make_glassnode_link)
}
