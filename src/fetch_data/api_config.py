from datetime import datetime, timezone
from data_structure import TopicMap

# ──────────────────────────────────────────────────────────────────────────────
# 0) cybotrade_datasource API Configurations
# ──────────────────────────────────────────────────────────────────────────────

API_KEY = "p85PodLHYEP2zIfquwiYUgRejWRl40tnKsVFza6peLfju4eg"
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

# No need to change this section, it is already fully configured.
COINGLASS_PARAMS = {
    "PROVIDER":       "coinglass",
    "EXCHANGE":       "Binance",
    "WINDOW":         "1h",
    "ON_CHAIN_COIN": "BTCUSDT",
}

# No need to change this section, it is already fully configured.


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
