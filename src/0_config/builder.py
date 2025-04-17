import asyncio
import json
from pathlib import Path
from config_manager import load_config

CONFIG_PATH = Path.cwd() / "src" / "0_config" / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Could not find config at {CONFIG_PATH!r}")
CONFIG = json.loads(CONFIG_PATH.read_text())


def make_link(provider: str, category: str, endpoint: str) -> str:
    cfg = CONFIG[provider]
    if provider == "cryptoquant":
        return f"{provider}|{cfg['coin']}/{category}/{endpoint}?exchange={cfg['exchange']}&window={cfg['window']}"
    if provider == "coinglass":
        return f"{provider}|futures/{category}/{endpoint}?exchange={cfg['exchange']}&symbol={cfg['symbol']}&interval={cfg['window']}"
    if provider == "glassnode":
        return f"{provider}|{category}/{endpoint}?a={cfg['crypto']}&i={cfg['window']}"


def build_topics() -> dict[str, str]:
    topics = {}
    for prov, info in CONFIG.items():
        for cat, eps in info["topics"].items():
            for ep in eps:
                key = f"{prov}_{cat}_{ep}"
                topics[key] = make_link(prov, cat, ep)
    return topics


async def main():
    cfg = await load_config()
    all_topics = build_topics(cfg)
    Path("links.json").write_text(json.dumps(all_topics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
