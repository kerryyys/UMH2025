# config_manager.py
import json
from pathlib import Path
from gnscraper import scrape_glassnode

DEFAULT_CONFIG_PATH = Path("src/0_config/config.json")


async def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = json.loads(path.read_text())
    glassnode_links = await scrape_glassnode()
    config["glassnode"] = {
        "provider":   "glassnode",
        "crypto":     "BTC",
        "window":     "1h",
        "topics":     glassnode_links
    }
    return config
