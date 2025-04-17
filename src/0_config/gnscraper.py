#!/usr/bin/env python3
import asyncio
import json
import re
from pathlib import Path

import httpx
from lxml import html
from tqdm import tqdm
from colorama import init as colorama_init, Fore, Style

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL = "https://docs.glassnode.com/basic-api/endpoints"
OUTPUT_FILE = Path("src/0_config/glassnode_links.json")
TIMEOUT = 20  # seconds

# ── SETUP ─────────────────────────────────────────────────────────────────────
colorama_init(autoreset=True)


async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a URL or raise on error, with a short timeout."""
    resp = await client.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


async def scrape_glassnode() -> dict[str, list[str]]:
    # If we've already saved once, reload and skip scraping.
    if OUTPUT_FILE.exists():
        print(Fore.CYAN + "Loading cached data from", OUTPUT_FILE)
        return json.loads(OUTPUT_FILE.read_text())

    async with httpx.AsyncClient() as client:
        # 1) get the list of endpoints
        print(Fore.YELLOW + "⚙️  Fetching index page…")
        index_html = await fetch_page(client, f"{BASE_URL}/indicators")
        tree = html.fromstring(index_html)
        raw_hrefs = tree.xpath(
            '//a[contains(@href, "/basic-api/endpoints/")]/@href')
        endpoints = {href.rsplit("/", 1)[1] for href in raw_hrefs}

        print(Fore.YELLOW +
              f"🔗 Found {len(endpoints)} endpoints, scraping metrics…")

        # prepare tasks
        async def fetch_endpoint(ep: str) -> tuple[str, list[str]]:
            url = f"{BASE_URL}/{ep}"
            page = await fetch_page(client, url)
            prefix = f"https://api.glassnode.com/v1/metrics/{ep}/"

            # capture everything after the prefix up to any quote/space/angle
            raw = re.findall(rf'{re.escape(prefix)}([^"\'\s<>]+)', page)
            # strip all leading/trailing slashes, remove templates, dedupe & sort
            clean = sorted({
                m.strip("/\\")
                for m in raw
                if "{" not in m
            })
            return ep, clean

        # run with progress bar
        results = {}
        tasks = [fetch_endpoint(ep) for ep in endpoints]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Endpoints"):
            ep, metrics = await coro
            results[ep] = metrics

    # cache out
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(Fore.MAGENTA + f"💾 Saved to {OUTPUT_FILE}")

    return results

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = asyncio.run(scrape_glassnode())
    print(Style.BRIGHT + Fore.BLUE +
          f"Completed: {len(data)} endpoints total.")
