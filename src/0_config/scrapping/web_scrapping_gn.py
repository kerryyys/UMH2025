import requests
from bs4 import BeautifulSoup
from typing import Dict, List


def fetch_html(url: str) -> str:
    """Fetch the HTML content from the given URL, raising an error on failure."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


if __name__ == "__main__":
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

        metric_endpoints = []

        formatted_html = endpoint_soup.prettify()

        # Split the formatted HTML into lines
        lines = formatted_html.splitlines()

        # Filter lines: keep only those that contain FILTER_STR and do not contain "{}"
        filtered = [
            line.strip()[len(filter_path):] for line in lines if filter_path in line and "{" not in line]

        links[endpoint] = filtered

    # Print the dictionary of endpoints with their metrics.
    print(links)
