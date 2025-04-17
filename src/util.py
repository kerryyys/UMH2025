from typing import Dict


def build_topics(self) -> Dict[str, str]:
    topics: Dict[str, str] = {}
    for provider, cats in self.links.items():
        for category, endpoints in cats.items():
            for ep in endpoints:
                name = f"{provider}_{category}_{ep}"
                topics[name] = self.link_builder(category, ep)
    return topics
