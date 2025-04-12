from typing import Callable, Dict, List

TopicMap = Dict[str, Dict[str, List[str]]]
LinkBuilder = Callable[[str, str], str]
