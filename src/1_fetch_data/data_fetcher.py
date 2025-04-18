import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict
from data_structure import TopicMap, LinkBuilder
from typing import List, Tuple, Callable
import pandas as pd
import cybotrade_datasource


import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import cybotrade_datasource
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


@dataclass
class TaskInfo:
    name: str
    provider: str
    topic: str
    year: str
    start: datetime
    end: datetime


class CybotradeCryptoDataFetcher:
    def __init__(self, api_key: str, crypto: str, years: Dict[str, Tuple[datetime, datetime]], links: Dict[str, str], output_dir: Path = Path("data"), concurrency_limit: int = 5):
        self.api_key = api_key
        self.crypto = crypto
        self.years = years
        self.links = links
        self.output_dir = output_dir
        self.concurrency_limit = concurrency_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_tasks(self) -> List[TaskInfo]:
        tasks = []
        for name, topic in self.links.items():
            provider = name.split("_", 1)[0]
            for year, (start, end) in self.years.items():
                tasks.append(TaskInfo(name, provider, topic, year, start, end))
        return tasks

    async def _download(self, task: TaskInfo, semaphore: asyncio.Semaphore, progress: Progress, progress_id: int, status_panel: Text):
        async with semaphore:
            try:
                data = await cybotrade_datasource.query_paginated(api_key=self.api_key, topic=task.topic, start_time=task.start, end_time=task.end)
                df = pd.DataFrame(data)
                path = self.output_dir / self.crypto / \
                    task.year / task.provider / f"{task.name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                status_panel.update(f"[green]Downloaded:[/] {path}")
            except Exception as e:
                status_panel.update(
                    f"[red]Error:[/] {task.name} ({task.year}): {e}")
            finally:
                progress.advance(progress_id)

    def run(self):
        tasks_info = self._build_tasks()
        total = len(tasks_info)
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        )
        status_panel = Text("Last downloaded: N/A")
        layout = Layout()
        layout.split_column(
            Layout(Panel(progress, title="Progress"), size=3),
            Layout(Panel(status_panel, title="Status"), ratio=1),
        )

        async def supervisor():
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            progress_id = progress.add_task("Downloading", total=total)
            await asyncio.gather(*(self._download(task, semaphore, progress, progress_id, status_panel) for task in tasks_info))
        with Live(layout, console=console, refresh_per_second=10):
            asyncio.run(supervisor())
