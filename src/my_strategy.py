import logging
import colorlog
import pandas as pd
from cybotrade.strategy import Strategy

class MyStrategy(Strategy):
    def __init__(self):
        print("Initializing MyStrategy...")  # Add logging here
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{MyStrategy.LOG_FORMAT}")
        )
        super().__init__(log_level=logging.INFO, handlers=[handler])
        self.collected_data = []
        print("MyStrategy initialized successfully")  # Add logging after initialization
