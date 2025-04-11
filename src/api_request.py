import nest_asyncio
import asyncio
from datetime import datetime, timezone
from cybotrade.models import RuntimeConfig, RuntimeMode
from cybotrade.permutation import Permutation
from my_strategy import MyStrategy
from config import API_KEY, API_SECRET
import pandas as pd

nest_asyncio.apply()

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[
        "cryptoquant|btc/exchange-flows/inflow?exchange=okx&window=hour"
    ],
    candle_topics=[],
    active_order_interval=15,
    start_time=datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 5, 2, 0, 0, 0, tzinfo=timezone.utc),
    data_count=50,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

async def collect_data_for_training():
    permutation = Permutation(config)
    hyper_parameters = {"dummy": [1]}

    try:
        # Attempt to initialize the strategy with permutation
        strategy: MyStrategy = await permutation.run(hyper_parameters, MyStrategy)

        # If strategy initialization fails, handle it
        if strategy is None:
            print("ERROR: Strategy is None — initialization failed.")
            return pd.DataFrame()  # Return empty DataFrame as fallback
        
        print("Strategy initialized successfully.")
        df = strategy.get_training_data()
        return df

    except Exception as e:
        # Catching any exception during strategy initialization
        print(f"ERROR: An error occurred during strategy initialization: {e}")
        return pd.DataFrame()  # Return empty DataFrame as fallback

async def main():
    # Collect data and ensure we return an empty DataFrame on failure
    df = await collect_data_for_training()

    if df.empty:
        print("No data retrieved. Exiting...")
    else:
        print(f"\n✅ Retrieved {len(df)} rows")
        print(df.head())
        df.to_csv("training_data.csv", index=False)
        print("Saved training data to training_data.csv")

if __name__ == "__main__":
    # Ensure async code runs properly within the event loop
    asyncio.run(main())
