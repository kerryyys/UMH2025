 # Configuration file (stores API key, endpoints, etc.)

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("CYBOTRADE_API_KEY")
API_SECRET = "secretkey"
BASE_URL = "https://api.datasource.cybotrade.rs"