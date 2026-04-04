import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    proxy_url: str = os.getenv("PROXY_URL", "")
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000/api/process")


settings = Settings()
