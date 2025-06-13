import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class DbConfig(BaseSettings):
    DB_USER: str = os.environ['DB_USER']
    DB_PASSWORD: str = os.environ['DB_PASSWORD']
    DB_HOST: str = os.environ['DB_HOST']
    DB_PORT: str = os.environ['DB_PORT']
    DB_NAME: str = os.environ['DB_NAME']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Database configuration loaded for host: {self.DB_HOST}")