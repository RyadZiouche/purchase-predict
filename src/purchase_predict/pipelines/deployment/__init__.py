from dotenv import load_dotenv
from .pipeline import create_pipeline

load_dotenv()  # <--- On charge les secrets du .env

__all__ = ["create_pipeline"]
__version__ = "0.1"
