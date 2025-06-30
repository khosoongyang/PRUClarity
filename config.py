
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o"
    MAX_SEARCH_RESULTS = 10
    SEARCH_LANGUAGE = "en"