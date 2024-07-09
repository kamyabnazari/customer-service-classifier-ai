from openai import OpenAI
from config import settings
from dependencies import get_database

client = OpenAI(api_key=settings.openai_api_key)
database = get_database()