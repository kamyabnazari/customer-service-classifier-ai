import json
import os
import re
from syslog import LOG_PERROR
from fastapi import HTTPException
from openai import OpenAI, OpenAIError
from app.core.config import settings
from app.dependencies import get_database
from app.models.database import predictions
from sqlalchemy import select
import time
import datetime
from typing import List, Dict

client = OpenAI(api_key=settings.openai_api_key)
database = get_database()

async def insert_prediction(database: database, name: str, request: str, response: str, status: str):
    query = predictions.insert().values(
        name=name,
        request=request,
        response=response,
        created_at=datetime.now(),
        status=status
    )
    try:
        last_record_id = await database.execute(query)
        return last_record_id
    except Exception as e:
        logging.error(f"Failed to insert prediction data: {e}")
        raise

def generate_joke(prompt: str):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {settings.openai_api_key}'
    }
    data = {
        "model": "text-davinci-002",
        "prompt": prompt,
        "max_tokens": 50
    }
    response = requests.post('https://api.openai.com/v1/completions', json=data, headers=headers)
    return response.json()