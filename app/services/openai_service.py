import requests
from config import OPENAI_API_KEY

def ask_openai(question: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/engines/davinci-codex/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "prompt": question,
            "max_tokens": 100
        }
    )
    response_data = response.json()
    return response_data['choices'][0]['text'].strip()
