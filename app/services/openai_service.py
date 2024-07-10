from openai import OpenAI
from config import OPENAI_API_KEY

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

def generate_response(prompt):
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"