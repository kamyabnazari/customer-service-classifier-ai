from openai import OpenAI
from config import OPENAI_API_KEY

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

def classify_with_gpt_3_5_turbo(text):
    return classify_inquiry(text, model="gpt-3.5-turbo")

def classify_with_gpt_4(text):
    return classify_inquiry(text, model="gpt-4")

def classify_with_gpt_4_turbo(text):
    return classify_inquiry(text, model="gpt-4-turbo")

def classify_with_gpt_4o(text):
    return classify_inquiry(text, model="gpt-4o")

def classify_inquiry(text, model):
    messages = [
        {"role": "system", "content": "You are a text classifier. Classify the following text into categories: Positive, Negative, Neutral."},
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=60,
        temperature=0.7,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    return classification