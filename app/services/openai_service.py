from openai import OpenAI
from config import OPENAI_API_KEY

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

def classify_with_gpt_3_5_turbo(text, categories):
    return classify_inquiry_zero_shot(text, categories, model="gpt-3.5-turbo")

def classify_with_gpt_4o(text, categories):
    return classify_inquiry_zero_shot(text, categories, model="gpt-4o")

def classify_inquiry_zero_shot(text, categories, model):
    categories_str = ", ".join(categories)
    messages = [
        {"role": "system", "content": f"You are a text classifier. Classify the following text into one of these categories: {categories_str}."},
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0.7,
        top_p=1.0,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    return classification

def classify_with_gpt_3_5_turbo_few_shot(text, categories):
    return classify_inquiry_few_shot(text, categories, model="gpt-3.5-turbo")

def classify_with_gpt_4o_few_shot(text, categories):
    return classify_inquiry_few_shot(text, categories, model="gpt-4o")

def classify_inquiry_few_shot(text, categories, model):
    categories_str = ", ".join(categories)
    examples = [
        {"text": "How do I locate my card?", "category": "card_arrival"},
        {"text": "I'm trying to figure out the current exchange rate.", "category": "exchange_rate"},
        {"text": "Why am I being charged more?", "category": "card_payment_wrong_exchange_rate"},
        {"text": "Send my card as soon as you are able to.", "category": "card_delivery_estimate"},
        {"text": "What can you do to unblock my pin?", "category": "pin_blocked"}
    ]

    example_messages = []
    for example in examples:
        example_messages.append({"role": "user", "content": f"Text: '{example['text']}'"})
        example_messages.append({"role": "assistant", "content": f"Category: {example['category']}"})
    
    messages = [
        {"role": "system", "content": f"You are a text classifier. Classify the following text into one of these categories: {categories_str}. Here are some examples:"},
    ] + example_messages + [
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0.7,
        top_p=1.0,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    return classification