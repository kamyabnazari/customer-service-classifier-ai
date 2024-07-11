from openai import OpenAI
from config import OPENAI_API_KEY
from .logging_service import write_log_to_file, write_results_to_csv

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

def classify_with_gpt_3_5_turbo_zero_shot(text, categories, temperature):
    return classify_inquiry_zero_shot(text, categories, model="gpt-3.5-turbo", classification_type="zero_shot", temperature=temperature)

def classify_with_gpt_3_5_turbo_few_shot(text, categories, temperature):
    return classify_inquiry_few_shot(text, categories, model="gpt-3.5-turbo", classification_type="few_shot", temperature=temperature)

def classify_with_gpt_3_5_turbo_fine_zero_shot(text, categories, temperature):
    return classify_inquiry_zero_shot(text, categories, model="ft:gpt-3.5-turbo", classification_type="fine_zero_shot", temperature=temperature)

def classify_with_gpt_3_5_turbo_fine_few_shot(text, categories, temperature):
    return classify_inquiry_few_shot(text, categories, model="ft:gpt-3.5-turbo", classification_type="fine_few_shot", temperature=temperature)

def classify_inquiry_zero_shot(text, categories, model, classification_type, temperature):
    categories_str = ", ".join(categories)
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry into one of these categories: {categories_str}. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=temperature,
        top_p=1.0,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, "manual", temperature, system_message, text, classification, usage)
    write_results_to_csv(model, classification_type, "manual", temperature, text, classification, usage)
    return classification

def classify_inquiry_few_shot(text, categories, model, classification_type, temperature):
    categories_str = ", ".join(categories)
    examples = [
        {"text": "How do I locate my card?", "category": "card_arrival"},
        {"text": "I'm trying to figure out the current exchange rate.", "category": "exchange_rate"},
        {"text": "Why am I being charged more?", "category": "card_payment_wrong_exchange_rate"},
        {"text": "Send my card as soon as you are able to.", "category": "card_delivery_estimate"},
        {"text": "What can you do to unblock my pin?", "category": "pin_blocked"}
    ]
    
    examples_str = "\n".join([f"Example {i+1} - Inquiry: '{example['text']}' Answer: '{example['category']}'" for i, example in enumerate(examples)])
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry into one of these categories: {categories_str}. Here are some examples:\n{examples_str}. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=temperature,
        top_p=1.0,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, "manual", temperature, system_message, text, classification, usage)
    write_results_to_csv(model, classification_type, "manual", temperature, text, classification, usage)
    return classification
